#!/usr/bin/env python3
"""Layerwise PPL evaluator for partial or full from-paper GSRQ codebooks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppl_utils import (  # noqa: E402
    capture_initial_hidden_states,
    layer_kwargs,
    load_eval_tokens,
)
from gsrq.cache_adapter import quantize_kv_heads  # noqa: E402
from gsrq.rope_surface import (  # noqa: E402
    key_quantization_input,
    key_states_for_attention,
    normalize_key_surface,
)


@dataclass
class PartialCodebookContext:
    layer: int
    head: int | None
    prefix_stages: int
    key_codebooks: torch.Tensor
    value_codebooks: torch.Tensor
    chunk_size: int
    codebook_bits: int
    current_token_fp16: bool
    key_surface: str
    provenance: dict[str, Any]
    key_assignment_whitener: torch.Tensor | None = None
    value_assignment_whitener: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(root / "models" / "swift" / "Meta-Llama-3-8B"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(root / "huggingface_cache_datasets" / "datasets"),
    )
    parser.add_argument(
        "--dataset", choices=["wikitext2", "c4"], default="wikitext2"
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fp16", "partial_gsrq", "full_gsrq"],
        default=["fp16", "partial_gsrq"],
    )
    parser.add_argument("--codebook-file")
    parser.add_argument(
        "--codebook-dir",
        help="Directory containing layer_00.pt ... layer_31.pt artifacts.",
    )
    parser.add_argument(
        "--key-override-dir",
        help=(
            "Optional M20 transformed-K artifact directory. Its raw K decode "
            "tables and frozen assignment whiteners replace only K; V remains "
            "from --codebook-dir."
        ),
    )
    parser.add_argument("--prefix-stages", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--quantize-current-token",
        action="store_true",
        help=(
            "Ablation: quantize the current query's own K/V before attention. "
            "The default quantizes only resident past K/V."
        ),
    )
    parser.add_argument(
        "--finite-diagnostics",
        action="store_true",
        help=(
            "Emit per-layer finite/max-absolute hidden-state diagnostics and "
            "fail at the first non-finite layer. Disabled for formal runs."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_partial_context(
    path: Path,
    prefix_stages: int,
    chunk_size: int,
    current_token_fp16: bool = True,
) -> PartialCodebookContext:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    provenance = artifact["provenance"]
    key_codebooks = artifact["artifacts"]["key"]["codebooks"].float()
    value_codebooks = artifact["artifacts"]["value"]["codebooks"].float()
    if key_codebooks.shape != value_codebooks.shape:
        raise ValueError("K/V codebook shapes differ")
    if prefix_stages <= 0:
        prefix_stages = int(key_codebooks.shape[0])
    if not 0 < prefix_stages <= key_codebooks.shape[0]:
        raise ValueError("prefix_stages is out of range")
    return PartialCodebookContext(
        layer=int(provenance["layer"]),
        head=int(provenance["head"]),
        prefix_stages=prefix_stages,
        key_codebooks=key_codebooks,
        value_codebooks=value_codebooks,
        chunk_size=chunk_size,
        codebook_bits=int(round(math.log2(key_codebooks.shape[-2]))),
        current_token_fp16=current_token_fp16,
        key_surface=normalize_key_surface(
            provenance.get("surface", "pre_rope")
        ),
        provenance=provenance,
    )


def load_full_context(
    directory: Path,
    layer: int,
    prefix_stages: int,
    chunk_size: int,
    current_token_fp16: bool = True,
    key_override_dir: Path | None = None,
) -> PartialCodebookContext:
    path = directory / f"layer_{layer:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    provenance = artifact["provenance"]
    if int(provenance["layer"]) != layer:
        raise ValueError(f"{path} has the wrong layer provenance")
    key_codebooks = artifact["codebooks"]["key"].float()
    value_codebooks = artifact["codebooks"]["value"].float()
    if key_codebooks.shape != value_codebooks.shape:
        raise ValueError(f"{path} has different K/V codebook shapes")
    if key_codebooks.ndim != 4:
        raise ValueError(f"{path} codebooks must be [H,R,K,D]")
    key_assignment_whitener = None
    value_assignment_whitener = artifact.get("value_assignment_whiteners")
    if value_assignment_whitener is not None:
        value_checks = {
            "transform": provenance.get("value_transform")
            == "m23_final_ce_sqrt_gradient_covariance",
            "shape": isinstance(value_assignment_whitener, torch.Tensor)
            and tuple(value_assignment_whitener.shape)
            == (
                value_codebooks.shape[0],
                value_codebooks.shape[-1],
                value_codebooks.shape[-1],
            ),
            "finite": isinstance(value_assignment_whitener, torch.Tensor)
            and bool(torch.isfinite(value_assignment_whitener).all()),
        }
        failed = [name for name, passed in value_checks.items() if not passed]
        if failed:
            raise ValueError(f"{path} failed M23 V-assignment checks: {failed}")
        value_assignment_whitener = value_assignment_whitener.float()
    override_provenance = None
    if key_override_dir is not None:
        override_path = key_override_dir / f"layer_{layer:02d}.pt"
        if not override_path.exists():
            raise FileNotFoundError(override_path)
        override = torch.load(
            override_path, map_location="cpu", weights_only=False
        )
        override_provenance = override.get("provenance", {})
        override_key = override.get("codebooks", {}).get("key")
        key_assignment_whitener = override.get(
            "key_assignment_whiteners"
        )
        checks = {
            "layer": int(override_provenance.get("layer", -1)) == layer,
            "model": str(Path(override_provenance.get("model", "")).resolve())
            == str(Path(provenance.get("model", "")).resolve()),
            "transform": override_provenance.get("key_transform")
            in {"m20_sqrt_fisher_fixed", "scf_spectral_fisher_fixed"},
            "weighted": bool(override_provenance.get("weighted")),
            "surface": normalize_key_surface(
                override_provenance.get("surface", "pre_rope")
            )
            == normalize_key_surface(provenance.get("surface", "pre_rope")),
            "key_shape": isinstance(override_key, torch.Tensor)
            and tuple(override_key.shape) == tuple(key_codebooks.shape),
            "whitener_shape": isinstance(
                key_assignment_whitener, torch.Tensor
            )
            and tuple(key_assignment_whitener.shape)
            == (
                key_codebooks.shape[0],
                key_codebooks.shape[-1],
                key_codebooks.shape[-1],
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{override_path} failed checks: {failed}")
        if not bool(torch.isfinite(override_key).all()) or not bool(
            torch.isfinite(key_assignment_whitener).all()
        ):
            raise ValueError(f"{override_path} contains non-finite tensors")
        key_codebooks = override_key.float()
    if prefix_stages <= 0:
        prefix_stages = int(key_codebooks.shape[1])
    if not 0 < prefix_stages <= key_codebooks.shape[1]:
        raise ValueError("prefix_stages is out of range")
    return PartialCodebookContext(
        layer=layer,
        head=None,
        prefix_stages=prefix_stages,
        key_codebooks=key_codebooks,
        value_codebooks=value_codebooks,
        chunk_size=chunk_size,
        codebook_bits=int(round(math.log2(key_codebooks.shape[-2]))),
        current_token_fp16=current_token_fp16,
        key_surface=normalize_key_surface(
            provenance.get("surface", "pre_rope")
        ),
        provenance={
            **provenance,
            "key_override": override_provenance,
            "key_override_dir": (
                str(key_override_dir.resolve())
                if key_override_dir is not None
                else None
            ),
        },
        key_assignment_whitener=key_assignment_whitener,
        value_assignment_whitener=value_assignment_whitener,
    )


@torch.no_grad()
def quantize_with_assignment_whitener(
    states: torch.Tensor,
    raw_codebooks: torch.Tensor,
    whiteners: torch.Tensor,
    *,
    prefix_stages: int,
    chunk_size: int,
) -> torch.Tensor:
    """Select in frozen transformed coordinates and decode raw K/V tables."""

    if states.ndim != 4 or raw_codebooks.ndim != 4:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,R,K,D]")
    batch, heads, tokens, dimension = map(int, states.shape)
    if tuple(raw_codebooks.shape[:1]) != (heads,) or int(
        raw_codebooks.shape[-1]
    ) != dimension:
        raise ValueError("raw codebook geometry differs from states")
    if tuple(whiteners.shape) != (heads, dimension, dimension):
        raise ValueError("assignment whitener geometry differs")
    device = states.device
    factors = whiteners.to(device=device, dtype=torch.float32)
    raw_tables = raw_codebooks.to(device=device, dtype=torch.float32)
    selector_states = torch.einsum(
        "hde,bhte->bhtd", factors, states.float()
    )
    selector_tables = torch.einsum(
        "hrke,hde->hrkd", raw_tables, factors
    )
    _, indices = quantize_kv_heads(
        selector_states,
        selector_tables,
        prefix_stages=prefix_stages,
        chunk_size=chunk_size,
        return_indices=True,
    )
    if indices is None:
        raise RuntimeError("transformed assignment did not return indices")
    indices = indices.reshape(batch, tokens, heads, prefix_stages).permute(
        0, 2, 1, 3
    )
    decoded = torch.zeros(
        batch, heads, tokens, dimension, device=device, dtype=torch.float32
    )
    head_rows = torch.arange(heads, device=device)[None, :, None]
    for stage in range(prefix_stages):
        decoded += raw_tables[:, stage][head_rows, indices[..., stage]]
    return decoded.to(dtype=states.dtype)


@torch.no_grad()
def quantize_keys_with_assignment_whitener(
    states: torch.Tensor,
    raw_codebooks: torch.Tensor,
    whiteners: torch.Tensor,
    *,
    prefix_stages: int,
    chunk_size: int,
) -> torch.Tensor:
    """Backward-compatible name for M20 transformed K assignment."""

    return quantize_with_assignment_whitener(
        states,
        raw_codebooks,
        whiteners,
        prefix_stages=prefix_stages,
        chunk_size=chunk_size,
    )


def gsrq_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    if past_key_value is not None or use_cache:
        raise ValueError("the layerwise evaluator requires use_cache=False")
    context: PartialCodebookContext = self._gsrq_context
    batch, query_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    raw_key_states = key_states
    exact_value_states = value_states

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    raw_query_states = query_states
    query_states, exact_key_states = apply_rotary_pos_emb(
        raw_query_states,
        raw_key_states,
        cos,
        sin,
    )
    key_states = key_quantization_input(
        raw_key_states,
        cos,
        sin,
        context.key_surface,
    )

    key_tables = (
        context.key_codebooks
        if context.head is None
        else context.key_codebooks[None]
    )
    value_tables = (
        context.value_codebooks
        if context.head is None
        else context.value_codebooks[None]
    )
    selected_heads = None if context.head is None else [context.head]
    if context.key_assignment_whitener is not None:
        if selected_heads is not None:
            raise ValueError("M20 assignment is supported only for full layers")
        key_states = quantize_with_assignment_whitener(
            key_states,
            key_tables,
            context.key_assignment_whitener,
            prefix_stages=context.prefix_stages,
            chunk_size=context.chunk_size,
        )
    else:
        key_states, _ = quantize_kv_heads(
            key_states,
            key_tables,
            prefix_stages=context.prefix_stages,
            head_indices=selected_heads,
            chunk_size=context.chunk_size,
        )
    if context.value_assignment_whitener is not None:
        if selected_heads is not None:
            raise ValueError("M23 V assignment is supported only for full layers")
        value_states = quantize_with_assignment_whitener(
            value_states,
            value_tables,
            context.value_assignment_whitener,
            prefix_stages=context.prefix_stages,
            chunk_size=context.chunk_size,
        )
    else:
        value_states, _ = quantize_kv_heads(
            value_states,
            value_tables,
            prefix_stages=context.prefix_stages,
            head_indices=selected_heads,
            chunk_size=context.chunk_size,
        )

    key_states = key_states_for_attention(
        key_states,
        cos,
        sin,
        context.key_surface,
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if context.current_token_fp16:
        exact_key_states = repeat_kv(
            exact_key_states, self.num_key_value_groups
        )
        exact_value_states = repeat_kv(
            exact_value_states, self.num_key_value_groups
        )
    weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    diagonal = None
    if context.current_token_fp16:
        diagonal_length = min(query_length, key_states.shape[-2])
        diagonal = torch.arange(
            diagonal_length, device=hidden_states.device
        )
        exact_diagonal_logits = (
            query_states[:, :, diagonal]
            * exact_key_states[:, :, diagonal]
        ).sum(dim=-1) / math.sqrt(self.head_dim)
        weights[:, :, diagonal, diagonal] = exact_diagonal_logits
    if attention_mask is not None:
        weights = weights + attention_mask[
            :, :, :, : key_states.shape[-2]
        ]
    weights = F.softmax(
        weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    weights = F.dropout(
        weights, p=self.attention_dropout, training=self.training
    )
    output = torch.matmul(weights, value_states)
    if context.current_token_fp16 and diagonal is not None:
        diagonal_probability = weights[:, :, diagonal, diagonal]
        value_delta = (
            exact_value_states[:, :, diagonal]
            - value_states[:, :, diagonal]
        )
        output[:, :, diagonal] += (
            diagonal_probability[..., None] * value_delta
        )
    if output.shape != (
        batch,
        self.num_heads,
        query_length,
        self.head_dim,
    ):
        raise RuntimeError(f"unexpected attention output {output.shape}")
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, self.hidden_size
    )
    output = self.o_proj(output)
    return output, weights if output_attentions else None, past_key_value


def bind_attention(
    attention: nn.Module,
    mode: str,
    context: PartialCodebookContext | None,
    layer_index: int,
) -> None:
    if mode in ("partial_gsrq", "full_gsrq") and context is not None:
        if layer_index == context.layer:
            attention._gsrq_context = context
            attention.forward = types.MethodType(
                gsrq_attention_forward, attention
            )
            return
    attention.forward = types.MethodType(
        type(attention).forward, attention
    )
    if hasattr(attention, "_gsrq_context"):
        del attention._gsrq_context


@torch.no_grad()
def evaluate(
    model: nn.Module,
    initial_inps: torch.Tensor,
    token_ids: torch.Tensor,
    cache: dict[str, Any],
    *,
    mode: str,
    context: PartialCodebookContext | None,
    codebook_dir: Path | None,
    key_override_dir: Path | None,
    device: torch.device,
    sequence_length: int,
    finite_diagnostics: bool = False,
) -> dict[str, Any]:
    start = time.time()
    inps = initial_inps.to(device)
    outs = torch.empty_like(inps)
    layers = model.model.layers
    for layer_index in range(len(layers)):
        layer_start = time.time()
        layer = layers[layer_index].to(device)
        layer_context = context
        if mode == "full_gsrq":
            if codebook_dir is None:
                raise ValueError("full_gsrq requires a codebook directory")
            layer_context = load_full_context(
                codebook_dir,
                layer_index,
                0 if context is None else context.prefix_stages,
                4096 if context is None else context.chunk_size,
                (
                    True
                    if context is None
                    else context.current_token_fp16
                ),
                key_override_dir,
            )
            if (
                context is not None
                and layer_context.key_surface != context.key_surface
            ):
                raise ValueError(
                    f"layer {layer_index} key surface "
                    f"{layer_context.key_surface!r} differs from "
                    f"{context.key_surface!r}"
                )
        if layer_context is not None:
            layer_context.key_codebooks = (
                layer_context.key_codebooks.to(device)
            )
            layer_context.value_codebooks = (
                layer_context.value_codebooks.to(device)
            )
            if layer_context.key_assignment_whitener is not None:
                layer_context.key_assignment_whitener = (
                    layer_context.key_assignment_whitener.to(device)
                )
            if layer_context.value_assignment_whitener is not None:
                layer_context.value_assignment_whitener = (
                    layer_context.value_assignment_whitener.to(device)
                )
        bind_attention(
            layer.self_attn, mode, layer_context, layer_index
        )
        for window in range(inps.shape[0]):
            hidden = inps[window].unsqueeze(0)
            outs[window].copy_(
                layer(
                    hidden,
                    **layer_kwargs(cache, model, hidden, device),
                    use_cache=False,
                )[0][0]
            )
        if hasattr(layer.self_attn, "_gsrq_context"):
            del layer.self_attn._gsrq_context
        layers[layer_index] = layer.cpu()
        del layer, layer_context
        inps, outs = outs, inps
        if finite_diagnostics:
            finite = torch.isfinite(inps)
            finite_count = int(finite.sum().cpu())
            total_count = int(inps.numel())
            finite_values = inps[finite]
            maximum_absolute = (
                float(finite_values.abs().max().cpu())
                if finite_count
                else None
            )
            diagnostic = {
                "event": "layer_hidden_numerics",
                "mode": mode,
                "layer": layer_index,
                "finite": finite_count == total_count,
                "finite_count": finite_count,
                "total_count": total_count,
                "nan_count": int(torch.isnan(inps).sum().cpu()),
                "inf_count": int(torch.isinf(inps).sum().cpu()),
                "max_abs_finite": maximum_absolute,
            }
            print(json.dumps(diagnostic), flush=True)
            if finite_count != total_count:
                raise FloatingPointError(
                    f"non-finite hidden states after layer {layer_index}"
                )
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "mode": mode,
                    "layer": layer_index,
                    "seconds": round(time.time() - layer_start, 3),
                }
            ),
            flush=True,
        )

    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    labels = token_ids.to(device)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    eval_tokens = 0
    window_metrics: list[dict[str, Any]] = []
    for window in range(inps.shape[0]):
        hidden = model.model.norm(inps[window].unsqueeze(0))
        logits = model.lm_head(hidden)
        shift_labels = labels[
            :,
            window * sequence_length + 1 : (window + 1) * sequence_length,
        ]
        window_nll = loss_function(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            shift_labels.reshape(-1),
        ).double()
        window_tokens = int(shift_labels.numel())
        nll_sum += window_nll
        eval_tokens += window_tokens
        window_metrics.append(
            {
                "window": window,
                "nll_sum": float(window_nll.cpu()),
                "eval_tokens": window_tokens,
                "mean_nll": float(window_nll.cpu()) / window_tokens,
            }
        )
    nll = float(nll_sum.cpu())
    mean_nll = nll / eval_tokens
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return {
        "status": "complete",
        "mode": mode,
        "ppl": math.exp(mean_nll),
        "nll_sum": nll,
        "mean_nll": mean_nll,
        "eval_tokens": eval_tokens,
        "evaluated_windows": int(initial_inps.shape[0]),
        "window_metrics": window_metrics,
        "elapsed_seconds": time.time() - start,
        "index_bits_per_activation": (
            context.prefix_stages
            * float(context.codebook_bits)
            / 128.0
            if mode == "full_gsrq" and context is not None
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    context = None
    if "partial_gsrq" in args.modes:
        if not args.codebook_file:
            raise ValueError("partial_gsrq requires --codebook-file")
        context = load_partial_context(
            Path(args.codebook_file),
            args.prefix_stages,
            args.chunk_size,
            not args.quantize_current_token,
        )
    codebook_dir = None
    key_override_dir = None
    if "full_gsrq" in args.modes:
        if "partial_gsrq" in args.modes:
            raise ValueError(
                "partial_gsrq and full_gsrq cannot share one invocation"
            )
        if not args.codebook_dir:
            raise ValueError("full_gsrq requires --codebook-dir")
        codebook_dir = Path(args.codebook_dir).resolve()
        key_override_dir = (
            Path(args.key_override_dir).resolve()
            if args.key_override_dir
            else None
        )
        first_context = load_full_context(
            codebook_dir,
            0,
            args.prefix_stages,
            args.chunk_size,
            not args.quantize_current_token,
            key_override_dir,
        )
        context = PartialCodebookContext(
            layer=-1,
            head=None,
            prefix_stages=first_context.prefix_stages,
            key_codebooks=torch.empty(0),
            value_codebooks=torch.empty(0),
            chunk_size=args.chunk_size,
            codebook_bits=first_context.codebook_bits,
            current_token_fp16=first_context.current_token_fp16,
            key_surface=first_context.key_surface,
            provenance={
                "codebook_dir": str(codebook_dir),
                "layer0": first_context.provenance,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    load_max_windows = args.max_windows
    if (
        args.dataset == "c4"
        and args.max_windows > 0
        and args.window_start > 0
    ):
        load_max_windows = args.window_start + args.max_windows
    token_ids, available_windows = load_eval_tokens(
        tokenizer,
        Path(args.dataset_root),
        args.dataset,
        args.sequence_length,
        load_max_windows,
    )
    end = available_windows
    if args.max_windows > 0:
        end = min(end, args.window_start + args.max_windows)
    token_ids = token_ids[
        :,
        args.window_start * args.sequence_length : end
        * args.sequence_length,
    ].contiguous()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    initial_inps, cache = capture_initial_hidden_states(
        model, token_ids, args.sequence_length, device
    )
    results = []
    for mode in args.modes:
        print(json.dumps({"event": "mode_start", "mode": mode}), flush=True)
        result = evaluate(
            model,
            initial_inps,
            token_ids,
            cache,
            mode=mode,
            context=context,
            codebook_dir=codebook_dir,
            key_override_dir=key_override_dir,
            device=device,
            sequence_length=args.sequence_length,
            finite_diagnostics=args.finite_diagnostics,
        )
        result.update(
            {
                "model": str(Path(args.model).resolve()),
                "dataset": args.dataset,
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "sequence_length": args.sequence_length,
                "available_windows": available_windows,
                "window_start": args.window_start,
                "window_end": end,
                "prefix_stages": (
                    context.prefix_stages if context is not None else None
                ),
                "surface": (
                    context.key_surface if context is not None else None
                ),
                "current_token_fp16": (
                    context.current_token_fp16
                    if context is not None
                    else None
                ),
                "codebook_dir": (
                    str(codebook_dir) if codebook_dir else None
                ),
                "key_override_dir": (
                    str(key_override_dir) if key_override_dir else None
                ),
            }
        )
        for window_row in result["window_metrics"]:
            window_row["window"] += args.window_start
        results.append(result)
        (output_dir / f"{mode}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({"event": "mode_complete", **result}), flush=True
        )
    provenance = {
        "model": str(Path(args.model).resolve()),
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "window_start": args.window_start,
        "window_end": end,
        "codebook_file": (
            str(Path(args.codebook_file).resolve())
            if args.codebook_file
            else None
        ),
        "codebook_dir": str(codebook_dir) if codebook_dir else None,
        "key_override_dir": (
            str(key_override_dir) if key_override_dir else None
        ),
        "partial_context": context.provenance if context else None,
        "prefix_stages": context.prefix_stages if context else None,
        "surface": context.key_surface if context else None,
        "current_token_fp16": (
            context.current_token_fp16 if context else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"provenance": provenance, "results": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
