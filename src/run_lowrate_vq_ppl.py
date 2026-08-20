#!/usr/bin/env python3
"""Layerwise C4/WikiText PPL evaluator for low-rate VQ baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from ppl_utils import (
    capture_initial_hidden_states,
    layer_kwargs,
    load_eval_tokens,
)
from lowrate_vq.codecs import quantize_fibquant, quantize_opq
from lowrate_vq.aqlm_kv import quantize_aqlm_kv
from lowrate_vq.cq import quantize_cq
from lowrate_vq.commvq import (
    CommVQValueCodec,
    quantize_commvq_key,
    quantize_commvq_value,
)
from lowrate_vq.m20_fibquant import (
    prepare_m20_fibquant_search,
    quantize_m20_fibquant_key,
)
from lowrate_vq.nova import quantize_nova_key


@dataclass
class CodecContext:
    method: str
    key_surface: str
    index_bpa: float
    metadata_bpa: float
    total_bpa: float
    chunk_size: int
    provenance: dict[str, Any]
    rotation: torch.Tensor | None = None
    codebook: torch.Tensor | None = None
    key_rotations: torch.Tensor | None = None
    value_rotations: torch.Tensor | None = None
    key_codebooks: torch.Tensor | None = None
    value_codebooks: torch.Tensor | None = None
    key_thetas: torch.Tensor | None = None
    value_codec: CommVQValueCodec | None = None
    key_decode_codebook: torch.Tensor | None = None
    key_block_whiteners: torch.Tensor | None = None
    key_selector_cross_codebooks: torch.Tensor | None = None
    key_selector_norms: torch.Tensor | None = None
    key_selector_whiteners: torch.Tensor | None = None
    key_forward: torch.Tensor | None = None
    key_inverse: torch.Tensor | None = None
    key_mean: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "models/LLM-Research/llama-2-7b"))
    parser.add_argument(
        "--dataset-root", default=str(root / "huggingface_cache_datasets/datasets")
    )
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="c4")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=[
            "fibquant",
            "m20_fibquant",
            "opq",
            "commvq",
            "cq",
            "m20_cq",
            "nova_cqv",
            "m20_nova_cqv",
            "aqlm_kv",
            "scf_aqlm_kv",
        ],
        required=True,
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--finite-diagnostics", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_context(method: str, artifact_dir: Path, layer: int, chunk_size: int) -> CodecContext:
    path = artifact_dir / ("codec.pt" if method == "fibquant" else f"layer_{layer:02d}.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    provenance = artifact["provenance"]
    common = dict(
        method=method,
        key_surface=str(provenance["key_surface"]),
        index_bpa=float(provenance["index_bpa"]),
        metadata_bpa=float(provenance["metadata_bpa"]),
        total_bpa=float(provenance["total_bpa"]),
        chunk_size=chunk_size,
        provenance={**provenance, "artifact": str(path.resolve())},
    )
    if method == "fibquant":
        return CodecContext(
            **common,
            rotation=artifact["rotation"].float(),
            codebook=artifact["codebook"].float(),
        )
    if method == "m20_fibquant":
        if int(provenance["layer"]) != layer:
            raise ValueError(f"{path} has the wrong layer provenance")
        if provenance.get("format") not in {
            "m20_fibquant_layer_v1",
            "m20_fibquant_layer_v2",
        }:
            raise ValueError(f"{path} is not an SCF-FibQuant layer artifact")
        if provenance.get("format") == "m20_fibquant_layer_v2" and not isinstance(
            provenance.get("fibquant_norm_squared_weighting"), bool
        ):
            raise ValueError(f"{path} omits the v2 norm-weighting contract")
        key_decode_codebook = artifact["key"]["decode_codebook"].float()
        key_block_whiteners = artifact["key"]["block_whiteners"].float()
        key_selector_cross_codebooks, key_selector_norms = (
            prepare_m20_fibquant_search(
                key_decode_codebook,
                key_block_whiteners,
            )
        )
        return CodecContext(
            **common,
            rotation=artifact["rotation"].float(),
            codebook=artifact["value"]["codebook"].float(),
            key_decode_codebook=key_decode_codebook,
            key_block_whiteners=key_block_whiteners,
            key_selector_cross_codebooks=key_selector_cross_codebooks,
            key_selector_norms=key_selector_norms,
        )
    if method == "commvq":
        if int(provenance["layer"]) != layer:
            raise ValueError(f"{path} has the wrong layer provenance")
        value_config = artifact["value"]["config"]
        value_codec = CommVQValueCodec(
            float(value_config["quant_bits"]), int(value_config["feature_dim"])
        )
        value_codec.load_state_dict(artifact["value"]["state_dict"])
        value_codec.eval()
        return CodecContext(
            **common,
            key_thetas=artifact["key"]["thetas"].float(),
            value_codec=value_codec,
        )
    if method in {"cq", "m20_cq"}:
        expected_formats = (
            {"cq_8c8b_layer_v1", "cq_layer_v2"}
            if method == "cq"
            else {"m20_cq_8c8b_layer_v1", "m20_cq_layer_v2"}
        )
        if provenance.get("format") not in expected_formats:
            raise ValueError(
                f"{path} has format {provenance.get('format')!r}, expected one of "
                f"{sorted(expected_formats)}"
            )
        if int(provenance["layer"]) != layer:
            raise ValueError(f"{path} has the wrong layer provenance")
        return CodecContext(
            **common,
            key_codebooks=artifact["key"]["codebooks"].float(),
            value_codebooks=artifact["value"]["codebooks"].float(),
            key_selector_whiteners=(
                artifact["key"]["selector_whiteners"].float()
                if method == "m20_cq"
                else None
            ),
        )
    if method in {"aqlm_kv", "scf_aqlm_kv"}:
        expected_format = (
            "aqlm_kv_layer_v1"
            if method == "aqlm_kv"
            else "scf_aqlm_kv_layer_v1"
        )
        if provenance.get("format") != expected_format:
            raise ValueError(
                f"{path} has format {provenance.get('format')!r}, expected "
                f"{expected_format!r}"
            )
        if int(provenance["layer"]) != layer:
            raise ValueError(f"{path} has the wrong layer provenance")
        return CodecContext(
            **common,
            key_codebooks=artifact["key"]["codebooks"].float(),
            value_codebooks=artifact["value"]["codebooks"].float(),
            key_selector_whiteners=(
                artifact["key"]["selector_whiteners"].float()
                if method == "scf_aqlm_kv"
                else None
            ),
        )
    if method in {"nova_cqv", "m20_nova_cqv"}:
        if provenance.get("format") != "nova_cqv_1bit_layer_v1":
            raise ValueError(f"{path} is not a NOVA-CQV 1-bit artifact")
        expected_method = (
            "NOVA-1+CQV" if method == "nova_cqv" else "M20-NOVA-1+CQV"
        )
        if provenance.get("method") != expected_method:
            raise ValueError(f"{path} has the wrong NOVA parent/child identity")
        if int(provenance["layer"]) != layer:
            raise ValueError(f"{path} has the wrong layer provenance")
        return CodecContext(
            **common,
            key_codebooks=artifact["key"]["codebooks"].float(),
            value_codebooks=artifact["value"]["codebooks"].float(),
            key_forward=artifact["key"]["forward"].float(),
            key_inverse=artifact["key"]["inverse"].float(),
            key_mean=artifact["key"]["mean"].float(),
        )
    if int(provenance["layer"]) != layer:
        raise ValueError(f"{path} has the wrong layer provenance")
    return CodecContext(
        **common,
        key_rotations=artifact["key"]["rotations"].float(),
        value_rotations=artifact["value"]["rotations"].float(),
        key_codebooks=artifact["key"]["codebooks"].float(),
        value_codebooks=artifact["value"]["codebooks"].float(),
    )


def move_context(context: CodecContext, device: torch.device) -> CodecContext:
    for name in (
        "rotation",
        "codebook",
        "key_rotations",
        "value_rotations",
        "key_codebooks",
        "value_codebooks",
        "key_thetas",
        "key_decode_codebook",
        "key_block_whiteners",
        "key_selector_cross_codebooks",
        "key_selector_norms",
        "key_selector_whiteners",
        "key_forward",
        "key_inverse",
        "key_mean",
    ):
        value = getattr(context, name)
        if value is not None:
            setattr(context, name, value.to(device))
    if context.value_codec is not None:
        context.value_codec = context.value_codec.to(device)
    return context


@torch.no_grad()
def quantize(states: torch.Tensor, context: CodecContext, kind: str) -> torch.Tensor:
    if context.method == "fibquant":
        assert context.rotation is not None and context.codebook is not None
        decoded, _ = quantize_fibquant(
            states,
            context.rotation,
            context.codebook,
            block_size=int(context.codebook.shape[-1]),
            chunk_size=context.chunk_size,
        )
        return decoded
    if context.method == "m20_fibquant":
        assert context.rotation is not None and context.codebook is not None
        if kind == "key":
            assert context.key_decode_codebook is not None
            assert context.key_block_whiteners is not None
            decoded, _ = quantize_m20_fibquant_key(
                states,
                context.rotation,
                context.key_decode_codebook,
                context.key_block_whiteners,
                block_size=int(context.key_decode_codebook.shape[-1]),
                chunk_size=context.chunk_size,
                selector_cross_codebooks=context.key_selector_cross_codebooks,
                selector_norms=context.key_selector_norms,
            )
            return decoded
        decoded, _ = quantize_fibquant(
            states,
            context.rotation,
            context.codebook,
            block_size=int(context.codebook.shape[-1]),
            chunk_size=context.chunk_size,
        )
        return decoded
    if context.method == "commvq":
        if kind == "key":
            assert context.key_thetas is not None
            decoded, _ = quantize_commvq_key(
                states,
                context.key_thetas,
                chunk_size=context.chunk_size,
            )
            return decoded
        assert context.value_codec is not None
        return quantize_commvq_value(states, context.value_codec)
    if context.method in {"cq", "m20_cq"}:
        codebooks = (
            context.key_codebooks if kind == "key" else context.value_codebooks
        )
        assert codebooks is not None
        decoded, _ = quantize_cq(
            states,
            codebooks,
            selector_whiteners=(
                context.key_selector_whiteners if kind == "key" else None
            ),
            chunk_size=context.chunk_size,
        )
        return decoded
    if context.method in {"aqlm_kv", "scf_aqlm_kv"}:
        codebooks = (
            context.key_codebooks if kind == "key" else context.value_codebooks
        )
        assert codebooks is not None
        decoded, _ = quantize_aqlm_kv(
            states,
            codebooks,
            selector_whiteners=(
                context.key_selector_whiteners
                if context.method == "scf_aqlm_kv" and kind == "key"
                else None
            ),
            chunk_size=context.chunk_size,
            problem_batch_size=16,
        )
        return decoded
    if context.method in {"nova_cqv", "m20_nova_cqv"}:
        if kind == "key":
            assert context.key_forward is not None
            assert context.key_inverse is not None
            assert context.key_mean is not None
            assert context.key_codebooks is not None
            decoded, _ = quantize_nova_key(
                states,
                context.key_forward,
                context.key_inverse,
                context.key_mean,
                context.key_codebooks,
                chunk_size=context.chunk_size,
            )
            return decoded
        assert context.value_codebooks is not None
        decoded, _ = quantize_cq(
            states,
            context.value_codebooks,
            chunk_size=context.chunk_size,
        )
        return decoded
    rotations = context.key_rotations if kind == "key" else context.value_rotations
    codebooks = context.key_codebooks if kind == "key" else context.value_codebooks
    assert rotations is not None and codebooks is not None
    decoded, _ = quantize_opq(
        states, rotations, codebooks, chunk_size=context.chunk_size
    )
    return decoded


def lowrate_attention_forward(
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
    context: CodecContext = self._lowrate_vq_context
    batch, query_length, _ = hidden_states.shape
    raw_queries = self.q_proj(hidden_states).view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    raw_keys = self.k_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    values = self.v_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    if position_embeddings is None:
        cos, sin = self.rotary_emb(values, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, post_rope_keys = apply_rotary_pos_emb(
        raw_queries, raw_keys, cos, sin
    )
    if context.key_surface == "pre_rope":
        keys = quantize(raw_keys, context, "key")
        _, keys = apply_rotary_pos_emb(raw_queries, keys, cos, sin)
    elif context.key_surface == "post_rope":
        keys = quantize(post_rope_keys, context, "key")
    else:
        raise ValueError(f"unsupported key surface {context.key_surface!r}")
    values = quantize(values, context, "value")
    keys = repeat_kv(keys, self.num_key_value_groups)
    values = repeat_kv(values, self.num_key_value_groups)
    weights = torch.matmul(query_states, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : keys.shape[-2]]
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
    output = torch.matmul(weights, values)
    if output.shape != (batch, self.num_heads, query_length, self.head_dim):
        raise RuntimeError(f"unexpected attention output {output.shape}")
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, self.hidden_size
    )
    return self.o_proj(output), weights if output_attentions else None, past_key_value


def bind_attention(attention: nn.Module, context: CodecContext | None) -> None:
    if context is not None:
        attention._lowrate_vq_context = context
        attention.forward = types.MethodType(lowrate_attention_forward, attention)
    else:
        attention.forward = types.MethodType(type(attention).forward, attention)
        if hasattr(attention, "_lowrate_vq_context"):
            del attention._lowrate_vq_context


@torch.no_grad()
def evaluate(
    model: nn.Module,
    initial_inps: torch.Tensor,
    token_ids: torch.Tensor,
    cache: dict[str, Any],
    *,
    method: str,
    artifact_dir: Path,
    device: torch.device,
    sequence_length: int,
    chunk_size: int,
    finite_diagnostics: bool,
) -> dict[str, Any]:
    started = time.time()
    inps = initial_inps.to(device)
    outs = torch.empty_like(inps)
    layers = model.model.layers
    first_context = None
    layer_provenance = []
    for layer_index in range(len(layers)):
        layer_started = time.time()
        context = move_context(
            load_context(method, artifact_dir, layer_index, chunk_size), device
        )
        if first_context is None:
            first_context = context
        layer_provenance.append(context.provenance)
        layer = layers[layer_index].to(device)
        bind_attention(layer.self_attn, context)
        for window in range(inps.shape[0]):
            hidden = inps[window].unsqueeze(0)
            outs[window].copy_(
                layer(
                    hidden,
                    **layer_kwargs(cache, model, hidden, device),
                    use_cache=False,
                )[0][0]
            )
        bind_attention(layer.self_attn, None)
        layers[layer_index] = layer.cpu()
        del layer, context
        inps, outs = outs, inps
        layer_is_finite = bool(torch.isfinite(inps).all())
        if not layer_is_finite:
            raise FloatingPointError(
                f"non-finite hidden states after layer {layer_index}; "
                "the result is invalid and must not be consolidated"
            )
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "method": method,
                    "layer": layer_index,
                    "seconds": round(time.time() - layer_started, 3),
                    "finite": layer_is_finite,
                    "max_abs": float(inps.abs().max()),
                }
            ),
            flush=True,
        )
    assert first_context is not None
    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    labels = token_ids.to(device)
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    eval_tokens = 0
    windows = []
    for window in range(inps.shape[0]):
        logits = model.lm_head(model.model.norm(inps[window].unsqueeze(0)))
        shift_labels = labels[
            :,
            window * sequence_length + 1 : (window + 1) * sequence_length,
        ]
        window_nll = loss_function(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            shift_labels.reshape(-1),
        ).double()
        if not bool(torch.isfinite(window_nll)):
            raise FloatingPointError(
                f"non-finite teacher-forced NLL in local window {window}"
            )
        count = int(shift_labels.numel())
        nll_sum += window_nll
        eval_tokens += count
        windows.append(
            {
                "window": window,
                "nll_sum": float(window_nll.cpu()),
                "eval_tokens": count,
                "mean_nll": float(window_nll.cpu()) / count,
            }
        )
    nll = float(nll_sum.cpu())
    mean_nll = nll / eval_tokens
    if not math.isfinite(mean_nll):
        raise FloatingPointError("non-finite aggregate teacher-forced NLL")
    try:
        ppl = math.exp(mean_nll)
    except OverflowError as error:
        raise FloatingPointError("teacher-forced PPL overflow") from error
    if not math.isfinite(ppl):
        raise FloatingPointError("non-finite teacher-forced PPL")
    result = {
        "status": "complete",
        "method": method,
        "ppl": ppl,
        "nll_sum": nll,
        "mean_nll": mean_nll,
        "eval_tokens": eval_tokens,
        "evaluated_windows": int(initial_inps.shape[0]),
        "window_metrics": windows,
        "elapsed_seconds": time.time() - started,
        "index_bpa": first_context.index_bpa,
        "metadata_bpa": first_context.metadata_bpa,
        "total_bpa": first_context.total_bpa,
        "key_surface": first_context.key_surface,
        "current_token_quantized": True,
        "layer_provenance": layer_provenance,
    }
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir).resolve()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    load_max_windows = args.max_windows
    if args.max_windows > 0 and args.window_start > 0:
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
        args.window_start * args.sequence_length : end * args.sequence_length,
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
    result = evaluate(
        model,
        initial_inps,
        token_ids,
        cache,
        method=args.method,
        artifact_dir=artifact_dir,
        device=device,
        sequence_length=args.sequence_length,
        chunk_size=args.chunk_size,
        finite_diagnostics=args.finite_diagnostics,
    )
    for row in result["window_metrics"]:
        row["window"] += args.window_start
    result.update(
        {
            "model": str(Path(args.model).resolve()),
            "dataset": args.dataset,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "sequence_length": args.sequence_length,
            "available_windows": available_windows,
            "window_start": args.window_start,
            "window_end": end,
            "artifact_dir": str(artifact_dir),
        }
    )
    atomic_json(output_dir / "result.json", result)
    atomic_json(
        output_dir / "summary.json",
        {"provenance": {k: result[k] for k in (
            "model", "dataset", "dataset_root", "sequence_length", "window_start",
            "window_end", "artifact_dir", "method", "index_bpa", "metadata_bpa",
            "total_bpa", "key_surface", "current_token_quantized"
        )}, "result": result},
    )
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
