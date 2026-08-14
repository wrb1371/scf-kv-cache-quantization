#!/usr/bin/env python3
"""Calibrate SCF-FibQuant on a frozen parent-FibQuant C4 trajectory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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

from calibrate_gsrq_codebooks import parse_layer_spec, sha256_file
from collect_gsrq_gradient_weights import load_manifest_windows
from facq_kvquant_aligned_ppl import (
    capture_initial_hidden_states,
    layer_kwargs,
)
from lowrate_vq.codecs import fibquant_rate, quantize_fibquant
from lowrate_vq.m20_fibquant import (
    build_block_metric_geometry,
    metric_codebook_diagnostics,
    normalize_fisher,
    post_rope_fisher_samples,
    refine_shared_metric_codebook,
)
from run_gsrq_activation_smoke import select_windows


@dataclass
class CalibrationContext:
    rotation: torch.Tensor
    codebook: torch.Tensor
    block_size: int
    chunk_size: int
    collect: bool
    fisher_samples_per_head: int
    fisher_outer_sum: torch.Tensor | None = None
    calls: int = 0
    key_batches: list[torch.Tensor] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "models/LLM-Research/llama-2-7b"))
    parser.add_argument(
        "--c4-train-arrow",
        default=str(
            root
            / "huggingface_cache_datasets/datasets/allenai___c4"
            / "default-b04fc8a0b8562884/0.0.0"
            / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            / "c4-train-00000-of-00002.arrow"
        ),
    )
    parser.add_argument(
        "--calibration-manifest",
        help="Frozen generic VQ manifest; when set, supersedes --c4-train-arrow.",
    )
    parser.add_argument("--parent-codec", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--fit-vectors-per-head", type=int, default=2048)
    parser.add_argument("--diagnostic-vectors-per-head", type=int, default=2048)
    parser.add_argument("--fisher-samples-per-head", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--layers", default="0,10,20,31")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spectral-power", type=float, default=0.5)
    parser.add_argument("--fisher-mixture", type=float, default=0.5)
    parser.add_argument("--method-name", default="SCF-FibQuant")
    parser.add_argument(
        "--norm-squared-weighting",
        action="store_true",
        help=(
            "Weight shared-codebook normal equations by each raw K-vector's "
            "squared norm; assignment is unchanged because this is a per-sample scalar."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def calibration_attention_forward(
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
        raise ValueError("SCF-FibQuant calibration requires use_cache=False")
    context: CalibrationContext = self._m20_fibquant_calibration_context
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
    queries, post_rope_keys = apply_rotary_pos_emb(raw_queries, raw_keys, cos, sin)
    quantized_keys, _ = quantize_fibquant(
        post_rope_keys,
        context.rotation,
        context.codebook,
        block_size=context.block_size,
        chunk_size=context.chunk_size,
    )
    quantized_values, _ = quantize_fibquant(
        values,
        context.rotation,
        context.codebook,
        block_size=context.block_size,
        chunk_size=context.chunk_size,
    )
    repeated_keys = repeat_kv(quantized_keys, self.num_key_value_groups)
    repeated_values = repeat_kv(quantized_values, self.num_key_value_groups)
    weights = torch.matmul(queries, repeated_keys.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : repeated_keys.shape[-2]]
    probabilities = F.softmax(weights, dim=-1, dtype=torch.float32).to(queries.dtype)
    if context.collect:
        samples = post_rope_fisher_samples(
            queries,
            probabilities,
            kv_heads=self.num_key_value_heads,
            samples_per_head=context.fisher_samples_per_head,
        )
        outer = torch.einsum(
            "hsd,hse->hde", samples.double(), samples.double()
        ) / context.fisher_samples_per_head
        if context.fisher_outer_sum is None:
            context.fisher_outer_sum = outer.cpu()
        else:
            context.fisher_outer_sum += outer.cpu()
        context.calls += 1
        context.key_batches.append(post_rope_keys.detach().half().cpu())
    probabilities = F.dropout(
        probabilities, p=self.attention_dropout, training=self.training
    )
    output = torch.matmul(probabilities, repeated_values)
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, self.hidden_size
    )
    return self.o_proj(output), probabilities if output_attentions else None, past_key_value


def bind_calibration(attention: nn.Module, context: CalibrationContext | None) -> None:
    if context is not None:
        attention._m20_fibquant_calibration_context = context
        attention.forward = types.MethodType(calibration_attention_forward, attention)
    else:
        attention.forward = types.MethodType(type(attention).forward, attention)
        if hasattr(attention, "_m20_fibquant_calibration_context"):
            del attention._m20_fibquant_calibration_context


@torch.no_grad()
def normalized_rotated_blocks(
    key_batches: list[torch.Tensor],
    rotation: torch.Tensor,
    *,
    fit_vectors: int,
    diagnostic_vectors: int,
    block_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    # Captured batches are [1,H,T,D]. Use the same token subset for every head
    # so paired head/block diagnostics have an identical support.
    keys = torch.cat(key_batches, dim=0)
    samples, heads, tokens, dimension = map(int, keys.shape)
    flattened = keys.permute(1, 0, 2, 3).reshape(heads, samples * tokens, dimension)
    requested = fit_vectors + diagnostic_vectors
    if requested > flattened.shape[1]:
        raise ValueError(
            f"requested {requested} vectors/head but only {flattened.shape[1]} exist"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(flattened.shape[1], generator=generator)[:requested]
    subset_hash = __import__("hashlib").sha256(
        selected.contiguous().numpy().tobytes()
    ).hexdigest()
    selected_vectors = flattened[:, selected].to(device=device, dtype=torch.float32)
    norms = torch.linalg.vector_norm(selected_vectors, dim=-1, keepdim=True)
    norm_squared = norms.squeeze(-1).double().square().cpu()
    normalized = selected_vectors / norms.clamp_min(torch.finfo(torch.float32).tiny)
    rotated = normalized @ rotation.to(device=device, dtype=torch.float32).T
    blocks = rotated.reshape(heads, requested, dimension // block_size, block_size)
    blocks = blocks.permute(0, 2, 1, 3).contiguous()
    return (
        blocks[:, :, :fit_vectors],
        blocks[:, :, fit_vectors:],
        norm_squared[:, :fit_vectors],
        norm_squared[:, fit_vectors:],
        subset_hash,
    )


def parent_layer_artifact(
    *,
    process: dict[str, Any],
    layer: int,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    heads: int,
    dimension: int,
    block_size: int,
) -> dict[str, Any]:
    blocks = dimension // block_size
    identity = torch.eye(block_size).expand(heads, blocks, block_size, block_size).clone()
    return {
        "provenance": {
            **process,
            "layer": layer,
            "key_mode": "exact_parent_identity",
            "calibrated": False,
        },
        "rotation": rotation.half().cpu(),
        "key": {
            "decode_codebook": codebook.half().cpu(),
            "block_whiteners": identity.float(),
        },
        "value": {"codebook": codebook.half().cpu()},
        "metrics": {
            "identity_parent": True,
            "finite": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sequence_length < 2:
        raise ValueError("samples and sequence length must be positive")
    if args.fit_vectors_per_head < 1 or args.diagnostic_vectors_per_head < 1:
        raise ValueError("fit/diagnostic vector counts must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    arrow_path = Path(args.c4_train_arrow).resolve()
    parent_path = Path(args.parent_codec).resolve()
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_provenance = parent["provenance"]
    rotation = parent["rotation"].float()
    codebook = parent["codebook"].float()
    block_size = int(codebook.shape[-1])
    config = json.loads((model_path / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_key_value_heads"])
    dimension = int(config["hidden_size"]) // int(config["num_attention_heads"])
    selected_layers = parse_layer_spec(args.layers, layers)
    if tuple(rotation.shape) != (dimension, dimension):
        raise ValueError("parent rotation differs from model head dimension")
    rate = fibquant_rate(
        dimension=dimension,
        block_size=block_size,
        codebook_size=len(codebook),
    )
    expected_rate = (
        float(parent_provenance["index_bpa"]),
        float(parent_provenance["metadata_bpa"]),
        float(parent_provenance["total_bpa"]),
    )
    if (rate.index_bpa, rate.metadata_bpa, rate.total_bpa) != expected_rate:
        raise ValueError("parent and reconstructed FibQuant rate ledgers differ")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    if args.calibration_manifest:
        frozen_manifest_path = Path(args.calibration_manifest).resolve()
        token_ids, frozen_manifest = load_manifest_windows(
            tokenizer, frozen_manifest_path
        )
        if str(Path(frozen_manifest.get("model", "")).resolve()) != str(model_path):
            raise ValueError("frozen calibration manifest model differs")
        if int(frozen_manifest["samples"]) != args.samples:
            raise ValueError("frozen calibration sample count differs")
        if int(frozen_manifest["sequence_length"]) != args.sequence_length:
            raise ValueError("frozen calibration sequence length differs")
        if int(frozen_manifest["seed"]) != args.seed:
            raise ValueError("frozen calibration seed differs")
        calibration_manifest = {
            **frozen_manifest,
            "frozen_manifest": str(frozen_manifest_path),
            "frozen_manifest_sha256": sha256_file(frozen_manifest_path),
        }
    else:
        token_ids, sample_manifest = select_windows(
            tokenizer,
            arrow_path,
            samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        calibration_manifest = {
            "model": str(model_path),
            "calibration_dataset": "c4_train",
            "calibration_arrow": str(arrow_path),
            "calibration_arrow_sha256": sha256_file(arrow_path),
            "samples": args.samples,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "sample_manifest": sample_manifest,
        }
    calibration_manifest = {
        **calibration_manifest,
        "parent_codec": str(parent_path),
        "parent_codec_sha256": sha256_file(parent_path),
    }
    manifest_path = output_dir / "calibration_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != calibration_manifest:
            raise ValueError("existing calibration manifest differs")
    else:
        atomic_json(manifest_path, calibration_manifest)
    process = {
        **calibration_manifest,
        "method": args.method_name,
        "format": "m20_fibquant_layer_v2",
        "selected_layers": selected_layers,
        "fit_vectors_per_head": args.fit_vectors_per_head,
        "diagnostic_vectors_per_head": args.diagnostic_vectors_per_head,
        "fisher_samples_per_head_window": args.fisher_samples_per_head,
        "iterations": args.iterations,
        "chunk_size": args.chunk_size,
        "kv_heads": heads,
        "head_dim": dimension,
        "block_size": block_size,
        "codebook_size": len(codebook),
        "key_surface": "post_rope",
        "index_bpa": rate.index_bpa,
        "metadata_bpa": rate.metadata_bpa,
        "total_bpa": rate.total_bpa,
        "key_codebook_scope": "per_layer_shared_across_heads_and_blocks",
        "value_codec": "exact_parent_fibquant",
        "metric": "post_rope_spectral_fisher_fixed_block_diagonal",
        "spectral_power": args.spectral_power,
        "fisher_mixture": args.fisher_mixture,
        "cache_representation_changed": False,
        "fibquant_norm_squared_weighting": bool(args.norm_squared_weighting),
    }
    atomic_json(output_dir / "process.json", process)

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    inps, cache = capture_initial_hidden_states(
        model, token_ids.reshape(1, -1), args.sequence_length, device
    )
    outs = torch.empty_like(inps)
    selected_set = set(selected_layers)
    layer_summaries = []
    for layer_index in range(layers):
        started = time.time()
        path = output_dir / f"layer_{layer_index:02d}.pt"
        reuse = path.exists() and not args.overwrite
        collect = layer_index in selected_set and not reuse
        if reuse:
            existing = torch.load(path, map_location="cpu", weights_only=False)
            provenance = existing.get("provenance", {})
            expected = {
                "format": process["format"],
                "model": process["model"],
                "layer": layer_index,
                "index_bpa": process["index_bpa"],
                "metadata_bpa": process["metadata_bpa"],
                "parent_codec_sha256": process["parent_codec_sha256"],
                "spectral_power": process["spectral_power"],
                "fisher_mixture": process["fisher_mixture"],
                "fibquant_norm_squared_weighting": process[
                    "fibquant_norm_squared_weighting"
                ],
            }
            mismatches = [
                name
                for name, value in expected.items()
                if provenance.get(
                    name,
                    0.5 if name in {"spectral_power", "fisher_mixture"} else None,
                )
                != value
            ]
            if mismatches:
                raise ValueError(f"existing {path} differs for {mismatches}")
        layer = model.model.layers[layer_index].to(device)
        context = CalibrationContext(
            rotation=rotation.to(device),
            codebook=codebook.to(device),
            block_size=block_size,
            chunk_size=args.chunk_size,
            collect=collect,
            fisher_samples_per_head=args.fisher_samples_per_head,
        )
        bind_calibration(layer.self_attn, context)
        with torch.no_grad():
            for window in range(inps.shape[0]):
                hidden = inps[window].unsqueeze(0).to(device)
                outs[window].copy_(
                    layer(
                        hidden,
                        **layer_kwargs(cache, model, hidden, device),
                        use_cache=False,
                    )[0][0].cpu()
                )
                del hidden
        bind_calibration(layer.self_attn, None)
        model.model.layers[layer_index] = layer.cpu()
        inps, outs = outs, inps
        if not bool(torch.isfinite(inps).all()):
            raise FloatingPointError(f"parent trajectory is non-finite after layer {layer_index}")
        if reuse:
            print(
                json.dumps(
                    {
                        "event": "layer_reused",
                        "layer": layer_index,
                        "artifact": str(path),
                        "artifact_sha256": sha256_file(path),
                        "seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
            del context
            torch.cuda.empty_cache()
            continue
        if collect:
            if context.fisher_outer_sum is None or context.calls != args.samples:
                raise ValueError("selected layer did not collect every calibration window")
            fisher = normalize_fisher(context.fisher_outer_sum / context.calls)
            geometry = build_block_metric_geometry(
                fisher.to(device),
                rotation.to(device),
                block_size=block_size,
                spectral_power=args.spectral_power,
                fisher_mixture=args.fisher_mixture,
            )
            (
                training,
                diagnostic,
                training_norm_squared,
                diagnostic_norm_squared,
                subset_hash,
            ) = normalized_rotated_blocks(
                context.key_batches,
                rotation,
                fit_vectors=args.fit_vectors_per_head,
                diagnostic_vectors=args.diagnostic_vectors_per_head,
                block_size=block_size,
                seed=args.seed + layer_index * 1000,
                device=device,
            )
            whiteners = geometry.block_whiteners.to(device)
            training_weights = (
                training_norm_squared if args.norm_squared_weighting else None
            )
            diagnostic_weights = (
                diagnostic_norm_squared if args.norm_squared_weighting else None
            )
            parent_train = metric_codebook_diagnostics(
                training,
                codebook.to(device),
                whiteners,
                sample_weights=training_weights,
                chunk_size=args.chunk_size,
            )
            parent_diagnostic = metric_codebook_diagnostics(
                diagnostic,
                codebook.to(device),
                whiteners,
                sample_weights=diagnostic_weights,
                chunk_size=args.chunk_size,
            )
            child_codebook, history = refine_shared_metric_codebook(
                training,
                codebook.to(device),
                whiteners,
                sample_weights=training_weights,
                iterations=args.iterations,
                chunk_size=args.chunk_size,
            )
            child_train = metric_codebook_diagnostics(
                training,
                child_codebook.to(device),
                whiteners,
                sample_weights=training_weights,
                chunk_size=args.chunk_size,
            )
            child_diagnostic = metric_codebook_diagnostics(
                diagnostic,
                child_codebook.to(device),
                whiteners,
                sample_weights=diagnostic_weights,
                chunk_size=args.chunk_size,
            )
            metric_delta = (
                child_diagnostic["metric_vector_mse"]
                / parent_diagnostic["metric_vector_mse"]
                - 1.0
            )
            metrics = {
                "parent_train": parent_train,
                "child_train": child_train,
                "parent_diagnostic": parent_diagnostic,
                "child_diagnostic": child_diagnostic,
                "diagnostic_relative_metric_mse_delta": metric_delta,
                "geometry": geometry.diagnostics,
                "history": history,
                "finite": bool(
                    parent_train["finite"]
                    and child_train["finite"]
                    and parent_diagnostic["finite"]
                    and child_diagnostic["finite"]
                    and geometry.diagnostics["finite"]
                ),
            }
            artifact = {
                "provenance": {
                    **process,
                    "layer": layer_index,
                    "key_mode": "m20_post_rope_block_metric",
                    "calibrated": True,
                    "training_subset_sha256": subset_hash,
                    "elapsed_seconds": time.time() - started,
                },
                "rotation": rotation.half().cpu(),
                "key": {
                    "decode_codebook": child_codebook.half().cpu(),
                    "block_whiteners": geometry.block_whiteners.float().cpu(),
                },
                "value": {"codebook": codebook.half().cpu()},
                "metrics": metrics,
            }
            atomic_torch(path, artifact)
            row = {
                "layer": layer_index,
                "artifact": str(path),
                "artifact_sha256": sha256_file(path),
                "diagnostic_relative_metric_mse_delta": metric_delta,
                "parent_diagnostic": parent_diagnostic,
                "child_diagnostic": child_diagnostic,
                "geometry_offblock_energy_fraction": geometry.diagnostics[
                    "offblock_frobenius_energy_fraction"
                ],
                "seconds": time.time() - started,
                "finite": metrics["finite"],
            }
            layer_summaries.append(row)
            print(json.dumps({"event": "calibrated_layer_complete", **row}), flush=True)
            del (
                training,
                diagnostic,
                training_norm_squared,
                diagnostic_norm_squared,
                fisher,
                geometry,
                whiteners,
                child_codebook,
                artifact,
            )
        else:
            artifact = parent_layer_artifact(
                process=process,
                layer=layer_index,
                rotation=rotation,
                codebook=codebook,
                heads=heads,
                dimension=dimension,
                block_size=block_size,
            )
            atomic_torch(path, artifact)
            print(
                json.dumps(
                    {
                        "event": "identity_layer_complete",
                        "layer": layer_index,
                        "artifact": str(path),
                        "artifact_sha256": sha256_file(path),
                        "seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
        del context
        torch.cuda.empty_cache()
    summary = {
        "status": "complete",
        "format": "m20_fibquant_calibration_summary_v2",
        "process": process,
        "layers": layer_summaries,
        "all_selected_finite": all(row["finite"] for row in layer_summaries),
        "all_selected_diagnostic_metric_improved": all(
            row["diagnostic_relative_metric_mse_delta"] < 0.0
            for row in layer_summaries
        ),
        "artifact_count": len(list(output_dir.glob("layer_*.pt"))),
    }
    atomic_json(output_dir / "calibration_summary.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
