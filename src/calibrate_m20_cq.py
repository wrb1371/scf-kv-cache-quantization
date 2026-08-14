#!/usr/bin/env python3
"""Calibrate paired Fisher-CQ and spectrum-tempered M20-CQ codebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibrate_cq_nova_1bit import (
    grouped_vectors,
    grouped_weights,
    load_cq_statistics,
)
from calibrate_gsrq_codebooks import parse_layer_spec, sha256_file
from collect_gsrq_gradient_weights import load_manifest_windows
from facq_kvquant_aligned_ppl import capture_initial_hidden_states, layer_kwargs
from lowrate_vq.cq import (
    GroupedKMeansFit,
    build_group_m20_whiteners,
    fit_batched_weighted_kmeans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--gradient-dir", required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--problem-batch-size",
        type=int,
        default=0,
        help="Independent head/group problems per fit call; 0 fits all together.",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--spectral-power", type=float, default=0.5)
    parser.add_argument("--fisher-mixture", type=float, default=0.5)
    parser.add_argument("--child-dir-name", default="m20_cq")
    parser.add_argument("--child-method-name", default="SCF-CQ")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def capture_layer_kv(
    layer: torch.nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    cache: dict[str, Any],
    model: torch.nn.Module,
    *,
    device: torch.device,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples, sequence_length, _ = inps.shape
    raw_keys = torch.empty(
        samples, sequence_length, kv_heads, head_dim, dtype=inps.dtype
    )
    values = torch.empty_like(raw_keys)
    captured: dict[str, torch.Tensor] = {}

    def store(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output.detach()
        return hook

    layer = layer.to(device)
    handles = [
        layer.self_attn.k_proj.register_forward_hook(store("k")),
        layer.self_attn.v_proj.register_forward_hook(store("v")),
    ]
    try:
        for sample in range(samples):
            captured.clear()
            hidden = inps[sample].unsqueeze(0).to(device)
            outs[sample].copy_(
                layer(
                    hidden,
                    **layer_kwargs(cache, model, hidden, device),
                    use_cache=False,
                )[0][0].cpu()
            )
            if set(captured) != {"k", "v"}:
                raise RuntimeError("K/V projection hooks did not both fire")
            raw_keys[sample].copy_(
                captured["k"].reshape(
                    1, sequence_length, kv_heads, head_dim
                )[0].cpu()
            )
            values[sample].copy_(
                captured["v"].reshape(
                    1, sequence_length, kv_heads, head_dim
                )[0].cpu()
            )
            del hidden
    finally:
        for handle in handles:
            handle.remove()
    return raw_keys, values


def merge_fits(fits: list[GroupedKMeansFit]) -> GroupedKMeansFit:
    history: list[dict[str, float | int]] = []
    iterations = len(fits[0].history)
    for iteration in range(iterations):
        rows = [fit.history[iteration] for fit in fits]
        history.append(
            {
                "iteration": iteration,
                "weighted_metric_mse": sum(
                    float(row["weighted_metric_mse"]) for row in rows
                ) / len(rows),
                "empty_clusters": sum(int(row["empty_clusters"]) for row in rows),
            }
        )
    diagnostics = {
        **fits[0].diagnostics,
        "problems": sum(int(fit.diagnostics["problems"]) for fit in fits),
        "final_weighted_metric_mse": sum(
            float(fit.diagnostics["final_weighted_metric_mse"]) for fit in fits
        ) / len(fits),
        "mean_code_utilization": sum(
            float(fit.diagnostics["mean_code_utilization"]) for fit in fits
        ) / len(fits),
        "mean_normalized_code_entropy": sum(
            float(fit.diagnostics["mean_normalized_code_entropy"]) for fit in fits
        ) / len(fits),
        "weight_positive_fraction": sum(
            float(fit.diagnostics["weight_positive_fraction"]) for fit in fits
        ) / len(fits),
        "finite": all(bool(fit.diagnostics["finite"]) for fit in fits),
    }
    return GroupedKMeansFit(
        codebooks=torch.cat([fit.codebooks for fit in fits]),
        initial_indices=torch.cat([fit.initial_indices for fit in fits]),
        assignments=torch.cat([fit.assignments for fit in fits]),
        history=history,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def fit_problem_batches(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    chunk_size: int,
    seed: int,
    problem_batch_size: int,
    whiteners: torch.Tensor | None = None,
    initial_indices: torch.Tensor | None = None,
) -> GroupedKMeansFit:
    problems = int(vectors.shape[0])
    batch = problems if problem_batch_size <= 0 else min(problem_batch_size, problems)
    fits = []
    for start in range(0, problems, batch):
        stop = min(start + batch, problems)
        fit = fit_batched_weighted_kmeans(
            vectors[start:stop],
            weights[start:stop],
            clusters=clusters,
            iterations=iterations,
            chunk_size=chunk_size,
            seed=seed + start,
            whiteners=(whiteners[start:stop] if whiteners is not None else None),
            initial_indices=(
                initial_indices[start:stop] if initial_indices is not None else None
            ),
        )
        fits.append(fit)
    return fits[0] if len(fits) == 1 else merge_fits(fits)


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.group_size < 1 or args.clusters < 2 or args.iterations < 1:
        raise ValueError("invalid CQ geometry or optimization budget")
    bits = math.log2(args.clusters)
    if not bits.is_integer():
        raise ValueError("CQ cluster count must be a power of two")
    index_bpa = bits / args.group_size
    model_path = Path(args.model).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    gradient_dir = Path(args.gradient_dir).resolve()
    output_root = Path(args.output_root).resolve()
    parent_dir = output_root / "cq_parent"
    child_dir = output_root / args.child_dir_name
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    token_matrix, normalized = load_manifest_windows(tokenizer, manifest_path)
    config = json.loads((model_path / "config.json").read_text())
    total_layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    if head_dim % args.group_size:
        raise ValueError("CQ group size must divide the attention head dimension")
    groups = head_dim // args.group_size
    selected_layers = parse_layer_spec(args.layers, total_layers)
    samples, sequence_length = map(int, token_matrix.shape)
    if samples != int(calibration["samples"]):
        raise ValueError("manifest sample count differs from loaded windows")
    statistics = load_cq_statistics(
        gradient_dir,
        normalized,
        total_layers=total_layers,
        samples=samples,
        sequence_length=sequence_length,
        kv_heads=kv_heads,
        groups=groups,
        group_size=args.group_size,
    )
    configuration = f"CQ-{args.group_size}c{int(bits)}b"
    static_codebook_bpa_at_paper_context = (
        16.0 * args.clusters / (16.0 * 65536.0)
    )
    process = {
        "format": "cq_m20_paired_calibration_v2",
        "model": str(model_path),
        "calibration_dataset": calibration["calibration_dataset"],
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "gradient_dir": str(gradient_dir),
        "gradient_manifest_sha256": statistics["gradient_manifest_sha256"],
        "samples": samples,
        "sequence_length": sequence_length,
        "seed": args.seed,
        "selected_layers": selected_layers,
        "configuration": configuration,
        "group_size": args.group_size,
        "clusters": args.clusters,
        "iterations": args.iterations,
        "initialization": "weighted_kmeans++",
        "problem_batch_size": args.problem_batch_size,
        "key_surface": "pre_rope",
        "index_bpa": index_bpa,
        "metadata_bpa": 0.0,
        "total_dynamic_bpa": index_bpa,
        "static_codebook_bpa_batch16_context65536": static_codebook_bpa_at_paper_context,
        "paper_accounted_bpa_batch16_context65536": (
            index_bpa + static_codebook_bpa_at_paper_context
        ),
        "spectral_power": args.spectral_power,
        "fisher_mixture": args.fisher_mixture,
        "child_dir_name": args.child_dir_name,
        "child_method_name": args.child_method_name,
    }
    process_path = output_root / "process.json"
    if process_path.exists() and not args.overwrite:
        existing_process = json.loads(process_path.read_text(encoding="utf-8"))
        if existing_process != process:
            raise ValueError("existing CQ calibration process contract differs")
    else:
        atomic_json(process_path, process)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
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
        model, token_matrix.reshape(1, -1), sequence_length, device
    )
    outs = torch.empty_like(inps)
    selected = set(selected_layers)
    summaries = []
    for layer_index in range(total_layers):
        layer_started = time.time()
        parent_path = parent_dir / f"layer_{layer_index:02d}.pt"
        child_path = child_dir / f"layer_{layer_index:02d}.pt"
        if layer_index in selected and parent_path.exists() != child_path.exists():
            raise ValueError(f"layer {layer_index} has a partial paired artifact set")
        layer = model.model.layers[layer_index]
        raw_keys, values = capture_layer_kv(
            layer,
            inps,
            outs,
            cache,
            model,
            device=device,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        model.model.layers[layer_index] = layer.cpu()
        inps, outs = outs, inps
        if not bool(torch.isfinite(inps).all()):
            raise FloatingPointError(f"FP16 trajectory is non-finite at layer {layer_index}")
        if layer_index not in selected or (
            parent_path.exists() and child_path.exists() and not args.overwrite
        ):
            print(json.dumps({"event": "layer_skip", "layer": layer_index}), flush=True)
            continue

        parent_parts: dict[str, dict[str, Any]] = {}
        parent_initial: dict[str, torch.Tensor] = {}
        for kind, activations, weights in (
            ("key", raw_keys, statistics["key_weights"][layer_index]),
            ("value", values, statistics["value_weights"][layer_index]),
        ):
            vectors = grouped_vectors(
                activations, group_size=args.group_size, device=device
            )
            scalar_weights = grouped_weights(weights, device)
            fit = fit_problem_batches(
                vectors,
                scalar_weights,
                clusters=args.clusters,
                iterations=args.iterations,
                chunk_size=args.chunk_size,
                seed=args.seed + layer_index * 1000 + (0 if kind == "key" else 100),
                problem_batch_size=args.problem_batch_size,
            )
            codebooks = fit.codebooks.reshape(
                kv_heads, groups, args.clusters, args.group_size
            ).half().cpu()
            parent_parts[kind] = {
                "codebooks": codebooks,
                "metrics": fit.diagnostics,
                "history": fit.history,
            }
            parent_initial[kind] = fit.initial_indices
            del vectors, scalar_weights, fit
            torch.cuda.empty_cache()

        key_whiteners, whitener_diagnostics = build_group_m20_whiteners(
            statistics["key_outer"][layer_index],
            spectral_power=args.spectral_power,
            fisher_mixture=args.fisher_mixture,
        )
        key_vectors = grouped_vectors(
            raw_keys, group_size=args.group_size, device=device
        )
        key_weights = grouped_weights(statistics["key_weights"][layer_index], device)
        flat_whiteners = key_whiteners.reshape(
            kv_heads * groups, args.group_size, args.group_size
        ).to(device)
        child_fit = fit_problem_batches(
            key_vectors,
            key_weights,
            clusters=args.clusters,
            iterations=args.iterations,
            chunk_size=args.chunk_size,
            seed=args.seed + layer_index * 1000,
            problem_batch_size=args.problem_batch_size,
            whiteners=flat_whiteners,
            initial_indices=parent_initial["key"].to(device),
        )
        child_key = child_fit.codebooks.reshape(
            kv_heads, groups, args.clusters, args.group_size
        ).half().cpu()
        value_hash = tensor_sha256(parent_parts["value"]["codebooks"])
        initial_hash = tensor_sha256(parent_initial["key"])
        common = {
            **process,
            "layer": layer_index,
            "elapsed_seconds": time.time() - layer_started,
        }
        parent_payload = {
            "provenance": {
                **common,
                "format": "cq_layer_v2",
                "method": configuration,
                "total_bpa": index_bpa,
            },
            "key": parent_parts["key"],
            "value": parent_parts["value"],
        }
        child_payload = {
            "provenance": {
                **common,
                "format": "m20_cq_layer_v2",
                "method": args.child_method_name,
                "total_bpa": index_bpa,
                "parent_value_sha256": value_hash,
                "shared_parent_key_initial_indices_sha256": initial_hash,
                "single_changed_component": "K selector/training metric",
            },
            "key": {
                "codebooks": child_key,
                "selector_whiteners": key_whiteners.float().cpu(),
                "whitener_diagnostics": whitener_diagnostics,
                "metrics": child_fit.diagnostics,
                "history": child_fit.history,
            },
            "value": parent_parts["value"],
        }
        if tensor_sha256(child_payload["value"]["codebooks"]) != value_hash:
            raise AssertionError("M20-CQ did not preserve parent V byte-for-byte")
        atomic_torch(parent_path, parent_payload)
        atomic_torch(child_path, child_payload)
        row = {
            "layer": layer_index,
            "parent": str(parent_path),
            "parent_sha256": sha256_file(parent_path),
            "child": str(child_path),
            "child_sha256": sha256_file(child_path),
            "value_sha256": value_hash,
            "finite": bool(
                parent_parts["key"]["metrics"]["finite"]
                and parent_parts["value"]["metrics"]["finite"]
                and child_fit.diagnostics["finite"]
            ),
            "seconds": time.time() - layer_started,
        }
        summaries.append(row)
        print(json.dumps({"event": "paired_layer_complete", **row}), flush=True)
        del raw_keys, values, key_vectors, key_weights, child_fit, child_key
        torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "format": "cq_m20_calibration_summary_v2",
        "process": process,
        "layers": summaries,
        "artifact_count_parent": len(list(parent_dir.glob("layer_*.pt"))),
        "artifact_count_child": len(list(child_dir.glob("layer_*.pt"))),
        "all_finite": all(row["finite"] for row in summaries),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output_root / "calibration_summary.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
