#!/usr/bin/env python3
"""Calibrate paired CQ/NOVA 1-bit parents and frozen M20 children."""

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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from calibrate_gsrq_codebooks import parse_layer_spec, sha256_file
from collect_gsrq_gradient_weights import CQ_GROUP_FORMAT, load_manifest_windows
from facq_kvquant_aligned_ppl import (
    capture_initial_hidden_states,
    layer_kwargs,
)
from lowrate_vq.cq import (
    build_group_m20_whiteners,
    fit_batched_weighted_kmeans,
)
from lowrate_vq.nova import (
    apply_stratified_grouping,
    build_qpca_basis,
    fp8_e5m2_roundtrip,
    m20_query_metric,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(root / "models/LLM-Research/llama-2-7b"))
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--gradient-dir", required=True)
    parser.add_argument("--cq-group-size", type=int, default=8)
    parser.add_argument("--cq-clusters", type=int, default=256)
    parser.add_argument("--cq-iterations", type=int, default=100)
    parser.add_argument("--nova-group-size", type=int, default=4)
    parser.add_argument("--nova-clusters", type=int, default=16)
    parser.add_argument("--nova-iterations", type=int, default=25)
    parser.add_argument("--nova-pool-stride", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
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


def load_cq_statistics(
    gradient_dir: Path,
    calibration: dict[str, Any],
    *,
    total_layers: int,
    samples: int,
    sequence_length: int,
    kv_heads: int,
    groups: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    manifest_path = gradient_dir / "gradient_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CQ_GROUP_FORMAT:
        raise ValueError("gradient directory is not a CQ group-statistics artifact")
    if manifest.get("selected_layers") != list(range(total_layers)):
        raise ValueError("CQ gradients do not cover every model layer")
    if int(manifest.get("cq_group_size", -1)) != group_size:
        raise ValueError("CQ gradient group size differs")
    if manifest.get("sample_manifest") != calibration["sample_manifest"]:
        raise ValueError("CQ gradients and activation calibration tokens differ")
    key_weights = [[] for _ in range(total_layers)]
    value_weights = [[] for _ in range(total_layers)]
    key_outer = torch.zeros(
        total_layers, kv_heads, groups, group_size, group_size, dtype=torch.float64
    )
    value_outer = torch.zeros_like(key_outer)
    for window in range(samples):
        path = gradient_dir / "windows" / f"window_{window:03d}.pt"
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        expected = calibration["sample_manifest"][window]
        if (
            artifact.get("format") != CQ_GROUP_FORMAT
            or artifact.get("token_sha256") != expected["token_sha256"]
        ):
            raise ValueError(f"gradient window {window} provenance differs")
        kw = artifact["key_group_gradient_squared_norms"].float()
        vw = artifact["value_group_gradient_squared_norms"].float()
        ko = artifact["key_group_gradient_outer_sum"].double()
        vo = artifact["value_group_gradient_outer_sum"].double()
        expected_weights = (total_layers, sequence_length, kv_heads, groups)
        expected_outer = (total_layers, kv_heads, groups, group_size, group_size)
        if tuple(kw.shape) != expected_weights or tuple(vw.shape) != expected_weights:
            raise ValueError("CQ group-weight geometry differs")
        if tuple(ko.shape) != expected_outer or tuple(vo.shape) != expected_outer:
            raise ValueError("CQ group-covariance geometry differs")
        for layer in range(total_layers):
            key_weights[layer].append(kw[layer])
            value_weights[layer].append(vw[layer])
        key_outer += ko
        value_outer += vo
    return {
        "key_weights": torch.stack(
            [torch.cat(rows, dim=0) for rows in key_weights]
        ),
        "value_weights": torch.stack(
            [torch.cat(rows, dim=0) for rows in value_weights]
        ),
        "key_outer": key_outer.float(),
        "value_outer": value_outer.float(),
        "gradient_manifest_sha256": sha256_file(manifest_path),
    }


@torch.no_grad()
def capture_layer_qkv(
    layer: torch.nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    cache: dict[str, Any],
    model: torch.nn.Module,
    *,
    device: torch.device,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    samples, sequence_length, _ = inps.shape
    raw_keys = torch.empty(
        samples, sequence_length, kv_heads, head_dim, dtype=inps.dtype
    )
    values = torch.empty_like(raw_keys)
    post_keys = torch.empty_like(raw_keys)
    post_queries = torch.empty(
        samples, sequence_length, query_heads, head_dim, dtype=inps.dtype
    )
    layer = layer.to(device)
    captured: dict[str, torch.Tensor] = {}

    def store(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output.detach()
        return hook

    handles = [
        layer.self_attn.q_proj.register_forward_hook(store("q")),
        layer.self_attn.k_proj.register_forward_hook(store("k")),
        layer.self_attn.v_proj.register_forward_hook(store("v")),
    ]
    try:
        for sample in range(samples):
            captured.clear()
            hidden = inps[sample].unsqueeze(0).to(device)
            kwargs = layer_kwargs(cache, model, hidden, device)
            outs[sample].copy_(
                layer(hidden, **kwargs, use_cache=False)[0][0].cpu()
            )
            if set(captured) != {"q", "k", "v"}:
                raise RuntimeError("Q/K/V projection hooks did not all fire")
            q = captured["q"].reshape(
                1, sequence_length, query_heads, head_dim
            ).transpose(1, 2)
            k = captured["k"].reshape(
                1, sequence_length, kv_heads, head_dim
            ).transpose(1, 2)
            v = captured["v"].reshape(
                1, sequence_length, kv_heads, head_dim
            )
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    raise RuntimeError("NOVA capture lacks RoPE positions")
                position_embeddings = layer.self_attn.rotary_emb(k, position_ids)
            q_post, k_post = apply_rotary_pos_emb(q, k, *position_embeddings)
            raw_keys[sample].copy_(k[0].transpose(0, 1).cpu())
            values[sample].copy_(v[0].cpu())
            post_keys[sample].copy_(k_post[0].transpose(0, 1).cpu())
            post_queries[sample].copy_(q_post[0].transpose(0, 1).cpu())
            del hidden, q, k, v, q_post, k_post
    finally:
        for handle in handles:
            handle.remove()
    return raw_keys, values, post_keys, post_queries


def grouped_vectors(
    activations: torch.Tensor, *, group_size: int, device: torch.device
) -> torch.Tensor:
    samples, tokens, heads, dimension = map(int, activations.shape)
    groups = dimension // group_size
    return (
        activations.reshape(samples * tokens, heads, groups, group_size)
        .permute(1, 2, 0, 3)
        .reshape(heads * groups, samples * tokens, group_size)
        .to(device=device, dtype=torch.float32)
    )


def grouped_weights(weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows, heads, groups = map(int, weights.shape)
    return weights.permute(1, 2, 0).reshape(heads * groups, rows).to(device)


@torch.no_grad()
def nova_moments(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples, tokens, _, dimension = map(int, queries.shape)
    repeats = query_heads // kv_heads
    sigma_q = torch.zeros(
        kv_heads, dimension, dimension, dtype=torch.float64, device=device
    )
    key_sum = torch.zeros(kv_heads, dimension, dtype=torch.float64, device=device)
    key_outer = torch.zeros_like(sigma_q)
    for sample in range(samples):
        q = queries[sample].to(device=device, dtype=torch.float64).reshape(
            tokens, kv_heads, repeats, dimension
        )
        k = keys[sample].to(device=device, dtype=torch.float64)
        sigma_q += torch.einsum("thrd,thre->hde", q, q)
        key_sum += k.sum(0)
        key_outer += torch.einsum("thd,the->hde", k, k)
    rows = samples * tokens
    mean = key_sum / rows
    covariance = key_outer / rows - torch.einsum("hd,he->hde", mean, mean)
    return (sigma_q / rows).float().cpu(), mean.float().cpu(), covariance.float().cpu()


def nova_initial_indices(
    *,
    heads: int,
    groups: int,
    rows: int,
    clusters: int,
    layer: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty(heads * groups, clusters, dtype=torch.long, device=device)
    for head in range(heads):
        generator = torch.Generator(device="cpu").manual_seed(
            seed + layer * heads + head
        )
        selected = torch.randperm(rows, generator=generator)[:clusters].to(device)
        output[head * groups : (head + 1) * groups] = selected
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = Path(args.model).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    gradient_dir = Path(args.gradient_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(calibration["samples"]) != 16 or int(calibration["sequence_length"]) != 2048:
        raise ValueError("CQ/NOVA contract requires exactly 16x2048 calibration")
    if int(calibration["seed"]) != args.seed:
        raise ValueError("calibration seed differs from requested seed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    token_matrix, normalized_calibration = load_manifest_windows(
        tokenizer, manifest_path
    )
    config = json.loads((model_path / "config.json").read_text())
    total_layers = int(config["num_hidden_layers"])
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["hidden_size"]) // query_heads
    selected_layers = parse_layer_spec(args.layers, total_layers)
    if args.cq_group_size != 8 or args.cq_clusters != 256:
        raise ValueError("the frozen CQ-8c8b parent requires group=8,K=256")
    if args.nova_group_size != 4 or args.nova_clusters != 16:
        raise ValueError("the frozen NOVA-1 adaptation requires group=4,K=16")
    if head_dim % args.cq_group_size or head_dim % args.nova_group_size:
        raise ValueError("group sizes must divide the attention head dimension")
    cq_groups = head_dim // args.cq_group_size
    nova_groups = head_dim // args.nova_group_size
    statistics = load_cq_statistics(
        gradient_dir,
        normalized_calibration,
        total_layers=total_layers,
        samples=16,
        sequence_length=2048,
        kv_heads=kv_heads,
        groups=cq_groups,
        group_size=args.cq_group_size,
    )
    process = {
        "format": "cq_nova_1bit_paired_calibration_v1",
        "model": str(model_path),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "gradient_dir": str(gradient_dir),
        "gradient_manifest_sha256": statistics["gradient_manifest_sha256"],
        "samples": 16,
        "sequence_length": 2048,
        "seed": args.seed,
        "selected_layers": selected_layers,
        "cq": {
            "configuration": "CQ-8c8b",
            "group_size": args.cq_group_size,
            "clusters": args.cq_clusters,
            "iterations": args.cq_iterations,
            "initialization": "weighted_kmeans++",
            "key_surface": "pre_rope",
            "index_bpa": 1.0,
            "metadata_bpa": 0.0,
            "total_bpa": 1.0,
        },
        "nova": {
            "source_commit": "d81c77b007d7a8e50ed608134fcb0feba0269ef8",
            "configuration": "G4_4bits_per_group_strict",
            "group_size": args.nova_group_size,
            "clusters": args.nova_clusters,
            "iterations": args.nova_iterations,
            "pool_stride": args.nova_pool_stride,
            "grouping": "stratified",
            "pertoken_norm": False,
            "codebook_roundtrip": "fp8_e5m2_per_group_amax",
            "key_surface": "post_rope",
            "value_codec": "byte_identical_CQ-8c8b_parent",
            "index_bpa": 1.0,
            "metadata_bpa": 0.0,
            "total_bpa": 1.0,
        },
    }
    atomic_json(output_root / "process.json", process)

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
        model, token_matrix.reshape(1, -1), 2048, device
    )
    outs = torch.empty_like(inps)
    selected = set(selected_layers)
    summaries = []
    directories = {
        "cq_parent": output_root / "cq_parent",
        "m20_cq": output_root / "m20_cq",
        "nova_parent": output_root / "nova_parent",
        "m20_nova": output_root / "m20_nova",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for layer_index in range(total_layers):
        layer_started = time.time()
        paths = {
            name: directory / f"layer_{layer_index:02d}.pt"
            for name, directory in directories.items()
        }
        existing = {name: path.exists() for name, path in paths.items()}
        if layer_index in selected and any(existing.values()) and not all(existing.values()):
            raise ValueError(f"layer {layer_index} has a partial paired artifact set")
        if layer_index not in selected or (all(existing.values()) and not args.overwrite):
            layer = model.model.layers[layer_index].to(device)
            with torch.no_grad():
                for sample in range(inps.shape[0]):
                    hidden = inps[sample].unsqueeze(0).to(device)
                    outs[sample].copy_(
                        layer(
                            hidden,
                            **layer_kwargs(cache, model, hidden, device),
                            use_cache=False,
                        )[0][0].cpu()
                    )
            model.model.layers[layer_index] = layer.cpu()
            inps, outs = outs, inps
            print(json.dumps({"event": "layer_skip", "layer": layer_index}), flush=True)
            continue

        layer = model.model.layers[layer_index]
        raw_keys, values, post_keys, post_queries = capture_layer_qkv(
            layer,
            inps,
            outs,
            cache,
            model,
            device=device,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        model.model.layers[layer_index] = layer.cpu()
        inps, outs = outs, inps
        if not bool(torch.isfinite(inps).all()):
            raise FloatingPointError(f"FP16 trajectory became non-finite at layer {layer_index}")

        # CQ-8c8b parent: paper Equation (6), separately for K and V.
        cq_artifacts: dict[str, dict[str, Any]] = {}
        parent_initial: dict[str, torch.Tensor] = {}
        for kind, activations, weights in (
            ("key", raw_keys, statistics["key_weights"][layer_index]),
            ("value", values, statistics["value_weights"][layer_index]),
        ):
            vectors = grouped_vectors(
                activations, group_size=args.cq_group_size, device=device
            )
            scalar_weights = grouped_weights(weights, device)
            fit = fit_batched_weighted_kmeans(
                vectors,
                scalar_weights,
                clusters=args.cq_clusters,
                iterations=args.cq_iterations,
                chunk_size=args.chunk_size,
                seed=args.seed + layer_index * 1000 + (0 if kind == "key" else 100),
            )
            codebooks = fit.codebooks.reshape(
                kv_heads, cq_groups, args.cq_clusters, args.cq_group_size
            ).half().cpu()
            cq_artifacts[kind] = {
                "codebooks": codebooks,
                "metrics": fit.diagnostics,
                "history": fit.history,
            }
            parent_initial[kind] = fit.initial_indices
            del vectors, scalar_weights, fit
            torch.cuda.empty_cache()

        # M20-CQ changes K geometry only; parent V is copied byte-for-byte.
        key_whiteners, whitener_diagnostics = build_group_m20_whiteners(
            statistics["key_outer"][layer_index]
        )
        key_vectors = grouped_vectors(
            raw_keys, group_size=args.cq_group_size, device=device
        )
        key_scalar_weights = grouped_weights(
            statistics["key_weights"][layer_index], device
        )
        child_fit = fit_batched_weighted_kmeans(
            key_vectors,
            key_scalar_weights,
            clusters=args.cq_clusters,
            iterations=args.cq_iterations,
            chunk_size=args.chunk_size,
            seed=args.seed + layer_index * 1000,
            whiteners=key_whiteners.reshape(
                kv_heads * cq_groups, args.cq_group_size, args.cq_group_size
            ).to(device),
            initial_indices=parent_initial["key"].to(device),
        )
        m20_cq_key = child_fit.codebooks.reshape(
            kv_heads, cq_groups, args.cq_clusters, args.cq_group_size
        ).half().cpu()
        m20_cq_metrics = child_fit.diagnostics
        m20_cq_history = child_fit.history
        parent_key_initial_hash = tensor_sha256(parent_initial["key"])
        del key_vectors, key_scalar_weights, child_fit
        torch.cuda.empty_cache()

        common = {
            **process,
            "layer": layer_index,
            "elapsed_seconds": time.time() - layer_started,
        }
        cq_parent_payload = {
            "provenance": {
                **common,
                "format": "cq_8c8b_layer_v1",
                "method": "CQ-8c8b",
                "key_surface": "pre_rope",
                "index_bpa": 1.0,
                "metadata_bpa": 0.0,
                "total_bpa": 1.0,
            },
            "key": cq_artifacts["key"],
            "value": cq_artifacts["value"],
        }
        atomic_torch(paths["cq_parent"], cq_parent_payload)
        parent_value_hash = tensor_sha256(cq_artifacts["value"]["codebooks"])
        m20_cq_payload = {
            "provenance": {
                **common,
                "format": "m20_cq_8c8b_layer_v1",
                "method": "M20-CQ-8c8b",
                "key_surface": "pre_rope",
                "index_bpa": 1.0,
                "metadata_bpa": 0.0,
                "total_bpa": 1.0,
                "parent_value_sha256": parent_value_hash,
                "shared_parent_key_initial_indices_sha256": parent_key_initial_hash,
                "single_changed_component": "K selector/training metric",
            },
            "key": {
                "codebooks": m20_cq_key,
                "selector_whiteners": key_whiteners.float().cpu(),
                "whitener_diagnostics": whitener_diagnostics,
                "metrics": m20_cq_metrics,
                "history": m20_cq_history,
            },
            "value": cq_artifacts["value"],
        }
        if tensor_sha256(m20_cq_payload["value"]["codebooks"]) != parent_value_hash:
            raise AssertionError("M20-CQ did not preserve parent V byte-for-byte")
        atomic_torch(paths["m20_cq"], m20_cq_payload)

        # Official NOVA QPCA geometry, strict-rate G4/K16 adaptation.
        sigma_q, key_mean, key_cov = nova_moments(
            post_queries,
            post_keys,
            query_heads=query_heads,
            kv_heads=kv_heads,
            device=device,
        )
        full_forward, full_inverse = build_qpca_basis(sigma_q, key_cov)
        full_forward, full_inverse, permutation = apply_stratified_grouping(
            full_forward, full_inverse, group_size=args.nova_group_size
        )
        tempered_sigma = m20_query_metric(sigma_q)
        child_forward, child_inverse = build_qpca_basis(tempered_sigma, key_cov)
        child_forward, child_inverse, child_permutation = apply_stratified_grouping(
            child_forward, child_inverse, group_size=args.nova_group_size
        )
        if not torch.equal(permutation, child_permutation):
            raise AssertionError("NOVA parent/child stratified permutations differ")
        identity = torch.eye(head_dim, dtype=torch.float64)
        inverse_errors = {
            "parent": float(
                (full_forward.double() @ full_inverse.double() - identity).abs().max()
            ),
            "child": float(
                (child_forward.double() @ child_inverse.double() - identity).abs().max()
            ),
        }
        pool_index = torch.arange(
            0, post_keys.shape[0] * post_keys.shape[1], args.nova_pool_stride
        )
        raw_pool = post_keys.reshape(-1, kv_heads, head_dim)[pool_index].permute(
            1, 0, 2
        ).to(device=device, dtype=torch.float32)
        pool_rows = int(raw_pool.shape[1])
        mean_device = key_mean.to(device)
        parent_codes = torch.bmm(
            raw_pool - mean_device[:, None], full_forward.to(device).float()
        )
        child_codes = torch.bmm(
            raw_pool - mean_device[:, None], child_forward.to(device).float()
        )
        parent_groups = parent_codes.reshape(
            kv_heads, pool_rows, nova_groups, args.nova_group_size
        ).permute(0, 2, 1, 3).reshape(
            kv_heads * nova_groups, pool_rows, args.nova_group_size
        )
        child_groups = child_codes.reshape(
            kv_heads, pool_rows, nova_groups, args.nova_group_size
        ).permute(0, 2, 1, 3).reshape(
            kv_heads * nova_groups, pool_rows, args.nova_group_size
        )
        nova_initial = nova_initial_indices(
            heads=kv_heads,
            groups=nova_groups,
            rows=pool_rows,
            clusters=args.nova_clusters,
            layer=layer_index,
            seed=args.seed,
            device=device,
        )
        unit_weights = torch.ones(
            kv_heads * nova_groups, pool_rows, device=device
        )
        nova_parent_fit = fit_batched_weighted_kmeans(
            parent_groups,
            unit_weights,
            clusters=args.nova_clusters,
            iterations=args.nova_iterations,
            chunk_size=args.chunk_size,
            seed=args.seed,
            initial_indices=nova_initial,
        )
        nova_child_fit = fit_batched_weighted_kmeans(
            child_groups,
            unit_weights,
            clusters=args.nova_clusters,
            iterations=args.nova_iterations,
            chunk_size=args.chunk_size,
            seed=args.seed,
            initial_indices=nova_initial,
        )
        nova_parent_codebooks = fp8_e5m2_roundtrip(
            nova_parent_fit.codebooks.reshape(
                kv_heads,
                nova_groups,
                args.nova_clusters,
                args.nova_group_size,
            )
        ).cpu()
        nova_child_codebooks = fp8_e5m2_roundtrip(
            nova_child_fit.codebooks.reshape(
                kv_heads,
                nova_groups,
                args.nova_clusters,
                args.nova_group_size,
            )
        ).cpu()
        shared_value = cq_artifacts["value"]["codebooks"]
        nova_common = {
            **common,
            "format": "nova_cqv_1bit_layer_v1",
            "key_surface": "post_rope",
            "index_bpa": 1.0,
            "metadata_bpa": 0.0,
            "total_bpa": 1.0,
            "value_codec": "CQ-8c8b",
            "value_codebook_sha256": tensor_sha256(shared_value),
            "rate_adaptation": "G4_K16_no_per_token_norm",
            "inverse_max_abs_errors": inverse_errors,
        }
        nova_parent_payload = {
            "provenance": {
                **nova_common,
                "method": "NOVA-1+CQV",
                "query_metric": "full_sigma_q",
            },
            "key": {
                "forward": full_forward.float().cpu(),
                "inverse": full_inverse.float().cpu(),
                "mean": key_mean.float().cpu(),
                "codebooks": nova_parent_codebooks,
                "metrics": nova_parent_fit.diagnostics,
                "permutation": permutation,
                "shared_initial_indices_sha256": tensor_sha256(
                    nova_initial.cpu()
                ),
            },
            "value": {"codebooks": shared_value},
        }
        nova_child_payload = {
            "provenance": {
                **nova_common,
                "method": "M20-NOVA-1+CQV",
                "query_metric": "0.5_identity_plus_trace_normalized_sqrt_sigma_q",
                "single_changed_component": "NOVA query metric spectrum",
            },
            "key": {
                "forward": child_forward.float().cpu(),
                "inverse": child_inverse.float().cpu(),
                "mean": key_mean.float().cpu(),
                "codebooks": nova_child_codebooks,
                "metrics": nova_child_fit.diagnostics,
                "permutation": permutation,
                "shared_initial_indices_sha256": tensor_sha256(
                    nova_initial.cpu()
                ),
            },
            "value": {"codebooks": shared_value},
        }
        if tensor_sha256(nova_parent_payload["value"]["codebooks"]) != tensor_sha256(
            nova_child_payload["value"]["codebooks"]
        ):
            raise AssertionError("NOVA parent/child V codebooks differ")
        atomic_torch(paths["nova_parent"], nova_parent_payload)
        atomic_torch(paths["m20_nova"], nova_child_payload)

        row = {
            "layer": layer_index,
            "seconds": time.time() - layer_started,
            "cq_parent_key_metric_mse": cq_artifacts["key"]["metrics"][
                "final_weighted_metric_mse"
            ],
            "nova_parent_metric_mse": nova_parent_fit.diagnostics[
                "final_weighted_metric_mse"
            ],
            "nova_child_metric_mse": nova_child_fit.diagnostics[
                "final_weighted_metric_mse"
            ],
            "nova_inverse_error": inverse_errors,
            "finite": True,
        }
        summaries.append(row)
        atomic_json(output_root / "calibration_progress.json", {
            "status": "running",
            "completed_layers": [entry["layer"] for entry in summaries],
            "latest": row,
            "updated_at": time.time(),
        })
        print("__DS_PROGRESS__ " + json.dumps({"event": "layer_complete", **row}), flush=True)
        del (
            raw_keys,
            values,
            post_keys,
            post_queries,
            cq_artifacts,
            m20_cq_key,
            key_whiteners,
            full_forward,
            full_inverse,
            child_forward,
            child_inverse,
            raw_pool,
            parent_codes,
            child_codes,
            parent_groups,
            child_groups,
            nova_parent_fit,
            nova_child_fit,
        )
        torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "format": "cq_nova_1bit_paired_calibration_v1",
        "completed_layers": selected_layers,
        "elapsed_seconds": time.time() - started,
        "layers": summaries,
        "output_directories": {name: str(path) for name, path in directories.items()},
    }
    atomic_json(output_root / "calibration_summary.json", summary)
    atomic_json(output_root / "calibration_progress.json", {
        "status": "complete",
        "completed_layers": selected_layers,
        "updated_at": time.time(),
    })
    print("__DS_PROGRESS__ " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
