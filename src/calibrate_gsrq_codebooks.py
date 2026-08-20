#!/usr/bin/env python3
"""Calibrate layer/head-local GSRQ codebooks on frozen training windows.

The paper does not publish its codebook-sharing granularity.  This executable
uses the literal per-layer, per-128D-subspace interpretation of "per-layer
key/value vectors" and stores one independently resumable artifact per layer.
Multiple processes may safely target disjoint `--layers` in the same output
directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppl_utils import (  # noqa: E402
    capture_initial_hidden_states,
    layer_kwargs,
)
from gsrq.rope_surface import (  # noqa: E402
    POST_ROPE_KEY_SURFACE,
    PRE_ROPE_KEY_SURFACE,
    key_quantization_input,
)
from gsrq.residual_quantizer import fit_residual_quantizer  # noqa: E402
from gsrq.gradient_weights import (  # noqa: E402
    GRADIENT_NORM_SCOPES,
    log_gradient_weights,
    select_gradient_norms,
)
from fw_gsrq.codec import fit_scfw_residual_quantizer  # noqa: E402
from run_gsrq_activation_smoke import select_windows  # noqa: E402


def parse_layer_spec(spec: str, total_layers: int) -> list[int]:
    if spec == "all":
        return list(range(total_layers))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid layer range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    layers = sorted(selected)
    if not layers or layers[0] < 0 or layers[-1] >= total_layers:
        raise ValueError(f"layers must be within [0,{total_layers - 1}]")
    return layers


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(root / "models" / "swift" / "Meta-Llama-3-8B"),
    )
    parser.add_argument(
        "--c4-train-arrow",
        default=str(
            root
            / "huggingface_cache_datasets"
            / "datasets"
            / "allenai___c4"
            / "default-b04fc8a0b8562884"
            / "0.0.0"
            / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            / "c4-train-00000-of-00002.arrow"
        ),
    )
    parser.add_argument(
        "--wikitext2-train-arrow",
        default=str(
            root
            / "huggingface_cache_datasets"
            / "datasets"
            / "Salesforce___wikitext"
            / "wikitext-2-raw-v1"
            / "0.0.0"
            / "b08601e04326c79dfdd32d625aee71d232d685c3"
            / "wikitext-train.arrow"
        ),
    )
    parser.add_argument(
        "--slimpajama-train-file",
        help="Pinned local SlimPajama Arrow or parquet training shard.",
    )
    parser.add_argument(
        "--calibration-dataset",
        choices=["c4", "wikitext2", "slimpajama"],
        default="c4",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--stages", type=int, default=32)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--initialization",
        choices=["sample", "kmeans++"],
        default="sample",
    )
    parser.add_argument(
        "--key-surface",
        choices=["pre_rope", "post_rope"],
        default="pre_rope",
        help="Fit K before or after the model's rotary position embedding.",
    )
    parser.add_argument(
        "--reuse-value-codebook-dir",
        help=(
            "Optional accepted GSRQ directory whose layer-matched V tables "
            "and metrics are copied exactly after provenance checks."
        ),
    )
    parser.add_argument(
        "--reuse-value-wait-seconds",
        type=int,
        default=0,
        help=(
            "Wait up to this many seconds for each layer-matched V artifact; "
            "useful when parent and transformed-K child calibrate concurrently."
        ),
    )
    parser.add_argument(
        "--gradient-weight-dir",
        help=(
            "Directory produced by collect_gsrq_gradient_weights.py. "
            "When present, paper-style log-gradient weights are applied "
            "at every residual stage."
        ),
    )
    parser.add_argument(
        "--gradient-weight-target-median",
        type=float,
        default=1.0,
        help="Paper Appendix-B tau for median normalization.",
    )
    parser.add_argument(
        "--gradient-weight-norm-scope",
        choices=GRADIENT_NORM_SCOPES,
        default="per_head",
        help=(
            "Use each D=128 subspace gradient norm independently, or share "
            "the norm of the concatenated whole-token KV gradient across heads."
        ),
    )
    parser.add_argument(
        "--key-metric-file",
        help=(
            "Optional frozen FARQ metric.pt. When present, train K with the "
            "SCF spectral-Fisher coordinate transform while retaining "
            "ordinary paper-style GSRQ for V."
        ),
    )
    parser.add_argument(
        "--fisher-spectral-power",
        type=float,
        default=0.5,
        help="Frozen exponent p in F^p; the SCF default is 0.5.",
    )
    parser.add_argument(
        "--fisher-mixture",
        type=float,
        default=0.5,
        help=(
            "Frozen alpha in (1-alpha)I + alpha*d*F^p/tr(F^p); "
            "the SCF default is 0.5."
        ),
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated indices/ranges, for example 0,8,16,24 or 0-3.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing selected layer artifacts.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_wikitext2_windows(
    tokenizer: Any,
    path: Path,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Select deterministic windows from the contiguous WikiText-2 train split."""
    data = Dataset.from_file(str(path))
    encoded = tokenizer(
        "\n\n".join(data["text"]),
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids
    max_start = encoded.shape[1] - sequence_length
    if max_start < 0:
        raise ValueError(
            "WikiText-2 train is shorter than the requested sequence length"
        )
    if samples > max_start + 1:
        raise ValueError("requested more unique windows than available starts")
    generator = random.Random(seed)
    starts = generator.sample(range(max_start + 1), samples)
    windows: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    for start in starts:
        window = encoded[:, start : start + sequence_length].contiguous()
        windows.append(window)
        manifest.append(
            {
                "token_start": start,
                "tokens": sequence_length,
                "token_sha256": hashlib.sha256(
                    window.numpy().tobytes()
                ).hexdigest(),
            }
        )
    return torch.cat(windows, dim=0), manifest


def capture_layer_activations(
    layer: torch.nn.Module,
    inps: torch.Tensor,
    outs: torch.Tensor,
    cache: dict[str, Any],
    model: torch.nn.Module,
    *,
    device: torch.device,
    kv_heads: int,
    head_dim: int,
    key_surface: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    samples, sequence_length, _ = inps.shape
    keys = torch.empty(
        samples,
        sequence_length,
        kv_heads,
        head_dim,
        dtype=inps.dtype,
        device="cpu",
    )
    values = torch.empty_like(keys)
    layer = layer.to(device)
    active_sample: dict[str, Any] = {
        "index": 0,
        "position_embeddings": None,
        "position_ids": None,
    }

    def capture_key(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        raw_keys = (
            output[0]
            .reshape(1, sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        position_embeddings = active_sample["position_embeddings"]
        if position_embeddings is None:
            position_ids = active_sample["position_ids"]
            rotary = getattr(layer.self_attn, "rotary_emb", None)
            if rotary is None or position_ids is None:
                raise RuntimeError(
                    "key capture lacks both external and attention-local RoPE"
                )
            position_embeddings = rotary(raw_keys, position_ids)
        cos, sin = position_embeddings
        fitted_keys = key_quantization_input(
            raw_keys,
            cos,
            sin,
            key_surface,
        )
        keys[active_sample["index"]].copy_(
            fitted_keys[0].transpose(0, 1).detach().cpu()
        )

    def capture_value(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        values[active_sample["index"]].copy_(
            output[0]
            .reshape(sequence_length, kv_heads, head_dim)
            .detach()
            .cpu()
        )

    key_handle = layer.self_attn.k_proj.register_forward_hook(capture_key)
    value_handle = layer.self_attn.v_proj.register_forward_hook(capture_value)
    try:
        for sample in range(samples):
            active_sample["index"] = sample
            hidden = inps[sample].unsqueeze(0).to(device)
            kwargs = layer_kwargs(cache, model, hidden, device)
            active_sample["position_embeddings"] = kwargs.get(
                "position_embeddings"
            )
            active_sample["position_ids"] = kwargs.get("position_ids")
            outs[sample].copy_(
                layer(
                    hidden,
                    **kwargs,
                    use_cache=False,
                )[0][0].cpu()
            )
            del hidden
    finally:
        key_handle.remove()
        value_handle.remove()
    return keys, values


def stage_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage": int(row["stage"]),
            "residual_vector_mse_before": float(
                row["residual_mse_before"]
            ),
            "residual_vector_mse_after": float(
                row["residual_mse_after"]
            ),
            "iterations": int(row["gskm_iterations"]),
            "converged": bool(row["gskm_converged"]),
            "last_iteration": row["gskm_last"],
        }
        for row in history
    ]


def fit_layer_codebooks(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    device: torch.device,
    stages: int,
    clusters: int,
    max_iterations: int,
    chunk_size: int,
    seed: int,
    layer_index: int,
    initialization: str,
    gradient_norms: dict[str, torch.Tensor] | None = None,
    gradient_weight_target_median: float = 1.0,
    gradient_weight_norm_scope: str = "per_head",
    reused_value_codebooks: torch.Tensor | None = None,
    reused_value_metrics: list[dict[str, Any]] | None = None,
    key_fisher: torch.Tensor | None = None,
    fisher_spectral_power: float = 0.5,
    fisher_mixture: float = 0.5,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    kv_heads = keys.shape[2]
    key_tables: list[torch.Tensor] = []
    value_tables: list[torch.Tensor] = []
    key_whiteners: list[torch.Tensor] = []
    metrics: dict[str, Any] = {"key": [], "value": []}
    for head in range(kv_heads):
        kinds = [("key", keys, key_tables, 0)]
        if reused_value_codebooks is None:
            kinds.append(("value", values, value_tables, 1))
        for kind, activations, tables, kind_index in kinds:
            vectors = (
                activations[:, :, head]
                .reshape(-1, activations.shape[-1])
                .to(device=device, dtype=torch.float32)
            )
            fit_seed = seed + layer_index * 10000 + head * 100 + kind_index * 50
            sample_weights = None
            weight_diagnostics = None
            if gradient_norms is not None:
                raw_norms = gradient_norms[kind]
                if tuple(raw_norms.shape) != (
                    activations.shape[0],
                    activations.shape[1],
                    kv_heads,
                ):
                    raise ValueError(
                        f"{kind} gradient-norm geometry differs from activations"
                    )
                selected_norms = select_gradient_norms(
                    raw_norms,
                    head=head,
                    scope=gradient_weight_norm_scope,
                )
                sample_weights, weight_diagnostics = log_gradient_weights(
                    selected_norms.to(device),
                    target_median=gradient_weight_target_median,
                )
                weight_diagnostics["gradient_norm_scope"] = (
                    gradient_weight_norm_scope
                )
            started = time.time()
            transform_diagnostics = None
            if kind == "key" and key_fisher is not None:
                rq_model, indices, history, transform_diagnostics = (
                    fit_scfw_residual_quantizer(
                        vectors,
                        key_fisher[head],
                        stages=stages,
                        clusters=clusters,
                        weights=sample_weights,
                        max_iterations=max_iterations,
                        chunk_size=chunk_size,
                        seed=fit_seed,
                        initialization=initialization,
                        spectral_power=fisher_spectral_power,
                        fisher_mixture=fisher_mixture,
                    )
                )
                tables.append(rq_model.decode_codebooks.half())
                key_whiteners.append(rq_model.whitener.float())
            else:
                rq_model, indices, history = fit_residual_quantizer(
                    vectors,
                    stages=stages,
                    clusters=clusters,
                    weights=sample_weights,
                    max_iterations=max_iterations,
                    chunk_size=chunk_size,
                    seed=fit_seed,
                    initialization=initialization,
                )
                tables.append(rq_model.codebooks.half())
            metrics[kind].append(
                {
                    "head": head,
                    "seed": fit_seed,
                    "vectors": int(vectors.shape[0]),
                    "initial_vector_mse": float(
                        vectors.square().sum(dim=-1).mean()
                    ),
                    "final_vector_mse": float(
                        history[-1]["residual_mse_after"]
                    ),
                    "elapsed_seconds": time.time() - started,
                    "gradient_weighting": weight_diagnostics,
                    "key_transform": transform_diagnostics,
                    "stages": stage_rows(history),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "head_complete",
                        "layer": layer_index,
                        "head": head,
                        "kind": kind,
                        "final_vector_mse": history[-1][
                            "residual_mse_after"
                        ],
                        "seconds": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
            del vectors, indices, rq_model, sample_weights
            torch.cuda.empty_cache()
    if reused_value_codebooks is not None:
        if reused_value_metrics is None:
            raise ValueError("reused V codebooks require reused V metrics")
        value_codebooks = reused_value_codebooks.detach().cpu().clone()
        metrics["value"] = copy.deepcopy(reused_value_metrics)
    else:
        value_codebooks = torch.stack(value_tables)
    codebooks = {
        "key": torch.stack(key_tables),
        "value": value_codebooks,
    }
    transform_artifacts = {}
    if key_fisher is not None:
        transform_artifacts["key_assignment_whiteners"] = torch.stack(
            key_whiteners
        )
    return codebooks, metrics, transform_artifacts


def load_layer_gradient_norms(
    directory: Path,
    layer_index: int,
    *,
    calibration_manifest: dict[str, Any],
    expected_heads: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load one layer's exact per-token K/V gradient norms with provenance checks."""

    manifest_path = directory / "gradient_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_samples = int(calibration_manifest["samples"])
    expected_length = int(calibration_manifest["sequence_length"])
    checks = {
        "format": (
            manifest.get("format"),
            "gsrq_final_ce_projection_gradient_norms_v2",
        ),
        "status": (manifest.get("status"), "complete"),
        "model": (
            str(Path(manifest["model"]).resolve()),
            str(Path(calibration_manifest["model"]).resolve()),
        ),
        "calibration_manifest_sha256": (
            manifest["calibration_manifest_sha256"],
            sha256_file(Path(manifest["calibration_manifest"])),
        ),
        "source_sha256": (
            manifest["calibration_source_sha256"],
            calibration_manifest["calibration_arrow_sha256"],
        ),
        "sample_manifest": (
            manifest["sample_manifest"],
            calibration_manifest["sample_manifest"],
        ),
        "samples": (int(manifest["samples"]), expected_samples),
        "sequence_length": (
            int(manifest["sequence_length"]), expected_length
        ),
        "kv_heads": (int(manifest["kv_heads"]), expected_heads),
        "forward_replay": (
            float(manifest["forward_replay_max_abs_logit_error"]), 0.0
        ),
        "saved_gradient_scale": (
            float(manifest.get("saved_gradient_scale", float("nan"))), 1.0
        ),
    }
    mismatches = [
        name for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(f"gradient-weight provenance differs: {mismatches}")
    selected_layers = [int(value) for value in manifest["selected_layers"]]
    if layer_index not in selected_layers:
        raise ValueError("gradient weights do not contain the requested layer")
    stored_layer = selected_layers.index(layer_index)
    kind_rows: dict[str, list[torch.Tensor]] = {"key": [], "value": []}
    for window, expected in enumerate(calibration_manifest["sample_manifest"]):
        path = directory / "windows" / f"window_{window:03d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != manifest["format"]
            or float(payload.get("loss_scale", float("nan")))
            != float(manifest["loss_scale"])
            or
            payload["token_sha256"] != expected["token_sha256"]
            or payload["selected_layers"] != selected_layers
        ):
            raise ValueError(f"gradient window {window} provenance differs")
        for kind in kind_rows:
            values = payload[f"{kind}_gradient_norms"][stored_layer]
            if tuple(values.shape) != (expected_length, expected_heads):
                raise ValueError(
                    f"gradient window {window} {kind} shape differs"
                )
            kind_rows[kind].append(values.float())
    gradients = {
        kind: torch.stack(rows).contiguous()
        for kind, rows in kind_rows.items()
    }
    return gradients, {
        "gradient_weight_manifest": str(manifest_path.resolve()),
        "gradient_weight_manifest_sha256": sha256_file(manifest_path),
        "gradient_loss": manifest["loss"],
        "gradient_surface": manifest["gradient_surface"],
    }


def load_reused_value_artifact(
    directory: Path,
    layer_index: int,
    *,
    calibration_manifest: dict[str, Any],
    expected_shape: tuple[int, int, int, int],
    expected_max_iterations: int,
    expected_gradient_manifest_sha256: str | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Load an accepted layer's V table only after exact contract checks."""
    source_path = (directory / f"layer_{layer_index:02d}.pt").resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    artifact = torch.load(source_path, map_location="cpu", weights_only=False)
    provenance = artifact["provenance"]
    checks = {
        "layer": (int(provenance["layer"]), layer_index),
        "model": (
            str(Path(provenance["model"]).resolve()),
            str(Path(calibration_manifest["model"]).resolve()),
        ),
        "samples": (
            int(provenance["samples"]),
            int(calibration_manifest["samples"]),
        ),
        "sequence_length": (
            int(provenance["sequence_length"]),
            int(calibration_manifest["sequence_length"]),
        ),
        "seed": (
            int(provenance["seed"]),
            int(calibration_manifest["seed"]),
        ),
        "calibration_arrow_sha256": (
            provenance.get(
                "calibration_arrow_sha256",
                provenance.get("c4_train_arrow_sha256"),
            ),
            calibration_manifest["calibration_arrow_sha256"],
        ),
        "sample_manifest": (
            provenance["sample_manifest"],
            calibration_manifest["sample_manifest"],
        ),
        "stages": (int(provenance["stages"]), expected_shape[1]),
        "clusters": (int(provenance["clusters"]), expected_shape[2]),
        "max_iterations": (
            int(provenance["max_iterations"]),
            expected_max_iterations,
        ),
    }
    if expected_gradient_manifest_sha256 is not None:
        checks.update(
            {
                "weighted": (bool(provenance.get("weighted")), True),
                "gradient_weight_manifest_sha256": (
                    provenance.get("gradient_weight_manifest_sha256"),
                    expected_gradient_manifest_sha256,
                ),
            }
        )
    mismatches = [
        name
        for name, (observed, expected) in checks.items()
        if observed != expected
    ]
    if mismatches:
        raise ValueError(
            f"{source_path} is not reusable for fields {mismatches}"
        )
    value_codebooks = artifact["codebooks"]["value"]
    if tuple(value_codebooks.shape) != expected_shape:
        raise ValueError(
            f"{source_path} V shape {tuple(value_codebooks.shape)} "
            f"does not match {expected_shape}"
        )
    if not bool(torch.isfinite(value_codebooks).all()):
        raise ValueError(f"{source_path} has nonfinite V codebooks")
    value_metrics = artifact["metrics"]["value"]
    if len(value_metrics) != expected_shape[0]:
        raise ValueError(f"{source_path} has incomplete V metrics")
    reuse_provenance = {
        "value_reused": True,
        "value_codebook_source": str(source_path),
        "value_codebook_source_sha256": sha256_file(source_path),
        "value_source_surface": provenance.get("surface"),
    }
    return value_codebooks, value_metrics, reuse_provenance


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sequence_length < 1:
        raise ValueError("samples and sequence length must be positive")
    if args.stages < 1 or args.clusters < 2:
        raise ValueError("stages and clusters are invalid")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model_path = Path(args.model).resolve()
    if args.calibration_dataset == "c4":
        arrow_path = Path(args.c4_train_arrow).resolve()
    elif args.calibration_dataset == "wikitext2":
        arrow_path = Path(args.wikitext2_train_arrow).resolve()
    else:
        if not args.slimpajama_train_file:
            raise ValueError(
                "slimpajama calibration requires --slimpajama-train-file"
            )
        arrow_path = Path(args.slimpajama_train_file).resolve()
    config = json.loads((model_path / "config.json").read_text())
    total_layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["hidden_size"]) // int(
        config["num_attention_heads"]
    )
    selected_layers = parse_layer_spec(args.layers, total_layers)
    key_surface = (
        POST_ROPE_KEY_SURFACE
        if args.key_surface == "post_rope"
        else PRE_ROPE_KEY_SURFACE
    )
    reuse_value_dir = (
        Path(args.reuse_value_codebook_dir).resolve()
        if args.reuse_value_codebook_dir
        else None
    )
    gradient_weight_dir = (
        Path(args.gradient_weight_dir).resolve()
        if args.gradient_weight_dir
        else None
    )
    key_metric_path = (
        Path(args.key_metric_file).resolve()
        if args.key_metric_file
        else None
    )
    if args.gradient_weight_target_median <= 0.0:
        raise ValueError("gradient weight target median must be positive")
    if (
        not math.isfinite(args.fisher_spectral_power)
        or args.fisher_spectral_power <= 0.0
    ):
        raise ValueError("fisher spectral power must be finite and positive")
    if (
        not math.isfinite(args.fisher_mixture)
        or not 0.0 <= args.fisher_mixture <= 1.0
    ):
        raise ValueError("fisher mixture must be finite and within [0,1]")
    if key_metric_path is None and (
        args.fisher_spectral_power != 0.5 or args.fisher_mixture != 0.5
    ):
        raise ValueError("spectral-Fisher options require --key-metric-file")
    if (
        reuse_value_dir is not None
        and gradient_weight_dir is not None
        and key_metric_path is None
    ):
        raise ValueError(
            "weighted GSRQ may reuse V only for a frozen transformed-K child"
        )
    if head_dim != 128:
        raise ValueError(f"GSRQ D=128 requires head_dim=128, got {head_dim}")

    key_metrics = None
    key_metric_provenance = None
    if key_metric_path is not None:
        metric_payload = torch.load(
            key_metric_path, map_location="cpu", weights_only=False
        )
        key_metric_provenance = metric_payload.get("provenance", {})
        key_metrics = metric_payload.get("metrics", {}).get("fisher")
        expected_metric_shape = (
            total_layers,
            kv_heads,
            head_dim,
            head_dim,
        )
        if not isinstance(key_metrics, torch.Tensor) or tuple(
            key_metrics.shape
        ) != expected_metric_shape:
            raise ValueError(
                f"key Fisher metric must be {expected_metric_shape}"
            )
        if not bool(torch.isfinite(key_metrics).all()):
            raise ValueError("key Fisher metric contains non-finite values")
        if str(Path(key_metric_provenance.get("model", "")).resolve()) != str(
            model_path
        ):
            raise ValueError("key Fisher metric model provenance differs")
        if key_metric_provenance.get("error_surface") != (
            "pre_rope_key_projection"
        ):
            raise ValueError("key Fisher metric is not defined on pre-RoPE K")
        if key_surface != PRE_ROPE_KEY_SURFACE:
            raise ValueError("M20 key training requires pre-RoPE K")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    if args.calibration_dataset in {"c4", "slimpajama"}:
        token_ids, sample_manifest = select_windows(
            tokenizer,
            arrow_path,
            samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
    else:
        token_ids, sample_manifest = select_wikitext2_windows(
            tokenizer,
            arrow_path,
            samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
    calibration_manifest = {
        "model": str(model_path),
        "calibration_dataset": args.calibration_dataset,
        "calibration_arrow": str(arrow_path),
        "calibration_arrow_sha256": sha256_file(arrow_path),
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "sample_manifest": sample_manifest,
    }
    manifest_path = output_dir / "calibration_manifest.json"
    manifest_text = json.dumps(calibration_manifest, indent=2) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != manifest_text:
            raise ValueError(
                "output directory has a different calibration manifest"
            )
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")

    run_config = {
        **calibration_manifest,
        "selected_layers": selected_layers,
        "total_layers": total_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "stages": args.stages,
        "clusters": args.clusters,
        "max_iterations": args.max_iterations,
        "chunk_size": args.chunk_size,
        "device": args.device,
        "surface": key_surface,
        "value_reused": reuse_value_dir is not None,
        "reuse_value_codebook_dir": (
            str(reuse_value_dir) if reuse_value_dir is not None else None
        ),
        "weighted": gradient_weight_dir is not None,
        "key_transform": (
            (
                "m20_sqrt_fisher_fixed"
                if args.fisher_spectral_power == 0.5
                and args.fisher_mixture == 0.5
                else "scf_spectral_fisher_fixed"
            )
            if key_metrics is not None
            else None
        ),
        "fisher_spectral_power": (
            args.fisher_spectral_power if key_metrics is not None else None
        ),
        "fisher_mixture": (
            args.fisher_mixture if key_metrics is not None else None
        ),
        "identity_mixture": (
            1.0 - args.fisher_mixture if key_metrics is not None else None
        ),
        "key_metric_file": (
            str(key_metric_path) if key_metric_path is not None else None
        ),
        "key_metric_file_sha256": (
            sha256_file(key_metric_path)
            if key_metric_path is not None
            else None
        ),
        "key_metric_provenance": key_metric_provenance,
        "weighting": (
            {
                "formula": "log1p(tau * gradient_norm / (median + epsilon))",
                "normalization_scope": "per_layer_head_kind_codebook",
                "gradient_norm_scope": args.gradient_weight_norm_scope,
                "target_median": args.gradient_weight_target_median,
                "gradient_weight_dir": str(gradient_weight_dir),
            }
            if gradient_weight_dir is not None
            else None
        ),
        "codebook_scope": "per_layer_per_kv_head",
        "initialization": "deterministic_normalized_sample",
        "reproduction_caveat": (
            "The paper leaves codebook scope and initialization implicit; "
            "this run freezes the literal per-layer/subspace interpretation."
        ),
    }
    run_config["initialization"] = args.initialization
    process_config = (
        output_dir
        / (
            "process_"
            + "_".join(str(layer) for layer in selected_layers)
            + ".json"
        )
    )
    process_config.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    # The shared layerwise catcher consumes one contiguous token stream and
    # slices it into windows internally; `select_windows` returns [S,T].
    inps, cache = capture_initial_hidden_states(
        model,
        token_ids.reshape(1, -1),
        args.sequence_length,
        device,
    )
    outs = torch.empty_like(inps)
    layers = model.model.layers
    max_selected = max(selected_layers)
    selected = set(selected_layers)

    for layer_index in range(max_selected + 1):
        layer_started = time.time()
        layer = layers[layer_index]
        keys, values = capture_layer_activations(
            layer,
            inps,
            outs,
            cache,
            model,
            device=device,
            kv_heads=kv_heads,
            head_dim=head_dim,
            key_surface=key_surface,
        )
        layers[layer_index] = layer.cpu()
        del layer
        inps, outs = outs, inps
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "layer_trajectory_complete",
                    "layer": layer_index,
                    "seconds": round(time.time() - layer_started, 3),
                }
            ),
            flush=True,
        )
        if layer_index not in selected:
            del keys, values
            continue

        artifact_path = output_dir / f"layer_{layer_index:02d}.pt"
        if artifact_path.exists() and not args.overwrite:
            print(
                json.dumps(
                    {
                        "event": "layer_skipped_existing",
                        "layer": layer_index,
                        "path": str(artifact_path),
                    }
                ),
                flush=True,
            )
            del keys, values
            continue
        reused_value_codebooks = None
        reused_value_metrics = None
        reuse_provenance: dict[str, Any] = {"value_reused": False}
        gradient_norms = None
        gradient_provenance: dict[str, Any] = {}
        if gradient_weight_dir is not None:
            gradient_norms, gradient_provenance = load_layer_gradient_norms(
                gradient_weight_dir,
                layer_index,
                calibration_manifest=calibration_manifest,
                expected_heads=kv_heads,
            )
        if reuse_value_dir is not None:
            source_value_path = (
                reuse_value_dir / f"layer_{layer_index:02d}.pt"
            )
            wait_started = time.time()
            while (
                not source_value_path.exists()
                and time.time() - wait_started
                < args.reuse_value_wait_seconds
            ):
                time.sleep(5)
            (
                reused_value_codebooks,
                reused_value_metrics,
                reuse_provenance,
            ) = load_reused_value_artifact(
                reuse_value_dir,
                layer_index,
                calibration_manifest=calibration_manifest,
                expected_shape=(
                    kv_heads,
                    args.stages,
                    args.clusters,
                    head_dim,
                ),
                expected_max_iterations=args.max_iterations,
                expected_gradient_manifest_sha256=(
                    gradient_provenance.get(
                        "gradient_weight_manifest_sha256"
                    )
                    if gradient_weight_dir is not None
                    else None
                ),
            )
        codebooks, metrics, transform_artifacts = fit_layer_codebooks(
            keys,
            values,
            device=device,
            stages=args.stages,
            clusters=args.clusters,
            max_iterations=args.max_iterations,
            chunk_size=args.chunk_size,
            seed=args.seed,
            layer_index=layer_index,
            initialization=args.initialization,
            gradient_norms=gradient_norms,
            gradient_weight_target_median=(
                args.gradient_weight_target_median
            ),
            gradient_weight_norm_scope=args.gradient_weight_norm_scope,
            reused_value_codebooks=reused_value_codebooks,
            reused_value_metrics=reused_value_metrics,
            key_fisher=(
                key_metrics[layer_index] if key_metrics is not None else None
            ),
            fisher_spectral_power=args.fisher_spectral_power,
            fisher_mixture=args.fisher_mixture,
        )
        artifact = {
            "provenance": {
                **run_config,
                **reuse_provenance,
                **gradient_provenance,
                "layer": layer_index,
                "calibration_manifest": str(manifest_path.resolve()),
            },
            "codebooks": codebooks,
            "metrics": metrics,
            **transform_artifacts,
        }
        temporary_path = artifact_path.with_suffix(".pt.partial")
        torch.save(artifact, temporary_path)
        temporary_path.replace(artifact_path)
        print(
            json.dumps(
                {
                    "event": "layer_calibration_complete",
                    "layer": layer_index,
                    "path": str(artifact_path),
                    "sha256": sha256_file(artifact_path),
                    "seconds": round(time.time() - layer_started, 3),
                }
            ),
            flush=True,
        )
        del keys, values, artifact, codebooks, metrics, gradient_norms
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
