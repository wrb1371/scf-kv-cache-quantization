#!/usr/bin/env python3
"""Fail-closed validation of a weighted parent and spectral-Fisher child."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fw_gsrq.codec import build_spectral_fisher_whitener  # noqa: E402
from gsrq.rope_surface import PRE_ROPE_KEY_SURFACE  # noqa: E402
from mpe_data import atomic_json_save, sha256_file  # noqa: E402


CLUSTERS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--child-dir", type=Path, required=True)
    parser.add_argument("--metric-file", type=Path, required=True)
    parser.add_argument("--gradient-manifest", type=Path, required=True)
    parser.add_argument("--expected-stages", type=int, default=12)
    parser.add_argument("--expected-rate", type=float, default=0.75)
    parser.add_argument("--expected-spectral-power", type=float, default=0.5)
    parser.add_argument("--expected-fisher-mixture", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_provenance(
    provenance: dict[str, Any],
    *,
    layer: int,
    gradient_hash: str,
    child: bool,
    expected_stages: int,
    expected_spectral_power: float = 0.5,
    expected_fisher_mixture: float = 0.5,
) -> list[str]:
    legacy_default = (
        expected_spectral_power == 0.5 and expected_fisher_mixture == 0.5
    )
    expected_transform = (
        "m20_sqrt_fisher_fixed" if legacy_default else "scf_spectral_fisher_fixed"
    )
    checks = {
        "layer": int(provenance.get("layer", -1)) == layer,
        "samples": int(provenance.get("samples", -1)) == 128,
        "sequence_length": int(provenance.get("sequence_length", -1))
        == 2048,
        "seed": int(provenance.get("seed", -1)) == 29,
        "stages": int(provenance.get("stages", -1)) == expected_stages,
        "clusters": int(provenance.get("clusters", -1)) == CLUSTERS,
        "iterations": int(provenance.get("max_iterations", -1)) == 40,
        "surface": provenance.get("surface") == PRE_ROPE_KEY_SURFACE,
        "weighted": bool(provenance.get("weighted")),
        "gradient": provenance.get("gradient_weight_manifest_sha256")
        == gradient_hash,
        "transform": provenance.get("key_transform")
        == (expected_transform if child else None),
    }
    if child and not legacy_default:
        checks.update(
            {
                "spectral_power": math.isclose(
                    float(provenance.get("fisher_spectral_power", math.nan)),
                    expected_spectral_power,
                ),
                "fisher_mixture": math.isclose(
                    float(provenance.get("fisher_mixture", math.nan)),
                    expected_fisher_mixture,
                ),
            }
        )
    return [name for name, passed in checks.items() if not passed]


def main() -> None:
    args = parse_args()
    parent_dir = args.parent_dir.resolve()
    child_dir = args.child_dir.resolve()
    metric_path = args.metric_file.resolve()
    gradient_path = args.gradient_manifest.resolve()
    gradient_hash = sha256_file(gradient_path)
    metric_hash = sha256_file(metric_path)
    metric_payload = torch.load(metric_path, map_location="cpu", weights_only=False)
    fisher = metric_payload["metrics"]["fisher"].float()
    if fisher.ndim != 4 or fisher.shape[-1] != fisher.shape[-2]:
        raise ValueError("Fisher metric must be [layers,heads,D,D]")
    layers, heads, dimension, _ = map(int, fisher.shape)
    if dimension != 128:
        raise ValueError(f"expected D=128, observed {dimension}")
    if args.expected_stages < 1:
        raise ValueError("expected stages must be positive")
    violations: list[str] = []
    layer_rows = []
    child_differs = False
    for layer in range(layers):
        parent_path = parent_dir / f"layer_{layer:02d}.pt"
        child_path = child_dir / f"layer_{layer:02d}.pt"
        if not parent_path.exists() or not child_path.exists():
            violations.append(f"layer {layer}: missing parent or child artifact")
            continue
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        child = torch.load(child_path, map_location="cpu", weights_only=False)
        for name, payload, is_child in (
            ("parent", parent, False),
            ("child", child, True),
        ):
            failures = validate_provenance(
                payload.get("provenance", {}),
                layer=layer,
                gradient_hash=gradient_hash,
                child=is_child,
                expected_stages=args.expected_stages,
                expected_spectral_power=args.expected_spectral_power,
                expected_fisher_mixture=args.expected_fisher_mixture,
            )
            if failures:
                violations.append(f"layer {layer} {name}: {failures}")
            codebooks = payload.get("codebooks", {})
            for kind in ("key", "value"):
                table = codebooks.get(kind)
                if not isinstance(table, torch.Tensor) or tuple(table.shape) != (
                    heads,
                    args.expected_stages,
                    CLUSTERS,
                    dimension,
                ):
                    violations.append(f"layer {layer} {name} {kind}: shape")
                elif table.dtype != torch.float16 or not bool(
                    torch.isfinite(table).all()
                ):
                    violations.append(
                        f"layer {layer} {name} {kind}: dtype/finite"
                    )
        parent_v = parent.get("codebooks", {}).get("value")
        child_v = child.get("codebooks", {}).get("value")
        if not isinstance(parent_v, torch.Tensor) or not isinstance(
            child_v, torch.Tensor
        ) or not torch.equal(parent_v, child_v):
            violations.append(f"layer {layer}: child V is not exact parent V")
        if isinstance(parent.get("codebooks", {}).get("key"), torch.Tensor) and isinstance(
            child.get("codebooks", {}).get("key"), torch.Tensor
        ):
            child_differs = child_differs or not torch.equal(
                parent["codebooks"]["key"], child["codebooks"]["key"]
            )
        whiteners = child.get("key_assignment_whiteners")
        if not isinstance(whiteners, torch.Tensor) or tuple(whiteners.shape) != (
            heads,
            dimension,
            dimension,
        ):
            violations.append(f"layer {layer}: whitener geometry")
        elif not bool(torch.isfinite(whiteners).all()):
            violations.append(f"layer {layer}: non-finite whiteners")
        else:
            maximum_error = 0.0
            for head in range(heads):
                expected = build_spectral_fisher_whitener(
                    fisher[layer, head],
                    spectral_power=args.expected_spectral_power,
                    fisher_mixture=args.expected_fisher_mixture,
                ).whitener
                maximum_error = max(
                    maximum_error,
                    float((whiteners[head] - expected).abs().max()),
                )
            if maximum_error > 1e-6:
                violations.append(
                    f"layer {layer}: frozen whitener error {maximum_error}"
                )
        child_provenance = child.get("provenance", {})
        if child_provenance.get("key_metric_file_sha256") != metric_hash:
            violations.append(f"layer {layer}: metric hash differs")
        if child_provenance.get("value_codebook_source_sha256") != sha256_file(
            parent_path
        ):
            violations.append(f"layer {layer}: V source hash differs")
        layer_rows.append(
            {
                "layer": layer,
                "parent_sha256": sha256_file(parent_path),
                "child_sha256": sha256_file(child_path),
                "value_bitwise_equal": bool(
                    isinstance(parent_v, torch.Tensor)
                    and isinstance(child_v, torch.Tensor)
                    and torch.equal(parent_v, child_v)
                ),
            }
        )
    if not child_differs:
        violations.append("spectral-Fisher child K tables never differ from parent K")
    rate = args.expected_stages * math.log2(CLUSTERS) / dimension
    if not math.isclose(rate, args.expected_rate):
        violations.append(
            f"static index rate {rate} differs from {args.expected_rate} BPA"
        )
    report = {
        "format": "scf_spectral_fisher_calibration_validation_v3",
        "status": "valid" if not violations else "invalid",
        "parent_dir": str(parent_dir),
        "child_dir": str(child_dir),
        "metric_sha256": metric_hash,
        "gradient_manifest_sha256": gradient_hash,
        "layers_checked": len(layer_rows),
        "expected_layers": layers,
        "heads": heads,
        "stages": args.expected_stages,
        "rate_bits_per_activation": rate,
        "fisher_spectral_power": args.expected_spectral_power,
        "fisher_mixture": args.expected_fisher_mixture,
        "per_token_metadata_bits": 0,
        "child_key_differs_from_parent": child_differs,
        "violations": violations,
        "layer_artifacts": layer_rows,
    }
    atomic_json_save(report, args.output.resolve())
    print(json.dumps(report, indent=2), flush=True)
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
