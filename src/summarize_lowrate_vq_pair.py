#!/usr/bin/env python3
"""Fail-closed paired NLL analysis for consolidated low-rate VQ results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"{path} is incomplete")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    parent_path = Path(args.parent).resolve()
    child_path = Path(args.child).resolve()
    parent = load(parent_path)
    child = load(child_path)
    parent_provenance = parent["provenance"]
    child_provenance = child["provenance"]
    identity_fields = (
        "model",
        "dataset",
        "dataset_root",
        "sequence_length",
        "index_bpa",
        "metadata_bpa",
        "total_bpa",
        "key_surface",
        "current_token_quantized",
    )
    mismatches = [
        name
        for name in identity_fields
        if parent_provenance.get(name) != child_provenance.get(name)
    ]
    if mismatches:
        raise ValueError(f"paired provenance differs for {mismatches}")
    parent_rows = {int(row["window"]): row for row in parent["window_metrics"]}
    child_rows = {int(row["window"]): row for row in child["window_metrics"]}
    if parent_rows.keys() != child_rows.keys():
        raise ValueError("parent/child window coverage differs")
    windows = sorted(parent_rows)
    if windows != list(range(len(windows))):
        raise ValueError("paired windows are not contiguous from zero")
    delta = np.array(
        [
            float(child_rows[index]["mean_nll"])
            - float(parent_rows[index]["mean_nll"])
            for index in windows
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(delta).all()):
        raise ValueError("paired NLL delta is non-finite")
    quarter_rows = []
    for quarter, indices in enumerate(np.array_split(np.arange(len(delta)), 4)):
        quarter_rows.append(
            {
                "quarter": quarter,
                "window_start": int(indices[0]),
                "window_end": int(indices[-1]) + 1,
                "mean_nll_delta": float(delta[indices].mean()),
            }
        )
    generator = np.random.default_rng(args.seed)
    bootstrap_means = np.empty(args.bootstrap_samples, dtype=np.float64)
    chunk = 10000
    for start in range(0, args.bootstrap_samples, chunk):
        stop = min(start + chunk, args.bootstrap_samples)
        sampled = generator.integers(
            0, len(delta), size=(stop - start, len(delta)), endpoint=False
        )
        bootstrap_means[start:stop] = delta[sampled].mean(axis=1)
    aggregate_delta = float(
        float(child["nll_sum"]) / int(child["eval_tokens"])
        - float(parent["nll_sum"]) / int(parent["eval_tokens"])
    )
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    payload = {
        "status": "complete",
        "format": "lowrate_vq_paired_nll_v1",
        "parent": str(parent_path),
        "child": str(child_path),
        "parent_method": parent["method"],
        "child_method": child["method"],
        "windows": len(delta),
        "aggregate_parent_ppl": float(parent["ppl"]),
        "aggregate_child_ppl": float(child["ppl"]),
        "relative_ppl_delta": float(child["ppl"]) / float(parent["ppl"]) - 1.0,
        "aggregate_mean_nll_delta": aggregate_delta,
        "paired_mean_nll_delta": float(delta.mean()),
        "wins": int((delta < 0.0).sum()),
        "ties": int((delta == 0.0).sum()),
        "losses": int((delta > 0.0).sum()),
        "quarters": quarter_rows,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "two_sided_95_percent_ci": [float(ci_low), float(ci_high)],
            "one_sided_95_percent_upper": float(
                np.quantile(bootstrap_means, 0.95)
            ),
        },
        "strong_gate": bool(
            aggregate_delta <= 0.0
            and all(row["mean_nll_delta"] <= 0.0 for row in quarter_rows)
            and float(np.quantile(bootstrap_means, 0.95)) < 0.0
        ),
        "finite": bool(
            math.isfinite(aggregate_delta)
            and np.isfinite(bootstrap_means).all()
        ),
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
