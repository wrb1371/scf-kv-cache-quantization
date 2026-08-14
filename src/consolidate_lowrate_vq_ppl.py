#!/usr/bin/env python3
"""Validate and consolidate low-rate VQ shards from NLL sufficient statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--expected-windows", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--expected-dataset", choices=["c4", "wikitext2"], required=True)
    parser.add_argument("--expected-index-bpa", type=float, required=True)
    parser.add_argument("--expected-metadata-bpa", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    provenance = None
    method = None
    for path in sorted(Path(args.shard_dir).glob("shard_*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise ValueError(f"incomplete shard {path}")
        if abs(float(result["index_bpa"]) - args.expected_index_bpa) > 1e-12:
            raise ValueError(f"index rate mismatch in {path}")
        if abs(float(result["metadata_bpa"]) - args.expected_metadata_bpa) > 1e-12:
            raise ValueError(f"metadata rate mismatch in {path}")
        if result.get("dataset") != args.expected_dataset:
            raise ValueError(f"dataset mismatch in {path}")
        if int(result.get("sequence_length", -1)) != args.sequence_length:
            raise ValueError(f"sequence length mismatch in {path}")
        current = {
            key: result[key]
            for key in (
                "method",
                "model",
                "dataset",
                "dataset_root",
                "sequence_length",
                "artifact_dir",
                "index_bpa",
                "metadata_bpa",
                "total_bpa",
                "key_surface",
                "current_token_quantized",
            )
        }
        if provenance is None:
            provenance = current
            method = result["method"]
        elif provenance != current:
            raise ValueError(f"provenance mismatch in {path}")
        rows.extend(result["window_metrics"])
    rows.sort(key=lambda row: int(row["window"]))
    expected = list(range(args.expected_windows))
    observed = [int(row["window"]) for row in rows]
    if observed != expected:
        raise ValueError(f"window coverage mismatch: {observed}")
    expected_tokens = args.expected_windows * (args.sequence_length - 1)
    tokens = sum(int(row["eval_tokens"]) for row in rows)
    if tokens != expected_tokens:
        raise ValueError(f"expected {expected_tokens} tokens, got {tokens}")
    nll = sum(float(row["nll_sum"]) for row in rows)
    mean_nll = nll / tokens
    payload = {
        "status": "complete",
        "method": method,
        "ppl": math.exp(mean_nll),
        "nll_sum": nll,
        "mean_nll": mean_nll,
        "eval_tokens": tokens,
        "evaluated_windows": args.expected_windows,
        "window_metrics": rows,
        "provenance": provenance,
        "validation": {
            "complete_unique_windows": True,
            "token_count_match": True,
            "finite": all(
                math.isfinite(float(row["nll_sum"]))
                and math.isfinite(float(row["mean_nll"]))
                for row in rows
            ),
            "index_bpa_match": True,
            "metadata_bpa_match": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({k: payload[k] for k in (
        "status", "method", "ppl", "mean_nll", "eval_tokens", "evaluated_windows"
    )}))


if __name__ == "__main__":
    main()
