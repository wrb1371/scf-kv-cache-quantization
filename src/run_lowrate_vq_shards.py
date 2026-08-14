#!/usr/bin/env python3
"""Run resumable, disjoint low-rate VQ perplexity shards on one GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset", choices=["c4", "wikitext2"], required=True)
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
        ],
        required=True,
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--expected-windows", type=int, default=256)
    parser.add_argument("--shard-windows", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--starts",
        help="Optional comma-separated shard starts; default covers all windows.",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_result(
    path: Path, start: int, count: int, sequence_length: int, dataset: str
) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        scalar_metrics = (
            result.get("ppl"),
            result.get("nll_sum"),
            result.get("mean_nll"),
        )
        window_metrics = result.get("window_metrics", [])
        finite_metrics = all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in scalar_metrics
        ) and all(
            math.isfinite(float(row.get("nll_sum", float("nan"))))
            and math.isfinite(float(row.get("mean_nll", float("nan"))))
            for row in window_metrics
        )
        return (
            result.get("status") == "complete"
            and int(result.get("window_start", -1)) == start
            and int(result.get("window_end", -1)) == start + count
            and int(result.get("evaluated_windows", -1)) == count
            and int(result.get("eval_tokens", -1)) == count * (sequence_length - 1)
            and result.get("dataset") == dataset
            and len(window_metrics) == count
            and finite_metrics
        )
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    if args.starts:
        starts = [int(value) for value in args.starts.split(",") if value.strip()]
    else:
        starts = list(range(0, args.expected_windows, args.shard_windows))
    state_path = (
        output_dir / f"lane_state_{'_'.join(str(start) for start in starts)}.json"
        if args.starts
        else output_dir / "lane_state.json"
    )
    evaluator = Path(__file__).resolve().parent / "run_lowrate_vq_ppl.py"
    for start in starts:
        count = min(args.shard_windows, args.expected_windows - start)
        shard = output_dir / f"shard_{start:03d}"
        result_path = shard / "result.json"
        if valid_result(
            result_path, start, count, args.sequence_length, args.dataset
        ):
            continue
        state = {
            "status": "running",
            "method": args.method,
            "device": args.device,
            "window_start": start,
            "window_count": count,
            "updated_at": time.time(),
        }
        atomic_json(state_path, state)
        command = [
            sys.executable,
            "-u",
            str(evaluator),
            "--model",
            args.model,
            "--dataset-root",
            args.dataset_root,
            "--dataset",
            args.dataset,
            "--sequence-length",
            str(args.sequence_length),
            "--max-windows",
            str(count),
            "--window-start",
            str(start),
            "--method",
            args.method,
            "--artifact-dir",
            args.artifact_dir,
            "--chunk-size",
            str(args.chunk_size),
            "--device",
            args.device,
            "--output-dir",
            str(shard),
        ]
        with (logs / f"shard_{start:03d}.log").open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode or not valid_result(
            result_path, start, count, args.sequence_length, args.dataset
        ):
            atomic_json(
                state_path,
                {
                    **state,
                    "status": "failed",
                    "returncode": completed.returncode,
                    "updated_at": time.time(),
                },
            )
            raise RuntimeError(f"shard {start} failed")
        atomic_json(
            state_path,
            {**state, "status": "shard_complete", "updated_at": time.time()},
        )
    atomic_json(
        state_path,
        {
            "status": "complete",
            "method": args.method,
            "device": args.device,
            "starts": starts,
            "updated_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
