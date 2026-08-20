#!/usr/bin/env python3
"""Interleaved parent/SCF FibQuant latency benchmark on one loaded model.

The two codec contexts are resident together, but only one is active for each
workload. Alternating execution order within every context length reduces bias
from a stable co-tenant GPU workload. Cache/static-memory accounting remains the
responsibility of the clean one-variant benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM
import transformers


SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from benchmark_scf_efficiency import _run_workload, _summarize  # noqa: E402
from mpe_data import atomic_json_save, sha256_file  # noqa: E402
from run_m20_backend_longbench import _build_contexts  # noqa: E402


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    calibration = (
        workspace
        / "FACQ_research_handoff_2026-07-28"
        / "artifacts"
        / "experiment"
        / "m20_backend_longbench_matrix_20260808"
        / "calibration"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(workspace / "models/LLM-Research/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument(
        "--parent-artifact-dir",
        default=str(calibration / "shared/fibquant_2p0"),
    )
    parser.add_argument(
        "--scf-artifact-dir",
        default=str(calibration / "fibquant/2p0/m20_fibquant"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[512, 2048, 4096]
    )
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--token-chunk-size", type=int, default=1024)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _activate(model: torch.nn.Module, contexts: list[Any]) -> None:
    layers = model.model.layers
    if len(layers) != len(contexts):
        raise ValueError("model/context layer counts differ")
    for layer, context in zip(layers, contexts):
        layer.self_attn._packed_codec_context = context


def main() -> None:
    args = parse_args()
    if (
        args.device.split(":", 1)[0] != "cuda"
        or args.warmups < 0
        or args.repetitions < 1
        or args.decode_steps < 1
        or any(length < 1 for length in args.context_lengths)
    ):
        raise ValueError("invalid paired benchmark protocol")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    parent_artifact = Path(args.parent_artifact_dir).resolve()
    scf_artifact = Path(args.scf_artifact_dir).resolve()

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    parent_contexts, parent_ledger = _build_contexts(
        model,
        method="fibquant",
        artifact_dir=parent_artifact,
        device=device,
        chunk_size=args.token_chunk_size,
    )
    scf_contexts, scf_ledger = _build_contexts(
        model,
        method="m20_fibquant",
        artifact_dir=scf_artifact,
        device=device,
        chunk_size=args.token_chunk_size,
    )
    variants = {"parent": parent_contexts, "scf": scf_contexts}
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result: dict[str, Any] = {
        "status": "running",
        "scope": "interleaved paired FibQuant parent/SCF latency",
        "model": str(model_path),
        "artifact_dirs": {
            "parent": str(parent_artifact),
            "scf": str(scf_artifact),
        },
        "protocol": {
            "batch_size": 1,
            "context_lengths": args.context_lengths,
            "decode_steps": args.decode_steps,
            "warmups_per_variant": args.warmups,
            "repetitions": args.repetitions,
            "execution_order": "alternating parent/scf within repetition",
            "attention_implementation": "eager packed-cache adapter",
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_bytes": int(total_bytes),
            "gpu_free_after_load_bytes": int(free_bytes),
        },
        "ledgers": {"parent": parent_ledger, "scf": scf_ledger},
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "started_unix": time.time(),
        "measurements": {},
    }
    atomic_json_save(result, output)

    for context_length in args.context_lengths:
        for warmup in range(args.warmups):
            for variant in ("parent", "scf"):
                _activate(model, variants[variant])
                _run_workload(
                    model,
                    device=device,
                    context_length=context_length,
                    decode_steps=args.decode_steps,
                    contexts=variants[variant],
                    measure_memory=False,
                )
            print(
                json.dumps(
                    {
                        "event": "paired_warmup_complete",
                        "context_length": context_length,
                        "warmup": warmup + 1,
                    }
                ),
                flush=True,
            )
        records: dict[str, list[dict[str, Any]]] = {"parent": [], "scf": []}
        pair_records = []
        for repetition in range(args.repetitions):
            order = ("parent", "scf") if repetition % 2 == 0 else ("scf", "parent")
            current_pair: dict[str, Any] = {"order": list(order)}
            for variant in order:
                _activate(model, variants[variant])
                record = _run_workload(
                    model,
                    device=device,
                    context_length=context_length,
                    decode_steps=args.decode_steps,
                    contexts=variants[variant],
                    measure_memory=False,
                )
                records[variant].append(record)
                current_pair[variant] = record
            pair_records.append(current_pair)
            print(
                json.dumps(
                    {
                        "event": "paired_measurement_complete",
                        "context_length": context_length,
                        "repetition": repetition + 1,
                        "order": list(order),
                        "parent_decode_ms_per_token": round(
                            current_pair["parent"]["decode_ms_per_token"], 3
                        ),
                        "scf_decode_ms_per_token": round(
                            current_pair["scf"]["decode_ms_per_token"], 3
                        ),
                    }
                ),
                flush=True,
            )
        result["measurements"][str(context_length)] = {
            "pairs": pair_records,
            "parent_summary": _summarize(records["parent"]),
            "scf_summary": _summarize(records["scf"]),
        }
        atomic_json_save(result, output)

    result.update({"status": "complete", "completed_unix": time.time()})
    atomic_json_save(result, output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
