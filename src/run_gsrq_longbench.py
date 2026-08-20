#!/usr/bin/env python3
"""Paper-aligned LongBench generation for GSRQ and SCF-GSRQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsrq_longbench_cache import (  # noqa: E402
    QuantizedLayerContext,
    build_gsrq_contexts,
    cache_bits_per_activation,
)
from longbench_utils import (  # noqa: E402
    TASKS,
    eos_ids,
    load_jsonl,
    load_official_assets,
    prepare_prompt,
    score_task,
    seed_everything,
)
from mpe_data import atomic_json_save, sha256_file  # noqa: E402


METHODS = ("gsrq", "gsrq_scf")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--model",
        default=str(workspace / "models/LLM-Research/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument(
        "--longbench-root",
        default=str(workspace / "huggingface_cache_datasets/longbench/data"),
    )
    parser.add_argument("--longbench-source", required=True)
    parser.add_argument("--codebook-dir", required=True)
    parser.add_argument(
        "--key-override-dir",
        help="SCF key codebooks and frozen assignment factors.",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-examples-per-task", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--token-chunk-size", type=int, default=256)
    parser.add_argument(
        "--prompt-wrapper",
        choices=["raw", "llama3_chat", "llama3_official"],
        default="llama3_official",
    )
    parser.add_argument(
        "--model-dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def _generate(
    model: nn.Module,
    contexts: list[QuantizedLayerContext],
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_ids: list[int],
    prohibit_first_eos: bool,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    for context in contexts:
        context.cache.reset()
    input_ids = input_ids.to(device)
    prompt_length = int(input_ids.shape[1])
    positions = torch.arange(prompt_length, device=device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones(
            (1, prompt_length), dtype=torch.long, device=device
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        use_cache=False,
    )
    logits = outputs.logits[:, -1].float()
    generated: list[int] = []
    for step in range(max_new_tokens):
        current_logits = logits
        if prohibit_first_eos and step == 0:
            current_logits = logits.clone()
            current_logits[:, stop_ids] = -float("inf")
        token = int(torch.argmax(current_logits, dim=-1).item())
        generated.append(token)
        if token in stop_ids or step + 1 == max_new_tokens:
            break
        absolute_position = prompt_length + step
        current_position = torch.tensor(
            [absolute_position], dtype=torch.long, device=device
        )
        outputs = model(
            input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
            attention_mask=torch.ones(
                (1, absolute_position + 1), dtype=torch.long, device=device
            ),
            position_ids=current_position.unsqueeze(0),
            cache_position=current_position,
            use_cache=False,
        )
        logits = outputs.logits[:, -1].float()

    lengths = {context.cache.length for context in contexts}
    if len(lengths) != 1:
        raise AssertionError("layer packed-cache lengths diverged")
    cache_length = next(iter(lengths))
    first = contexts[0]
    heads, stages, clusters, dimension = map(int, first.key_codebooks.shape)
    expected = 2 * heads * cache_length * stages
    payloads = {context.cache.logical_payload_bytes() for context in contexts}
    if payloads != {expected}:
        raise AssertionError("cache payload differs from the exact uint8 ledger")
    return torch.tensor(generated, dtype=torch.long), {
        "logical_cache_tokens": cache_length,
        "logical_kv_payload_bytes_per_layer": expected,
        "total_bpa": cache_bits_per_activation(
            stages=stages, clusters=clusters, dimension=dimension
        ),
        "cache_dtype": "uint8",
        "per_token_state_bytes": 0,
    }


def main() -> None:
    args = parse_args()
    if args.max_examples_per_task < 0 or args.max_input_length < 1:
        raise ValueError("example limit/max input length is invalid")
    if args.method == "gsrq_scf" and not args.key_override_dir:
        raise ValueError("gsrq_scf requires --key-override-dir")
    if args.method == "gsrq" and args.key_override_dir:
        raise ValueError("plain gsrq does not use --key-override-dir")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("paper-aligned LongBench requires CUDA")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    codebook_dir = Path(args.codebook_dir).resolve()
    key_override_dir = (
        Path(args.key_override_dir).resolve() if args.key_override_dir else None
    )
    dataset_root = Path(args.longbench_root).resolve()
    source = Path(args.longbench_source).resolve()
    prompts, max_generations, official_metrics = load_official_assets(source)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    contexts, ledger = build_gsrq_contexts(
        model,
        mode="gsrq",
        codebook_dir=codebook_dir,
        parent_path=None,
        rbg_path=None,
        metric_path=None,
        device=device,
        token_chunk_size=args.token_chunk_size,
        key_override_dir=key_override_dir,
    )

    source_files = {
        "dataset2prompt": source / "config/dataset2prompt.json",
        "dataset2maxlen": source / "config/dataset2maxlen.json",
        "metrics": source / "metrics.py",
    }
    manifest = {
        "status": "running",
        "method": args.method,
        "model": str(model_path),
        "model_dtype": args.model_dtype,
        "model_directory_mutated": False,
        "codebook_dir": str(codebook_dir),
        "key_override_dir": (
            str(key_override_dir) if key_override_dir is not None else None
        ),
        "longbench_source": str(source),
        "longbench_source_commit": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        "official_source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "dataset_root": str(dataset_root),
        "dataset_source_sha256": {
            task: sha256_file(dataset_root / f"{task}.jsonl")
            for task in args.tasks
        },
        "tasks": args.tasks,
        "max_examples_per_task": args.max_examples_per_task,
        "max_input_length": args.max_input_length,
        "prompt_wrapper": args.prompt_wrapper,
        "generation": "greedy, num_beams=1, do_sample=False, temperature=1.0",
        "seed": args.seed,
        "device": str(device),
        "token_chunk_size": args.token_chunk_size,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cache_adapter_sha256": sha256_file(
            Path(__file__).with_name("gsrq_longbench_cache.py")
        ),
        "ledger": ledger,
        "started_unix": time.time(),
    }
    atomic_json_save(manifest, output_dir / "run_manifest.json")

    task_scores: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    for task in args.tasks:
        data = load_jsonl(dataset_root / f"{task}.jsonl")
        if args.max_examples_per_task:
            data = data[: args.max_examples_per_task]
        prediction_path = output_dir / f"{task}.jsonl"
        completed = load_jsonl(prediction_path) if prediction_path.exists() else []
        if len(completed) > len(data) or any(
            int(row["row"]) != index for index, row in enumerate(completed)
        ):
            raise ValueError(f"{prediction_path} is not a contiguous resumable prefix")
        with prediction_path.open("a", encoding="utf-8") as writer:
            for row_index in range(len(completed), len(data)):
                row = data[row_index]
                _, input_ids, was_truncated = prepare_prompt(
                    task,
                    row,
                    prompts[task],
                    tokenizer,
                    max_input_length=args.max_input_length,
                    prompt_wrapper=args.prompt_wrapper,
                )
                started = time.time()
                generated, cache_info = _generate(
                    model,
                    contexts,
                    input_ids,
                    max_new_tokens=int(max_generations[task]),
                    stop_ids=eos_ids(tokenizer, task),
                    prohibit_first_eos=(task == "samsum"),
                    device=device,
                )
                record = {
                    "task": task,
                    "row": row_index,
                    "pred": tokenizer.decode(generated, skip_special_tokens=True),
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row["length"],
                    "input_tokens": int(input_ids.shape[1]),
                    "middle_truncated": was_truncated,
                    "generated_tokens": int(generated.numel()),
                    "seconds": time.time() - started,
                    "cache": cache_info,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.flush()
                completed.append(record)
                print(
                    json.dumps(
                        {
                            "event": "longbench_example_complete",
                            "method": args.method,
                            "task": task,
                            "row": row_index,
                            "total": len(data),
                            "seconds": round(record["seconds"], 3),
                        }
                    ),
                    flush=True,
                )
        task_scores[task] = score_task(task, completed, official_metrics)
        task_counts[task] = len(completed)
        atomic_json_save(
            {
                "status": "running",
                "method": args.method,
                "scores": task_scores,
                "counts": task_counts,
                "macro_average_completed_tasks": round(
                    float(np.mean(list(task_scores.values()))), 2
                ),
                "ledger": ledger,
            },
            output_dir / "results.json",
        )
    result = {
        "status": "complete",
        "method": args.method,
        "scores": task_scores,
        "counts": task_counts,
        "macro_average": round(
            float(np.mean([task_scores[task] for task in args.tasks])), 2
        ),
        "ledger": ledger,
        "metric": "official LongBench QA-F1/ROUGE-L F1, percentage points",
    }
    atomic_json_save(result, output_dir / "results.json")
    manifest.update(
        {"status": "complete", "completed_unix": time.time(), "result": result}
    )
    atomic_json_save(manifest, output_dir / "run_manifest.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
