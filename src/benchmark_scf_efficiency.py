#!/usr/bin/env python3
"""Measure real packed-cache inference cost for parent/SCF VQ backends.

The benchmark runs one configuration per process so model and codec memory are
accounted from a clean CUDA allocator.  It intentionally uses the same eager
attention and packed-cache adapters as the sealed LongBench runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM
import transformers


SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from mpe_data import atomic_json_save, sha256_file  # noqa: E402
from gsrq_longbench_cache import (  # noqa: E402
    build_gsrq_contexts as _build_gsrq_contexts,
)
from run_m20_backend_longbench import _build_contexts as _build_grouped_contexts  # noqa: E402


BACKENDS = ("fp16", "gsrq", "cq", "fibquant")
VARIANTS = ("fp16", "parent", "scf")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--model",
        default=str(workspace / "models/LLM-Research/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument("--artifact-dir")
    parser.add_argument(
        "--parent-artifact-dir",
        help=(
            "Base GSRQ codebook directory used with --backend gsrq --variant scf; "
            "the SCF directory supplied through --artifact-dir is loaded as the K-side override"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--token-chunk-size", type=int, default=1024)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.backend == "fp16" and args.variant != "fp16":
        raise ValueError("backend=fp16 requires variant=fp16")
    if args.backend != "fp16" and args.variant == "fp16":
        raise ValueError("quantized backends require variant=parent or scf")
    if args.backend != "fp16" and not args.artifact_dir:
        raise ValueError("quantized backends require --artifact-dir")
    if args.backend == "gsrq" and args.variant == "scf" and not args.parent_artifact_dir:
        raise ValueError("GSRQ-SCF requires --parent-artifact-dir and --artifact-dir")
    if any(length < 1 for length in args.context_lengths):
        raise ValueError("context lengths must be positive")
    if args.context_lengths != sorted(set(args.context_lengths)):
        raise ValueError("context lengths must be sorted and unique")
    if args.decode_steps < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("decode/warmup/repetition values are invalid")


def _tensor_storage_bytes(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return int(storage.data_ptr()), int(storage.nbytes())


def _unique_tensor_bytes(tensors: Iterable[torch.Tensor | None]) -> int:
    storages: dict[int, int] = {}
    for tensor in tensors:
        if tensor is None:
            continue
        pointer, size = _tensor_storage_bytes(tensor)
        storages[pointer] = max(storages.get(pointer, 0), size)
    return int(sum(storages.values()))


def _context_tensors(contexts: list[Any]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for context in contexts:
        for value in vars(context).values():
            if isinstance(value, torch.Tensor):
                tensors.append(value)
    return tensors


def _factor_tensors(backend: str, contexts: list[Any]) -> list[torch.Tensor]:
    field = {
        "gsrq": "key_assignment_whitener",
        "cq": "key_selector_whiteners",
        "fibquant": "key_block_whiteners",
    }[backend]
    return [
        tensor
        for context in contexts
        for tensor in [getattr(context, field, None)]
        if isinstance(tensor, torch.Tensor)
    ]


def _named_context_tensors(contexts: list[Any], field: str) -> list[torch.Tensor]:
    return [
        tensor
        for context in contexts
        for tensor in [getattr(context, field, None)]
        if isinstance(tensor, torch.Tensor)
    ]


def _pretransformed_codeword_bytes(backend: str, contexts: list[Any]) -> int:
    """FP32 bytes needed when SCF candidate tables are materialized once."""

    if backend == "gsrq":
        return int(
            sum(
                context.key_codebooks.numel() * torch.empty((), dtype=torch.float32).element_size()
                for context in contexts
                if context.key_assignment_whitener is not None
            )
        )
    if backend == "cq":
        return int(
            sum(
                context.key_codebooks.numel() * torch.empty((), dtype=torch.float32).element_size()
                for context in contexts
                if context.key_selector_whiteners is not None
            )
        )
    if backend == "fibquant":
        total = 0
        for context in contexts:
            whiteners = context.key_block_whiteners
            codebook = context.key_decode_codebook
            if whiteners is None or codebook is None:
                continue
            heads, blocks, subdim, _ = map(int, whiteners.shape)
            clusters = int(codebook.shape[0])
            total += heads * blocks * clusters * subdim * 4
        return int(total)
    return 0


def _cache_state(contexts: list[Any]) -> dict[str, int]:
    if not contexts:
        return {"tokens": 0, "logical_bytes": 0, "resident_bytes": 0}
    lengths = {int(context.cache.length) for context in contexts}
    if len(lengths) != 1:
        raise RuntimeError("layer cache lengths diverged")
    logical = 0
    resident = 0
    for context in contexts:
        cache = context.cache
        if hasattr(cache, "payload_bytes"):
            logical += int(cache.payload_bytes())
            resident += int(cache.allocated_bytes())
        else:
            logical += int(cache.logical_payload_bytes())
            resident += _unique_tensor_bytes([cache.key, cache.value])
    return {
        "tokens": next(iter(lengths)),
        "logical_bytes": logical,
        "resident_bytes": resident,
    }


def _past_tensors(past_key_values: Any) -> list[torch.Tensor]:
    if past_key_values is None:
        return []
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(past_key_values.key_cache) + list(past_key_values.value_cache)
    tensors: list[torch.Tensor] = []
    for layer in past_key_values:
        if isinstance(layer, torch.Tensor):
            tensors.append(layer)
        elif isinstance(layer, (tuple, list)):
            tensors.extend(value for value in layer if isinstance(value, torch.Tensor))
    return tensors


def _fixed_tokens(length: int, vocab_size: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset, offset + length, device=device, dtype=torch.long)
    # Stay away from Llama-3 special-token IDs while retaining varied embeddings.
    values = (values * 104729 + 1543).remainder(max(vocab_size - 512, 1)) + 10
    return values.unsqueeze(0)


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@torch.inference_mode()
def _run_workload(
    model: torch.nn.Module,
    *,
    device: torch.device,
    context_length: int,
    decode_steps: int,
    contexts: list[Any],
    measure_memory: bool,
) -> dict[str, Any]:
    for context in contexts:
        context.cache.reset()
    prompt = _fixed_tokens(context_length, int(model.config.vocab_size), device)
    continuation = _fixed_tokens(
        decode_steps, int(model.config.vocab_size), device, offset=context_length
    )
    prompt_positions = torch.arange(context_length, device=device, dtype=torch.long)
    prompt_mask = torch.ones((1, context_length), dtype=torch.long, device=device)

    gc.collect()
    torch.cuda.synchronize(device)
    if measure_memory:
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))

    prefill_start, prefill_end = _event_pair()
    wall_start = time.perf_counter()
    prefill_start.record()
    if contexts:
        outputs = model(
            input_ids=prompt,
            attention_mask=prompt_mask,
            position_ids=prompt_positions.unsqueeze(0),
            cache_position=prompt_positions,
            use_cache=False,
        )
        past_key_values = None
    else:
        outputs = model(
            input_ids=prompt,
            attention_mask=prompt_mask,
            position_ids=prompt_positions.unsqueeze(0),
            cache_position=prompt_positions,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
    prefill_end.record()
    prefill_end.synchronize()
    prefill_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    prefill_cuda_ms = float(prefill_start.elapsed_time(prefill_end))
    peak_after_prefill_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_after_prefill_reserved = int(torch.cuda.max_memory_reserved(device))
    del outputs

    decode_start, decode_end = _event_pair()
    wall_start = time.perf_counter()
    decode_start.record()
    for step in range(decode_steps):
        position = context_length + step
        current_position = torch.tensor([position], dtype=torch.long, device=device)
        current_mask = torch.ones((1, position + 1), dtype=torch.long, device=device)
        if contexts:
            outputs = model(
                input_ids=continuation[:, step : step + 1],
                attention_mask=current_mask,
                position_ids=current_position.unsqueeze(0),
                cache_position=current_position,
                use_cache=False,
            )
        else:
            outputs = model(
                input_ids=continuation[:, step : step + 1],
                attention_mask=current_mask,
                position_ids=current_position.unsqueeze(0),
                cache_position=current_position,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
        del outputs
    decode_end.record()
    decode_end.synchronize()
    decode_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    decode_cuda_ms = float(decode_start.elapsed_time(decode_end))

    if contexts:
        cache = _cache_state(contexts)
    else:
        past_tensors = _past_tensors(past_key_values)
        cache = {
            "tokens": context_length + decode_steps,
            "logical_bytes": _unique_tensor_bytes(past_tensors),
            "resident_bytes": _unique_tensor_bytes(past_tensors),
        }
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if not contexts:
        del past_key_values
    del prompt, continuation, prompt_positions, prompt_mask
    torch.cuda.synchronize(device)
    return {
        "prefill_cuda_ms": prefill_cuda_ms,
        "prefill_wall_ms": prefill_wall_ms,
        "prefill_tokens_per_second": context_length / (prefill_cuda_ms / 1000.0),
        "decode_cuda_ms": decode_cuda_ms,
        "decode_wall_ms": decode_wall_ms,
        "decode_ms_per_token": decode_cuda_ms / decode_steps,
        "decode_tokens_per_second": decode_steps / (decode_cuda_ms / 1000.0),
        "end_to_end_cuda_ms": prefill_cuda_ms + decode_cuda_ms,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_prefill_allocated_bytes": peak_after_prefill_allocated,
        "peak_prefill_reserved_bytes": peak_after_prefill_reserved,
        "peak_end_to_end_allocated_bytes": peak_allocated,
        "peak_end_to_end_reserved_bytes": peak_reserved,
        "cache": cache,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "prefill_cuda_ms",
        "prefill_wall_ms",
        "prefill_tokens_per_second",
        "decode_cuda_ms",
        "decode_wall_ms",
        "decode_ms_per_token",
        "decode_tokens_per_second",
        "end_to_end_cuda_ms",
        "peak_prefill_allocated_bytes",
        "peak_end_to_end_allocated_bytes",
        "peak_end_to_end_reserved_bytes",
    ]
    summary = {name: _distribution([float(record[name]) for record in records]) for name in metrics}
    cache_values = {json.dumps(record["cache"], sort_keys=True) for record in records}
    if len(cache_values) != 1:
        raise RuntimeError("cache accounting changed across repetitions")
    summary["cache"] = records[0]["cache"]
    return summary


def _build_runtime(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[Any], dict[str, Any]]:
    if args.backend == "fp16":
        model.config.use_cache = True
        return [], {
            "cache": "Hugging Face FP16 DynamicCache",
            "kernel_fusion": "stock eager attention; no VQ encoder",
        }
    artifact_dir = Path(args.artifact_dir).resolve()
    if args.backend == "gsrq":
        base_codebook_dir = (
            Path(args.parent_artifact_dir).resolve() if args.variant == "scf" else artifact_dir
        )
        key_override_dir = artifact_dir if args.variant == "scf" else None
        contexts, ledger = _build_gsrq_contexts(
            model,
            mode="gsrq",
            codebook_dir=base_codebook_dir,
            parent_path=None,
            rbg_path=None,
            metric_path=None,
            device=device,
            token_chunk_size=args.token_chunk_size,
            key_override_dir=key_override_dir,
        )
    else:
        method = {
            ("cq", "parent"): "cq",
            ("cq", "scf"): "m20_cq",
            ("fibquant", "parent"): "fibquant",
            ("fibquant", "scf"): "m20_fibquant",
        }[(args.backend, args.variant)]
        contexts, ledger = _build_grouped_contexts(
            model,
            method=method,
            artifact_dir=artifact_dir,
            device=device,
            chunk_size=args.token_chunk_size,
        )
    model.config.use_cache = False
    factors = _factor_tensors(args.backend, contexts)
    selector_cross_codebooks = _named_context_tensors(
        contexts, "key_selector_cross_codebooks"
    )
    selector_norms = _named_context_tensors(contexts, "key_selector_norms")
    ledger = {
        **ledger,
        "runtime_static_tensor_bytes_unique": _unique_tensor_bytes(_context_tensors(contexts)),
        "scf_factor_bytes_actual": _unique_tensor_bytes(factors),
        "pretransformed_codeword_bytes_actual": _unique_tensor_bytes(
            selector_cross_codebooks
        ),
        "pretransformed_codeword_norm_bytes_actual": _unique_tensor_bytes(
            selector_norms
        ),
        "pretransformed_codeword_bytes_fp32_if_materialized": _pretransformed_codeword_bytes(
            args.backend, contexts
        ),
        "kernel_fusion": (
            "none; PyTorch eager with batched SCF-FibQuant search and exact "
            "8-bit cast-based packing"
        ),
    }
    return contexts, ledger


def main() -> None:
    args = parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("efficiency benchmark requires CUDA")
    torch.cuda.set_device(device)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()

    started = time.time()
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    contexts, ledger = _build_runtime(model, args, device)
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result: dict[str, Any] = {
        "status": "running",
        "backend": args.backend,
        "variant": args.variant,
        "model": str(model_path),
        "artifact_dir": str(Path(args.artifact_dir).resolve()) if args.artifact_dir else None,
        "parent_artifact_dir": (
            str(Path(args.parent_artifact_dir).resolve()) if args.parent_artifact_dir else None
        ),
        "protocol": {
            "batch_size": 1,
            "context_lengths": args.context_lengths,
            "decode_steps": args.decode_steps,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "token_chunk_size": args.token_chunk_size,
            "model_dtype": "float16",
            "attention_implementation": "eager",
            "timing": "CUDA events plus synchronized wall clock",
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
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "ledger": ledger,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "started_unix": started,
        "measurements": {},
    }
    atomic_json_save(result, output)

    for context_length in args.context_lengths:
        for warmup in range(args.warmups):
            _run_workload(
                model,
                device=device,
                context_length=context_length,
                decode_steps=args.decode_steps,
                contexts=contexts,
                measure_memory=False,
            )
            print(
                "__DS_PROGRESS__ "
                + json.dumps(
                    {
                        "event": "warmup_complete",
                        "backend": args.backend,
                        "variant": args.variant,
                        "context_length": context_length,
                        "warmup": warmup + 1,
                        "total": args.warmups,
                    }
                ),
                flush=True,
            )
        records = []
        for repetition in range(args.repetitions):
            record = _run_workload(
                model,
                device=device,
                context_length=context_length,
                decode_steps=args.decode_steps,
                contexts=contexts,
                measure_memory=True,
            )
            records.append(record)
            print(
                "__DS_PROGRESS__ "
                + json.dumps(
                    {
                        "event": "measurement_complete",
                        "backend": args.backend,
                        "variant": args.variant,
                        "context_length": context_length,
                        "repetition": repetition + 1,
                        "total": args.repetitions,
                        "prefill_ms": round(record["prefill_cuda_ms"], 3),
                        "decode_ms_per_token": round(record["decode_ms_per_token"], 3),
                    }
                ),
                flush=True,
            )
        result["measurements"][str(context_length)] = {
            "records": records,
            "summary": _summarize(records),
        }
        atomic_json_save(result, output)

    result.update({"status": "complete", "completed_unix": time.time()})
    atomic_json_save(result, output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
