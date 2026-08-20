#!/usr/bin/env python3
"""Frozen LongBench generation with true packed CQ/FibQuant KV caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsrq.rope_surface import (  # noqa: E402
    key_states_for_attention,
    normalize_key_surface,
)
from m20_longbench_cache import (  # noqa: E402
    FixedWidthKVCache,
    PackedCodecContext,
    context_ledger,
    decode_cq,
    decode_fibquant,
    encode_cq,
    encode_fibquant,
)
from longbench_utils import (  # noqa: E402
    TASKS,
    eos_ids as _eos_ids,
    load_jsonl as _load_jsonl,
    load_official_assets as _load_official_assets,
    prepare_prompt as _prepare_prompt,
    score_task as _score_task,
    seed_everything as _seed_everything,
)
from mpe_data import atomic_json_save, sha256_file  # noqa: E402
from run_lowrate_vq_ppl import load_context, move_context  # noqa: E402


METHODS = ("cq", "m20_cq", "fibquant", "m20_fibquant")


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    handoff = Path(__file__).resolve().parents[1]
    frozen = handoff / "artifacts/experiment/p3_longbench_gsrq_protocol_20260801"
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
    parser.add_argument(
        "--longbench-source",
        default=str(frozen / "source/LongBench/LongBench"),
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--max-examples-per-task", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--token-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--prompt-wrapper",
        choices=["raw", "llama3_chat", "llama3_official"],
        default="llama3_official",
    )
    parser.add_argument("--model-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _integer_log2(value: int) -> int:
    if value < 2 or value & (value - 1):
        raise ValueError("codebook size must be a power of two")
    return value.bit_length() - 1


def _build_contexts(
    model: nn.Module,
    *,
    method: str,
    artifact_dir: Path,
    device: torch.device,
    chunk_size: int,
) -> tuple[list[PackedCodecContext], dict[str, Any]]:
    contexts: list[PackedCodecContext] = []
    expected_rate: tuple[float, float, float, int, int] | None = None
    artifact_files: list[str] = []
    for layer, decoder_layer in enumerate(model.model.layers):
        loaded = move_context(load_context(method, artifact_dir, layer, chunk_size), device)
        surface = normalize_key_surface(loaded.key_surface)
        if method in {"cq", "m20_cq"}:
            if loaded.key_codebooks is None or loaded.value_codebooks is None:
                raise ValueError("CQ layer artifact is incomplete")
            heads, groups, clusters, subdim = map(int, loaded.key_codebooks.shape)
            if tuple(loaded.value_codebooks.shape) != tuple(loaded.key_codebooks.shape):
                raise ValueError("CQ K/V codebook geometry differs")
            if groups * subdim != 128:
                raise ValueError("LongBench CQ expects 128-D KV heads")
            bits = _integer_log2(clusters)
            index_count = groups
            store_norm = False
            context = PackedCodecContext(
                layer=layer,
                method=method,
                key_surface=surface,
                index_bpa=float(loaded.index_bpa),
                metadata_bpa=float(loaded.metadata_bpa),
                total_bpa=float(loaded.total_bpa),
                bits_per_index=bits,
                index_count=index_count,
                chunk_size=chunk_size,
                cache=FixedWidthKVCache(
                    bits_per_index=bits, index_count=index_count, store_norm=store_norm
                ),
                provenance=loaded.provenance,
                key_codebooks=loaded.key_codebooks,
                value_codebooks=loaded.value_codebooks,
                key_selector_whiteners=loaded.key_selector_whiteners,
            )
            artifact_files.append(str((artifact_dir / f"layer_{layer:02d}.pt").resolve()))
        else:
            if loaded.rotation is None or loaded.codebook is None:
                raise ValueError("FibQuant layer context is incomplete")
            key_codebook = (
                loaded.key_decode_codebook
                if method == "m20_fibquant"
                else loaded.codebook
            )
            if key_codebook is None:
                raise ValueError("FibQuant K codebook is missing")
            clusters, block_size = map(int, key_codebook.shape)
            if 128 % block_size:
                raise ValueError("FibQuant block size must divide 128")
            bits = _integer_log2(clusters)
            index_count = 128 // block_size
            context = PackedCodecContext(
                layer=layer,
                method=method,
                key_surface=surface,
                index_bpa=float(loaded.index_bpa),
                metadata_bpa=float(loaded.metadata_bpa),
                total_bpa=float(loaded.total_bpa),
                bits_per_index=bits,
                index_count=index_count,
                chunk_size=chunk_size,
                cache=FixedWidthKVCache(
                    bits_per_index=bits, index_count=index_count, store_norm=True
                ),
                provenance=loaded.provenance,
                rotation=loaded.rotation,
                value_codebook=loaded.codebook,
                key_decode_codebook=key_codebook,
                key_block_whiteners=loaded.key_block_whiteners,
                key_selector_cross_codebooks=loaded.key_selector_cross_codebooks,
                key_selector_norms=loaded.key_selector_norms,
            )
            artifact_files.append(
                str(
                    (
                        artifact_dir / ("codec.pt" if method == "fibquant" else f"layer_{layer:02d}.pt")
                    ).resolve()
                )
            )
        expected_heads = int(model.config.num_key_value_heads)
        actual_heads = (
            int(context.key_codebooks.shape[0])
            if context.key_codebooks is not None
            else expected_heads
        )
        if actual_heads != expected_heads:
            raise ValueError("codec KV-head count differs from the model")
        represented = context.bits_per_index * context.index_count / 128
        if not math.isclose(represented, context.index_bpa):
            raise ValueError("artifact index BPA differs from fixed-width geometry")
        if context.cache.store_norm and not math.isclose(context.metadata_bpa, 0.125):
            raise ValueError("FibQuant must ledger one FP16 norm per vector")
        if not context.cache.store_norm and context.metadata_bpa != 0.0:
            raise ValueError("CQ cache unexpectedly claims per-token metadata")
        rate = (
            context.index_bpa,
            context.metadata_bpa,
            context.total_bpa,
            context.bits_per_index,
            context.index_count,
        )
        if expected_rate is None:
            expected_rate = rate
        elif rate != expected_rate:
            raise ValueError("layer rate geometries differ")
        decoder_layer.self_attn._packed_codec_context = context
        decoder_layer.self_attn.forward = types.MethodType(
            packed_cache_attention_forward, decoder_layer.self_attn
        )
        contexts.append(context)
    if not contexts:
        raise ValueError("model has no decoder layers")
    first = contexts[0]
    ledger = {
        **context_ledger(first),
        "layers": len(contexts),
        "kv_heads": int(model.config.num_key_value_heads),
        "head_dimension": 128,
        "artifact_dir": str(artifact_dir),
        "artifact_files": artifact_files,
        "persistent_payload": "packed fixed-width indices plus optional FP16 norms",
    }
    return contexts, ledger


def packed_cache_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, None]:
    if past_key_value is not None or use_cache:
        raise ValueError("packed attention owns its cache and requires use_cache=False")
    context: PackedCodecContext = self._packed_codec_context
    batch, query_length, _ = hidden_states.shape
    if batch != 1 or position_ids is None:
        raise ValueError("LongBench packed attention requires batch one and positions")
    if tuple(position_ids.shape) != (1, query_length):
        raise ValueError("position IDs have the wrong packed-attention geometry")
    # The outer generation/evaluation loops own the monotone position sequence.
    # Rebuilding an expected GPU tensor and synchronizing torch.equal in every
    # layer added two validation kernels per attention call without protecting
    # any token-dependent cache state.

    queries = self.q_proj(hidden_states).view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    raw_keys = self.k_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    values = self.v_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    if position_embeddings is None:
        current_cos, current_sin = self.rotary_emb(values, position_ids)
    else:
        current_cos, current_sin = position_embeddings
    query_states, post_rope_keys = apply_rotary_pos_emb(
        queries, raw_keys, current_cos, current_sin
    )

    if context.method in {"cq", "m20_cq"}:
        assert context.key_codebooks is not None and context.value_codebooks is not None
        key_indices = encode_cq(
            raw_keys,
            context.key_codebooks,
            selector_whiteners=context.key_selector_whiteners,
            chunk_size=context.chunk_size,
        )
        value_indices = encode_cq(
            values,
            context.value_codebooks,
            selector_whiteners=None,
            chunk_size=context.chunk_size,
        )
        context.cache.append(key_indices, value_indices)
    else:
        key_indices, key_norm = encode_fibquant(post_rope_keys, context, kind="key")
        value_indices, value_norm = encode_fibquant(values, context, kind="value")
        context.cache.append(
            key_indices,
            value_indices,
            key_norm=key_norm,
            value_norm=value_norm,
        )

    cached_keys, cached_values, key_norms, value_norms = context.cache.tensors()
    if context.method in {"cq", "m20_cq"}:
        assert context.key_codebooks is not None and context.value_codebooks is not None
        keys = decode_cq(
            cached_keys, context.key_codebooks, validate_indices=False
        )
        decoded_values = decode_cq(
            cached_values, context.value_codebooks, validate_indices=False
        )
        full_positions = torch.arange(
            context.cache.length, device=position_ids.device
        ).unsqueeze(0)
        full_cos, full_sin = self.rotary_emb(values, full_positions)
        keys = key_states_for_attention(keys, full_cos, full_sin, context.key_surface)
    else:
        assert key_norms is not None and value_norms is not None
        assert context.rotation is not None
        assert context.key_decode_codebook is not None
        assert context.value_codebook is not None
        keys = decode_fibquant(
            cached_keys,
            key_norms,
            context.rotation,
            context.key_decode_codebook,
            validate_indices=False,
        )
        decoded_values = decode_fibquant(
            cached_values,
            value_norms,
            context.rotation,
            context.value_codebook,
            validate_indices=False,
        )
    keys = repeat_kv(keys, self.num_key_value_groups)
    decoded_values = repeat_kv(decoded_values, self.num_key_value_groups)
    logits = torch.matmul(
        query_states.float(), keys.float().transpose(2, 3)
    ) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : keys.shape[-2]].float()
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    output = torch.matmul(probabilities, decoded_values.float()).to(hidden_states.dtype)
    if output.shape != (batch, self.num_heads, query_length, self.head_dim):
        raise RuntimeError(f"unexpected packed attention output {output.shape}")
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, self.hidden_size
    )
    return self.o_proj(output), probabilities if output_attentions else None, None


@torch.no_grad()
def _generate(
    model: nn.Module,
    contexts: list[PackedCodecContext],
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_ids: list[int],
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
        attention_mask=torch.ones((1, prompt_length), dtype=torch.long, device=device),
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
            current_logits[:, eos_ids] = -float("inf")
        token = int(torch.argmax(current_logits, dim=-1).item())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        absolute_position = prompt_length + step
        current = torch.tensor([[token]], dtype=torch.long, device=device)
        current_position = torch.tensor([absolute_position], dtype=torch.long, device=device)
        outputs = model(
            input_ids=current,
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
    heads = int(model.config.num_key_value_heads)
    expected = 2 * heads * cache_length * (
        first.cache.bytes_per_index_vector + (2 if first.cache.store_norm else 0)
    )
    payloads = {context.cache.payload_bytes() for context in contexts}
    if payloads != {expected}:
        raise AssertionError("packed cache payload differs from the exact ledger")
    return torch.tensor(generated, dtype=torch.long), {
        "logical_cache_tokens": cache_length,
        "logical_kv_payload_bytes_per_layer": expected,
        "resident_allocated_bytes_per_layer": first.cache.allocated_bytes(),
        "index_bpa": first.index_bpa,
        "metadata_bpa": first.metadata_bpa,
        "total_bpa": first.total_bpa,
        "cache_dtype": "packed_uint8+fp16_norm" if first.cache.store_norm else "packed_uint8",
        "per_token_state_bytes": 0,
    }


def main() -> None:
    args = parse_args()
    if args.max_examples_per_task < 0 or args.max_input_length < 1:
        raise ValueError("example limit/max input length is invalid")
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("paper-aligned LongBench requires CUDA")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    dataset_root = Path(args.longbench_root).resolve()
    source = Path(args.longbench_source).resolve()
    prompts, max_generations, official_metrics = _load_official_assets(source)
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
    contexts, ledger = _build_contexts(
        model,
        method=args.method,
        artifact_dir=artifact_dir,
        device=device,
        chunk_size=args.token_chunk_size,
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
        "artifact_dir": str(artifact_dir),
        "longbench_source": str(source),
        "longbench_source_commit": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        "official_source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "dataset_root": str(dataset_root),
        "dataset_source_sha256": {
            task: sha256_file(dataset_root / f"{task}.jsonl") for task in args.tasks
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
            Path(__file__).with_name("m20_longbench_cache.py")
        ),
        "ledger": ledger,
        "started_unix": time.time(),
    }
    atomic_json_save(manifest, output_dir / "run_manifest.json")

    task_scores: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    for task in args.tasks:
        data = _load_jsonl(dataset_root / f"{task}.jsonl")
        if args.max_examples_per_task:
            data = data[: args.max_examples_per_task]
        prediction_path = output_dir / f"{task}.jsonl"
        completed = _load_jsonl(prediction_path) if prediction_path.exists() else []
        if len(completed) > len(data) or any(
            int(row["row"]) != index for index, row in enumerate(completed)
        ):
            raise ValueError(f"{prediction_path} is not a contiguous resumable prefix")
        with prediction_path.open("a", encoding="utf-8") as writer:
            for row_index in range(len(completed), len(data)):
                row = data[row_index]
                _, input_ids, was_truncated = _prepare_prompt(
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
                    eos_ids=_eos_ids(tokenizer, task),
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
                    "__DS_PROGRESS__ "
                    + json.dumps(
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
        task_scores[task] = _score_task(task, completed, official_metrics)
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
    manifest.update({"status": "complete", "completed_unix": time.time(), "result": result})
    atomic_json_save(manifest, output_dir / "run_manifest.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
