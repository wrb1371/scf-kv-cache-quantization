#!/usr/bin/env python3
"""Bounded real-activation smoke test for the from-paper GSRQ primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsrq.residual_quantizer import fit_residual_quantizer  # noqa: E402


class CaptureComplete(Exception):
    pass


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
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--holdout-samples",
        type=int,
        default=0,
        help="Additional disjoint C4 windows used only for reconstruction audit.",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_text_dataset(path: Path) -> Dataset:
    if path.suffix == ".parquet":
        return Dataset.from_parquet(str(path))
    return Dataset.from_file(str(path))


def select_windows(
    tokenizer: Any,
    path: Path,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    data = _load_text_dataset(path)
    generator = random.Random(seed)
    windows: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    attempted: set[int] = set()
    while len(windows) < samples:
        row = generator.randrange(len(data))
        if row in attempted:
            continue
        attempted.add(row)
        text = data[row]["text"]
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        ).input_ids
        if encoded.shape[1] < sequence_length:
            continue
        max_start = encoded.shape[1] - sequence_length
        start = generator.randint(0, max_start)
        window = encoded[:, start : start + sequence_length].contiguous()
        windows.append(window)
        manifest.append(
            {
                "row": row,
                "token_start": start,
                "tokens": sequence_length,
                "token_sha256": hashlib.sha256(
                    window.numpy().tobytes()
                ).hexdigest(),
            }
        )
    return torch.cat(windows, dim=0), manifest


@torch.no_grad()
def capture_pre_rope_kv(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    layer_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_index]
    captured: dict[str, torch.Tensor] = {}

    def capture_key(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["key"] = output.detach().float().cpu()

    def capture_value(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["value"] = output.detach().float().cpu()
        raise CaptureComplete

    key_handle = layer.self_attn.k_proj.register_forward_hook(capture_key)
    value_handle = layer.self_attn.v_proj.register_forward_hook(
        capture_value
    )
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    try:
        for sample in input_ids:
            try:
                model(sample[None].to(device), use_cache=False)
            except CaptureComplete:
                pass
            if "key" not in captured or "value" not in captured:
                raise RuntimeError("projection hooks did not both fire")
            keys.append(captured.pop("key"))
            values.append(captured.pop("value"))
    finally:
        key_handle.remove()
        value_handle.remove()
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def reconstruction_rows(
    original: torch.Tensor,
    model: Any,
    indices: torch.Tensor,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    original_float = original.float()
    original_norm = torch.linalg.vector_norm(original_float, dim=-1)
    for prefix in range(model.stages + 1):
        reconstruction = model.decode(indices, prefix_stages=prefix)
        error = original_float - reconstruction
        rows.append(
            {
                "prefix": prefix,
                "bpa": prefix * model.codebook_bits / model.dimension,
                "mse": float(error.square().mean()),
                "vector_mse": float(error.square().sum(dim=-1).mean()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        original_float, reconstruction, dim=-1, eps=1e-8
                    ).mean()
                )
                if prefix
                else 0.0,
                "gain_mae": float(
                    (
                        original_norm
                        - torch.linalg.vector_norm(
                            reconstruction, dim=-1
                        )
                    )
                    .abs()
                    .mean()
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    if args.samples < 1 or args.holdout_samples < 0:
        raise ValueError("samples must be positive and holdout non-negative")
    total_samples = args.samples + args.holdout_samples
    token_ids, sample_manifest = select_windows(
        tokenizer,
        Path(args.c4_train_arrow),
        samples=total_samples,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    print(
        "__DS_PROGRESS__ "
        + json.dumps(
            {
                "event": "samples_ready",
                "samples": args.samples,
                "holdout_samples": args.holdout_samples,
                "sequence_length": args.sequence_length,
            }
        ),
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    keys, values = capture_pre_rope_kv(
        model,
        token_ids,
        layer_index=args.layer,
        device=device,
    )
    del model
    torch.cuda.empty_cache()

    kv_width = keys.shape[-1]
    model_config = json.loads((Path(args.model) / "config.json").read_text())
    kv_heads = int(model_config["num_key_value_heads"])
    head_dim = kv_width // kv_heads
    if head_dim != 128:
        raise ValueError(f"expected D=128, got {head_dim}")
    if not 0 <= args.head < kv_heads:
        raise ValueError("head is out of range")
    keys = keys.reshape(
        total_samples, args.sequence_length, kv_heads, head_dim
    )[:, :, args.head].reshape(-1, head_dim)
    values = values.reshape(
        total_samples, args.sequence_length, kv_heads, head_dim
    )[:, :, args.head].reshape(-1, head_dim)

    results: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for name, vectors in (("key", keys), ("value", values)):
        fit_count = args.samples * args.sequence_length
        fit_vectors = vectors[:fit_count]
        holdout_vectors = vectors[fit_count:]
        print(
            "__DS_PROGRESS__ "
            + json.dumps(
                {
                    "event": "fit_start",
                    "kind": name,
                    "vectors": fit_vectors.shape[0],
                    "holdout_vectors": holdout_vectors.shape[0],
                    "dimension": fit_vectors.shape[1],
                }
            ),
            flush=True,
        )
        rq_model, indices, history = fit_residual_quantizer(
            fit_vectors.to(device),
            stages=args.stages,
            clusters=args.clusters,
            max_iterations=args.max_iterations,
            chunk_size=args.chunk_size,
            seed=args.seed + (0 if name == "key" else 1000),
        )
        fit_rows = reconstruction_rows(
            fit_vectors, rq_model, indices
        )
        holdout_indices = None
        holdout_rows = None
        if holdout_vectors.shape[0]:
            holdout_indices = rq_model.encode(
                holdout_vectors.to(device), chunk_size=args.chunk_size
            ).cpu()
            holdout_rows = reconstruction_rows(
                holdout_vectors, rq_model, holdout_indices
            )
        results[name] = {
            "fit_prefix_metrics": fit_rows,
            "holdout_prefix_metrics": holdout_rows,
            "fit_history": history,
        }
        artifacts[name] = {
            "codebooks": rq_model.codebooks.half(),
            "fit_indices": indices.to(torch.uint8)
            if args.clusters <= 256
            else indices,
            "holdout_indices": (
                holdout_indices.to(torch.uint8)
                if holdout_indices is not None and args.clusters <= 256
                else holdout_indices
            ),
        }
        print(
            "__DS_PROGRESS__ "
            + json.dumps(
                {
                    "event": "fit_complete",
                    "kind": name,
                    "fit_final_mse": fit_rows[-1]["mse"],
                    "holdout_final_mse": (
                        holdout_rows[-1]["mse"]
                        if holdout_rows is not None
                        else None
                    ),
                }
            ),
            flush=True,
        )

    provenance = {
        "model": str(Path(args.model).resolve()),
        "c4_train_arrow": str(Path(args.c4_train_arrow).resolve()),
        "samples": args.samples,
        "holdout_samples": args.holdout_samples,
        "sequence_length": args.sequence_length,
        "layer": args.layer,
        "head": args.head,
        "stages": args.stages,
        "clusters": args.clusters,
        "max_iterations": args.max_iterations,
        "seed": args.seed,
        "surface": "pre_rope_key_and_value_projection",
        "weighted": False,
        "reproduction_caveat": (
            "bounded unweighted activation smoke, not accepted GSRQ baseline"
        ),
        "calibration_sample_manifest": sample_manifest[: args.samples],
        "holdout_sample_manifest": sample_manifest[args.samples :],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps({"provenance": provenance, "results": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {"provenance": provenance, "artifacts": artifacts},
        output_dir / "codebooks.pt",
    )
    print(json.dumps({"event": "complete", **provenance}), flush=True)


if __name__ == "__main__":
    main()
