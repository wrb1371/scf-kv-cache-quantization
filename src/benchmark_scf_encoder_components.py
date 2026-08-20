#!/usr/bin/env python3
"""Microbenchmark the deployed SCF encoder and its matrix/search components.

This complements the end-to-end benchmark.  It uses the exact frozen layer-0
tables from the LongBench artifacts and synthetic tensors only to control the
token shape; latency does not depend on token values in these dense kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import numpy as np
import torch


SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from gsrq_longbench_cache import (  # noqa: E402
    greedy_encode_parent,
    greedy_encode_with_assignment_whitener,
)
from lowrate_vq.codecs import quantize_fibquant, squared_distances  # noqa: E402
from lowrate_vq.cq import quantize_cq  # noqa: E402
from lowrate_vq.m20_fibquant import (  # noqa: E402
    prepare_m20_fibquant_search,
    quantize_m20_fibquant_key,
)
from mpe_data import atomic_json_save, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    root = workspace / "FACQ_research_handoff_2026-07-28" / "artifacts" / "experiment"
    matrix = root / "m20_backend_longbench_matrix_20260808" / "calibration"
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 512, 2048])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--gsrq-parent",
        default=str(matrix / "scfw/1p0/weighted_gsrq/layer_00.pt"),
    )
    parser.add_argument(
        "--gsrq-scf",
        default=str(matrix / "scfw/1p0/scfw_gsrq/layer_00.pt"),
    )
    parser.add_argument(
        "--cq-parent",
        default=str(matrix / "cq/2p0/codebooks/cq_parent/layer_00.pt"),
    )
    parser.add_argument(
        "--cq-scf",
        default=str(matrix / "cq/2p0/codebooks/m20_cq/layer_00.pt"),
    )
    parser.add_argument(
        "--fib-parent",
        default=str(matrix / "shared/fibquant_2p0/codec.pt"),
    )
    parser.add_argument(
        "--fib-scf",
        default=str(matrix / "fibquant/2p0/m20_fibquant/layer_00.pt"),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "q1_ms": float(np.percentile(array, 25)),
        "q3_ms": float(np.percentile(array, 75)),
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
        "mean_ms": float(np.mean(array)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


@torch.inference_mode()
def _time_cuda(
    function: Callable[[], Any], *, warmups: int, repetitions: int
) -> dict[str, float]:
    for _ in range(warmups):
        output = function()
        del output
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
        del output
    return _summary(values)


def _states(tokens: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(20260814 + tokens)
    return torch.randn(
        1, 8, tokens, 128, generator=generator, device=device, dtype=torch.float16
    )


def _cq_layout(
    states: torch.Tensor, codebooks: torch.Tensor, factors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, tokens, dimension = map(int, states.shape)
    _, groups, clusters, subdim = map(int, codebooks.shape)
    problems = heads * groups
    raw = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, groups, subdim)
        .permute(0, 2, 1, 3)
        .reshape(problems, batch * tokens, subdim)
        .float()
    )
    tables = codebooks.reshape(problems, clusters, subdim).float()
    work_factors = factors.reshape(problems, subdim, subdim).float()
    if dimension != groups * subdim:
        raise ValueError("invalid CQ layout")
    return raw, tables, work_factors


@torch.inference_mode()
def _cq_search(
    selector_rows: torch.Tensor,
    selector_tables: torch.Tensor,
    raw_tables: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    problems, rows, subdim = map(int, selector_rows.shape)
    table_norms = selector_tables.square().sum(-1).unsqueeze(1)
    decoded = torch.empty_like(selector_rows)
    indices = torch.empty(problems, rows, device=selector_rows.device, dtype=torch.long)
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        current = selector_rows[:, start:stop]
        distances = (
            current.square().sum(-1, keepdim=True)
            + table_norms
            - 2.0 * torch.bmm(current, selector_tables.transpose(1, 2))
        ).clamp_min_(0.0)
        selected = distances.argmin(-1)
        indices[:, start:stop] = selected
        decoded[:, start:stop] = torch.gather(
            raw_tables,
            1,
            selected.unsqueeze(-1).expand(-1, -1, subdim),
        )
    return indices, decoded


def _fib_layout(
    states: torch.Tensor, rotation: torch.Tensor, block_size: int
) -> torch.Tensor:
    batch, heads, tokens, dimension = map(int, states.shape)
    flat = states.reshape(-1, dimension).float()
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    rotated = (flat / norms.clamp_min(torch.finfo(torch.float32).tiny)) @ rotation.float().T
    return rotated.reshape(batch, heads, tokens, dimension // block_size, block_size)


@torch.inference_mode()
def _fib_transform_rows(
    grouped: torch.Tensor, whiteners: torch.Tensor
) -> torch.Tensor:
    output = torch.empty_like(grouped)
    for head in range(int(grouped.shape[1])):
        for block in range(int(grouped.shape[3])):
            output[:, head, :, block] = (
                grouped[:, head, :, block].float() @ whiteners[head, block].float().T
            )
    return output


@torch.inference_mode()
def _fib_transform_tables(centers: torch.Tensor, whiteners: torch.Tensor) -> torch.Tensor:
    heads, blocks = map(int, whiteners.shape[:2])
    output = torch.empty(
        heads,
        blocks,
        int(centers.shape[0]),
        int(centers.shape[1]),
        device=centers.device,
        dtype=torch.float32,
    )
    for head in range(heads):
        for block in range(blocks):
            output[head, block] = centers.float() @ whiteners[head, block].float().T
    return output


@torch.inference_mode()
def _fib_search(
    transformed: torch.Tensor, selectors: torch.Tensor, centers: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, tokens, blocks, subdim = map(int, transformed.shape)
    indices = torch.empty(
        batch, heads, tokens, blocks, device=transformed.device, dtype=torch.long
    )
    decoded = torch.empty_like(transformed)
    for head in range(heads):
        for block in range(blocks):
            vectors = transformed[:, head, :, block].reshape(-1, subdim)
            selected = squared_distances(vectors, selectors[head, block]).argmin(-1)
            indices[:, head, :, block] = selected.reshape(batch, tokens)
            decoded[:, head, :, block] = centers[selected].reshape(batch, tokens, subdim)
    return indices, decoded


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.repetitions < 1 or any(t < 1 for t in args.token_counts):
        raise ValueError("invalid timing protocol")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    g_parent = _load(args.gsrq_parent)
    g_scf = _load(args.gsrq_scf)
    c_parent = _load(args.cq_parent)
    c_scf = _load(args.cq_scf)
    f_parent = _load(args.fib_parent)
    f_scf = _load(args.fib_scf)
    tensors = {
        "g_parent": g_parent["codebooks"]["key"].to(device),
        "g_scf": g_scf["codebooks"]["key"].to(device),
        "g_factor": g_scf["key_assignment_whiteners"].to(device),
        "c_parent": c_parent["key"]["codebooks"].to(device),
        "c_scf": c_scf["key"]["codebooks"].to(device),
        "c_factor": c_scf["key"]["selector_whiteners"].to(device),
        "f_parent_rotation": f_parent["rotation"].to(device),
        "f_parent_codebook": f_parent["codebook"].to(device),
        "f_scf_rotation": f_scf["rotation"].to(device),
        "f_scf_codebook": f_scf["key"]["decode_codebook"].to(device),
        "f_factor": f_scf["key"]["block_whiteners"].to(device),
    }
    torch.cuda.synchronize()
    f_cross_tables, f_selector_norms = prepare_m20_fibquant_search(
        tensors["f_scf_codebook"], tensors["f_factor"]
    )
    results: dict[str, Any] = {}
    timing = {"warmups": args.warmups, "repetitions": args.repetitions}

    for tokens in args.token_counts:
        states = _states(tokens, device)
        current: dict[str, Any] = {}

        g_selector_states = torch.einsum(
            "hde,bhte->bhtd", tensors["g_factor"].float(), states.float()
        )
        g_selector_tables = torch.einsum(
            "hrke,hde->hrkd", tensors["g_scf"].float(), tensors["g_factor"].float()
        )
        current["gsrq"] = {
            "parent_full": _time_cuda(
                lambda: greedy_encode_parent(
                    states, tensors["g_parent"], token_chunk_size=args.chunk_size
                ),
                **timing,
            ),
            "scf_current_full": _time_cuda(
                lambda: greedy_encode_with_assignment_whitener(
                    states,
                    tensors["g_scf"],
                    tensors["g_factor"],
                    token_chunk_size=args.chunk_size,
                ),
                **timing,
            ),
            "scf_state_transform": _time_cuda(
                lambda: torch.einsum(
                    "hde,bhte->bhtd", tensors["g_factor"].float(), states.float()
                ),
                **timing,
            ),
            "scf_table_transform": _time_cuda(
                lambda: torch.einsum(
                    "hrke,hde->hrkd",
                    tensors["g_scf"].float(),
                    tensors["g_factor"].float(),
                ),
                **timing,
            ),
            "nearest_search_pretransformed": _time_cuda(
                lambda: greedy_encode_parent(
                    g_selector_states,
                    g_selector_tables,
                    token_chunk_size=args.chunk_size,
                ),
                **timing,
            ),
        }

        c_raw, c_tables, c_factors = _cq_layout(
            states, tensors["c_scf"], tensors["c_factor"]
        )
        c_selector_rows = torch.matmul(c_raw, c_factors.transpose(-1, -2))
        c_selector_tables = torch.matmul(c_tables, c_factors.transpose(-1, -2))
        current["cq"] = {
            "parent_full": _time_cuda(
                lambda: quantize_cq(
                    states,
                    tensors["c_parent"],
                    chunk_size=args.chunk_size,
                    return_indices=True,
                ),
                **timing,
            ),
            "scf_current_full": _time_cuda(
                lambda: quantize_cq(
                    states,
                    tensors["c_scf"],
                    selector_whiteners=tensors["c_factor"],
                    chunk_size=args.chunk_size,
                    return_indices=True,
                ),
                **timing,
            ),
            "scf_state_transform": _time_cuda(
                lambda: torch.matmul(c_raw, c_factors.transpose(-1, -2)), **timing
            ),
            "scf_table_transform": _time_cuda(
                lambda: torch.matmul(c_tables, c_factors.transpose(-1, -2)), **timing
            ),
            "nearest_search_pretransformed": _time_cuda(
                lambda: _cq_search(
                    c_selector_rows,
                    c_selector_tables,
                    c_tables,
                    chunk_size=args.chunk_size,
                ),
                **timing,
            ),
        }

        f_grouped = _fib_layout(states, tensors["f_scf_rotation"], block_size=4)
        f_transformed = _fib_transform_rows(f_grouped, tensors["f_factor"])
        f_selectors = _fib_transform_tables(tensors["f_scf_codebook"], tensors["f_factor"])
        current["fibquant"] = {
            "parent_full": _time_cuda(
                lambda: quantize_fibquant(
                    states,
                    tensors["f_parent_rotation"],
                    tensors["f_parent_codebook"],
                    block_size=4,
                    chunk_size=args.chunk_size,
                    return_indices=True,
                ),
                **timing,
            ),
            "scf_current_full": _time_cuda(
                lambda: quantize_m20_fibquant_key(
                    states,
                    tensors["f_scf_rotation"],
                    tensors["f_scf_codebook"],
                    tensors["f_factor"],
                    block_size=4,
                    chunk_size=args.chunk_size,
                    return_indices=True,
                    selector_cross_codebooks=f_cross_tables,
                    selector_norms=f_selector_norms,
                ),
                **timing,
            ),
            "common_normalize_rotate": _time_cuda(
                lambda: _fib_layout(states, tensors["f_scf_rotation"], block_size=4),
                **timing,
            ),
            "scf_state_transform": _time_cuda(
                lambda: _fib_transform_rows(f_grouped, tensors["f_factor"]), **timing
            ),
            "scf_table_transform": _time_cuda(
                lambda: _fib_transform_tables(
                    tensors["f_scf_codebook"], tensors["f_factor"]
                ),
                **timing,
            ),
            "nearest_search_pretransformed": _time_cuda(
                lambda: _fib_search(
                    f_transformed, f_selectors, tensors["f_scf_codebook"].float()
                ),
                **timing,
            ),
        }
        results[str(tokens)] = current
        print(json.dumps({"event": "tokens_complete", "tokens": tokens}), flush=True)

    output = {
        "status": "complete",
        "scope": "layer-0 K encoder microbenchmark",
        "input_note": "synthetic values with exact deployed artifact geometry",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "protocol": {
            "token_counts": args.token_counts,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "chunk_size": args.chunk_size,
        },
        "artifacts": {
            name: str(Path(getattr(args, name.replace("-", "_"), "")).resolve())
            for name in (
                "gsrq_parent",
                "gsrq_scf",
                "cq_parent",
                "cq_scf",
                "fib_parent",
                "fib_scf",
            )
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "measurements": results,
    }
    atomic_json_save(output, Path(args.output).resolve())
    print(json.dumps({"status": "complete", "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
