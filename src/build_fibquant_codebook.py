#!/usr/bin/env python3
"""Build the paper-specified universal FibQuant codebook and Haar rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaincinv, ndtri
import torch

from lowrate_vq.codecs import fibquant_rate, squared_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--training-multiple", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def orthogonal_matrix(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


def generalized_golden_ratio(dimension: int) -> float:
    return brentq(lambda value: value ** (dimension + 1) - value - 1, 1, 2)


def roberts_kronecker_directions(count: int, dimension: int) -> torch.Tensor:
    phi = generalized_golden_ratio(dimension)
    n = np.arange(1, count + 1, dtype=np.float64)[:, None] - 0.5
    powers = phi ** (-np.arange(1, dimension + 1, dtype=np.float64))[None]
    uniforms = np.mod(n * powers, 1.0)
    eps = np.finfo(np.float64).eps
    gaussian = ndtri(np.clip(uniforms, eps, 1.0 - eps))
    gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
    return torch.from_numpy(gaussian).float()


def beta_quantile_initialization(
    dimension: int, block_size: int, count: int
) -> torch.Tensor:
    q = (np.arange(1, count + 1, dtype=np.float64) - 0.5) / count
    beta_shape = (
        block_size / (block_size + 2.0)
        * (dimension - block_size - 2.0)
        / 2.0
        + 1.0
    )
    radii = np.sqrt(betaincinv(block_size / 2.0, beta_shape, q))
    directions = roberts_kronecker_directions(count, block_size)
    return directions * torch.from_numpy(radii).float().unsqueeze(-1)


def spherical_beta_samples(
    count: int, dimension: int, block_size: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn(count, dimension, generator=generator)
    samples /= torch.linalg.vector_norm(samples, dim=-1, keepdim=True)
    return samples[:, :block_size].contiguous()


def repair_empty_cells(
    samples: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if not len(empty):
        return centers
    assigned_error = (samples - centers[assignments]).square().sum(dim=-1)
    for target in empty.tolist():
        donor_scores = torch.zeros(len(centers), dtype=torch.float64)
        donor_scores.scatter_add_(0, assignments, assigned_error.double())
        donor_scores[counts < 2] = -1
        donor = int(donor_scores.argmax())
        member_mask = assignments == donor
        member_indices = torch.nonzero(member_mask, as_tuple=False).flatten()
        if not len(member_indices):
            replacement = int(assigned_error.argmax())
        else:
            local = assigned_error[member_indices].argmax()
            replacement = int(member_indices[local])
        direction = samples[replacement] - centers[donor]
        centers[target] = centers[donor] + 0.5 * direction
        centers[donor] = centers[donor] - 0.5 * direction
    return centers


def lloyd_refine(
    samples: torch.Tensor, initial: torch.Tensor, iterations: int
) -> tuple[torch.Tensor, float, list[float]]:
    centers = initial.clone()
    history: list[float] = []
    for _ in range(iterations):
        distances = squared_distances(samples, centers)
        assignments = distances.argmin(dim=-1)
        counts = torch.bincount(assignments, minlength=len(centers))
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples)
        occupied = counts > 0
        centers[occupied] = sums[occupied] / counts[occupied, None]
        centers = repair_empty_cells(
            samples, centers, assignments, counts
        )
        distortion = float(
            squared_distances(samples, centers).min(dim=-1).values.mean()
        )
        history.append(distortion)
    return centers, history[-1], history


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.dimension % args.block_size:
        raise ValueError("block size must divide dimension")
    if args.codebook_size < 2 or args.codebook_size & (args.codebook_size - 1):
        raise ValueError("codebook size must be a power of two")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    initial = beta_quantile_initialization(
        args.dimension, args.block_size, args.codebook_size
    )
    samples = spherical_beta_samples(
        args.training_multiple * args.codebook_size,
        args.dimension,
        args.block_size,
        args.seed + 1,
    )
    best = None
    restart_rows = []
    for restart in range(args.restarts):
        omega = orthogonal_matrix(args.block_size, args.seed + 100 + restart)
        centers, distortion, history = lloyd_refine(
            samples, initial @ omega.transpose(0, 1), args.iterations
        )
        restart_rows.append(
            {"restart": restart, "distortion": distortion, "history": history}
        )
        if best is None or distortion < best[0]:
            best = (distortion, centers.clone(), restart)
    assert best is not None
    rotation = orthogonal_matrix(args.dimension, args.seed)
    rate = fibquant_rate(
        dimension=args.dimension,
        block_size=args.block_size,
        codebook_size=args.codebook_size,
    )
    artifact = {
        "provenance": {
            "method": "FibQuant",
            "paper": "arXiv:2605.11478v1",
            "dimension": args.dimension,
            "block_size": args.block_size,
            "codebook_size": args.codebook_size,
            "training_samples": len(samples),
            "training_multiple": args.training_multiple,
            "restarts": args.restarts,
            "iterations": args.iterations,
            "seed": args.seed,
            "initialization": "beta_quantile_roberts_kronecker",
            "refinement": "multi_restart_lloyd_max",
            "key_surface": "post_rope",
            "index_bpa": rate.index_bpa,
            "metadata_bpa": rate.metadata_bpa,
            "total_bpa": rate.total_bpa,
            "rate_metadata": rate.metadata,
            "best_restart": best[2],
            "training_distortion": best[0],
            "elapsed_seconds": time.time() - started,
        },
        "rotation": rotation.half(),
        "codebook": best[1].half(),
        "diagnostics": {"restarts": restart_rows},
    }
    path = output_dir / "codec.pt"
    temporary = output_dir / "codec.pt.partial"
    torch.save(artifact, temporary)
    temporary.replace(path)
    summary = {
        **artifact["provenance"],
        "artifact": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "rotation_orthogonality_max_abs": float(
            (rotation @ rotation.transpose(0, 1) - torch.eye(args.dimension))
            .abs()
            .max()
        ),
        "finite": bool(torch.isfinite(best[1]).all()),
    }
    summary_path = output_dir / "summary.json"
    temporary_summary = output_dir / "summary.json.partial"
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    temporary_summary.replace(summary_path)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

