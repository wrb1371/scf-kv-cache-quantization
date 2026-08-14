"""Gain--Shape K-means from arXiv:2607.01065.

The paper fully specifies the unweighted update but only describes a weighted
extension in prose.  This implementation uses the conventional weighted-mean
extension for both the normalized shape mean and raw gain projection.  That
choice is a reproduction assumption and must not be hidden in paper-facing
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class GSKMResult:
    """Result of one GSKM fit."""

    codebook: torch.Tensor
    shapes: torch.Tensor
    gains: torch.Tensor
    assignments: torch.Tensor
    history: list[dict[str, Any]]
    converged: bool
    iterations: int
    initialization: str
    weighted: bool


def _validate_data(
    data: torch.Tensor,
    clusters: int,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 2:
        raise ValueError(f"data must be [N,D], got {tuple(data.shape)}")
    if not data.is_floating_point():
        raise TypeError("data must be floating point")
    if data.shape[0] < clusters:
        raise ValueError(
            f"need at least K={clusters} samples, got N={data.shape[0]}"
        )
    if clusters < 1:
        raise ValueError("clusters must be positive")
    work = data.float()
    if not torch.isfinite(work).all():
        raise ValueError("data must be finite")
    if weights is None:
        sample_weights = torch.ones(
            work.shape[0], device=work.device, dtype=torch.float32
        )
    else:
        if weights.ndim != 1 or weights.shape[0] != work.shape[0]:
            raise ValueError("weights must have shape [N]")
        sample_weights = weights.to(
            device=work.device, dtype=torch.float32
        )
        if not torch.isfinite(sample_weights).all():
            raise ValueError("weights must be finite")
        if torch.any(sample_weights < 0):
            raise ValueError("weights must be non-negative")
        if not bool(torch.sum(sample_weights) > 0):
            raise ValueError("at least one weight must be positive")
    return work, sample_weights


def _sample_initial_shapes(
    data: torch.Tensor,
    clusters: int,
    seed: int,
    eps: float,
) -> torch.Tensor:
    generator = torch.Generator(device=data.device)
    generator.manual_seed(seed)
    indices = torch.randperm(
        data.shape[0], generator=generator, device=data.device
    )[:clusters]
    shapes = F.normalize(data[indices], dim=-1, eps=eps)
    zero = torch.linalg.vector_norm(shapes, dim=-1) <= eps
    if torch.any(zero):
        replacements = torch.zeros(
            int(zero.sum()), data.shape[1], device=data.device
        )
        replacement_rows = torch.arange(
            int(zero.sum()), device=data.device
        )
        replacements[
            replacement_rows,
            replacement_rows.remainder(data.shape[1]),
        ] = 1.0
        shapes[zero] = replacements
    return shapes


def _kmeans_plus_plus_initial_shapes(
    data: torch.Tensor,
    clusters: int,
    seed: int,
    eps: float,
) -> torch.Tensor:
    """Select unit-vector seeds with exact D2 k-means++ sampling."""

    normalized = F.normalize(data, dim=-1, eps=eps)
    generator = torch.Generator(device=data.device)
    generator.manual_seed(seed)
    first = torch.randint(
        normalized.shape[0],
        (1,),
        generator=generator,
        device=data.device,
    )
    chosen = [normalized[first[0]]]
    nearest_square = (
        normalized - chosen[0][None]
    ).square().sum(dim=-1)
    for _ in range(1, clusters):
        total = nearest_square.sum()
        if not bool(torch.isfinite(total)) or not bool(total > eps):
            index = torch.randint(
                normalized.shape[0],
                (1,),
                generator=generator,
                device=data.device,
            )[0]
        else:
            index = torch.multinomial(
                nearest_square,
                1,
                replacement=False,
                generator=generator,
            )[0]
        center = normalized[index]
        chosen.append(center)
        distance_square = (
            normalized - center[None]
        ).square().sum(dim=-1)
        nearest_square = torch.minimum(
            nearest_square, distance_square
        )
    return torch.stack(chosen)


@torch.no_grad()
def assign_codewords(
    data: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Return exact nearest-codeword assignments in squared Euclidean distance."""

    if data.ndim != 2 or codebook.ndim != 2:
        raise ValueError("data and codebook must both be matrices")
    if data.shape[1] != codebook.shape[1]:
        raise ValueError("data and codebook dimensions differ")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    work = data.float()
    centers = codebook.to(device=work.device, dtype=torch.float32)
    center_square = centers.square().sum(dim=-1)
    assignments: list[torch.Tensor] = []
    for start in range(0, work.shape[0], chunk_size):
        batch = work[start : start + chunk_size]
        distances = (
            batch.square().sum(dim=-1, keepdim=True)
            + center_square[None, :]
            - 2.0 * torch.matmul(batch, centers.T)
        )
        assignments.append(torch.argmin(distances, dim=-1))
    return torch.cat(assignments)


@torch.no_grad()
def _assign_gain_shape(
    data: torch.Tensor,
    shapes: torch.Tensor,
    gains: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    assignments: list[torch.Tensor] = []
    gain_square = gains.square()
    for start in range(0, data.shape[0], chunk_size):
        batch = data[start : start + chunk_size]
        scores = (
            2.0
            * torch.matmul(batch, shapes.T)
            * gains.unsqueeze(0)
            - gain_square.unsqueeze(0)
        )
        assignments.append(torch.argmax(scores, dim=-1))
    return torch.cat(assignments)


@torch.no_grad()
def _weighted_update(
    data: torch.Tensor,
    normalized: torch.Tensor,
    sample_weights: torch.Tensor,
    assignments: torch.Tensor,
    old_shapes: torch.Tensor,
    old_gains: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    clusters, dimension = old_shapes.shape
    weighted_normalized = normalized * sample_weights[:, None]
    weighted_raw = data * sample_weights[:, None]
    shape_sum = torch.zeros(
        clusters, dimension, device=data.device, dtype=torch.float32
    )
    raw_sum = torch.zeros_like(shape_sum)
    weight_sum = torch.zeros(
        clusters, device=data.device, dtype=torch.float32
    )
    shape_sum.index_add_(0, assignments, weighted_normalized)
    raw_sum.index_add_(0, assignments, weighted_raw)
    weight_sum.index_add_(0, assignments, sample_weights)

    shapes = old_shapes.clone()
    gains = old_gains.clone()
    shape_norm = torch.linalg.vector_norm(shape_sum, dim=-1)
    valid = (weight_sum > eps) & (shape_norm > eps)
    shapes[valid] = shape_sum[valid] / shape_norm[valid, None]
    projected = torch.sum(raw_sum[valid] * shapes[valid], dim=-1)
    gains[valid] = torch.clamp(
        projected / weight_sum[valid], min=0.0
    )
    empty = int((~valid).sum().item())
    return shapes, gains, empty


@torch.no_grad()
def _weighted_mse(
    data: torch.Tensor,
    codebook: torch.Tensor,
    assignments: torch.Tensor,
    sample_weights: torch.Tensor,
) -> float:
    error_square = (
        data - codebook.index_select(0, assignments)
    ).square().sum(dim=-1)
    return float(
        torch.sum(error_square * sample_weights)
        / torch.sum(sample_weights)
    )


@torch.no_grad()
def fit_gskm(
    data: torch.Tensor,
    *,
    clusters: int = 256,
    weights: torch.Tensor | None = None,
    max_iterations: int = 100,
    chunk_size: int = 16384,
    seed: int = 0,
    tolerance: float = 0.0,
    eps: float = 1e-8,
    initialization: str = "sample",
) -> GSKMResult:
    """Fit Gain--Shape K-means.

    `initialization="sample"` selects normalized input samples.  The paper does
    not freeze an initialization; this deterministic option is deliberately
    exposed in every run manifest.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if initialization not in ("sample", "kmeans++"):
        raise ValueError(f"unsupported initialization: {initialization}")

    work, sample_weights = _validate_data(data, clusters, weights)
    normalized = F.normalize(work, dim=-1, eps=eps)
    if initialization == "sample":
        shapes = _sample_initial_shapes(work, clusters, seed, eps)
    else:
        shapes = _kmeans_plus_plus_initial_shapes(
            work, clusters, seed, eps
        )
    initial_assignments = _assign_gain_shape(
        work,
        shapes,
        torch.ones(clusters, device=work.device),
        chunk_size,
    )
    shapes, gains, _ = _weighted_update(
        work,
        normalized,
        sample_weights,
        initial_assignments,
        shapes,
        torch.ones(clusters, device=work.device),
        eps,
    )

    previous_assignments: torch.Tensor | None = None
    previous_mse: float | None = None
    history: list[dict[str, Any]] = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        assignments = _assign_gain_shape(
            work, shapes, gains, chunk_size
        )
        changed = (
            assignments.numel()
            if previous_assignments is None
            else int((assignments != previous_assignments).sum().item())
        )
        shapes, gains, empty = _weighted_update(
            work,
            normalized,
            sample_weights,
            assignments,
            shapes,
            gains,
            eps,
        )
        codebook = gains[:, None] * shapes
        mse = _weighted_mse(
            work, codebook, assignments, sample_weights
        )
        relative_improvement = None
        if previous_mse is not None:
            relative_improvement = (previous_mse - mse) / max(
                abs(previous_mse), eps
            )
        history.append(
            {
                "iteration": iteration,
                "weighted_mse": mse,
                "changed_assignments": changed,
                "empty_or_degenerate_clusters": empty,
                "relative_improvement": relative_improvement,
            }
        )
        if changed == 0:
            converged = True
            break
        if (
            tolerance > 0
            and relative_improvement is not None
            and 0 <= relative_improvement <= tolerance
        ):
            converged = True
            break
        previous_assignments = assignments
        previous_mse = mse

    codebook = gains[:, None] * shapes
    final_assignments = assign_codewords(
        work, codebook, chunk_size=chunk_size
    )
    return GSKMResult(
        codebook=codebook,
        shapes=shapes,
        gains=gains,
        assignments=final_assignments,
        history=history,
        converged=converged,
        iterations=len(history),
        initialization=initialization,
        weighted=weights is not None,
    )
