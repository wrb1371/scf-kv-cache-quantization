"""Strict-rate grouped VQ kernels for CQ and spectrum-tempered CQ.

CQ-8c8b stores one 8-bit index for every contiguous eight channels.  The
paper's Fisher weight is scalar per token/group.  The paired M20 variant keeps
those weights but optionally changes the assignment geometry with one frozen
positive-definite matrix per head/group; decoded centroids remain in raw KV
coordinates and the cache representation is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from fw_gsrq.codec import build_spectral_fisher_whitener


@dataclass(frozen=True)
class GroupedKMeansFit:
    codebooks: torch.Tensor
    initial_indices: torch.Tensor
    assignments: torch.Tensor
    history: list[dict[str, float | int]]
    diagnostics: dict[str, Any]


def _validate_float(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite floating point")


def _transform(
    rows: torch.Tensor, whiteners: torch.Tensor | None
) -> torch.Tensor:
    if whiteners is None:
        return rows.float()
    return torch.matmul(rows.float(), whiteners.float().transpose(-1, -2))


@torch.no_grad()
def kmeans_plus_plus_indices(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    *,
    clusters: int,
    seed: int,
    whiteners: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched weighted k-means++ sample indices for ``[P,N,S]`` rows."""

    if vectors.ndim != 3 or weights.shape != vectors.shape[:2]:
        raise ValueError("vectors/weights must have shape [P,N,S]/[P,N]")
    problems, rows, subdim = map(int, vectors.shape)
    if not 1 <= clusters <= rows:
        raise ValueError("cluster count must lie in [1,N]")
    _validate_float(vectors, "vectors")
    _validate_float(weights, "weights")
    if bool(torch.any(weights < 0.0)) or bool(torch.any(weights.sum(-1) <= 0.0)):
        raise ValueError("every problem must have positive nonnegative weights")
    if whiteners is not None:
        if tuple(whiteners.shape) != (problems, subdim, subdim):
            raise ValueError("whiteners have the wrong geometry")
        _validate_float(whiteners, "whiteners")

    device = vectors.device
    generator = torch.Generator(device=device).manual_seed(seed)
    transformed = _transform(vectors, whiteners)
    selected = torch.empty(problems, clusters, dtype=torch.long, device=device)
    first = torch.multinomial(weights.float(), 1, generator=generator).squeeze(1)
    selected[:, 0] = first
    problem = torch.arange(problems, device=device)
    center = transformed[problem, first]
    minimum = (transformed - center[:, None]).square().sum(-1)
    chosen = torch.zeros(problems, rows, dtype=torch.bool, device=device)
    chosen[problem, first] = True
    for cluster in range(1, clusters):
        scores = weights.float() * minimum
        scores.masked_fill_(chosen, 0.0)
        totals = scores.sum(-1)
        degenerate = totals <= 0.0
        if bool(degenerate.any()):
            fallback = weights.float().masked_fill(chosen, 0.0)
            fallback_totals = fallback.sum(-1)
            if bool(torch.any(fallback_totals[degenerate] <= 0.0)):
                fallback = (~chosen).float()
            scores[degenerate] = fallback[degenerate]
        current = torch.multinomial(scores, 1, generator=generator).squeeze(1)
        selected[:, cluster] = current
        chosen[problem, current] = True
        center = transformed[problem, current]
        distance = (transformed - center[:, None]).square().sum(-1)
        # ``Tensor.minimum_`` is not available in the deployed PyTorch build.
        # Keep the preallocated tensor while using the portable functional op.
        minimum.copy_(torch.minimum(minimum, distance))
    return selected


@torch.no_grad()
def fit_batched_weighted_kmeans(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    chunk_size: int,
    seed: int,
    whiteners: torch.Tensor | None = None,
    initial_indices: torch.Tensor | None = None,
) -> GroupedKMeansFit:
    """Fit independent weighted group codebooks with optional M20 geometry."""

    if vectors.ndim != 3 or weights.shape != vectors.shape[:2]:
        raise ValueError("vectors/weights must have shape [P,N,S]/[P,N]")
    problems, rows, subdim = map(int, vectors.shape)
    if iterations < 1 or chunk_size < 1:
        raise ValueError("iterations and chunk size must be positive")
    _validate_float(vectors, "vectors")
    _validate_float(weights, "weights")
    if bool(torch.any(weights < 0.0)) or bool(torch.any(weights.sum(-1) <= 0.0)):
        raise ValueError("every problem must have positive nonnegative weights")
    # Per-problem rescaling is objective-equivalent and avoids final-CE
    # magnitudes near FP32 underflow from destabilizing the Lloyd sums.
    positive_mean = weights.sum(-1) / (weights > 0.0).sum(-1).clamp_min(1)
    work_weights = weights.float() / positive_mean[:, None].clamp_min(
        torch.finfo(torch.float32).tiny
    )
    if whiteners is not None:
        if tuple(whiteners.shape) != (problems, subdim, subdim):
            raise ValueError("whiteners have the wrong geometry")
        _validate_float(whiteners, "whiteners")
        work_whiteners = whiteners.to(vectors.device, dtype=torch.float32)
    else:
        work_whiteners = None
    if initial_indices is None:
        initial_indices = kmeans_plus_plus_indices(
            vectors,
            work_weights,
            clusters=clusters,
            seed=seed,
            whiteners=work_whiteners,
        )
    if tuple(initial_indices.shape) != (problems, clusters):
        raise ValueError("initial indices have the wrong geometry")
    if int(initial_indices.min()) < 0 or int(initial_indices.max()) >= rows:
        raise ValueError("initial indices are out of range")

    device = vectors.device
    problem = torch.arange(problems, device=device)[:, None]
    centers = vectors.float()[problem, initial_indices.to(device)].clone()
    transformed_vectors = _transform(vectors, work_whiteners)
    offsets = (
        torch.arange(problems, device=device, dtype=torch.long) * clusters
    )[:, None]
    history: list[dict[str, float | int]] = []
    assignments = torch.empty(problems, rows, dtype=torch.long, device=device)
    for iteration in range(iterations):
        transformed_centers = _transform(centers, work_whiteners)
        center_norms = transformed_centers.square().sum(-1).unsqueeze(1)
        sums = torch.zeros(
            problems * clusters, subdim, device=device, dtype=torch.float32
        )
        counts = torch.zeros(problems * clusters, device=device, dtype=torch.float32)
        minimum_errors = torch.empty(problems, rows, device=device)
        distortion = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            current = transformed_vectors[:, start:stop]
            distances = (
                current.square().sum(-1, keepdim=True)
                + center_norms
                - 2.0 * torch.bmm(current, transformed_centers.transpose(1, 2))
            ).clamp_min_(0.0)
            error, selected = distances.min(-1)
            assignments[:, start:stop] = selected
            minimum_errors[:, start:stop] = error
            current_weights = work_weights[:, start:stop]
            flat_selected = (selected + offsets).reshape(-1)
            counts.index_add_(0, flat_selected, current_weights.reshape(-1))
            weighted_rows = (
                vectors[:, start:stop].float() * current_weights.unsqueeze(-1)
            )
            sums.index_add_(0, flat_selected, weighted_rows.reshape(-1, subdim))
            distortion += (error.double() * current_weights.double()).sum()
        sums = sums.reshape(problems, clusters, subdim)
        counts = counts.reshape(problems, clusters)
        occupied = counts > 0.0
        means = sums / counts.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        centers = torch.where(occupied.unsqueeze(-1), means, centers)
        empty_total = int((~occupied).sum())
        if empty_total:
            weighted_error = minimum_errors * work_weights
            for p in range(problems):
                empty = torch.nonzero(~occupied[p], as_tuple=False).flatten()
                if len(empty):
                    ranked = torch.argsort(weighted_error[p], descending=True)
                    centers[p, empty] = vectors[p, ranked[: len(empty)]].float()
        history.append(
            {
                "iteration": iteration,
                "weighted_metric_mse": float(
                    distortion / work_weights.sum().double()
                ),
                "empty_clusters": empty_total,
            }
        )

    # Reassign once to the deployed FP32 centers for utilization diagnostics.
    transformed_centers = _transform(centers, work_whiteners)
    center_norms = transformed_centers.square().sum(-1).unsqueeze(1)
    final_distortion = torch.zeros((), device=device, dtype=torch.float64)
    counts = torch.zeros(problems, clusters, device=device, dtype=torch.long)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        current = transformed_vectors[:, start:stop]
        distances = (
            current.square().sum(-1, keepdim=True)
            + center_norms
            - 2.0 * torch.bmm(current, transformed_centers.transpose(1, 2))
        ).clamp_min_(0.0)
        error, selected = distances.min(-1)
        assignments[:, start:stop] = selected
        counts.scatter_add_(1, selected, torch.ones_like(selected))
        final_distortion += (
            error.double() * work_weights[:, start:stop].double()
        ).sum()
    probabilities = counts.double() / counts.sum(-1, keepdim=True).clamp_min(1)
    entropy_terms = torch.where(
        probabilities > 0.0,
        -probabilities * probabilities.clamp_min(1e-300).log2(),
        torch.zeros_like(probabilities),
    )
    diagnostics: dict[str, Any] = {
        "problems": problems,
        "rows_per_problem": rows,
        "subdimension": subdim,
        "clusters": clusters,
        "iterations": iterations,
        "metric": "m20" if work_whiteners is not None else "euclidean",
        "final_weighted_metric_mse": float(
            final_distortion / work_weights.sum().double()
        ),
        "mean_code_utilization": float((counts > 0).float().mean()),
        "mean_normalized_code_entropy": float(
            (entropy_terms.sum(-1) / math.log2(clusters)).mean()
        ),
        "weight_positive_fraction": float((weights > 0.0).float().mean()),
        "finite": bool(torch.isfinite(centers).all()),
    }
    return GroupedKMeansFit(
        codebooks=centers,
        initial_indices=initial_indices.detach().cpu(),
        assignments=assignments,
        history=history,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def build_group_m20_whiteners(
    group_outer_sum: torch.Tensor,
    *,
    spectral_power: float = 0.5,
    fisher_mixture: float = 0.5,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Build spectral-Fisher factors for ``[H,G,S,S]`` empirical sums.

    The defaults reproduce SCF exactly.  Exposing the two geometry parameters
    keeps attribution baselines on the same CQ calibration and optimization
    path instead of introducing a second implementation.
    """

    if group_outer_sum.ndim != 4 or group_outer_sum.shape[-1] != group_outer_sum.shape[-2]:
        raise ValueError("group covariance must have shape [H,G,S,S]")
    _validate_float(group_outer_sum, "group covariance")
    heads, groups, subdim, _ = map(int, group_outer_sum.shape)
    factors = torch.empty_like(group_outer_sum, dtype=torch.float32)
    diagnostics: list[dict[str, float]] = []
    for head in range(heads):
        for group in range(groups):
            transform = build_spectral_fisher_whitener(
                group_outer_sum[head, group].float(),
                spectral_power=spectral_power,
                fisher_mixture=fisher_mixture,
            )
            factors[head, group] = transform.whitener
            diagnostics.append(
                {"head": float(head), "group": float(group), **transform.diagnostics}
            )
    return factors, diagnostics


@torch.no_grad()
def quantize_cq(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    selector_whiteners: torch.Tensor | None = None,
    chunk_size: int = 4096,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Roundtrip ``[B,H,T,D]`` using per-head contiguous group tables."""

    if states.ndim != 4 or codebooks.ndim != 4:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,G,K,S]")
    batch, heads, tokens, dimension = map(int, states.shape)
    cb_heads, groups, clusters, subdim = map(int, codebooks.shape)
    if cb_heads != heads or groups * subdim != dimension:
        raise ValueError("CQ codebook geometry differs from states")
    if selector_whiteners is not None and tuple(selector_whiteners.shape) != (
        heads,
        groups,
        subdim,
        subdim,
    ):
        raise ValueError("CQ selector whiteners have the wrong geometry")
    problems = heads * groups
    rows = batch * tokens
    raw = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, rows, groups, subdim)
        .permute(0, 2, 1, 3)
        .reshape(problems, rows, subdim)
        .float()
    )
    tables = codebooks.to(states.device, dtype=torch.float32).reshape(
        problems, clusters, subdim
    )
    factors = (
        selector_whiteners.to(states.device, dtype=torch.float32).reshape(
            problems, subdim, subdim
        )
        if selector_whiteners is not None
        else None
    )
    selector_rows = _transform(raw, factors)
    selector_tables = _transform(tables, factors)
    table_norms = selector_tables.square().sum(-1).unsqueeze(1)
    decoded = torch.empty_like(raw)
    all_indices = torch.empty(
        problems, rows, dtype=torch.long, device=states.device
    )
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        current = selector_rows[:, start:stop]
        distance = (
            current.square().sum(-1, keepdim=True)
            + table_norms
            - 2.0 * torch.bmm(current, selector_tables.transpose(1, 2))
        ).clamp_min_(0.0)
        selected = distance.argmin(-1)
        all_indices[:, start:stop] = selected
        decoded[:, start:stop] = torch.gather(
            tables,
            1,
            selected.unsqueeze(-1).expand(-1, -1, subdim),
        )
    output = (
        decoded.reshape(heads, groups, batch, tokens, subdim)
        .permute(2, 0, 3, 1, 4)
        .reshape(batch, heads, tokens, dimension)
        .to(states.dtype)
    )
    indices = None
    if return_indices:
        indices = (
            all_indices.reshape(heads, groups, batch, tokens)
            .permute(2, 0, 3, 1)
            .contiguous()
        )
    return output, indices
