"""Paper-faithful log-gradient sample weights for GSRQ codebook fitting."""

from __future__ import annotations

from typing import Any

import torch


GRADIENT_NORM_SCOPES = ("per_head", "whole_token_shared")


def select_gradient_norms(
    gradient_norms: torch.Tensor,
    *,
    head: int,
    scope: str,
) -> torch.Tensor:
    """Select paper-plausible sample norms for one D-dimensional subspace.

    ``per_head`` uses the norm of the gradient corresponding to that D=128
    subspace. ``whole_token_shared`` reconstructs the norm of the concatenated
    KV vector from its head-wise norms and shares it across all subspaces.
    """

    if gradient_norms.ndim != 3:
        raise ValueError("gradient_norms must have shape [samples, tokens, heads]")
    if scope not in GRADIENT_NORM_SCOPES:
        raise ValueError(f"unknown gradient-norm scope: {scope}")
    if head < 0 or head >= gradient_norms.shape[-1]:
        raise ValueError("head index is out of range")
    values = gradient_norms.detach().to(dtype=torch.float32)
    if scope == "per_head":
        return values[:, :, head].reshape(-1)
    return values.square().sum(dim=-1).sqrt().reshape(-1)


def log_gradient_weights(
    gradient_norms: torch.Tensor,
    *,
    target_median: float = 1.0,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``log(1 + lambda * ||grad||)`` with median-normalized lambda.

    The paper defines ``lambda = tau / (median(||grad||) + epsilon)`` and
    selects ``tau=1``.  Normalization is deliberately applied to exactly the
    samples consumed by one layer/head codebook, which keeps the transform
    auditable despite the paper not specifying a cross-codebook aggregation.
    """

    if gradient_norms.numel() < 1:
        raise ValueError("gradient_norms must be non-empty")
    if target_median <= 0.0 or epsilon <= 0.0:
        raise ValueError("target_median and epsilon must be positive")
    raw = gradient_norms.detach().to(dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("gradient norms must be finite")
    if bool(torch.any(raw < 0.0)):
        raise ValueError("gradient norms must be non-negative")
    median = raw.median()
    if not bool(median > 0.0):
        raise ValueError("gradient-norm median must be positive")
    scale = float(target_median) / (median + float(epsilon))
    weights = torch.log1p(raw * scale)
    if not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0.0):
        raise ValueError("log-gradient weights are invalid")
    positive = raw[raw > 0.0]
    diagnostics = {
        "formula": "log1p(tau * gradient_norm / (median + epsilon))",
        "normalization_scope": "per_layer_head_kind_codebook",
        "target_median": float(target_median),
        "epsilon": float(epsilon),
        "raw_min": float(raw.min()),
        "raw_median": float(median),
        "raw_max": float(raw.max()),
        "raw_positive_min": float(positive.min()),
        "raw_zero_fraction": float((raw == 0.0).float().mean()),
        "lambda": float(scale),
        "weight_min": float(weights.min()),
        "weight_median": float(weights.median()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_max_to_median": float(
            weights.max() / weights.median().clamp_min(float(epsilon))
        ),
    }
    return weights, diagnostics
