"""Spectrum-compressed Fisher geometry for the FibQuant product codec.

The cache representation is unchanged: each normalized 128-D vector stores
one 8-bit index per 4-D block plus the original FP16 norm.  Only the frozen K
selector geometry and its layer-local shared decode table change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from fw_gsrq.codec import build_spectral_fisher_whitener
from lowrate_vq.codecs import squared_distances


@dataclass(frozen=True)
class BlockMetricGeometry:
    raw_metrics: torch.Tensor
    rotated_metrics: torch.Tensor
    block_metrics: torch.Tensor
    block_whiteners: torch.Tensor
    diagnostics: dict[str, Any]


def _validate_finite(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite floating point")


def _expanded_sample_weights(
    sample_weights: torch.Tensor | None,
    *,
    heads: int,
    groups: int,
    samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Return nonnegative [H,G,N] weights with global mean one.

    A global rescaling does not change assignments or the unregularized normal
    equations, but mean-one normalization keeps the numerical ridge on the same
    scale across calibration sets.
    """

    if sample_weights is None:
        return torch.ones(heads, groups, samples, device=device, dtype=torch.float64)
    if sample_weights.ndim == 2 and tuple(sample_weights.shape) == (heads, samples):
        expanded = sample_weights[:, None, :].expand(heads, groups, samples)
    elif sample_weights.ndim == 3 and tuple(sample_weights.shape) == (
        heads,
        groups,
        samples,
    ):
        expanded = sample_weights
    else:
        raise ValueError("sample_weights must have shape [H,N] or [H,G,N]")
    expanded = expanded.to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(expanded).all()) or bool(torch.any(expanded < 0.0)):
        raise ValueError("sample_weights must be finite and nonnegative")
    mean = expanded.mean()
    if float(mean) <= 0.0:
        raise ValueError("sample_weights must have positive mean")
    return expanded / mean


@torch.no_grad()
def build_block_metric_geometry(
    fisher: torch.Tensor,
    rotation: torch.Tensor,
    *,
    block_size: int,
    spectral_power: float = 0.5,
    fisher_mixture: float = 0.5,
) -> BlockMetricGeometry:
    """Apply M20 in raw K space, rotate it, then take 4-D principal blocks."""

    if fisher.ndim != 3 or fisher.shape[-1] != fisher.shape[-2]:
        raise ValueError("fisher must have shape [H,D,D]")
    heads, dimension, _ = map(int, fisher.shape)
    if tuple(rotation.shape) != (dimension, dimension):
        raise ValueError("rotation has the wrong geometry")
    if block_size < 1 or dimension % block_size:
        raise ValueError("block_size must divide the dimension")
    _validate_finite(fisher, "fisher")
    _validate_finite(rotation, "rotation")

    raw_metrics = []
    whitener_diagnostics = []
    for head in range(heads):
        transform = build_spectral_fisher_whitener(
            fisher[head].float(),
            spectral_power=spectral_power,
            fisher_mixture=fisher_mixture,
        )
        raw_metrics.append(transform.metric)
        whitener_diagnostics.append({"head": head, **transform.diagnostics})
    raw = torch.stack(raw_metrics).float()
    work_rotation = rotation.to(device=raw.device, dtype=torch.float32)
    # Row vectors use y=xR^T and x=yR, so e_x=e_y R and
    # e_x M e_x^T=e_y (R M R^T) e_y^T.
    rotated = torch.matmul(
        torch.matmul(work_rotation.unsqueeze(0), raw),
        work_rotation.T.unsqueeze(0),
    )
    blocks = dimension // block_size
    block_metrics = torch.empty(
        heads,
        blocks,
        block_size,
        block_size,
        dtype=torch.float32,
        device=raw.device,
    )
    block_whiteners = torch.empty_like(block_metrics)
    diagonal_reconstruction = torch.zeros_like(rotated)
    minimum = math.inf
    maximum = -math.inf
    factorization_error = 0.0
    traces = []
    for block in range(blocks):
        start = block * block_size
        stop = start + block_size
        metric = rotated[:, start:stop, start:stop].contiguous()
        eigenvalues = torch.linalg.eigvalsh(metric.double())
        minimum = min(minimum, float(eigenvalues.min()))
        maximum = max(maximum, float(eigenvalues.max()))
        if bool(torch.any(eigenvalues <= 0.0)):
            raise ValueError("a projected FibQuant metric block is not positive definite")
        # cholesky returns A with M=A A^T; L=A^T gives M=L^T L.
        whitener = torch.linalg.cholesky(metric.double()).transpose(-1, -2)
        block_metrics[:, block] = metric
        block_whiteners[:, block] = whitener.float()
        reconstructed = torch.matmul(whitener.transpose(-1, -2), whitener)
        factorization_error = max(
            factorization_error,
            float((reconstructed - metric.double()).abs().max()),
        )
        traces.append(metric.diagonal(dim1=-2, dim2=-1).sum(-1))
        diagonal_reconstruction[:, start:stop, start:stop] = metric
    total_energy = rotated.double().square().sum().clamp_min(
        torch.finfo(torch.float64).tiny
    )
    offblock_energy = (rotated - diagonal_reconstruction).double().square().sum()
    trace_tensor = torch.stack(traces, dim=1)
    diagnostics: dict[str, Any] = {
        "heads": heads,
        "dimension": dimension,
        "block_size": block_size,
        "blocks": blocks,
        "m20_spectral_power": spectral_power,
        "m20_fisher_mixture": fisher_mixture,
        "m20_identity_mixture": 1.0 - fisher_mixture,
        "block_minimum_eigenvalue": minimum,
        "block_maximum_eigenvalue": maximum,
        "block_trace_minimum": float(trace_tensor.min()),
        "block_trace_maximum": float(trace_tensor.max()),
        "block_factorization_max_abs_error": factorization_error,
        "offblock_frobenius_energy_fraction": float(offblock_energy / total_energy),
        "whiteners": whitener_diagnostics,
        "finite": bool(
            torch.isfinite(raw).all()
            and torch.isfinite(rotated).all()
            and torch.isfinite(block_whiteners).all()
        ),
    }
    return BlockMetricGeometry(
        raw_metrics=raw.cpu(),
        rotated_metrics=rotated.cpu(),
        block_metrics=block_metrics.cpu(),
        block_whiteners=block_whiteners.cpu(),
        diagnostics=diagnostics,
    )


@torch.no_grad()
def post_rope_fisher_samples(
    query_states: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    kv_heads: int,
    samples_per_head: int,
) -> torch.Tensor:
    """Return deterministic p(1-p)-weighted post-RoPE query samples."""

    if query_states.ndim != 4 or probabilities.ndim != 4:
        raise ValueError("queries and probabilities must be rank four")
    batch, query_heads, tokens, dimension = map(int, query_states.shape)
    if tuple(probabilities.shape) != (batch, query_heads, tokens, tokens):
        raise ValueError("probability geometry differs from queries")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if samples_per_head < 1:
        raise ValueError("samples_per_head must be positive")
    _validate_finite(query_states, "query_states")
    _validate_finite(probabilities, "probabilities")
    repeats = query_heads // kv_heads
    causal = torch.ones(tokens, tokens, device=query_states.device, dtype=torch.bool).tril()
    causal_flat = torch.nonzero(causal.reshape(-1), as_tuple=False).squeeze(1)
    pair_queries = torch.div(causal_flat, tokens, rounding_mode="floor")
    pairs = int(causal_flat.numel())
    total = batch * repeats * pairs
    output = []
    for kv_head in range(kv_heads):
        start = kv_head * repeats
        head_probabilities = probabilities[:, start : start + repeats]
        causal_probabilities = head_probabilities[..., causal].reshape(-1).float()
        if int(causal_probabilities.numel()) != total:
            raise AssertionError("causal flattening changed its deterministic order")
        weights = causal_probabilities * (1.0 - causal_probabilities)
        cumulative = torch.cumsum(weights.double(), dim=0)
        weight_sum = cumulative[-1]
        if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
            raise ValueError("post-RoPE Fisher sampling has nonpositive weight")
        targets = (
            torch.arange(samples_per_head, device=cumulative.device, dtype=torch.float64)
            + 0.5
        ) * (weight_sum / samples_per_head)
        selected = torch.searchsorted(cumulative, targets, right=False)
        selected.clamp_(max=total - 1)
        flattened_query_head = torch.div(selected, pairs, rounding_mode="floor")
        pair_offset = selected.remainder(pairs)
        batch_index = torch.div(flattened_query_head, repeats, rounding_mode="floor")
        local_head = flattened_query_head.remainder(repeats)
        query_position = pair_queries[pair_offset]
        sampled = query_states[
            batch_index,
            start + local_head,
            query_position,
        ].float() / math.sqrt(dimension)
        output.append(sampled)
    result = torch.stack(output, dim=0)
    expected = (kv_heads, samples_per_head, dimension)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise ValueError("post-RoPE Fisher sample output is invalid")
    return result


@torch.no_grad()
def normalize_fisher(samples_outer_mean: torch.Tensor) -> torch.Tensor:
    """Symmetrize, project to PSD, and normalize every head to trace D."""

    if samples_outer_mean.ndim != 3 or samples_outer_mean.shape[-1] != samples_outer_mean.shape[-2]:
        raise ValueError("Fisher input must have shape [H,D,D]")
    _validate_finite(samples_outer_mean, "samples_outer_mean")
    dimension = int(samples_outer_mean.shape[-1])
    symmetric = 0.5 * (
        samples_outer_mean.double() + samples_outer_mean.double().transpose(-1, -2)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric.cpu())
    scale = torch.maximum(torch.ones_like(eigenvalues[:, 0]), eigenvalues.abs().amax(-1))
    tolerance = 4096.0 * torch.finfo(torch.float64).eps * dimension * scale
    if bool(torch.any(eigenvalues.amin(-1) < -tolerance)):
        raise ValueError("Fisher has a materially negative eigenvalue")
    eigenvalues = eigenvalues.clamp_min(0.0)
    reconstructed = torch.matmul(
        eigenvectors * eigenvalues.unsqueeze(-2), eigenvectors.transpose(-1, -2)
    )
    traces = reconstructed.diagonal(dim1=-2, dim2=-1).sum(-1)
    if bool(torch.any(traces <= 0.0)):
        raise ValueError("Fisher has nonpositive trace")
    return (reconstructed * (dimension / traces)[:, None, None]).float().contiguous()


@torch.no_grad()
def metric_codebook_diagnostics(
    blocks: torch.Tensor,
    codebook: torch.Tensor,
    block_whiteners: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Measure deployed shared-table distortion and index usage."""

    if blocks.ndim != 4:
        raise ValueError("blocks must have shape [H,M,N,S]")
    heads, groups, samples, block_size = map(int, blocks.shape)
    if tuple(block_whiteners.shape) != (heads, groups, block_size, block_size):
        raise ValueError("block whiteners have the wrong geometry")
    if codebook.ndim != 2 or int(codebook.shape[1]) != block_size:
        raise ValueError("codebook has the wrong geometry")
    device = blocks.device
    centers = codebook.to(device=device, dtype=torch.float32)
    whiteners = block_whiteners.to(device=device, dtype=torch.float32)
    weights = _expanded_sample_weights(
        sample_weights,
        heads=heads,
        groups=groups,
        samples=samples,
        device=device,
    )
    counts = torch.zeros(len(centers), dtype=torch.long, device=device)
    metric_sse = torch.zeros((), dtype=torch.float64, device=device)
    euclidean_sse = torch.zeros((), dtype=torch.float64, device=device)
    observations = 0
    weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    for head in range(heads):
        for group in range(groups):
            vectors = blocks[head, group].float()
            whitener = whiteners[head, group]
            group_weights = weights[head, group]
            selector = centers @ whitener.T
            for start in range(0, samples, chunk_size):
                stop = min(start + chunk_size, samples)
                current = vectors[start:stop]
                transformed = current @ whitener.T
                distances = squared_distances(transformed, selector)
                indices = distances.argmin(-1)
                decoded = centers[indices]
                current_weights = group_weights[start:stop]
                counts += torch.bincount(indices, minlength=len(centers))
                metric_sse += (
                    distances.gather(1, indices[:, None]).squeeze(1).double()
                    * current_weights
                ).sum()
                euclidean_sse += (
                    (current - decoded).double().square().sum(-1) * current_weights
                ).sum()
                observations += int(stop - start)
                weight_sum += current_weights.sum()
    probabilities = counts.double() / max(observations, 1)
    occupied = probabilities > 0
    entropy = -(probabilities[occupied] * probabilities[occupied].log2()).sum()
    normalized_entropy = entropy / math.log2(len(centers))
    return {
        "observations": observations,
        "metric_vector_mse": float(metric_sse / weight_sum),
        "euclidean_vector_mse": float(euclidean_sse / weight_sum),
        "sample_weighting": (
            "uniform" if sample_weights is None else "norm_squared_mean_one"
        ),
        "sample_weight_minimum": float(weights.min()),
        "sample_weight_maximum": float(weights.max()),
        "code_utilization": float((counts > 0).float().mean()),
        "normalized_code_entropy": float(normalized_entropy),
        "minimum_code_count": int(counts.min()),
        "maximum_code_count": int(counts.max()),
        "finite": bool(
            torch.isfinite(metric_sse)
            and torch.isfinite(euclidean_sse)
            and torch.isfinite(normalized_entropy)
        ),
    }


@torch.no_grad()
def refine_shared_metric_codebook(
    training_blocks: torch.Tensor,
    initial_codebook: torch.Tensor,
    block_whiteners: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    iterations: int,
    chunk_size: int = 1024,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Metric Lloyd updates for one decode table shared by all head/blocks."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    if training_blocks.ndim != 4:
        raise ValueError("training_blocks must have shape [H,M,N,S]")
    heads, groups, samples, block_size = map(int, training_blocks.shape)
    if tuple(block_whiteners.shape) != (heads, groups, block_size, block_size):
        raise ValueError("block whiteners have the wrong geometry")
    if initial_codebook.ndim != 2 or int(initial_codebook.shape[1]) != block_size:
        raise ValueError("initial codebook has the wrong geometry")
    _validate_finite(training_blocks, "training_blocks")
    _validate_finite(initial_codebook, "initial_codebook")
    _validate_finite(block_whiteners, "block_whiteners")
    device = training_blocks.device
    centers = initial_codebook.to(device=device, dtype=torch.float32).clone()
    whiteners = block_whiteners.to(device=device, dtype=torch.float32)
    weights = _expanded_sample_weights(
        sample_weights,
        heads=heads,
        groups=groups,
        samples=samples,
        device=device,
    )
    identity = torch.eye(block_size, device=device, dtype=torch.float64)
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        normal = torch.zeros(
            len(centers), block_size, block_size, device=device, dtype=torch.float64
        )
        right = torch.zeros(
            len(centers), block_size, device=device, dtype=torch.float64
        )
        counts = torch.zeros(len(centers), device=device, dtype=torch.long)
        total_weight = torch.zeros((), device=device, dtype=torch.float64)
        distortion = torch.zeros((), device=device, dtype=torch.float64)
        for head in range(heads):
            for group in range(groups):
                vectors = training_blocks[head, group].float()
                whitener = whiteners[head, group]
                metric = (whitener.T @ whitener).double()
                selector = centers @ whitener.T
                group_counts = torch.zeros_like(counts)
                group_weight_sums = torch.zeros(
                    len(centers), device=device, dtype=torch.float64
                )
                group_sums = torch.zeros(
                    len(centers), block_size, device=device, dtype=torch.float64
                )
                for start in range(0, samples, chunk_size):
                    stop = min(start + chunk_size, samples)
                    current = vectors[start:stop]
                    transformed = current @ whitener.T
                    distances = squared_distances(transformed, selector)
                    indices = distances.argmin(-1)
                    current_weights = weights[head, group, start:stop]
                    group_counts += torch.bincount(indices, minlength=len(centers))
                    group_weight_sums.index_add_(0, indices, current_weights)
                    group_sums.index_add_(
                        0, indices, current.double() * current_weights[:, None]
                    )
                    distortion += (
                        distances.gather(1, indices[:, None]).squeeze(1).double()
                        * current_weights
                    ).sum()
                    total_weight += current_weights.sum()
                counts += group_counts
                normal += group_weight_sums[:, None, None] * metric[None]
                right += group_sums @ metric
        occupied = counts > 0
        if bool(occupied.any()):
            systems = normal[occupied] + ridge * identity[None]
            solutions = torch.linalg.solve(systems, right[occupied].unsqueeze(-1))
            centers[occupied] = solutions.squeeze(-1).float()
        if not bool(torch.isfinite(centers).all()):
            raise FloatingPointError("metric codebook refinement produced non-finite centers")
        history.append(
            {
                "iteration": iteration,
                "metric_vector_mse_before_update": float(
                    distortion / total_weight
                ),
                "sample_weighting": (
                    "uniform" if sample_weights is None else "norm_squared_mean_one"
                ),
                "occupied_codes": int(occupied.sum()),
                "minimum_code_count": int(counts.min()),
                "maximum_code_count": int(counts.max()),
            }
        )
    # The real decoder stores FP16. Selector tables must be derived from that
    # rounded table, not the unavailable FP32 optimizer state.
    deployed = centers.half().float().contiguous()
    return deployed.cpu(), history


@torch.no_grad()
def quantize_m20_fibquant_key(
    states: torch.Tensor,
    rotation: torch.Tensor,
    decode_codebook: torch.Tensor,
    block_whiteners: torch.Tensor,
    *,
    block_size: int,
    chunk_size: int = 8192,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Deployment-faithful SCF-FibQuant K selector and ordinary lookup decode."""

    if states.ndim != 4:
        raise ValueError("states must have shape [B,H,T,D]")
    batch, heads, tokens, dimension = map(int, states.shape)
    blocks = dimension // block_size
    if dimension % block_size:
        raise ValueError("block size must divide dimension")
    if tuple(rotation.shape) != (dimension, dimension):
        raise ValueError("rotation has the wrong geometry")
    if tuple(block_whiteners.shape) != (heads, blocks, block_size, block_size):
        raise ValueError("block whiteners have the wrong geometry")
    if decode_codebook.ndim != 2 or int(decode_codebook.shape[1]) != block_size:
        raise ValueError("decode codebook has the wrong geometry")
    flat = states.reshape(-1, dimension).float()
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(torch.finfo(torch.float32).tiny)
    rotated = (flat / safe_norms) @ rotation.float().T
    grouped = rotated.reshape(batch, heads, tokens, blocks, block_size)
    decoded = torch.empty_like(grouped)
    indices_out = (
        torch.empty(
            batch, heads, tokens, blocks, dtype=torch.long, device=states.device
        )
        if return_indices
        else None
    )
    centers = decode_codebook.to(device=states.device, dtype=torch.float32)
    whiteners = block_whiteners.to(device=states.device, dtype=torch.float32)
    for head in range(heads):
        for block in range(blocks):
            vectors = grouped[:, head, :, block].reshape(-1, block_size)
            output = decoded[:, head, :, block].reshape(-1, block_size)
            index_output = (
                indices_out[:, head, :, block].reshape(-1)
                if indices_out is not None
                else None
            )
            whitener = whiteners[head, block]
            selector = centers @ whitener.T
            for start in range(0, len(vectors), chunk_size):
                stop = min(start + chunk_size, len(vectors))
                transformed = vectors[start:stop].float() @ whitener.T
                indices = squared_distances(transformed, selector).argmin(-1)
                output[start:stop] = centers[indices]
                if index_output is not None:
                    index_output[start:stop] = indices
    decoded_rotated = decoded.reshape(-1, dimension)
    reconstructed = (decoded_rotated @ rotation.float()) * norms
    zero = norms.squeeze(-1) == 0
    if bool(zero.any()):
        reconstructed[zero] = 0
    return reconstructed.reshape_as(states).to(states.dtype), indices_out
