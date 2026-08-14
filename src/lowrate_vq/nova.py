"""Rate-generalized NOVA QPCA transform and grouped-key VQ.

The QPCA equations and stratified permutation follow the official NOVA-KV
calibration code.  The strict 1-bit adaptation uses four coordinates and four
index bits per group (16 centroids), disables per-token RMS metadata, and can
replace the query metric by the one frozen M20 spectrum-tempering rule.
"""

from __future__ import annotations

import math

import torch

from fw_gsrq.codec import build_sqrt_fisher_whitener
from lowrate_vq.cq import quantize_cq


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


@torch.no_grad()
def m20_query_metric(sigma_q: torch.Tensor) -> torch.Tensor:
    """Apply the fixed M20 power-1/2 plus identity rule head by head."""

    if sigma_q.ndim != 3 or sigma_q.shape[-1] != sigma_q.shape[-2]:
        raise ValueError("sigma_q must have shape [H,D,D]")
    metrics = []
    for head in range(sigma_q.shape[0]):
        metrics.append(build_sqrt_fisher_whitener(sigma_q[head].float()).metric)
    return torch.stack(metrics)


@torch.no_grad()
def build_qpca_basis(
    sigma_q: torch.Tensor, k_cov: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official NOVA non-orthogonal transform under a supplied query metric."""

    if sigma_q.shape != k_cov.shape or sigma_q.ndim != 3:
        raise ValueError("sigma_q and k_cov must share shape [H,D,D]")
    sq = _sym(sigma_q.to(torch.float64))
    sk = _sym(k_cov.to(torch.float64))
    eigenvalues, eigenvectors = torch.linalg.eigh(sq)
    eigenvalues = eigenvalues.clamp_min(1e-30)
    sqrt_mq = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.pow(0.5))
        @ eigenvectors.transpose(-1, -2)
    )
    inverse_sqrt_mq = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.pow(-0.5))
        @ eigenvectors.transpose(-1, -2)
    )
    balanced = _sym(sqrt_mq @ sk @ sqrt_mq)
    spectrum, basis = torch.linalg.eigh(balanced)
    order = torch.argsort(spectrum, dim=-1, descending=True)
    basis = torch.gather(
        basis,
        -1,
        order.unsqueeze(-2).expand(*basis.shape[:-1], -1),
    )
    return sqrt_mq @ basis, basis.transpose(-1, -2) @ inverse_sqrt_mq


def stratified_permutation(dimension: int, group_size: int) -> torch.Tensor:
    groups = math.ceil(dimension / group_size)
    permutation = [
        group + offset * groups
        for group in range(groups)
        for offset in range(group_size)
        if group + offset * groups < dimension
    ]
    if sorted(permutation) != list(range(dimension)):
        raise AssertionError("NOVA stratified ordering is not a permutation")
    return torch.tensor(permutation, dtype=torch.long)


@torch.no_grad()
def apply_stratified_grouping(
    forward: torch.Tensor,
    inverse: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = int(forward.shape[-1])
    permutation = stratified_permutation(dimension, group_size)
    return (
        forward[:, :, permutation].contiguous(),
        inverse[:, permutation, :].contiguous(),
        permutation,
    )


@torch.no_grad()
def fp8_e5m2_roundtrip(codebooks: torch.Tensor) -> torch.Tensor:
    """Official per-group amax-scaled E5M2 deployment roundtrip."""

    if codebooks.ndim != 4:
        raise ValueError("codebooks must have shape [H,G,K,S]")
    work = codebooks.float()
    scale = work.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / 57344.0
    return ((work / scale).to(torch.float8_e5m2).float() * scale).half()


@torch.no_grad()
def quantize_nova_key(
    states: torch.Tensor,
    forward: torch.Tensor,
    inverse: torch.Tensor,
    mean: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    chunk_size: int = 4096,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Roundtrip post-RoPE K with the official row-vector convention."""

    if states.ndim != 4:
        raise ValueError("states must have shape [B,H,T,D]")
    batch, heads, tokens, dimension = map(int, states.shape)
    if tuple(forward.shape) != (heads, dimension, dimension):
        raise ValueError("NOVA forward transform has the wrong geometry")
    if tuple(inverse.shape) != (heads, dimension, dimension):
        raise ValueError("NOVA inverse transform has the wrong geometry")
    if tuple(mean.shape) != (heads, dimension):
        raise ValueError("NOVA key mean has the wrong geometry")
    rows = states.permute(1, 0, 2, 3).reshape(heads, batch * tokens, dimension)
    transformed = torch.bmm(
        rows.float() - mean.to(states.device, dtype=torch.float32)[:, None],
        forward.to(states.device, dtype=torch.float32),
    )
    grouped_states = transformed.reshape(heads, batch, tokens, dimension).permute(
        1, 0, 2, 3
    )
    decoded_grouped, indices = quantize_cq(
        grouped_states,
        codebooks,
        chunk_size=chunk_size,
        return_indices=return_indices,
    )
    decoded_rows = decoded_grouped.permute(1, 0, 2, 3).reshape(
        heads, batch * tokens, dimension
    )
    reconstructed = torch.bmm(
        decoded_rows.float(), inverse.to(states.device, dtype=torch.float32)
    ) + mean.to(states.device, dtype=torch.float32)[:, None]
    return (
        reconstructed.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .to(states.dtype),
        indices,
    )

