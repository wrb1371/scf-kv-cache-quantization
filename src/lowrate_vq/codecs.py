"""Fixed-rate low-bit VQ codec kernels used by the baseline reproductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RateLedger:
    index_bpa: float
    metadata_bpa: float
    total_bpa: float
    metadata: dict[str, Any]


def squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared distances without materializing coordinate deltas."""

    return (
        x.float().square().sum(dim=-1, keepdim=True)
        + centers.float().square().sum(dim=-1).unsqueeze(0)
        - 2.0 * x.float() @ centers.float().transpose(0, 1)
    ).clamp_min_(0.0)


@torch.no_grad()
def quantize_fibquant(
    states: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    *,
    block_size: int,
    chunk_size: int = 8192,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize [B,H,T,D] vectors with the FibQuant normalize/rotate codec."""

    if states.ndim != 4:
        raise ValueError("states must have shape [B,H,T,D]")
    dimension = int(states.shape[-1])
    if tuple(rotation.shape) != (dimension, dimension):
        raise ValueError("FibQuant rotation has the wrong geometry")
    if codebook.ndim != 2 or int(codebook.shape[-1]) != block_size:
        raise ValueError("FibQuant codebook has the wrong geometry")
    if dimension % block_size:
        raise ValueError("block size must divide the head dimension")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    original_shape = states.shape
    flat = states.reshape(-1, dimension).float()
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    safe_norms = norms.clamp_min(torch.finfo(torch.float32).tiny)
    rotated = (flat / safe_norms) @ rotation.float().transpose(0, 1)
    blocks = rotated.reshape(-1, block_size)
    decoded_blocks = torch.empty_like(blocks)
    all_indices = (
        torch.empty(blocks.shape[0], dtype=torch.long, device=blocks.device)
        if return_indices
        else None
    )
    table = codebook.to(device=blocks.device, dtype=torch.float32)
    for start in range(0, blocks.shape[0], chunk_size):
        end = min(start + chunk_size, blocks.shape[0])
        distances = squared_distances(blocks[start:end], table)
        indices = distances.argmin(dim=-1)
        decoded_blocks[start:end] = table[indices]
        if all_indices is not None:
            all_indices[start:end] = indices
    decoded_rotated = decoded_blocks.reshape(-1, dimension)
    decoded = (decoded_rotated @ rotation.float()) * norms
    zero = norms.squeeze(-1) == 0
    if bool(zero.any()):
        decoded[zero] = 0
    decoded = decoded.reshape(original_shape).to(dtype=states.dtype)
    if all_indices is not None:
        all_indices = all_indices.reshape(
            *original_shape[:-1], dimension // block_size
        )
    return decoded, all_indices


@torch.no_grad()
def quantize_opq(
    states: torch.Tensor,
    rotations: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    chunk_size: int = 8192,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize [B,H,T,D] vectors with head-local OPQ transforms and PQ tables."""

    if states.ndim != 4:
        raise ValueError("states must have shape [B,H,T,D]")
    batch, heads, tokens, dimension = map(int, states.shape)
    if tuple(rotations.shape) != (heads, dimension, dimension):
        raise ValueError("OPQ rotations have the wrong geometry")
    if codebooks.ndim != 4 or int(codebooks.shape[0]) != heads:
        raise ValueError("OPQ codebooks must have shape [H,M,K,S]")
    subspaces, _, subdim = map(int, codebooks.shape[1:])
    if subspaces * subdim != dimension:
        raise ValueError("OPQ subspaces do not cover the head dimension")

    rotated = torch.einsum(
        "hde,bhte->bhtd", rotations.float(), states.float()
    )
    rotated = rotated.reshape(batch, heads, tokens, subspaces, subdim)
    decoded = torch.empty_like(rotated)
    indices_out = (
        torch.empty(
            batch,
            heads,
            tokens,
            subspaces,
            dtype=torch.long,
            device=states.device,
        )
        if return_indices
        else None
    )
    tables = codebooks.to(device=states.device, dtype=torch.float32)
    for head in range(heads):
        for subspace in range(subspaces):
            vectors = rotated[:, head, :, subspace].reshape(-1, subdim)
            table = tables[head, subspace]
            decoded_vectors = decoded[:, head, :, subspace].reshape(
                -1, subdim
            )
            index_vectors = (
                indices_out[:, head, :, subspace].reshape(-1)
                if indices_out is not None
                else None
            )
            for start in range(0, vectors.shape[0], chunk_size):
                end = min(start + chunk_size, vectors.shape[0])
                selected = squared_distances(
                    vectors[start:end], table
                ).argmin(dim=-1)
                decoded_vectors[start:end] = table[selected]
                if index_vectors is not None:
                    index_vectors[start:end] = selected
    decoded = decoded.reshape(batch, heads, tokens, dimension)
    decoded = torch.einsum(
        "hde,bhtd->bhte", rotations.float(), decoded
    ).to(dtype=states.dtype)
    return decoded, indices_out


def fibquant_rate(
    *, dimension: int, block_size: int, codebook_size: int, norm_bits: int = 16
) -> RateLedger:
    if 2 ** (codebook_size.bit_length() - 1) != codebook_size:
        raise ValueError("codebook_size must be a power of two")
    # A power of two N needs log2(N), not Python's integer bit length.
    index_bits = (dimension // block_size) * (codebook_size.bit_length() - 1)
    index_bpa = index_bits / dimension
    metadata_bpa = norm_bits / dimension
    return RateLedger(
        index_bpa=index_bpa,
        metadata_bpa=metadata_bpa,
        total_bpa=index_bpa + metadata_bpa,
        metadata={
            "index_bits_per_vector": index_bits,
            "norm_bits_per_vector": norm_bits,
            "dimension": dimension,
            "block_size": block_size,
            "codebook_size": codebook_size,
        },
    )


def opq_rate(
    *, dimension: int, subspaces: int, centroids: int
) -> RateLedger:
    if centroids < 2 or centroids & (centroids - 1):
        raise ValueError("centroids must be a power of two")
    bits_per_index = centroids.bit_length() - 1
    index_bits = subspaces * bits_per_index
    index_bpa = index_bits / dimension
    return RateLedger(
        index_bpa=index_bpa,
        metadata_bpa=0.0,
        total_bpa=index_bpa,
        metadata={
            "index_bits_per_vector": index_bits,
            "bits_per_index": bits_per_index,
            "dimension": dimension,
            "subspaces": subspaces,
            "centroids": centroids,
        },
    )
