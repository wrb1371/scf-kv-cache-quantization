"""Non-parametric OPQ (CVPR 2013 Algorithm 1) for batched KV heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OPQFit:
    rotations: torch.Tensor
    codebooks: torch.Tensor
    indices: torch.Tensor
    history: list[dict[str, float | int]]
    diagnostics: dict[str, float | int | bool]


def _assign(
    vectors: torch.Tensor,
    centers: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign [H,N,S] to [H,K,S], returning indices and squared errors."""

    heads, rows, _ = vectors.shape
    assignments = torch.empty(heads, rows, dtype=torch.long, device=vectors.device)
    errors = torch.empty(heads, rows, dtype=torch.float32, device=vectors.device)
    center_norms = centers.float().square().sum(dim=-1).unsqueeze(1)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        chunk = vectors[:, start:end].float()
        distances = (
            chunk.square().sum(dim=-1, keepdim=True)
            + center_norms
            - 2.0 * torch.bmm(chunk, centers.float().transpose(1, 2))
        ).clamp_min_(0.0)
        selected_error, selected = distances.min(dim=-1)
        assignments[:, start:end] = selected
        errors[:, start:end] = selected_error
    return assignments, errors


def _lloyd_step(
    vectors: torch.Tensor,
    centers: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments, errors = _assign(vectors, centers, chunk_size=chunk_size)
    heads, _, subdim = vectors.shape
    clusters = int(centers.shape[1])
    sums = torch.zeros(heads, clusters, subdim, device=vectors.device)
    counts = torch.zeros(heads, clusters, device=vectors.device)
    for start in range(0, vectors.shape[1], chunk_size):
        end = min(start + chunk_size, vectors.shape[1])
        selected = assignments[:, start:end]
        sums.scatter_add_(
            1,
            selected.unsqueeze(-1).expand(-1, -1, subdim),
            vectors[:, start:end].float(),
        )
        counts.scatter_add_(1, selected, torch.ones_like(selected, dtype=torch.float32))
    updated = centers.float().clone()
    occupied = counts > 0
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    updated = torch.where(occupied.unsqueeze(-1), means, updated)
    for head in range(heads):
        empty = torch.nonzero(~occupied[head], as_tuple=False).flatten()
        if len(empty):
            ranked = torch.argsort(errors[head], descending=True)
            for offset, cluster in enumerate(empty.tolist()):
                updated[head, cluster] = vectors[head, ranked[offset]].float()
    assignments, errors = _assign(vectors, updated, chunk_size=chunk_size)
    return updated, assignments, errors


def _encode_decode(
    rotated: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads, rows, dimension = rotated.shape
    subspaces, _, subdim = map(int, codebooks.shape[1:])
    decoded = torch.empty_like(rotated, dtype=torch.float32)
    indices = torch.empty(
        heads, rows, subspaces, dtype=torch.long, device=rotated.device
    )
    errors = torch.empty(
        heads, rows, subspaces, dtype=torch.float32, device=rotated.device
    )
    for subspace in range(subspaces):
        start = subspace * subdim
        end = start + subdim
        selected, selected_error = _assign(
            rotated[:, :, start:end],
            codebooks[:, subspace],
            chunk_size=chunk_size,
        )
        indices[:, :, subspace] = selected
        errors[:, :, subspace] = selected_error
        decoded[:, :, start:end] = torch.gather(
            codebooks[:, subspace],
            1,
            selected.unsqueeze(-1).expand(-1, -1, subdim),
        )
    if decoded.shape[-1] != dimension:
        raise RuntimeError("decoded OPQ dimension mismatch")
    return decoded, indices, errors


def _utilization(indices: torch.Tensor, clusters: int) -> tuple[float, float]:
    fractions = []
    entropies = []
    denominator = torch.log(torch.tensor(float(clusters), device=indices.device))
    for head in range(indices.shape[0]):
        for subspace in range(indices.shape[2]):
            counts = torch.bincount(
                indices[head, :, subspace], minlength=clusters
            ).float()
            fractions.append(float((counts > 0).float().mean()))
            probabilities = counts[counts > 0] / counts.sum()
            entropies.append(float(-(probabilities * probabilities.log()).sum() / denominator))
    return sum(fractions) / len(fractions), sum(entropies) / len(entropies)


@torch.no_grad()
def fit_opq(
    vectors: torch.Tensor,
    *,
    subspaces: int,
    clusters: int,
    iterations: int,
    chunk_size: int,
    seed: int,
) -> OPQFit:
    """Fit independent OPQ models for H matrices in [H,N,D]."""

    if vectors.ndim != 3:
        raise ValueError("vectors must have shape [H,N,D]")
    heads, rows, dimension = map(int, vectors.shape)
    if dimension % subspaces:
        raise ValueError("subspaces must divide dimension")
    if rows < clusters:
        raise ValueError("training rows must be at least the cluster count")
    if iterations < 1 or chunk_size < 1:
        raise ValueError("iterations and chunk_size must be positive")
    vectors = vectors.float()
    subdim = dimension // subspaces
    rotations = torch.eye(dimension, device=vectors.device).repeat(heads, 1, 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codebooks = torch.empty(
        heads, subspaces, clusters, subdim, device=vectors.device
    )
    for head in range(heads):
        for subspace in range(subspaces):
            selected = torch.randperm(rows, generator=generator)[:clusters].to(vectors.device)
            start = subspace * subdim
            codebooks[head, subspace] = vectors[head, selected, start : start + subdim]

    history: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        rotated = torch.einsum("hde,hne->hnd", rotations, vectors)
        for subspace in range(subspaces):
            start = subspace * subdim
            codebooks[:, subspace], _, _ = _lloyd_step(
                rotated[:, :, start : start + subdim],
                codebooks[:, subspace],
                chunk_size=chunk_size,
            )
        decoded, indices, errors = _encode_decode(
            rotated, codebooks, chunk_size=chunk_size
        )
        cross = torch.einsum("hnd,hne->hde", vectors, decoded)
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        # For row vectors, A=R^T=U V^T solves min ||X A - Y||_F.
        rotations = (u @ vh).transpose(1, 2).contiguous()
        objective = float(errors.sum(dim=-1).mean())
        orthogonality = float(
            (
                rotations @ rotations.transpose(1, 2)
                - torch.eye(dimension, device=vectors.device)
            )
            .abs()
            .max()
        )
        history.append(
            {
                "iteration": iteration,
                "rotated_vector_mse": objective,
                "rotation_orthogonality_max_abs": orthogonality,
            }
        )

    rotated = torch.einsum("hde,hne->hnd", rotations, vectors)
    decoded, indices, errors = _encode_decode(rotated, codebooks, chunk_size=chunk_size)
    raw_decoded = torch.einsum("hde,hnd->hne", rotations, decoded)
    utilization, normalized_entropy = _utilization(indices, clusters)
    diagnostics: dict[str, float | int | bool] = {
        "heads": heads,
        "training_vectors_per_head": rows,
        "dimension": dimension,
        "subspaces": subspaces,
        "clusters": clusters,
        "final_vector_mse": float((vectors - raw_decoded).square().sum(dim=-1).mean()),
        "mean_code_utilization": utilization,
        "mean_normalized_code_entropy": normalized_entropy,
        "rotation_orthogonality_max_abs": float(
            (
                rotations @ rotations.transpose(1, 2)
                - torch.eye(dimension, device=vectors.device)
            )
            .abs()
            .max()
        ),
        "finite": bool(
            torch.isfinite(rotations).all() and torch.isfinite(codebooks).all()
        ),
    }
    return OPQFit(rotations, codebooks, indices, history, diagnostics)
