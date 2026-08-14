"""Successive residual quantization using GSKM codebooks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch

from .gskm import assign_codewords, fit_gskm


@dataclass
class ResidualQuantizerModel:
    """A fixed sequence of full-dimensional residual codebooks."""

    codebooks: torch.Tensor
    codebook_bits: int = 8

    def __post_init__(self) -> None:
        if self.codebooks.ndim != 3:
            raise ValueError("codebooks must have shape [R,K,D]")
        if self.codebook_bits < 1:
            raise ValueError("codebook_bits must be positive")
        expected = 1 << self.codebook_bits
        if self.codebooks.shape[1] != expected:
            raise ValueError(
                f"K={self.codebooks.shape[1]} does not match "
                f"{self.codebook_bits} index bits"
            )
        if not torch.isfinite(self.codebooks).all():
            raise ValueError("codebooks must be finite")

    @property
    def stages(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def clusters(self) -> int:
        return int(self.codebooks.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.codebooks.shape[2])

    @property
    def bits_per_activation(self) -> float:
        return self.stages * self.codebook_bits / self.dimension

    @torch.no_grad()
    def encode(
        self, data: torch.Tensor, *, chunk_size: int = 16384
    ) -> torch.Tensor:
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError("data has the wrong shape")
        residual = data.float().clone()
        indices: list[torch.Tensor] = []
        for codebook in self.codebooks:
            stage_indices = assign_codewords(
                residual,
                codebook.to(residual.device),
                chunk_size=chunk_size,
            )
            indices.append(stage_indices)
            residual -= codebook.to(residual.device).index_select(
                0, stage_indices
            )
        return torch.stack(indices, dim=-1)

    @torch.no_grad()
    def decode(
        self,
        indices: torch.Tensor,
        *,
        prefix_stages: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [N,R]")
        prefix = self.stages if prefix_stages is None else prefix_stages
        if not 0 <= prefix <= self.stages:
            raise ValueError("prefix_stages is out of range")
        if indices.shape[1] < prefix:
            raise ValueError("indices do not contain the requested prefix")
        output = torch.zeros(
            indices.shape[0],
            self.dimension,
            device=indices.device,
            dtype=torch.float32,
        )
        for stage in range(prefix):
            output += self.codebooks[stage].to(
                indices.device
            ).index_select(0, indices[:, stage].long())
        return output.to(dtype=dtype)

    @torch.no_grad()
    def prefix_reconstructions(
        self,
        indices: torch.Tensor,
        prefixes: Iterable[int],
    ) -> dict[int, torch.Tensor]:
        return {
            int(prefix): self.decode(
                indices, prefix_stages=int(prefix)
            )
            for prefix in prefixes
        }


@torch.no_grad()
def fit_residual_quantizer(
    data: torch.Tensor,
    *,
    stages: int = 32,
    clusters: int = 256,
    weights: torch.Tensor | None = None,
    max_iterations: int = 100,
    chunk_size: int = 16384,
    seed: int = 0,
    tolerance: float = 0.0,
    initialization: str = "sample",
) -> tuple[ResidualQuantizerModel, torch.Tensor, list[dict[str, Any]]]:
    """Fit one GSKM codebook to the residual at each stage."""

    if stages < 1:
        raise ValueError("stages must be positive")
    residual = data.float().clone()
    codebooks: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    stage_history: list[dict[str, Any]] = []
    for stage in range(stages):
        before_mse = float(residual.square().sum(dim=-1).mean())
        fitted = fit_gskm(
            residual,
            clusters=clusters,
            weights=weights,
            max_iterations=max_iterations,
            chunk_size=chunk_size,
            seed=seed + stage,
            tolerance=tolerance,
            initialization=initialization,
        )
        codebook = fitted.codebook
        indices = assign_codewords(
            residual, codebook, chunk_size=chunk_size
        )
        residual -= codebook.index_select(0, indices)
        after_mse = float(residual.square().sum(dim=-1).mean())
        codebooks.append(codebook.cpu())
        all_indices.append(indices.cpu())
        stage_history.append(
            {
                "stage": stage,
                "residual_mse_before": before_mse,
                "residual_mse_after": after_mse,
                "gskm_iterations": fitted.iterations,
                "gskm_converged": fitted.converged,
                "gskm_last": fitted.history[-1],
            }
        )
    codebook_bits = int(round(math.log2(clusters)))
    if 1 << codebook_bits != clusters:
        raise ValueError("clusters must be a power of two")
    model = ResidualQuantizerModel(
        torch.stack(codebooks), codebook_bits=codebook_bits
    )
    return model, torch.stack(all_indices, dim=-1), stage_history
