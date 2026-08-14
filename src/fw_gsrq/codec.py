"""Deployment-faithful core for Fisher-whitened GSRQ.

The cache stores the same fixed-width residual indices as GSRQ.  A frozen
per-layer/head transform changes the coordinate system used to train the key
codebooks and select indices.  Decode tables remain in the original pre-RoPE
key coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from gsrq.residual_quantizer import (
    ResidualQuantizerModel,
    fit_residual_quantizer,
)


@dataclass(frozen=True)
class WhiteningTransform:
    """Fixed `M = L^T L` factorization used by one layer/head."""

    metric: torch.Tensor
    whitener: torch.Tensor
    diagnostics: dict[str, float]


def _validate_rows(data: torch.Tensor, dimension: int, name: str) -> None:
    if data.ndim != 2 or int(data.shape[1]) != dimension:
        raise ValueError(f"{name} must have shape [N,{dimension}]")
    if not data.is_floating_point() or not bool(torch.isfinite(data).all()):
        raise ValueError(f"{name} must be finite floating point")


@torch.no_grad()
def build_spectral_fisher_whitener(
    fisher: torch.Tensor,
    *,
    spectral_power: float = 0.5,
    fisher_mixture: float = 0.5,
) -> WhiteningTransform:
    """Build ``M=(1-alpha)I+alpha*d*F^p/tr(F^p)`` deterministically.

    This general form exists for attribution and sensitivity experiments.  The
    paper default remains ``p=0.5, alpha=0.5``.  In particular, Euclidean
    geometry is represented explicitly by ``I`` rather than by ``F**0``, whose
    value is ambiguous for a rank-deficient empirical Fisher matrix.

    The input is symmetrized and negative eigenvalues caused by storage
    rounding are clamped.  No diagonal jitter or hidden eigenvalue floor is
    added: a singular ``alpha=1`` boundary therefore fails closed.
    """

    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("fisher must be a square matrix")
    if not fisher.is_floating_point() or not bool(torch.isfinite(fisher).all()):
        raise ValueError("fisher must be finite floating point")
    if not math.isfinite(spectral_power) or spectral_power <= 0.0:
        raise ValueError("spectral_power must be finite and positive")
    if not math.isfinite(fisher_mixture) or not 0.0 <= fisher_mixture <= 1.0:
        raise ValueError("fisher_mixture must be finite and within [0,1]")
    dimension = int(fisher.shape[0])
    work = fisher.detach().to(dtype=torch.float64)
    symmetric = 0.5 * (work + work.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    negative_count = int((eigenvalues < 0.0).sum().item())
    minimum_before = float(eigenvalues.min().item())
    powered_eigenvalues = eigenvalues.clamp_min(0.0).pow(spectral_power)
    powered = (eigenvectors * powered_eigenvalues.unsqueeze(0)) @ eigenvectors.T
    trace = torch.trace(powered)
    if not bool(trace > 0.0):
        raise ValueError("powered Fisher must have positive trace")
    normalized = powered * (float(dimension) / trace)
    identity = torch.eye(dimension, dtype=torch.float64, device=work.device)
    metric = (1.0 - fisher_mixture) * identity + fisher_mixture * normalized
    # torch.linalg.cholesky returns A with M=A A^T.  L=A^T therefore satisfies
    # the column-vector convention M=L^T L and z=Lx.
    cholesky, info = torch.linalg.cholesky_ex(metric)
    if int(info.item()) != 0:
        minimum_metric_eigenvalue = float(torch.linalg.eigvalsh(metric).min().item())
        raise ValueError(
            "spectral Fisher metric is not positive definite without an "
            f"identity floor (minimum eigenvalue={minimum_metric_eigenvalue:.6g})"
        )
    whitener = cholesky.T.contiguous()
    reconstruction = whitener.T @ whitener
    metric_eigenvalues = torch.linalg.eigvalsh(metric)
    diagnostics = {
        "dimension": float(dimension),
        "fisher_spectral_power": float(spectral_power),
        "fisher_mixture": float(fisher_mixture),
        "identity_mixture": float(1.0 - fisher_mixture),
        "negative_eigenvalues_clamped": float(negative_count),
        "minimum_fisher_eigenvalue_before_clamp": minimum_before,
        "normalized_powered_fisher_trace": float(torch.trace(normalized).item()),
        "metric_trace": float(torch.trace(metric).item()),
        "metric_minimum_eigenvalue": float(metric_eigenvalues.min().item()),
        "metric_maximum_eigenvalue": float(metric_eigenvalues.max().item()),
        "metric_effective_dimension": float(
            metric_eigenvalues.sum().square()
            / metric_eigenvalues.square().sum()
        ),
        "factorization_max_abs_error": float(
            (reconstruction - metric).abs().max().item()
        ),
    }
    return WhiteningTransform(
        metric=metric.float(),
        whitener=whitener.float(),
        diagnostics=diagnostics,
    )


@torch.no_grad()
def build_fixed_whitener(fisher: torch.Tensor) -> WhiteningTransform:
    """Build the legacy full-Fisher metric ``M=(I+F)/2``."""

    built = build_spectral_fisher_whitener(
        fisher,
        spectral_power=1.0,
        fisher_mixture=0.5,
    )
    diagnostics = dict(built.diagnostics)
    diagnostics["normalized_fisher_trace"] = diagnostics[
        "normalized_powered_fisher_trace"
    ]
    return WhiteningTransform(
        metric=built.metric,
        whitener=built.whitener,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def build_sqrt_fisher_whitener(fisher: torch.Tensor) -> WhiteningTransform:
    """Build SCF's fixed spectrum-compressed Fisher metric.

    `sqrt(F)` is trace-normalized before the same 50/50 identity mixture.  The
    square root is a fixed matrix-power tempering rule, not a fitted mixture
    coefficient or a scan over spectral caps.
    """

    built = build_spectral_fisher_whitener(
        fisher,
        spectral_power=0.5,
        fisher_mixture=0.5,
    )
    diagnostics = dict(built.diagnostics)
    diagnostics["normalized_tempered_trace"] = diagnostics[
        "normalized_powered_fisher_trace"
    ]
    return WhiteningTransform(
        metric=built.metric,
        whitener=built.whitener,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def transform_rows(data: torch.Tensor, whitener: torch.Tensor) -> torch.Tensor:
    """Apply `z=Lx` to row-major vectors."""

    if whitener.ndim != 2 or whitener.shape[0] != whitener.shape[1]:
        raise ValueError("whitener must be square")
    dimension = int(whitener.shape[0])
    _validate_rows(data, dimension, "data")
    return data.float() @ whitener.to(device=data.device, dtype=torch.float32).T


@torch.no_grad()
def inverse_transform_rows(
    transformed: torch.Tensor,
    whitener: torch.Tensor,
) -> torch.Tensor:
    """Map row-major selector coordinates back to raw key coordinates."""

    if whitener.ndim != 2 or whitener.shape[0] != whitener.shape[1]:
        raise ValueError("whitener must be square")
    dimension = int(whitener.shape[0])
    _validate_rows(transformed, dimension, "transformed")
    factor = whitener.to(device=transformed.device, dtype=torch.float32)
    return torch.linalg.solve_triangular(
        factor,
        transformed.float().T,
        upper=True,
    ).T.contiguous()


@torch.no_grad()
def roundtrip_selector_codebooks(
    selector_codebooks: torch.Tensor,
    whitener: torch.Tensor,
    *,
    decode_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create FP16 raw decode tables and their exact deployed selector view."""

    if selector_codebooks.ndim != 3:
        raise ValueError("selector_codebooks must have shape [R,K,D]")
    stages, clusters, dimension = map(int, selector_codebooks.shape)
    if tuple(whitener.shape) != (dimension, dimension):
        raise ValueError("whitener and codebook dimensions differ")
    if decode_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported decode dtype")
    flat_selector = selector_codebooks.float().reshape(-1, dimension)
    raw_fp32 = inverse_transform_rows(flat_selector, whitener)
    decode = raw_fp32.to(dtype=decode_dtype).reshape(
        stages, clusters, dimension
    ).contiguous()
    deployed_selector = transform_rows(
        decode.float().reshape(-1, dimension), whitener
    ).reshape_as(selector_codebooks).contiguous()
    denominator = flat_selector.square().sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    diagnostics = {
        "selector_roundtrip_relative_sse": float(
            (deployed_selector - selector_codebooks.float()).square().sum()
            / denominator
        ),
        "selector_roundtrip_max_abs_error": float(
            (deployed_selector - selector_codebooks.float()).abs().max()
        ),
        "decode_finite": float(bool(torch.isfinite(decode).all())),
    }
    return decode, deployed_selector, diagnostics


@dataclass
class FWResidualQuantizerModel:
    """One-head FW-GSRQ selector/decode pair."""

    selector_codebooks: torch.Tensor
    decode_codebooks: torch.Tensor
    whitener: torch.Tensor
    metric: torch.Tensor
    codebook_bits: int = 8

    def __post_init__(self) -> None:
        if self.selector_codebooks.ndim != 3:
            raise ValueError("selector_codebooks must have shape [R,K,D]")
        if self.decode_codebooks.shape != self.selector_codebooks.shape:
            raise ValueError("selector/decode codebook geometries differ")
        dimension = int(self.selector_codebooks.shape[-1])
        if tuple(self.whitener.shape) != (dimension, dimension):
            raise ValueError("whitener has wrong geometry")
        if tuple(self.metric.shape) != (dimension, dimension):
            raise ValueError("metric has wrong geometry")
        expected_clusters = 1 << int(self.codebook_bits)
        if int(self.selector_codebooks.shape[1]) != expected_clusters:
            raise ValueError("cluster count and codebook_bits differ")
        tensors = (
            self.selector_codebooks,
            self.decode_codebooks,
            self.whitener,
            self.metric,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("FW-GSRQ tensors must be finite")

    @property
    def stages(self) -> int:
        return int(self.selector_codebooks.shape[0])

    @property
    def clusters(self) -> int:
        return int(self.selector_codebooks.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.selector_codebooks.shape[2])

    @property
    def bits_per_activation(self) -> float:
        return self.stages * self.codebook_bits / self.dimension

    @property
    def serialized_static_bytes(self) -> int:
        """Bytes for FP16 raw tables plus FP32 L; selector tables are derived."""

        return (
            self.decode_codebooks.numel() * self.decode_codebooks.element_size()
            + self.whitener.numel() * self.whitener.element_size()
        )

    @property
    def derived_selector_bytes(self) -> int:
        return self.selector_codebooks.numel() * self.selector_codebooks.element_size()

    @torch.no_grad()
    def encode_selector(
        self,
        transformed: torch.Tensor,
        *,
        chunk_size: int = 16384,
    ) -> torch.Tensor:
        _validate_rows(transformed, self.dimension, "transformed")
        selector = ResidualQuantizerModel(
            self.selector_codebooks,
            codebook_bits=self.codebook_bits,
        )
        return selector.encode(transformed, chunk_size=chunk_size)

    @torch.no_grad()
    def encode(
        self,
        data: torch.Tensor,
        *,
        chunk_size: int = 16384,
    ) -> torch.Tensor:
        return self.encode_selector(
            transform_rows(data, self.whitener),
            chunk_size=chunk_size,
        )

    @torch.no_grad()
    def decode(
        self,
        indices: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        decoder = ResidualQuantizerModel(
            self.decode_codebooks,
            codebook_bits=self.codebook_bits,
        )
        return decoder.decode(indices, dtype=dtype)

    @torch.no_grad()
    def decode_selector(self, indices: torch.Tensor) -> torch.Tensor:
        selector = ResidualQuantizerModel(
            self.selector_codebooks,
            codebook_bits=self.codebook_bits,
        )
        return selector.decode(indices)


@torch.no_grad()
def fit_fw_residual_quantizer(
    data: torch.Tensor,
    fisher: torch.Tensor,
    *,
    stages: int = 32,
    clusters: int = 256,
    weights: torch.Tensor | None = None,
    max_iterations: int = 100,
    chunk_size: int = 16384,
    seed: int = 0,
    tolerance: float = 0.0,
    initialization: str = "sample",
    decode_dtype: torch.dtype = torch.float16,
) -> tuple[FWResidualQuantizerModel, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Fit GSKM/RQ in fixed Fisher-whitened coordinates."""

    transform = build_fixed_whitener(fisher.to(device=data.device))
    transformed = transform_rows(data, transform.whitener)
    trained, training_indices, history = fit_residual_quantizer(
        transformed,
        stages=stages,
        clusters=clusters,
        weights=weights,
        max_iterations=max_iterations,
        chunk_size=chunk_size,
        seed=seed,
        tolerance=tolerance,
        initialization=initialization,
    )
    decode, deployed_selector, roundtrip = roundtrip_selector_codebooks(
        trained.codebooks.to(data.device),
        transform.whitener,
        decode_dtype=decode_dtype,
    )
    deployed_quantizer = ResidualQuantizerModel(
        deployed_selector,
        codebook_bits=int(round(math.log2(clusters))),
    )
    deployed_indices = deployed_quantizer.encode(
        transformed,
        chunk_size=chunk_size,
    ).cpu()
    model = FWResidualQuantizerModel(
        selector_codebooks=deployed_selector.cpu(),
        decode_codebooks=decode.cpu(),
        whitener=transform.whitener.cpu(),
        metric=transform.metric.cpu(),
        codebook_bits=int(round(math.log2(clusters))),
    )
    diagnostics: dict[str, Any] = {
        "whitener": transform.diagnostics,
        "roundtrip": roundtrip,
        "training_to_deployment_index_change_fraction": float(
            (training_indices.cpu() != deployed_indices).float().mean()
        ),
        "cache_bits_per_activation": model.bits_per_activation,
        "serialized_static_bytes": model.serialized_static_bytes,
        "derived_selector_bytes": model.derived_selector_bytes,
    }
    return model, deployed_indices, history, diagnostics


@torch.no_grad()
def fit_scfw_residual_quantizer(
    data: torch.Tensor,
    fisher: torch.Tensor,
    *,
    stages: int = 32,
    clusters: int = 256,
    weights: torch.Tensor | None = None,
    max_iterations: int = 100,
    chunk_size: int = 16384,
    seed: int = 0,
    tolerance: float = 0.0,
    initialization: str = "sample",
    decode_dtype: torch.dtype = torch.float16,
    spectral_power: float = 0.5,
    fisher_mixture: float = 0.5,
) -> tuple[FWResidualQuantizerModel, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Fit SCF-GSRQ in a frozen spectral-Fisher geometry."""

    transform = build_spectral_fisher_whitener(
        fisher.to(device=data.device),
        spectral_power=spectral_power,
        fisher_mixture=fisher_mixture,
    )
    transformed = transform_rows(data, transform.whitener)
    trained, training_indices, history = fit_residual_quantizer(
        transformed,
        stages=stages,
        clusters=clusters,
        weights=weights,
        max_iterations=max_iterations,
        chunk_size=chunk_size,
        seed=seed,
        tolerance=tolerance,
        initialization=initialization,
    )
    decode, deployed_selector, roundtrip = roundtrip_selector_codebooks(
        trained.codebooks.to(data.device),
        transform.whitener,
        decode_dtype=decode_dtype,
    )
    codebook_bits = int(round(math.log2(clusters)))
    deployed_quantizer = ResidualQuantizerModel(
        deployed_selector,
        codebook_bits=codebook_bits,
    )
    deployed_indices = deployed_quantizer.encode(
        transformed,
        chunk_size=chunk_size,
    ).cpu()
    model = FWResidualQuantizerModel(
        selector_codebooks=deployed_selector.cpu(),
        decode_codebooks=decode.cpu(),
        whitener=transform.whitener.cpu(),
        metric=transform.metric.cpu(),
        codebook_bits=codebook_bits,
    )
    diagnostics: dict[str, Any] = {
        "whitener": transform.diagnostics,
        "roundtrip": roundtrip,
        "training_to_deployment_index_change_fraction": float(
            (training_indices.cpu() != deployed_indices).float().mean()
        ),
        "cache_bits_per_activation": model.bits_per_activation,
        "serialized_static_bytes": model.serialized_static_bytes,
        "derived_selector_bytes": model.derived_selector_bytes,
    }
    return model, deployed_indices, history, diagnostics


@torch.no_grad()
def fold_whitener_into_projection(
    projection_weight: torch.Tensor,
    whiteners: torch.Tensor,
) -> torch.Tensor:
    """Return selector projection weights `blockdiag(L_h) W_K`.

    `projection_weight` may be `[H*D,I]` or `[H,D,I]`; the returned tensor has
    the same shape.  Bias-free Llama/Mistral K projections need no additional
    state.  A biased projection can apply the same transform to its reshaped
    bias separately.
    """

    if whiteners.ndim != 3 or whiteners.shape[1] != whiteners.shape[2]:
        raise ValueError("whiteners must have shape [H,D,D]")
    heads, dimension, _ = map(int, whiteners.shape)
    original_shape = projection_weight.shape
    if projection_weight.ndim == 2:
        if int(projection_weight.shape[0]) != heads * dimension:
            raise ValueError("flat projection output dimension differs")
        reshaped = projection_weight.reshape(
            heads, dimension, projection_weight.shape[1]
        )
    elif projection_weight.ndim == 3:
        if tuple(projection_weight.shape[:2]) != (heads, dimension):
            raise ValueError("projection head geometry differs")
        reshaped = projection_weight
    else:
        raise ValueError("projection_weight must be rank 2 or 3")
    folded = torch.bmm(
        whiteners.to(device=reshaped.device, dtype=torch.float32),
        reshaped.float(),
    )
    return folded.reshape(original_shape)
