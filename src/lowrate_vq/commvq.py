"""Paper-faithful CommVQ primitives adapted to arbitrary LLaMA KV width."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lowrate_vq.codecs import RateLedger, squared_distances


def commvq_rate(
    *, full_dimension: int, group_pairs: int, levels: int, stages: int
) -> RateLedger:
    if levels < 2 or levels & (levels - 1):
        raise ValueError("levels must be a power of two")
    bits_per_group_stage = 2 * (levels.bit_length() - 1)
    group_width = 2 * group_pairs
    if full_dimension % group_width:
        raise ValueError("CommVQ group width must divide the full KV dimension")
    groups = full_dimension // group_width
    index_bits = groups * stages * bits_per_group_stage
    index_bpa = index_bits / full_dimension
    metadata_bpa = 16.0 / full_dimension
    return RateLedger(
        index_bpa=index_bpa,
        metadata_bpa=metadata_bpa,
        total_bpa=index_bpa + metadata_bpa,
        metadata={
            "full_dimension": full_dimension,
            "groups": groups,
            "group_pairs": group_pairs,
            "levels": levels,
            "stages": stages,
            "index_bits_per_vector": index_bits,
            "norm_bits_per_vector": 16,
        },
    )


def commutative_design(levels: int, group_pairs: int, device: torch.device) -> torch.Tensor:
    """Return the official T matrix mapping 2*N parameters to N^2 centers."""

    rows = 2 * levels * levels
    columns = 2 * levels
    design = torch.zeros(rows, columns, device=device)
    for first in range(levels):
        for second in range(levels):
            center = first * levels + second
            design[2 * center, 2 * first] = 1
            design[2 * center, 2 * second + 1] = -1
            design[2 * center + 1, 2 * second] = 1
            design[2 * center + 1, 2 * first + 1] = 1
    return design


def theta_to_centers(theta: torch.Tensor) -> torch.Tensor:
    """Decode official [2*N,g] parameters to [N^2,2g] centers."""

    if theta.ndim != 2 or theta.shape[0] % 2:
        raise ValueError("theta must have shape [2*N,g]")
    levels = theta.shape[0] // 2
    pairs = theta.shape[1]
    base = theta.reshape(levels, 2, pairs)
    first = torch.arange(levels, device=theta.device)[:, None]
    second = torch.arange(levels, device=theta.device)[None, :]
    real = base[first, 0] - base[second, 1]
    imaginary = base[second, 0] + base[first, 1]
    return torch.stack((real, imaginary), dim=-1).reshape(
        levels * levels, 2 * pairs
    )


def _m_step(
    samples: torch.Tensor,
    assignments: torch.Tensor,
    design: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Constrained weighted least-squares M-step with a stable gauge solve.

    The upstream implementation explicitly inverts the normal matrix.  At the
    adapted 0.75-BPA setting there are as many centers as fitting vectors, so
    annealing can make some soft counts tiny and the float32 normal matrix
    numerically singular.  A ridge relative to the mean diagonal preserves the
    same constrained objective while remaining representable at its scale.
    """

    weights = assignments.float()
    counts = weights.sum(dim=0).clamp_min(1e-8)
    means = (weights.transpose(0, 1) @ samples.float()) / counts[:, None]
    repeated_counts = counts[:, None].expand(-1, 2).reshape(-1)
    pair_means = means.reshape(means.shape[0], -1, 2).permute(0, 2, 1).reshape(
        -1, means.shape[1] // 2
    )
    weighted_design_t = design.transpose(0, 1) * repeated_counts.unsqueeze(0)
    normal = weighted_design_t @ design
    rhs = weighted_design_t @ pair_means
    eye = torch.eye(normal.shape[0], device=normal.device, dtype=normal.dtype)
    diagonal_scale = normal.diagonal().abs().mean().clamp_min(1.0)
    relative_ridge = float(ridge)
    solver = "cholesky"
    theta = None
    for _ in range(7):
        regularized = normal + (relative_ridge * diagonal_scale) * eye
        factor, info = torch.linalg.cholesky_ex(regularized)
        if int(info.max()) == 0:
            theta = torch.cholesky_solve(rhs, factor)
            break
        relative_ridge *= 10.0
    if theta is None:
        # This should be rare and is deliberately explicit in the artifact.
        solver = "cpu_float64_pinv"
        regularized_cpu = (
            normal.double().cpu()
            + (relative_ridge * float(diagonal_scale))
            * torch.eye(normal.shape[0], dtype=torch.float64)
        )
        theta = (torch.linalg.pinv(regularized_cpu) @ rhs.double().cpu()).to(
            device=normal.device, dtype=normal.dtype
        )
    diagnostics = {
        "solver": solver,
        "relative_ridge": relative_ridge,
        "diagonal_scale": float(diagonal_scale),
        "finite": bool(torch.isfinite(theta).all()),
    }
    if not diagnostics["finite"]:
        raise FloatingPointError("CommVQ constrained M-step produced non-finite theta")
    return theta, theta_to_centers(theta), diagnostics


@torch.no_grad()
def fit_commutative_stage(
    samples: torch.Tensor,
    *,
    levels: int,
    iterations: int,
    initial_temperature: float,
    final_temperature: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[dict]]:
    if samples.ndim != 2 or samples.shape[1] % 2:
        raise ValueError("samples must be [N,2g]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    pairs = samples.shape[1] // 2
    design = commutative_design(levels, pairs, samples.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randint(
        0, samples.shape[0], (levels * levels,), generator=generator
    ).to(samples.device)
    initial_centers = samples[selected]
    initial_weights = F.softmax(-torch.cdist(samples.float(), initial_centers.float()), dim=1)
    theta, centers, solve = _m_step(samples, initial_weights, design)
    solve_diagnostics = [solve]
    history = []
    for iteration in range(iterations):
        fraction = iteration / max(iterations - 1, 1)
        temperature = initial_temperature * (
            final_temperature / initial_temperature
        ) ** fraction
        distances = torch.cdist(samples.float(), centers.float())
        weights = F.softmax(-distances / temperature, dim=1)
        hard = distances.argmin(dim=1)
        history.append(float((samples - centers[hard]).square().mean()))
        theta, centers, solve = _m_step(samples, weights, design)
        solve_diagnostics.append(solve)
    return theta, centers, history, solve_diagnostics


@torch.no_grad()
def fit_commvq_key_groups(
    normalized_groups: torch.Tensor,
    *,
    levels: int,
    stages: int,
    em_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Fit official residual EM independently for [N,G,2g] groups."""

    if normalized_groups.ndim != 3:
        raise ValueError("normalized_groups must have shape [N,G,2g]")
    _, groups, width = normalized_groups.shape
    all_theta = []
    group_rows = []
    for group in range(groups):
        residual = normalized_groups[:, group].float().clone()
        stage_theta = []
        stage_rows = []
        temperature_scale = 1.0
        for stage in range(stages):
            theta, centers, history, solve_diagnostics = fit_commutative_stage(
                residual,
                levels=levels,
                iterations=em_iterations,
                initial_temperature=1e-3 * temperature_scale,
                final_temperature=1e-5 * temperature_scale,
                seed=seed + group * 1000 + stage,
            )
            selected = torch.cdist(residual.float(), centers.float()).argmin(dim=1)
            residual -= centers[selected]
            stage_theta.append(theta.cpu())
            stage_rows.append(
                {
                    "stage": stage,
                    "em_history": history,
                    "residual_scalar_mse": float(residual.square().mean()),
                    "m_step_solvers": solve_diagnostics,
                }
            )
            if stage % 4 == 0:
                temperature_scale /= 10.0
        all_theta.append(torch.stack(stage_theta))
        group_rows.append(
            {
                "group": group,
                "stages": stage_rows,
                "final_scalar_mse": float(residual.square().mean()),
            }
        )
    thetas = torch.stack(all_theta)
    return thetas, {
        "groups": groups,
        "width": width,
        "levels": levels,
        "stages": stages,
        "em_iterations": em_iterations,
        "finite": bool(torch.isfinite(thetas).all()),
        "group_metrics": group_rows,
    }


@torch.no_grad()
def quantize_commvq_key(
    states: torch.Tensor,
    thetas: torch.Tensor,
    *,
    chunk_size: int = 4096,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Explicitly encode/decode [B,H,T,D] with compact CommVQ K tables."""

    batch, heads, tokens, head_dim = map(int, states.shape)
    full_dimension = heads * head_dim
    groups, stages, _, pairs = map(int, thetas.shape)
    group_width = 2 * pairs
    if groups * group_width != full_dimension:
        raise ValueError("CommVQ theta geometry differs from KV states")
    full = states.transpose(1, 2).reshape(batch * tokens, full_dimension).float()
    norms = torch.linalg.vector_norm(full, dim=-1, keepdim=True)
    normalized = full / norms.clamp_min(torch.finfo(torch.float32).tiny)
    grouped = normalized.reshape(batch * tokens, groups, group_width)
    decoded = torch.zeros_like(grouped)
    indices = (
        torch.empty(
            batch * tokens,
            groups,
            stages,
            dtype=torch.long,
            device=states.device,
        )
        if return_indices
        else None
    )
    for group in range(groups):
        residual = grouped[:, group].clone()
        for stage in range(stages):
            centers = theta_to_centers(thetas[group, stage].float()).to(states.device)
            for start in range(0, residual.shape[0], chunk_size):
                end = min(start + chunk_size, residual.shape[0])
                selected = squared_distances(residual[start:end], centers).argmin(dim=-1)
                chosen = centers[selected]
                decoded[start:end, group] += chosen
                residual[start:end] -= chosen
                if indices is not None:
                    indices[start:end, group, stage] = selected
    decoded = decoded.reshape(batch * tokens, full_dimension) * norms
    decoded = decoded.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    return decoded.to(dtype=states.dtype), indices


class CommVQValueCodec(nn.Module):
    """The official CommVQ ValueCompressor with deterministic inference."""

    def __init__(self, quant_bits: float, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.quant_bits = quant_bits
        self.code_bits = 2
        self.code_range = 4
        self.code_groups = int(feature_dim * quant_bits) // self.code_bits
        self.embedding_dim = self.code_groups * self.code_range
        self.embedding = nn.Parameter(
            torch.randn(self.code_groups * self.code_bits, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.intermediate_dim = (
            self.embedding_dim
            if self.embedding_dim == 6144
            else min(4096, self.embedding_dim)
        )
        self.encoder = nn.Linear(feature_dim, self.intermediate_dim)
        self.output = nn.Linear(self.intermediate_dim, self.embedding_dim)
        self.register_buffer("channel_scale", torch.ones(feature_dim))
        self.epsilon = 1e-6

    def encode_bits(self, x: torch.Tensor, *, stochastic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = x / (self.channel_scale + self.epsilon)
        prescale = torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
        normalized = normalized / (prescale + self.epsilon)
        logits = self.output(F.gelu(self.encoder(self.norm(normalized))))
        logits = logits.reshape(*logits.shape[:-1], self.code_groups, self.code_range)
        if stochastic:
            onehot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            selected = logits.argmax(dim=-1)
            onehot = F.one_hot(selected, num_classes=self.code_range).to(logits.dtype)
        onehot[..., :2] += onehot[..., 2:3]
        return onehot[..., :2].flatten(-2), prescale

    def decode_bits(self, bits: torch.Tensor, prescale: torch.Tensor) -> torch.Tensor:
        decoded = bits @ self.embedding
        postscale = torch.linalg.vector_norm(decoded, dim=-1, keepdim=True)
        decoded = decoded / (postscale + self.epsilon)
        return decoded * (prescale + self.epsilon) * (self.channel_scale + self.epsilon)

    def forward(self, x: torch.Tensor, *, stochastic: bool = True) -> torch.Tensor:
        return self.decode_bits(*self.encode_bits(x, stochastic=stochastic))


def quantize_commvq_value(states: torch.Tensor, codec: CommVQValueCodec) -> torch.Tensor:
    batch, heads, tokens, head_dim = map(int, states.shape)
    full = states.transpose(1, 2).reshape(batch, tokens, heads * head_dim)
    decoded = codec(full.float(), stochastic=False)
    return decoded.reshape(batch, tokens, heads, head_dim).transpose(1, 2).to(states.dtype)
