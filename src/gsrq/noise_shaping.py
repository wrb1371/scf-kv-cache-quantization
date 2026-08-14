"""Causal encoder-only noise shaping for frozen GSRQ codebooks.

The functions in this module only change index assignment.  Returned indices
have the same chronological layout and decode with the ordinary sum-of-RQ-
codewords path used by independent GSRQ.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class FeedbackEncoding:
    """Reconstruction, ordinary indices, routing, and state diagnostics."""

    reconstructed: torch.Tensor
    indices: torch.Tensor
    bucket_ids: torch.Tensor
    selected_state_norms: torch.Tensor
    max_bucket_state_norms: torch.Tensor
    selected_cumulative_error_norms: torch.Tensor
    page_end_state_norms: torch.Tensor
    state_cumulative_max_abs_difference: float


def fixed_orthogonal_simhash_directions(
    heads: int,
    dimension: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct two deterministic orthonormal hash directions per head."""

    if heads < 1:
        raise ValueError("heads must be positive")
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        heads,
        dimension,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
    directions = orthogonal.transpose(1, 2).contiguous().float()
    return directions.to(device=device)


def simhash_bucket_ids(
    post_rope_keys: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """Return two-bit SimHash buckets for `[B,H,T,D]` post-RoPE keys."""

    if post_rope_keys.ndim != 4:
        raise ValueError("post_rope_keys must be [B,H,T,D]")
    if directions.shape != (
        post_rope_keys.shape[1],
        2,
        post_rope_keys.shape[-1],
    ):
        raise ValueError("directions must be [H,2,D]")
    projections = torch.einsum(
        "bhtd,hkd->bhtk",
        post_rope_keys.float(),
        directions.to(
            device=post_rope_keys.device,
            dtype=torch.float32,
        ),
    )
    bits = projections >= 0
    return bits[..., 0].long() + 2 * bits[..., 1].long()


def random_bucket_ids(
    batch: int,
    heads: int,
    tokens: int,
    *,
    buckets: int,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a deterministic activation-independent routing control."""

    if min(batch, heads, tokens, buckets) < 1:
        raise ValueError("routing dimensions and buckets must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = torch.randint(
        buckets,
        (batch, heads, tokens),
        generator=generator,
        dtype=torch.long,
    )
    return output.to(device=device)


def chronological_bucket_ids(states: torch.Tensor) -> torch.Tensor:
    """Route every vector to the single chronological feedback stream."""

    if states.ndim != 4:
        raise ValueError("states must be [B,H,T,D]")
    return torch.zeros(
        states.shape[:3],
        device=states.device,
        dtype=torch.long,
    )


def decode_gsrq_indices(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    batch: int,
    tokens: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode ordinary `[B*T,H,R]` GSRQ indices."""

    if indices.ndim != 3:
        raise ValueError("indices must be [B*T,H,R]")
    if codebooks.ndim != 4:
        raise ValueError("codebooks must be [H,R,K,D]")
    heads, stages, _, dimension = codebooks.shape
    if indices.shape != (batch * tokens, heads, stages):
        raise ValueError("indices and codebooks have incompatible shapes")
    tables = codebooks.to(device=indices.device, dtype=torch.float32)
    decoded = torch.zeros(
        batch * tokens,
        heads,
        dimension,
        device=indices.device,
        dtype=torch.float32,
    )
    head_index = torch.arange(heads, device=indices.device)[None, :]
    for stage in range(stages):
        decoded += tables[
            head_index,
            stage,
            indices[:, :, stage].long(),
        ]
    return (
        decoded.reshape(batch, tokens, heads, dimension)
        .permute(0, 2, 1, 3)
        .contiguous()
        .to(dtype=dtype)
    )


@torch.no_grad()
def encode_gsrq_feedback(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    bucket_ids: torch.Tensor,
    *,
    buckets: int,
    page_size: int = 64,
    state_dtype: torch.dtype = torch.float16,
) -> FeedbackEncoding:
    """Causally encode `V + state` using frozen per-head RQ codebooks.

    Pages are evaluated in parallel because their states reset independently;
    positions inside a page remain strictly sequential.
    """

    if states.ndim != 4:
        raise ValueError("states must be [B,H,T,D]")
    if codebooks.ndim != 4:
        raise ValueError("codebooks must be [H,R,K,D]")
    batch, heads, tokens, dimension = states.shape
    if codebooks.shape[0] != heads or codebooks.shape[-1] != dimension:
        raise ValueError("states and codebooks have incompatible shapes")
    if bucket_ids.shape != (batch, heads, tokens):
        raise ValueError("bucket_ids must be [B,H,T]")
    if buckets < 1 or page_size < 1:
        raise ValueError("buckets and page_size must be positive")
    if bucket_ids.numel() and (
        int(bucket_ids.min()) < 0 or int(bucket_ids.max()) >= buckets
    ):
        raise ValueError("bucket id is out of range")
    if state_dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError("state_dtype must be a floating-point state type")

    stages, clusters = codebooks.shape[1:3]
    pages = math.ceil(tokens / page_size)
    padded_tokens = pages * page_size
    pad = padded_tokens - tokens
    padded_states = torch.nn.functional.pad(
        states, (0, 0, 0, pad)
    )
    padded_buckets = torch.nn.functional.pad(
        bucket_ids, (0, pad)
    )
    page_states = (
        padded_states.reshape(
            batch, heads, pages, page_size, dimension
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * pages, heads, page_size, dimension)
    )
    page_buckets = (
        padded_buckets.reshape(batch, heads, pages, page_size)
        .permute(0, 2, 1, 3)
        .reshape(batch * pages, heads, page_size)
    )
    valid = (
        torch.arange(padded_tokens, device=states.device)
        .reshape(pages, page_size)
        .expand(batch, -1, -1)
        .reshape(batch * pages, page_size)
        < tokens
    )

    tables = codebooks.to(device=states.device, dtype=torch.float32)
    table_norms = tables.square().sum(dim=-1)
    feedback = torch.zeros(
        batch * pages,
        heads,
        buckets,
        dimension,
        device=states.device,
        dtype=state_dtype,
    )
    cumulative = torch.zeros_like(feedback, dtype=torch.float32)
    reconstructed = torch.zeros_like(page_states)
    saved_indices = torch.zeros(
        batch * pages,
        heads,
        page_size,
        stages,
        device=states.device,
        dtype=torch.long,
    )
    selected_norms = torch.zeros(
        batch * pages,
        heads,
        page_size,
        device=states.device,
        dtype=torch.float32,
    )
    maximum_norms = torch.zeros_like(selected_norms)
    cumulative_norms = torch.zeros_like(selected_norms)
    max_state_cumulative_difference = 0.0
    head_index = torch.arange(heads, device=states.device)[None, :]

    for offset in range(page_size):
        route = page_buckets[:, :, offset]
        route_index = route[..., None, None].expand(
            -1, -1, 1, dimension
        )
        selected = torch.gather(
            feedback, 2, route_index
        ).squeeze(2)
        original = page_states[:, :, offset].float()
        residual = original + selected.float()
        decoded = torch.zeros_like(residual)
        per_stage = []
        for stage in range(stages):
            table = tables[:, stage]
            distances = (
                residual.square().sum(dim=-1, keepdim=True)
                + table_norms[:, stage][None, :, :]
                - 2.0
                * torch.einsum("bhd,hkd->bhk", residual, table)
            )
            stage_indices = torch.argmin(distances, dim=-1)
            codewords = table[head_index, stage_indices]
            decoded += codewords
            residual -= codewords
            per_stage.append(stage_indices)

        is_valid = valid[:, offset][:, None, None]
        updated = (selected.float() + original - decoded).to(
            dtype=state_dtype
        )
        previous_selected = torch.gather(
            feedback, 2, route_index
        )
        feedback.scatter_(
            2,
            route_index,
            torch.where(
                is_valid[..., None],
                updated[:, :, None, :],
                previous_selected,
            ),
        )

        cumulative_selected = torch.gather(
            cumulative, 2, route_index
        ).squeeze(2)
        cumulative_updated = cumulative_selected + original - decoded
        previous_cumulative = torch.gather(
            cumulative, 2, route_index
        )
        cumulative.scatter_(
            2,
            route_index,
            torch.where(
                is_valid[..., None],
                cumulative_updated[:, :, None, :],
                previous_cumulative,
            ),
        )

        reconstructed[:, :, offset] = torch.where(
            is_valid,
            decoded,
            reconstructed[:, :, offset].float(),
        ).to(dtype=states.dtype)
        saved_indices[:, :, offset] = torch.stack(
            per_stage, dim=-1
        )
        selected_after = torch.gather(
            feedback, 2, route_index
        ).squeeze(2).float()
        cumulative_after = torch.gather(
            cumulative, 2, route_index
        ).squeeze(2)
        selected_norms[:, :, offset] = torch.where(
            is_valid.squeeze(-1),
            torch.linalg.vector_norm(selected_after, dim=-1),
            torch.zeros_like(selected_norms[:, :, offset]),
        )
        maximum_norms[:, :, offset] = torch.where(
            is_valid.squeeze(-1),
            torch.linalg.vector_norm(
                feedback.float(), dim=-1
            ).amax(dim=-1),
            torch.zeros_like(maximum_norms[:, :, offset]),
        )
        cumulative_norms[:, :, offset] = torch.where(
            is_valid.squeeze(-1),
            torch.linalg.vector_norm(cumulative_after, dim=-1),
            torch.zeros_like(cumulative_norms[:, :, offset]),
        )
        if bool(is_valid.any()):
            difference = (
                selected_after - cumulative_after
            ).abs().masked_fill(
                ~is_valid.squeeze(-1)[..., None], 0.0
            )
            max_state_cumulative_difference = max(
                max_state_cumulative_difference,
                float(difference.max()),
            )

    def restore_tokens(tensor: torch.Tensor) -> torch.Tensor:
        trailing = tensor.shape[3:]
        restored = (
            tensor.reshape(batch, pages, heads, page_size, *trailing)
            .permute(0, 2, 1, 3, *range(4, 4 + len(trailing)))
            .reshape(batch, heads, padded_tokens, *trailing)
        )
        return restored[:, :, :tokens]

    reconstructed_out = restore_tokens(reconstructed)
    indices_out = (
        restore_tokens(saved_indices)
        .permute(0, 2, 1, 3)
        .reshape(batch * tokens, heads, stages)
        .contiguous()
    )
    buckets_out = restore_tokens(page_buckets[..., None]).squeeze(-1)
    return FeedbackEncoding(
        reconstructed=reconstructed_out,
        indices=indices_out,
        bucket_ids=buckets_out,
        selected_state_norms=restore_tokens(
            selected_norms[..., None]
        ).squeeze(-1),
        max_bucket_state_norms=restore_tokens(
            maximum_norms[..., None]
        ).squeeze(-1),
        selected_cumulative_error_norms=restore_tokens(
            cumulative_norms[..., None]
        ).squeeze(-1),
        page_end_state_norms=torch.linalg.vector_norm(
            feedback.float(), dim=-1
        ).reshape(batch, pages, heads, buckets),
        state_cumulative_max_abs_difference=(
            max_state_cumulative_difference
        ),
    )


def feedback_state_rate_ledger(
    *,
    tokens: int,
    heads: int,
    dimension: int,
    buckets: int,
    stages: int = 32,
    index_bits: int = 8,
    state_bits: int = 16,
) -> dict[str, Any]:
    """Return an exact integer-bit ledger for paying the active-page state."""

    if min(
        tokens,
        heads,
        dimension,
        buckets,
        stages,
        index_bits,
        state_bits,
    ) < 1:
        raise ValueError("all rate arguments must be positive")
    activation_elements = 2 * tokens * heads * dimension
    full_index_count = 2 * tokens * heads * stages
    full_index_bits = full_index_count * index_bits
    state_storage_bits = buckets * heads * dimension * state_bits
    omitted_indices = math.ceil(state_storage_bits / index_bits)
    if omitted_indices > full_index_count:
        raise ValueError("state cannot be paid from the available indices")
    rebated_index_bits = omitted_indices * index_bits
    effective_bits = (
        full_index_bits - rebated_index_bits + state_storage_bits
    )
    return {
        "tokens": tokens,
        "heads": heads,
        "dimension": dimension,
        "buckets": buckets,
        "stages": stages,
        "index_bits": index_bits,
        "state_bits": state_bits,
        "activation_elements_k_plus_v": activation_elements,
        "full_index_count": full_index_count,
        "full_index_bits": full_index_bits,
        "state_storage_bits": state_storage_bits,
        "state_overhead_bpa": state_storage_bits / activation_elements,
        "omitted_indices_required": omitted_indices,
        "rebated_index_bits": rebated_index_bits,
        "rebated_index_bpa": (
            full_index_bits - rebated_index_bits
        )
        / activation_elements,
        "effective_bpa": effective_bits / activation_elements,
        "rate_gate_pass": (
            effective_bits / activation_elements <= 2.0
            and rebated_index_bits >= state_storage_bits
        ),
    }

