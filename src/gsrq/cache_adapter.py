"""Runtime decoding helpers for GSRQ codebooks."""

from __future__ import annotations

import torch

from .gskm import assign_codewords


@torch.no_grad()
def _quantize_all_heads_batched(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_stages: int,
    return_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Batched equivalent of independent per-head RQ encoding."""

    batch, heads, tokens, dimension = states.shape
    residual = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
        .float()
        .clone()
    )
    decoded = torch.zeros_like(residual)
    tables = codebooks[:, :prefix_stages].to(
        device=states.device, dtype=torch.float32
    )
    saved: list[torch.Tensor] = []
    for stage in range(prefix_stages):
        table = tables[:, stage]
        distances = (
            residual.square().sum(dim=-1, keepdim=True)
            + table.square().sum(dim=-1)[:, None, :]
            - 2.0 * torch.bmm(residual, table.transpose(1, 2))
        )
        indices = torch.argmin(distances, dim=-1)
        codeword = torch.gather(
            table,
            1,
            indices[..., None].expand(-1, -1, dimension),
        )
        decoded += codeword
        residual -= codeword
        if return_indices:
            saved.append(indices)
    reconstructed = (
        decoded.reshape(heads, batch, tokens, dimension)
        .permute(1, 0, 2, 3)
        .to(dtype=states.dtype)
    )
    if not return_indices:
        return reconstructed, None
    indices_out = (
        torch.stack(saved, dim=-1).permute(1, 0, 2)
        if saved
        else torch.empty(
            batch * tokens,
            heads,
            0,
            device=states.device,
            dtype=torch.long,
        )
    )
    return reconstructed, indices_out


@torch.no_grad()
def quantize_kv_heads(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    prefix_stages: int | None = None,
    head_indices: list[int] | None = None,
    chunk_size: int = 16384,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize selected heads with per-head residual codebooks.

    `states` is `[batch, kv_heads, tokens, head_dim]`. `codebooks` is
    `[coded_heads, stages, K, head_dim]`. When `head_indices` is provided,
    codebook row j applies to `head_indices[j]`.
    """

    if states.ndim != 4:
        raise ValueError("states must be [B,H,T,D]")
    if codebooks.ndim != 4:
        raise ValueError("codebooks must be [H,R,K,D]")
    if states.shape[-1] != codebooks.shape[-1]:
        raise ValueError("head dimensions differ")
    heads = (
        list(range(states.shape[1]))
        if head_indices is None
        else list(head_indices)
    )
    if len(heads) != codebooks.shape[0]:
        raise ValueError("head_indices and codebook head count differ")
    if len(set(heads)) != len(heads):
        raise ValueError("head_indices must be unique")
    if any(head < 0 or head >= states.shape[1] for head in heads):
        raise ValueError("head index is out of range")
    prefix = (
        codebooks.shape[1]
        if prefix_stages is None
        else prefix_stages
    )
    if not 0 <= prefix <= codebooks.shape[1]:
        raise ValueError("prefix_stages is out of range")
    if head_indices is None and codebooks.shape[0] == states.shape[1]:
        return _quantize_all_heads_batched(
            states,
            codebooks,
            prefix_stages=prefix,
            return_indices=return_indices,
        )

    reconstructed = states.clone()
    saved_indices: list[torch.Tensor] = []
    for codebook_row, head in enumerate(heads):
        original = states[:, head].reshape(-1, states.shape[-1]).float()
        residual = original.clone()
        decoded = torch.zeros_like(residual)
        per_stage: list[torch.Tensor] = []
        for stage in range(prefix):
            table = codebooks[codebook_row, stage].to(
                device=states.device, dtype=torch.float32
            )
            indices = assign_codewords(
                residual, table, chunk_size=chunk_size
            )
            codeword = table.index_select(0, indices)
            decoded += codeword
            residual -= codeword
            if return_indices:
                per_stage.append(indices)
        reconstructed[:, head] = decoded.reshape_as(
            states[:, head]
        ).to(dtype=states.dtype)
        if return_indices:
            saved_indices.append(
                torch.stack(per_stage, dim=-1)
                if per_stage
                else torch.empty(
                    original.shape[0],
                    0,
                    device=states.device,
                    dtype=torch.long,
                )
            )

    if not return_indices:
        return reconstructed, None
    return reconstructed, torch.stack(saved_indices, dim=1)


def index_bits_per_activation(
    *,
    prefix_stages: int,
    codebook_size: int,
    dimension: int,
) -> float:
    if codebook_size < 1 or codebook_size & (codebook_size - 1):
        raise ValueError("codebook_size must be a power of two")
    if prefix_stages < 0 or dimension < 1:
        raise ValueError("invalid rate arguments")
    bits_per_index = codebook_size.bit_length() - 1
    return prefix_stages * bits_per_index / dimension
