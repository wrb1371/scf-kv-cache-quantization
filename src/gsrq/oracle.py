"""Exact replay and budget helpers for analysis-only RQ stage oracles."""

from __future__ import annotations

import math
from typing import Iterable

import torch


@torch.no_grad()
def decode_batched_head_prefixes(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    batch: int,
    tokens: int,
    prefixes: Iterable[int],
    dtype: torch.dtype = torch.float32,
) -> dict[int, torch.Tensor]:
    """Decode stored full-RQ indices at multiple prefix depths.

    `indices` follows the cache adapter layout `[B*T,H,R]`; `codebooks` is
    `[H,R,K,D]`.  Returned tensors are `[B,H,T,D]`.
    """

    if indices.ndim != 3:
        raise ValueError("indices must be [B*T,H,R]")
    if codebooks.ndim != 4:
        raise ValueError("codebooks must be [H,R,K,D]")
    if indices.shape[0] != batch * tokens:
        raise ValueError("indices do not match batch*tokens")
    heads, stages, clusters, dimension = codebooks.shape
    if indices.shape[1:] != (heads, stages):
        raise ValueError("index/codebook head or stage dimensions differ")
    requested = sorted(set(int(prefix) for prefix in prefixes))
    if not requested or requested[0] < 0 or requested[-1] > stages:
        raise ValueError("prefix is out of range")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= clusters
    ):
        raise ValueError("index is out of codebook range")

    per_head = indices.permute(1, 0, 2).long()
    tables = codebooks.to(device=indices.device, dtype=torch.float32)
    decoded = torch.zeros(
        heads,
        batch * tokens,
        dimension,
        device=indices.device,
        dtype=torch.float32,
    )
    outputs: dict[int, torch.Tensor] = {}
    if 0 in requested:
        outputs[0] = (
            decoded.reshape(heads, batch, tokens, dimension)
            .permute(1, 0, 2, 3)
            .to(dtype=dtype)
            .clone()
        )
    for stage in range(stages):
        table = tables[:, stage]
        stage_indices = per_head[:, :, stage]
        decoded += torch.gather(
            table,
            1,
            stage_indices[..., None].expand(-1, -1, dimension),
        )
        prefix = stage + 1
        if prefix in requested:
            outputs[prefix] = (
                decoded.reshape(heads, batch, tokens, dimension)
                .permute(1, 0, 2, 3)
                .to(dtype=dtype)
                .clone()
            )
    return outputs


def logical_read_bpa(
    key_stages: float,
    value_stages: float,
    *,
    index_bits: int = 8,
    dimension: int = 128,
) -> float:
    """Logical read bits per scalar averaged over K and V."""

    if key_stages < 0 or value_stages < 0:
        raise ValueError("stage counts must be nonnegative")
    if index_bits < 1 or dimension < 1:
        raise ValueError("index_bits and dimension must be positive")
    return index_bits * (key_stages + value_stages) / (2 * dimension)


def multiple_choice_budget_oracle(
    losses: torch.Tensor,
    cost_units: torch.Tensor,
    *,
    max_total_units: int,
) -> dict[str, object]:
    """Solve an exact multiple-choice budget allocation by dynamic programming.

    One candidate is selected independently for every row of `losses`, while
    the integer costs sum to at most `max_total_units`.
    """

    if losses.ndim != 2:
        raise ValueError("losses must be [items,candidates]")
    if cost_units.ndim != 1 or cost_units.shape[0] != losses.shape[1]:
        raise ValueError("cost_units must match candidate count")
    if max_total_units < 0:
        raise ValueError("max_total_units must be nonnegative")
    if not bool(torch.isfinite(losses).all()):
        raise ValueError("losses must be finite")
    costs = cost_units.to(dtype=torch.long, device="cpu")
    if bool((costs < 0).any()):
        raise ValueError("cost units must be nonnegative")
    values = losses.detach().to(dtype=torch.float64, device="cpu")

    infinity = torch.tensor(float("inf"), dtype=torch.float64)
    dp = torch.full((max_total_units + 1,), infinity)
    dp[0] = 0.0
    parents: list[torch.Tensor] = []
    for item in range(values.shape[0]):
        updated = torch.full_like(dp, infinity)
        chosen = torch.full(
            (max_total_units + 1,), -1, dtype=torch.long
        )
        for candidate, cost_tensor in enumerate(costs):
            cost = int(cost_tensor)
            if cost > max_total_units:
                continue
            proposal = dp[: max_total_units + 1 - cost] + values[
                item, candidate
            ]
            target = updated[cost:]
            better = proposal < target
            target[better] = proposal[better]
            chosen_target = chosen[cost:]
            chosen_target[better] = candidate
        dp = updated
        parents.append(chosen)

    best_budget = int(torch.argmin(dp))
    if not math.isfinite(float(dp[best_budget])):
        raise ValueError("no feasible allocation")
    selected = [0] * values.shape[0]
    budget = best_budget
    for item in range(values.shape[0] - 1, -1, -1):
        candidate = int(parents[item][budget])
        if candidate < 0:
            raise RuntimeError("oracle backtracking failed")
        selected[item] = candidate
        budget -= int(costs[candidate])
    if budget != 0:
        raise RuntimeError("oracle backtracking left an invalid budget")
    return {
        "selected_candidates": selected,
        "total_loss": float(dp[best_budget]),
        "total_cost_units": best_budget,
    }
