"""Additive multi-codebook quantization for frozen KV-cache vectors.

This is a source-informed transfer of AQLM's representation and joint encoder,
not a reproduction of AQLM's weight checkpoints or blockwise weight objective.
Each contiguous KV block is represented as a sum of frozen codewords.  The
dynamic cache stores only the fixed-width code indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from lowrate_vq.codecs import RateLedger
from lowrate_vq.cq import fit_batched_weighted_kmeans


@dataclass(frozen=True)
class AdditiveCodebookFit:
    codebooks: torch.Tensor
    initial_indices: torch.Tensor | None
    assignments: torch.Tensor
    history: list[dict[str, float | int]]
    diagnostics: dict[str, Any]


def _power_of_two(value: int) -> bool:
    return value >= 2 and value & (value - 1) == 0


def aqlm_kv_rate(
    *,
    dimension: int,
    group_size: int,
    num_codebooks: int,
    entries_per_codebook: int,
) -> RateLedger:
    if dimension < 1 or group_size < 1 or dimension % group_size:
        raise ValueError("group_size must divide dimension")
    if num_codebooks < 1:
        raise ValueError("num_codebooks must be positive")
    if not _power_of_two(entries_per_codebook):
        raise ValueError("entries_per_codebook must be a power of two")
    groups = dimension // group_size
    bits_per_index = entries_per_codebook.bit_length() - 1
    index_bits = groups * num_codebooks * bits_per_index
    index_bpa = index_bits / dimension
    return RateLedger(
        index_bpa=index_bpa,
        metadata_bpa=0.0,
        total_bpa=index_bpa,
        metadata={
            "dimension": dimension,
            "group_size": group_size,
            "groups": groups,
            "num_codebooks": num_codebooks,
            "entries_per_codebook": entries_per_codebook,
            "bits_per_index": bits_per_index,
            "index_bits_per_vector": index_bits,
            "per_token_metadata_bits": 0,
        },
    )


def additive_index_tuples(
    num_codebooks: int,
    entries_per_codebook: int,
    *,
    device: torch.device,
    max_candidates: int = 65536,
) -> torch.Tensor:
    if num_codebooks < 1 or entries_per_codebook < 2:
        raise ValueError("invalid additive-codebook geometry")
    candidates = entries_per_codebook**num_codebooks
    if candidates > max_candidates:
        raise ValueError(
            f"exact joint encoder needs {candidates} candidates, above "
            f"the frozen limit {max_candidates}"
        )
    flat = torch.arange(candidates, device=device, dtype=torch.long)
    columns = []
    divisor = 1
    for _ in range(num_codebooks):
        columns.append((flat // divisor).remainder(entries_per_codebook))
        divisor *= entries_per_codebook
    return torch.stack(columns, dim=-1)


def additive_sum_table(
    codebooks: torch.Tensor,
    *,
    max_candidates: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all joint sums for codebooks shaped ``[P,M,K,S]``."""

    if codebooks.ndim != 4 or not codebooks.is_floating_point():
        raise ValueError("codebooks must have shape [P,M,K,S]")
    if not bool(torch.isfinite(codebooks).all()):
        raise ValueError("codebooks must be finite")
    problems, num_codebooks, entries, subdim = map(int, codebooks.shape)
    tuples = additive_index_tuples(
        num_codebooks,
        entries,
        device=codebooks.device,
        max_candidates=max_candidates,
    )
    table = torch.zeros(
        problems,
        len(tuples),
        subdim,
        device=codebooks.device,
        dtype=torch.float32,
    )
    work = codebooks.float()
    for codebook in range(num_codebooks):
        table += work[:, codebook, tuples[:, codebook]]
    return table, tuples


@torch.no_grad()
def assign_additive_codes(
    vectors: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    selector_whiteners: torch.Tensor | None = None,
    chunk_size: int = 1024,
    max_candidates: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly assign ``[P,N,S]`` vectors to additive codeword sums.

    When ``selector_whiteners`` is provided, the exact joint search minimizes
    ``||L(x-c)||^2`` for one frozen ``L`` per problem.  Returned reconstructions
    remain in raw coordinates, so cache decode and attention do not change.
    """

    if vectors.ndim != 3 or codebooks.ndim != 4:
        raise ValueError("vectors/codebooks must be [P,N,S]/[P,M,K,S]")
    problems, rows, subdim = map(int, vectors.shape)
    if int(codebooks.shape[0]) != problems or int(codebooks.shape[-1]) != subdim:
        raise ValueError("vector and codebook geometry differs")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not vectors.is_floating_point() or not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite floating point")
    if selector_whiteners is not None:
        if tuple(selector_whiteners.shape) != (problems, subdim, subdim):
            raise ValueError("selector whiteners have the wrong geometry")
        if not selector_whiteners.is_floating_point() or not bool(
            torch.isfinite(selector_whiteners).all()
        ):
            raise ValueError("selector whiteners must be finite floating point")
    table, tuples = additive_sum_table(
        codebooks, max_candidates=max_candidates
    )
    if selector_whiteners is None:
        selector_table = table
        selector_vectors = vectors.float()
    else:
        factor = selector_whiteners.to(vectors.device, dtype=torch.float32)
        selector_table = torch.matmul(table, factor.transpose(-1, -2))
        selector_vectors = torch.matmul(vectors.float(), factor.transpose(-1, -2))
    table_norms = selector_table.square().sum(-1).unsqueeze(1)
    joint = torch.empty(problems, rows, dtype=torch.long, device=vectors.device)
    decoded = torch.empty(problems, rows, subdim, device=vectors.device)
    errors = torch.empty(problems, rows, device=vectors.device)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        current = selector_vectors[:, start:stop]
        distances = (
            current.square().sum(-1, keepdim=True)
            + table_norms
            - 2.0 * torch.bmm(current, selector_table.transpose(1, 2))
        ).clamp_min_(0.0)
        error, selected = distances.min(-1)
        joint[:, start:stop] = selected
        errors[:, start:stop] = error
        decoded[:, start:stop] = torch.gather(
            table,
            1,
            selected.unsqueeze(-1).expand(-1, -1, subdim),
        )
    assignments = tuples[joint]
    return assignments, decoded, errors


@torch.no_grad()
def decode_aqlm_indices(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    """Decode ``[B,H,T,G,M]`` indices using ``[H,G,M,K,S]`` tables."""

    if indices.ndim != 5 or codebooks.ndim != 5:
        raise ValueError("indices/codebooks must be [B,H,T,G,M]/[H,G,M,K,S]")
    batch, heads, tokens, groups, num_codebooks = map(int, indices.shape)
    cb_heads, cb_groups, cb_count, entries, subdim = map(int, codebooks.shape)
    if (heads, groups, num_codebooks) != (cb_heads, cb_groups, cb_count):
        raise ValueError("index and codebook geometry differs")
    if int(indices.min()) < 0 or int(indices.max()) >= entries:
        raise ValueError("additive index is out of range")
    tables = codebooks.to(indices.device, dtype=torch.float32)
    output = torch.zeros(
        batch, heads, tokens, groups, subdim,
        device=indices.device,
        dtype=torch.float32,
    )
    head_indices = torch.arange(heads, device=indices.device)[None, :, None, None]
    group_indices = torch.arange(groups, device=indices.device)[None, None, None, :]
    for codebook in range(num_codebooks):
        current = indices[..., codebook]
        # Advanced indexing broadcasts the static head/group axes without
        # materializing a [B,H,T,G,K,S] expanded table.
        output += tables[
            head_indices,
            group_indices,
            codebook,
            current,
        ]
    return output.reshape(batch, heads, tokens, groups * subdim)


@torch.no_grad()
def quantize_aqlm_kv(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    selector_whiteners: torch.Tensor | None = None,
    chunk_size: int = 1024,
    problem_batch_size: int = 16,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Roundtrip ``[B,H,T,D]`` with a joint non-greedy additive encoder."""

    if states.ndim != 4 or codebooks.ndim != 5:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,G,M,K,S]")
    batch, heads, tokens, dimension = map(int, states.shape)
    cb_heads, groups, num_codebooks, _entries, subdim = map(int, codebooks.shape)
    if cb_heads != heads or groups * subdim != dimension:
        raise ValueError("AQLM-KV codebooks do not cover the state geometry")
    if selector_whiteners is not None:
        expected = (heads, groups, subdim, subdim)
        if tuple(selector_whiteners.shape) != expected:
            raise ValueError("AQLM-KV selector whiteners have the wrong geometry")
        if not selector_whiteners.is_floating_point() or not bool(
            torch.isfinite(selector_whiteners).all()
        ):
            raise ValueError("AQLM-KV selector whiteners must be finite")
    if problem_batch_size < 1:
        raise ValueError("problem_batch_size must be positive")
    rows = batch * tokens
    problems = heads * groups
    vectors = (
        states.permute(1, 0, 2, 3)
        .reshape(heads, rows, groups, subdim)
        .permute(0, 2, 1, 3)
        .reshape(problems, rows, subdim)
        .float()
    )
    tables = codebooks.to(states.device, dtype=torch.float32).reshape(
        problems, num_codebooks, -1, subdim
    )
    flat_whiteners = (
        selector_whiteners.to(states.device, dtype=torch.float32).reshape(
            problems, subdim, subdim
        )
        if selector_whiteners is not None
        else None
    )
    decoded = torch.empty_like(vectors)
    assignments = torch.empty(
        problems, rows, num_codebooks, dtype=torch.long, device=states.device
    )
    for start in range(0, problems, problem_batch_size):
        stop = min(start + problem_batch_size, problems)
        codes, reconstructed, _ = assign_additive_codes(
            vectors[start:stop],
            tables[start:stop],
            selector_whiteners=(
                flat_whiteners[start:stop] if flat_whiteners is not None else None
            ),
            chunk_size=chunk_size,
        )
        assignments[start:stop] = codes
        decoded[start:stop] = reconstructed
    output = (
        decoded.reshape(heads, groups, batch, tokens, subdim)
        .permute(2, 0, 3, 1, 4)
        .reshape(batch, heads, tokens, dimension)
        .to(states.dtype)
    )
    indices = None
    if return_indices:
        indices = (
            assignments.reshape(heads, groups, batch, tokens, num_codebooks)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
    return output, indices


@torch.no_grad()
def _fit_batch(
    vectors: torch.Tensor,
    *,
    num_codebooks: int,
    entries_per_codebook: int,
    iterations: int,
    kmeans_iterations: int,
    chunk_size: int,
    seed: int,
    selector_whiteners: torch.Tensor | None = None,
    initial_indices: torch.Tensor | None = None,
    initial_codebooks: torch.Tensor | None = None,
) -> AdditiveCodebookFit:
    problems, rows, subdim = map(int, vectors.shape)
    if selector_whiteners is not None:
        if tuple(selector_whiteners.shape) != (problems, subdim, subdim):
            raise ValueError("selector whiteners have the wrong geometry")
        if not selector_whiteners.is_floating_point() or not bool(
            torch.isfinite(selector_whiteners).all()
        ):
            raise ValueError("selector whiteners must be finite")
    if initial_indices is not None:
        expected = (problems, num_codebooks, entries_per_codebook)
        if tuple(initial_indices.shape) != expected:
            raise ValueError("initial additive indices have the wrong geometry")
        if int(initial_indices.min()) < 0 or int(initial_indices.max()) >= rows:
            raise ValueError("initial additive indices are out of range")
    if initial_codebooks is not None:
        expected = (problems, num_codebooks, entries_per_codebook, subdim)
        if tuple(initial_codebooks.shape) != expected:
            raise ValueError("initial additive codebooks have the wrong geometry")
        if not initial_codebooks.is_floating_point() or not bool(
            torch.isfinite(initial_codebooks).all()
        ):
            raise ValueError("initial additive codebooks must be finite")
        if initial_indices is not None:
            raise ValueError("initial indices and codebooks are mutually exclusive")
    weights = torch.ones(problems, rows, device=vectors.device)
    selected_initial = None
    if initial_codebooks is not None:
        codebooks = initial_codebooks.to(
            vectors.device, dtype=torch.float32
        ).clone()
    else:
        residual = vectors.float().clone()
        initial_tables = []
        selected_initial_list = []
        for codebook in range(num_codebooks):
            fit = fit_batched_weighted_kmeans(
                residual,
                weights,
                clusters=entries_per_codebook,
                iterations=kmeans_iterations,
                chunk_size=chunk_size,
                seed=seed + 1009 * codebook,
                whiteners=selector_whiteners,
                initial_indices=(
                    initial_indices[:, codebook].to(vectors.device)
                    if initial_indices is not None
                    else None
                ),
            )
            centers = fit.codebooks.float()
            selected = fit.assignments.to(vectors.device)
            initial_tables.append(centers)
            selected_initial_list.append(fit.initial_indices.to(vectors.device))
            residual -= torch.gather(
                centers,
                1,
                selected.unsqueeze(-1).expand(-1, -1, subdim),
            )
        codebooks = torch.stack(initial_tables, dim=1)
        selected_initial = torch.stack(selected_initial_list, dim=1).cpu()
    history: list[dict[str, float | int]] = []
    assignments = torch.empty(
        problems, rows, num_codebooks, dtype=torch.long, device=vectors.device
    )
    for iteration in range(iterations):
        assignments, decoded, errors = assign_additive_codes(
            vectors,
            codebooks,
            selector_whiteners=selector_whiteners,
            chunk_size=chunk_size,
        )
        before = float(errors.double().mean())
        for codebook in range(num_codebooks):
            other = decoded - torch.gather(
                codebooks[:, codebook],
                1,
                assignments[..., codebook].unsqueeze(-1).expand(-1, -1, subdim),
            )
            targets = vectors.float() - other
            offsets = (
                torch.arange(problems, device=vectors.device)[:, None]
                * entries_per_codebook
            )
            flat_indices = (assignments[..., codebook] + offsets).reshape(-1)
            sums = torch.zeros(
                problems * entries_per_codebook,
                subdim,
                device=vectors.device,
                dtype=torch.float32,
            )
            counts = torch.zeros(
                problems * entries_per_codebook,
                device=vectors.device,
                dtype=torch.float32,
            )
            sums.index_add_(0, flat_indices, targets.reshape(-1, subdim))
            counts.index_add_(
                0,
                flat_indices,
                torch.ones_like(flat_indices, dtype=torch.float32),
            )
            sums = sums.reshape(problems, entries_per_codebook, subdim)
            counts = counts.reshape(problems, entries_per_codebook)
            occupied = counts > 0
            means = sums / counts.clamp_min(1.0).unsqueeze(-1)
            codebooks[:, codebook] = torch.where(
                occupied.unsqueeze(-1), means, codebooks[:, codebook]
            )
            # Keep subsequent coordinate updates consistent with the new table.
            assignments, decoded, _ = assign_additive_codes(
                vectors,
                codebooks,
                selector_whiteners=selector_whiteners,
                chunk_size=chunk_size,
            )
        assignments, decoded, errors = assign_additive_codes(
            vectors,
            codebooks,
            selector_whiteners=selector_whiteners,
            chunk_size=chunk_size,
        )
        history.append(
            {
                "iteration": iteration,
                "joint_mse_before_update": before,
                "joint_mse_after_update": float(errors.double().mean()),
            }
        )
    assignments, _decoded, errors = assign_additive_codes(
        vectors,
        codebooks,
        selector_whiteners=selector_whiteners,
        chunk_size=chunk_size,
    )
    utilization = []
    normalized_entropy = []
    for codebook in range(num_codebooks):
        counts = torch.zeros(
            problems, entries_per_codebook, device=vectors.device, dtype=torch.long
        )
        counts.scatter_add_(
            1,
            assignments[..., codebook],
            torch.ones_like(assignments[..., codebook]),
        )
        probabilities = counts.double() / counts.sum(-1, keepdim=True).clamp_min(1)
        entropy = torch.where(
            probabilities > 0,
            -probabilities * probabilities.clamp_min(1e-300).log2(),
            torch.zeros_like(probabilities),
        ).sum(-1)
        utilization.append(float((counts > 0).float().mean()))
        normalized_entropy.append(
            float((entropy / math.log2(entries_per_codebook)).mean())
        )
    diagnostics = {
        "problems": problems,
        "rows_per_problem": rows,
        "subdimension": subdim,
        "num_codebooks": num_codebooks,
        "entries_per_codebook": entries_per_codebook,
        "joint_candidates": entries_per_codebook**num_codebooks,
        "encoder": "exact_joint",
        "metric": "m20" if selector_whiteners is not None else "euclidean",
        "initialization": (
            "provided_codebooks"
            if initial_codebooks is not None
            else "shared_initial_indices"
            if initial_indices is not None
            else "deterministic_residual_kmeans"
        ),
        "final_vector_mse": float(errors.double().mean()),
        "final_metric_vector_mse": float(errors.double().mean()),
        "mean_code_utilization": sum(utilization) / len(utilization),
        "mean_normalized_code_entropy": sum(normalized_entropy) / len(normalized_entropy),
        "per_codebook_utilization": utilization,
        "per_codebook_normalized_entropy": normalized_entropy,
        "finite": bool(torch.isfinite(codebooks).all() and torch.isfinite(errors).all()),
    }
    return AdditiveCodebookFit(
        codebooks=codebooks,
        initial_indices=selected_initial,
        assignments=assignments,
        history=history,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def fit_additive_codebooks(
    vectors: torch.Tensor,
    *,
    num_codebooks: int,
    entries_per_codebook: int,
    iterations: int = 5,
    kmeans_iterations: int = 10,
    chunk_size: int = 1024,
    problem_batch_size: int = 16,
    seed: int = 29,
    selector_whiteners: torch.Tensor | None = None,
    initial_indices: torch.Tensor | None = None,
    initial_codebooks: torch.Tensor | None = None,
) -> AdditiveCodebookFit:
    """Fit independent additive codebooks for ``[P,N,S]`` problems.

    A frozen per-problem factor makes initialization, exact joint assignment,
    and coordinate reassignment optimize ``||L(x-c)||^2``.  The raw centroid
    update remains exact because the SPD metric is constant per problem.
    """

    if vectors.ndim != 3 or not vectors.is_floating_point():
        raise ValueError("vectors must have shape [P,N,S]")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")
    if min(num_codebooks, iterations, kmeans_iterations, chunk_size) < 1:
        raise ValueError("fit budgets must be positive")
    if problem_batch_size < 1:
        raise ValueError("problem_batch_size must be positive")
    if not _power_of_two(entries_per_codebook):
        raise ValueError("entries_per_codebook must be a power of two")
    problems, rows, subdim = map(int, vectors.shape)
    if selector_whiteners is not None:
        if tuple(selector_whiteners.shape) != (problems, subdim, subdim):
            raise ValueError("selector whiteners have the wrong geometry")
        if not selector_whiteners.is_floating_point() or not bool(
            torch.isfinite(selector_whiteners).all()
        ):
            raise ValueError("selector whiteners must be finite")
    if initial_indices is not None:
        expected = (problems, num_codebooks, entries_per_codebook)
        if tuple(initial_indices.shape) != expected:
            raise ValueError("initial additive indices have the wrong geometry")
        if int(initial_indices.min()) < 0 or int(initial_indices.max()) >= rows:
            raise ValueError("initial additive indices are out of range")
    if initial_codebooks is not None:
        expected = (problems, num_codebooks, entries_per_codebook, subdim)
        if tuple(initial_codebooks.shape) != expected:
            raise ValueError("initial additive codebooks have the wrong geometry")
        if not initial_codebooks.is_floating_point() or not bool(
            torch.isfinite(initial_codebooks).all()
        ):
            raise ValueError("initial additive codebooks must be finite")
        if initial_indices is not None:
            raise ValueError("initial indices and codebooks are mutually exclusive")
    fits = []
    for start in range(0, len(vectors), problem_batch_size):
        stop = min(start + problem_batch_size, len(vectors))
        fits.append(
            _fit_batch(
                vectors[start:stop],
                num_codebooks=num_codebooks,
                entries_per_codebook=entries_per_codebook,
                iterations=iterations,
                kmeans_iterations=kmeans_iterations,
                chunk_size=chunk_size,
                seed=seed + start,
                selector_whiteners=(
                    selector_whiteners[start:stop]
                    if selector_whiteners is not None
                    else None
                ),
                initial_indices=(
                    initial_indices[start:stop]
                    if initial_indices is not None
                    else None
                ),
                initial_codebooks=(
                    initial_codebooks[start:stop]
                    if initial_codebooks is not None
                    else None
                ),
            )
        )
    history = []
    for iteration in range(iterations):
        history.append(
            {
                "iteration": iteration,
                "joint_mse_before_update": sum(
                    float(fit.history[iteration]["joint_mse_before_update"])
                    for fit in fits
                ) / len(fits),
                "joint_mse_after_update": sum(
                    float(fit.history[iteration]["joint_mse_after_update"])
                    for fit in fits
                ) / len(fits),
            }
        )
    diagnostics = {
        **fits[0].diagnostics,
        "problems": sum(int(fit.diagnostics["problems"]) for fit in fits),
        "final_vector_mse": sum(
            float(fit.diagnostics["final_vector_mse"])
            * int(fit.diagnostics["problems"])
            for fit in fits
        ) / len(vectors),
        "mean_code_utilization": sum(
            float(fit.diagnostics["mean_code_utilization"])
            * int(fit.diagnostics["problems"])
            for fit in fits
        ) / len(vectors),
        "mean_normalized_code_entropy": sum(
            float(fit.diagnostics["mean_normalized_code_entropy"])
            * int(fit.diagnostics["problems"])
            for fit in fits
        ) / len(vectors),
        "finite": all(bool(fit.diagnostics["finite"]) for fit in fits),
    }
    return AdditiveCodebookFit(
        codebooks=torch.cat([fit.codebooks for fit in fits]),
        initial_indices=(
            torch.cat(
                [fit.initial_indices for fit in fits if fit.initial_indices is not None]
            )
            if all(fit.initial_indices is not None for fit in fits)
            else None
        ),
        assignments=torch.cat([fit.assignments for fit in fits]),
        history=history,
        diagnostics=diagnostics,
    )
