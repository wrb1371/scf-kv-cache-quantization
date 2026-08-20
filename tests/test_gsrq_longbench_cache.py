from __future__ import annotations

from pathlib import Path
import sys

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from gsrq_longbench_cache import (  # noqa: E402
    LayerIndexCache,
    base_codebook_static_bytes,
    cache_bits_per_activation,
    greedy_encode_parent,
    greedy_encode_with_assignment_whitener,
)


def test_layer_index_cache_appends_and_resets_without_changing_payload() -> None:
    cache = LayerIndexCache()
    first = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.long)
    second = torch.tensor([[[[5, 6]]]], dtype=torch.long)
    cache.append(first, first + 1)
    cache.append(second, second + 1)
    key, value = cache.tensors()
    assert cache.length == 3
    assert cache.logical_payload_bytes() == 12
    assert torch.equal(key.long(), torch.cat([first, second], dim=2))
    assert torch.equal(value.long(), torch.cat([first + 1, second + 1], dim=2))
    allocated = cache.key
    cache.reset()
    assert cache.length == 0
    assert cache.logical_payload_bytes() == 0
    assert cache.key is allocated


def test_identity_selector_matches_parent_assignments() -> None:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(2, 3, 5, 4, generator=generator)
    codebooks = torch.randn(3, 2, 8, 4, generator=generator)
    identity = torch.eye(4).repeat(3, 1, 1)
    parent = greedy_encode_parent(states, codebooks, token_chunk_size=3)
    selected = greedy_encode_with_assignment_whitener(
        states,
        codebooks,
        identity,
        token_chunk_size=3,
    )
    assert torch.equal(parent, selected)


def test_nontrivial_selector_matches_explicit_transformed_problem() -> None:
    generator = torch.Generator().manual_seed(11)
    states = torch.randn(1, 2, 7, 3, generator=generator)
    codebooks = torch.randn(2, 3, 4, 3, generator=generator)
    factors = torch.randn(2, 3, 3, generator=generator)
    actual = greedy_encode_with_assignment_whitener(
        states,
        codebooks,
        factors,
        token_chunk_size=4,
    )
    transformed_states = torch.einsum("hde,bhte->bhtd", factors, states)
    transformed_codebooks = torch.einsum(
        "hrke,hde->hrkd", codebooks, factors
    )
    expected = greedy_encode_parent(
        transformed_states,
        transformed_codebooks,
        token_chunk_size=4,
    )
    assert torch.equal(actual, expected)


def test_gsrq_rate_and_static_ledger() -> None:
    assert cache_bits_per_activation(stages=16, clusters=256, dimension=128) == 1.0
    assert base_codebook_static_bytes(
        layers=2,
        heads=4,
        stages=3,
        clusters=8,
        dimension=16,
    ) == 2 * 4 * 3 * 8 * 16 * 2 * 2
