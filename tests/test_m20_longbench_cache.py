from __future__ import annotations

import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))

from m20_longbench_cache import (  # noqa: E402
    FixedWidthKVCache,
    decode_cq,
    decode_fibquant,
    pack_fixed_width,
    unpack_fixed_width,
)
from lowrate_vq.codecs import quantize_fibquant  # noqa: E402
from lowrate_vq.cq import quantize_cq  # noqa: E402
from gsrq_longbench_cache import (  # noqa: E402
    greedy_encode_parent,
    greedy_encode_with_assignment_whitener,
)


def test_pack_roundtrip_all_matrix_widths() -> None:
    generator = torch.Generator().manual_seed(7)
    for bits, count in ((6, 32), (7, 32), (8, 16), (8, 32), (12, 8), (12, 16)):
        values = torch.randint(
            0, 1 << bits, (2, 3, 5, count), generator=generator
        )
        packed = pack_fixed_width(values, bits)
        assert packed.dtype == torch.uint8
        assert packed.shape[-1] == bits * count // 8
        restored = unpack_fixed_width(packed, bits, count)
        assert torch.equal(restored, values)


def test_cache_ledgers_match_requested_rates() -> None:
    for bits, count, norms, expected_bpa in (
        (12, 8, False, 0.75),
        (12, 16, False, 1.5),
        (8, 16, False, 1.0),
        (8, 32, False, 2.0),
        (6, 32, True, 1.625),
        (7, 32, True, 1.875),
        (8, 32, True, 2.125),
    ):
        cache = FixedWidthKVCache(
            bits_per_index=bits, index_count=count, store_norm=norms
        )
        indices = torch.zeros(1, 8, 11, count, dtype=torch.long)
        vector_norms = torch.ones(1, 8, 11, dtype=torch.float16) if norms else None
        cache.append(
            indices,
            indices,
            key_norm=vector_norms,
            value_norm=vector_norms,
        )
        expected = 2 * 8 * 11 * int(expected_bpa * 128 / 8)
        assert cache.payload_bytes() == expected


def test_cq_explicit_decode_matches_codec_decode() -> None:
    torch.manual_seed(11)
    states = torch.randn(1, 2, 7, 16)
    codebooks = torch.randn(2, 8, 8, 2)
    expected, indices = quantize_cq(states, codebooks, return_indices=True)
    assert indices is not None
    restored = unpack_fixed_width(pack_fixed_width(indices, 3), 3, 8)
    assert torch.equal(restored, indices)
    actual = decode_cq(indices, codebooks)
    torch.testing.assert_close(actual, expected.float(), atol=0, rtol=0)


def test_fibquant_explicit_decode_matches_fp16_norm_deployment() -> None:
    torch.manual_seed(13)
    states = torch.randn(1, 2, 5, 8)
    states = states / torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    rotation = torch.eye(8)
    codebook = torch.randn(64, 4)
    _, indices = quantize_fibquant(
        states,
        rotation,
        codebook,
        block_size=4,
        return_indices=True,
    )
    assert indices is not None
    norms = torch.linalg.vector_norm(states.float(), dim=-1).half()
    actual = decode_fibquant(indices, norms, rotation, codebook)
    selected = codebook[indices].reshape_as(states) * norms.float().unsqueeze(-1)
    torch.testing.assert_close(actual, selected, atol=0, rtol=0)


def test_fibquant_zero_norm_needs_no_mask_kernel() -> None:
    torch.manual_seed(14)
    indices = torch.randint(0, 64, (1, 2, 5, 2))
    norms = torch.zeros(1, 2, 5, dtype=torch.float16)
    rotation = torch.randn(8, 8)
    codebook = torch.randn(64, 4)
    actual = decode_fibquant(indices, norms, rotation, codebook)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_scfw_indices_equal_explicit_transformed_greedy() -> None:
    torch.manual_seed(17)
    states = torch.randn(1, 2, 5, 8)
    codebooks = torch.randn(2, 3, 8, 8)
    whiteners = torch.randn(2, 8, 8)
    transformed_states = torch.einsum("hde,bhte->bhtd", whiteners, states)
    transformed_tables = torch.einsum("hrke,hde->hrkd", codebooks, whiteners)
    expected = greedy_encode_parent(transformed_states, transformed_tables)
    actual = greedy_encode_with_assignment_whitener(
        states, codebooks, whiteners, token_chunk_size=32
    )
    assert torch.equal(actual, expected)
