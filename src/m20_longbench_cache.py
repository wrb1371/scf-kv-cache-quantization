"""Fixed-width packed KV-cache adapters for low-rate LongBench evaluation.

Only packed indices and, for FibQuant, one FP16 norm per vector persist across
autoregressive steps. Codebooks, rotations, and selector metrics are static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lowrate_vq.codecs import quantize_fibquant
from lowrate_vq.cq import quantize_cq
from lowrate_vq.m20_fibquant import quantize_m20_fibquant_key


def _validate_width(bits_per_index: int) -> None:
    if not 1 <= bits_per_index <= 16:
        raise ValueError("bits_per_index must be in [1,16]")


@torch.no_grad()
def pack_fixed_width(
    indices: torch.Tensor,
    bits_per_index: int,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Pack the last index dimension into an exact fixed-width byte stream."""

    _validate_width(bits_per_index)
    if indices.ndim < 1:
        raise ValueError("indices must have at least one dimension")
    units = int(indices.shape[-1])
    total_bits = units * bits_per_index
    if total_bits % 8:
        raise ValueError("this cache requires an integral byte count per vector")
    limit = 1 << bits_per_index
    values = indices.long()
    if validate and values.numel() and (
        int(values.min()) < 0 or int(values.max()) >= limit
    ):
        raise ValueError("an index exceeds its fixed-width representation")
    if bits_per_index == 8:
        # CQ/FibQuant use one byte per index. The generic bit loop would launch
        # hundreds of tiny kernels even though packing is exactly a dtype cast.
        return values.to(torch.uint8).contiguous()
    packed = torch.zeros(
        *values.shape[:-1], total_bits // 8, dtype=torch.uint8, device=values.device
    )
    for unit in range(units):
        value = values[..., unit]
        offset = unit * bits_per_index
        for bit in range(bits_per_index):
            byte = (offset + bit) // 8
            shift = (offset + bit) % 8
            packed[..., byte].bitwise_or_(
                (((value >> bit) & 1).to(torch.uint8)) << shift
            )
    return packed.contiguous()


@torch.no_grad()
def unpack_fixed_width(
    packed: torch.Tensor,
    bits_per_index: int,
    index_count: int,
) -> torch.Tensor:
    """Invert :func:`pack_fixed_width` without retaining an unpacked cache."""

    _validate_width(bits_per_index)
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("packed indices must be a uint8 tensor")
    if index_count < 1 or index_count * bits_per_index != int(packed.shape[-1]) * 8:
        raise ValueError("packed byte geometry differs from index count/width")
    if bits_per_index == 8:
        return packed.long()
    output = torch.zeros(
        *packed.shape[:-1], index_count, dtype=torch.long, device=packed.device
    )
    for unit in range(index_count):
        offset = unit * bits_per_index
        value = output[..., unit]
        for bit in range(bits_per_index):
            byte = (offset + bit) // 8
            shift = (offset + bit) % 8
            value.bitwise_or_(
                (((packed[..., byte] >> shift) & 1).long()) << bit
            )
    return output


class FixedWidthKVCache:
    """Append-only packed K/V indices with optional FP16 vector norms."""

    def __init__(self, *, bits_per_index: int, index_count: int, store_norm: bool) -> None:
        _validate_width(bits_per_index)
        if index_count < 1 or index_count * bits_per_index % 8:
            raise ValueError("cache index vector must occupy an integral byte count")
        self.bits_per_index = bits_per_index
        self.index_count = index_count
        self.bytes_per_index_vector = index_count * bits_per_index // 8
        self.store_norm = store_norm
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.key_norm: torch.Tensor | None = None
        self.value_norm: torch.Tensor | None = None
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def _ensure_capacity(self, packed: torch.Tensor, required: int) -> None:
        batch, heads, _, width = map(int, packed.shape)
        if width != self.bytes_per_index_vector:
            raise ValueError("packed vector width differs from cache geometry")
        if self.key is not None:
            same = tuple(self.key.shape[:2]) == (batch, heads) and int(
                self.key.shape[-1]
            ) == width
            if same and int(self.key.shape[2]) >= required:
                return
        old_key, old_value, old_length = self.key, self.value, self.length
        old_key_norm, old_value_norm = self.key_norm, self.value_norm
        old_capacity = int(old_key.shape[2]) if old_key is not None else 0
        capacity = max(required, max(1024, old_capacity * 2))
        shape = (batch, heads, capacity, width)
        self.key = torch.empty(shape, dtype=torch.uint8, device=packed.device)
        self.value = torch.empty_like(self.key)
        if self.store_norm:
            norm_shape = (batch, heads, capacity)
            self.key_norm = torch.empty(norm_shape, dtype=torch.float16, device=packed.device)
            self.value_norm = torch.empty_like(self.key_norm)
        if old_key is not None and old_length:
            self.key[:, :, :old_length].copy_(old_key[:, :, :old_length])
            assert old_value is not None
            self.value[:, :, :old_length].copy_(old_value[:, :, :old_length])
            if self.store_norm:
                assert self.key_norm is not None and self.value_norm is not None
                assert old_key_norm is not None and old_value_norm is not None
                self.key_norm[:, :, :old_length].copy_(old_key_norm[:, :, :old_length])
                self.value_norm[:, :, :old_length].copy_(old_value_norm[:, :, :old_length])

    def append(
        self,
        key_indices: torch.Tensor,
        value_indices: torch.Tensor,
        *,
        key_norm: torch.Tensor | None = None,
        value_norm: torch.Tensor | None = None,
    ) -> None:
        if key_indices.shape != value_indices.shape or key_indices.ndim != 4:
            raise ValueError("K/V indices must share [B,H,T,U] geometry")
        if int(key_indices.shape[-1]) != self.index_count:
            raise ValueError("index count differs from cache geometry")
        # Encoder assignments are argmin outputs with a statically validated
        # codebook width, so avoid two synchronizing min/max checks per tensor.
        packed_key = pack_fixed_width(
            key_indices, self.bits_per_index, validate=False
        )
        packed_value = pack_fixed_width(
            value_indices, self.bits_per_index, validate=False
        )
        query_length = int(key_indices.shape[2])
        required = self.length + query_length
        self._ensure_capacity(packed_key, required)
        assert self.key is not None and self.value is not None
        destination = slice(self.length, required)
        self.key[:, :, destination].copy_(packed_key)
        self.value[:, :, destination].copy_(packed_value)
        if self.store_norm:
            expected = tuple(key_indices.shape[:3])
            if key_norm is None or value_norm is None:
                raise ValueError("this cache requires K/V FP16 norms")
            if tuple(key_norm.shape) != expected or tuple(value_norm.shape) != expected:
                raise ValueError("norm geometry differs from indices")
            assert self.key_norm is not None and self.value_norm is not None
            self.key_norm[:, :, destination].copy_(key_norm.to(torch.float16))
            self.value_norm[:, :, destination].copy_(value_norm.to(torch.float16))
        elif key_norm is not None or value_norm is not None:
            raise ValueError("this cache does not store norms")
        self.length = required

    def tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.key is None or self.value is None or self.length < 1:
            raise ValueError("packed cache is empty")
        key = unpack_fixed_width(
            self.key[:, :, : self.length], self.bits_per_index, self.index_count
        )
        value = unpack_fixed_width(
            self.value[:, :, : self.length], self.bits_per_index, self.index_count
        )
        if not self.store_norm:
            return key, value, None, None
        assert self.key_norm is not None and self.value_norm is not None
        return (
            key,
            value,
            self.key_norm[:, :, : self.length],
            self.value_norm[:, :, : self.length],
        )

    def payload_bytes(self) -> int:
        if self.key is None:
            return 0
        batch, heads = map(int, self.key.shape[:2])
        per_kind = batch * heads * self.length * self.bytes_per_index_vector
        if self.store_norm:
            per_kind += batch * heads * self.length * 2
        return 2 * per_kind

    def allocated_bytes(self) -> int:
        tensors = (self.key, self.value, self.key_norm, self.value_norm)
        return sum(
            int(t.numel() * t.element_size()) for t in tensors if t is not None
        )


@dataclass
class PackedCodecContext:
    layer: int
    method: str
    key_surface: str
    index_bpa: float
    metadata_bpa: float
    total_bpa: float
    bits_per_index: int
    index_count: int
    chunk_size: int
    cache: FixedWidthKVCache
    provenance: dict[str, Any]
    key_codebooks: torch.Tensor | None = None
    value_codebooks: torch.Tensor | None = None
    key_selector_whiteners: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    value_codebook: torch.Tensor | None = None
    key_decode_codebook: torch.Tensor | None = None
    key_block_whiteners: torch.Tensor | None = None
    key_selector_cross_codebooks: torch.Tensor | None = None
    key_selector_norms: torch.Tensor | None = None


@torch.no_grad()
def encode_cq(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    selector_whiteners: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    _, indices = quantize_cq(
        states,
        codebooks,
        selector_whiteners=selector_whiteners,
        chunk_size=chunk_size,
        return_indices=True,
    )
    if indices is None:
        raise RuntimeError("CQ encoder did not return indices")
    return indices


@torch.no_grad()
def decode_cq(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> torch.Tensor:
    if indices.ndim != 4 or codebooks.ndim != 4:
        raise ValueError("indices/codebooks must be [B,H,T,G]/[H,G,K,S]")
    batch, heads, tokens, groups = map(int, indices.shape)
    cb_heads, cb_groups, clusters, subdim = map(int, codebooks.shape)
    if (heads, groups) != (cb_heads, cb_groups):
        raise ValueError("CQ cache/codebook geometry differs")
    if validate_indices and indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= clusters
    ):
        raise ValueError("CQ cache contains an out-of-range index")
    head_rows = torch.arange(heads, device=indices.device)[None, :, None, None]
    group_rows = torch.arange(groups, device=indices.device)[None, None, None, :]
    decoded = codebooks.to(indices.device, dtype=torch.float32)[
        head_rows, group_rows, indices.long()
    ]
    return decoded.reshape(batch, heads, tokens, groups * subdim)


@torch.no_grad()
def encode_fibquant(
    states: torch.Tensor,
    context: PackedCodecContext,
    *,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.rotation is None:
        raise ValueError("FibQuant rotation is missing")
    if kind == "key" and context.method == "m20_fibquant":
        if context.key_decode_codebook is None or context.key_block_whiteners is None:
            raise ValueError("SCF-FibQuant K selector is incomplete")
        _, indices = quantize_m20_fibquant_key(
            states,
            context.rotation,
            context.key_decode_codebook,
            context.key_block_whiteners,
            block_size=int(context.key_decode_codebook.shape[-1]),
            chunk_size=context.chunk_size,
            return_indices=True,
            selector_cross_codebooks=context.key_selector_cross_codebooks,
            selector_norms=context.key_selector_norms,
        )
    else:
        codebook = (
            context.value_codebook
            if kind == "value" or context.method == "fibquant"
            else context.key_decode_codebook
        )
        if codebook is None:
            raise ValueError("FibQuant decode codebook is missing")
        _, indices = quantize_fibquant(
            states,
            context.rotation,
            codebook,
            block_size=int(codebook.shape[-1]),
            chunk_size=context.chunk_size,
            return_indices=True,
        )
    if indices is None:
        raise RuntimeError("FibQuant encoder did not return indices")
    norms = torch.linalg.vector_norm(states.float(), dim=-1).to(torch.float16)
    return indices, norms


@torch.no_grad()
def decode_fibquant(
    indices: torch.Tensor,
    norms: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    *,
    validate_indices: bool = True,
) -> torch.Tensor:
    if indices.ndim != 4 or tuple(norms.shape) != tuple(indices.shape[:3]):
        raise ValueError("FibQuant indices/norm geometry differs")
    batch, heads, tokens, blocks = map(int, indices.shape)
    clusters, block_size = map(int, codebook.shape)
    dimension = blocks * block_size
    if tuple(rotation.shape) != (dimension, dimension):
        raise ValueError("FibQuant rotation geometry differs")
    if validate_indices and indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= clusters
    ):
        raise ValueError("FibQuant cache contains an out-of-range index")
    decoded_rotated = codebook.to(indices.device, dtype=torch.float32)[
        indices.long()
    ].reshape(batch, heads, tokens, dimension)
    decoded = torch.matmul(decoded_rotated, rotation.to(indices.device).float())
    decoded *= norms.float().unsqueeze(-1)
    # Multiplication by the stored norm already maps zero-norm vectors to zero;
    # a separate comparison plus masked-fill only adds two eager kernels.
    return decoded


def context_ledger(context: PackedCodecContext) -> dict[str, Any]:
    return {
        "method": context.method,
        "index_bpa": context.index_bpa,
        "metadata_bpa": context.metadata_bpa,
        "total_bpa": context.total_bpa,
        "bits_per_index": context.bits_per_index,
        "indices_per_vector": context.index_count,
        "packed_index_bytes_per_vector": context.cache.bytes_per_index_vector,
        "norm_bytes_per_vector": 2 if context.cache.store_norm else 0,
        "per_token_state_bytes": 0,
        "variable_length_address_bytes": 0,
    }
