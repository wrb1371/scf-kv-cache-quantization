"""Packed GSRQ/SCF-GSRQ cache used by LongBench and efficiency runs.

The persistent cache contains only uint8 residual-codebook indices.  SCF
changes key assignment geometry through a frozen per-head factor; decoding
continues to gather the original GSRQ centroids.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from gsrq.rope_surface import (
    PRE_ROPE_KEY_SURFACE,
    normalize_key_surface,
    rotate_key_states,
)


def cache_bits_per_activation(
    *, stages: int, clusters: int, dimension: int
) -> float:
    if stages < 1 or dimension < 1:
        raise ValueError("stages and dimension must be positive")
    if clusters < 1 or clusters & (clusters - 1):
        raise ValueError("clusters must be a power of two")
    return stages * int(math.log2(clusters)) / dimension


def base_codebook_static_bytes(
    *,
    layers: int,
    heads: int,
    stages: int,
    clusters: int,
    dimension: int,
    kinds: int = 2,
    bytes_per_scalar: int = 2,
) -> int:
    values = (
        layers,
        heads,
        stages,
        clusters,
        dimension,
        kinds,
        bytes_per_scalar,
    )
    if min(values) < 1:
        raise ValueError("static-ledger arguments must be positive")
    return math.prod(values)


def _head_gather(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if table.ndim < 3 or indices.ndim != 3:
        raise ValueError("table/indices must be [H,K,...]/[B,H,T]")
    if table.shape[0] != indices.shape[1]:
        raise ValueError("head count differs between table and indices")
    heads = torch.arange(table.shape[0], device=indices.device)[None, :, None]
    return table[heads, indices.long()]


def _assign(residual: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    distances = (
        residual.square().sum(dim=-1, keepdim=True)
        + table.square().sum(dim=-1)[:, None, :]
        - 2.0 * torch.bmm(residual, table.transpose(1, 2))
    )
    return torch.argmin(distances, dim=-1)


@torch.no_grad()
def greedy_encode_parent(
    states: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    token_chunk_size: int = 32,
) -> torch.Tensor:
    """Return residual-VQ indices with shape ``[B,H,T,R]``."""

    if states.ndim != 4 or codebooks.ndim != 4:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,R,K,D]")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    heads, stages, _, dimension = map(int, codebooks.shape)
    if states.shape[1] != heads or states.shape[-1] != dimension:
        raise ValueError("state geometry differs from codebooks")
    batch, _, tokens, _ = states.shape
    flat = (
        states.detach()
        .to(device=codebooks.device, dtype=torch.float32)
        .permute(1, 0, 2, 3)
        .reshape(heads, batch * tokens, dimension)
    )
    tables = codebooks.detach().to(dtype=torch.float32)
    chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[1], token_chunk_size):
        residual = flat[:, start : start + token_chunk_size].clone()
        saved: list[torch.Tensor] = []
        for stage in range(stages):
            table = tables[:, stage]
            indices = _assign(residual, table)
            residual -= _head_gather(table, indices.unsqueeze(0))[0]
            saved.append(indices)
        chunks.append(torch.stack(saved, dim=-1))
    return (
        torch.cat(chunks, dim=1)
        .reshape(heads, batch, tokens, stages)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


@torch.no_grad()
def greedy_encode_with_assignment_whitener(
    states: torch.Tensor,
    raw_codebooks: torch.Tensor,
    whiteners: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Select in SCF coordinates while retaining raw-table decoding."""

    if states.ndim != 4 or raw_codebooks.ndim != 4:
        raise ValueError("states/codebooks must be [B,H,T,D]/[H,R,K,D]")
    heads, _, _, dimension = map(int, raw_codebooks.shape)
    if tuple(states.shape[1::2]) != (heads, dimension):
        raise ValueError("state geometry differs from codebooks")
    if tuple(whiteners.shape) != (heads, dimension, dimension):
        raise ValueError("assignment whitener geometry differs")
    factors = whiteners.to(device=states.device, dtype=torch.float32)
    raw_tables = raw_codebooks.to(device=states.device, dtype=torch.float32)
    selector_states = torch.einsum("hde,bhte->bhtd", factors, states.float())
    selector_tables = torch.einsum("hrke,hde->hrkd", raw_tables, factors)
    return greedy_encode_parent(
        selector_states,
        selector_tables,
        token_chunk_size=token_chunk_size,
    )


def _decode(indices: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 4 or codebooks.ndim != 4:
        raise ValueError("indices/codebooks must be [B,H,T,R]/[H,R,K,D]")
    heads, stages, clusters, dimension = map(int, codebooks.shape)
    if tuple(indices.shape[1::2]) != (heads, stages):
        raise ValueError("index geometry differs from codebooks")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= clusters
    ):
        raise ValueError("indices are out of range")
    output = torch.zeros(
        *indices.shape[:3],
        dimension,
        device=indices.device,
        dtype=torch.float32,
    )
    tables = codebooks.to(device=indices.device, dtype=torch.float32)
    for stage in range(stages):
        output += _head_gather(tables[:, stage], indices[..., stage])
    return output


def _explicit_attention(
    query_states: torch.Tensor,
    key_indices: torch.Tensor,
    value_indices: torch.Tensor,
    key_codebooks: torch.Tensor,
    value_codebooks: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = _decode(key_indices, key_codebooks)
    values = _decode(value_indices, value_codebooks)
    query_heads = int(query_states.shape[1])
    kv_heads = int(keys.shape[1])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = query_heads // kv_heads
    keys = rotate_key_states(keys, cos, sin).repeat_interleave(repeats, dim=1)
    values = values.repeat_interleave(repeats, dim=1)
    logits = torch.matmul(
        query_states.float(), keys.transpose(-1, -2)
    ) / math.sqrt(keys.shape[-1])
    if attention_mask is not None:
        logits = logits + attention_mask[..., : keys.shape[-2]].float()
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, values), probabilities


class LayerIndexCache:
    """Append-only uint8 K/V residual-stage indices."""

    def __init__(self) -> None:
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.length = 0

    def reset(self) -> None:
        self.length = 0

    def _ensure_capacity(self, indices: torch.Tensor, required: int) -> None:
        batch, heads, _, stages = map(int, indices.shape)
        if self.key is not None:
            compatible = tuple(self.key.shape[:2]) == (batch, heads) and int(
                self.key.shape[-1]
            ) == stages
            if compatible and int(self.key.shape[2]) >= required:
                return
        old_key, old_value, old_length = self.key, self.value, self.length
        old_capacity = int(old_key.shape[2]) if old_key is not None else 0
        capacity = max(required, max(1024, old_capacity * 2))
        shape = (batch, heads, capacity, stages)
        self.key = torch.empty(shape, dtype=torch.uint8, device=indices.device)
        self.value = torch.empty_like(self.key)
        if old_key is not None and old_length:
            self.key[:, :, :old_length].copy_(old_key[:, :, :old_length])
            assert old_value is not None
            self.value[:, :, :old_length].copy_(old_value[:, :, :old_length])

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("K/V indices must share [B,H,T,R] geometry")
        if key.numel() and (int(key.min()) < 0 or int(key.max()) > 255):
            raise ValueError("indices cannot be represented by uint8")
        query_length = int(key.shape[2])
        required = self.length + query_length
        self._ensure_capacity(key, required)
        assert self.key is not None and self.value is not None
        destination = slice(self.length, required)
        self.key[:, :, destination].copy_(key.to(dtype=torch.uint8))
        self.value[:, :, destination].copy_(value.to(dtype=torch.uint8))
        self.length = required

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None or self.value is None or self.length < 1:
            raise ValueError("index cache is empty")
        return self.key[:, :, : self.length], self.value[:, :, : self.length]

    def logical_payload_bytes(self) -> int:
        if self.key is None:
            return 0
        batch, heads, _, stages = map(int, self.key.shape)
        return 2 * batch * heads * self.length * stages


@dataclass
class QuantizedLayerContext:
    layer: int
    key_codebooks: torch.Tensor
    value_codebooks: torch.Tensor
    token_chunk_size: int
    cache: LayerIndexCache
    key_assignment_whitener: torch.Tensor | None = None


@dataclass
class _LoadedGSRQContext:
    key_codebooks: torch.Tensor
    value_codebooks: torch.Tensor
    key_surface: str
    key_assignment_whitener: torch.Tensor | None


def _load_gsrq_context(
    directory: Path,
    layer: int,
    *,
    key_override_dir: Path | None,
) -> _LoadedGSRQContext:
    """Load only the artifact fields required by the packed runtime."""

    path = directory / f"layer_{layer:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    provenance = artifact["provenance"]
    if int(provenance["layer"]) != layer:
        raise ValueError(f"{path} has the wrong layer provenance")
    key_codebooks = artifact["codebooks"]["key"].float()
    value_codebooks = artifact["codebooks"]["value"].float()
    if key_codebooks.shape != value_codebooks.shape or key_codebooks.ndim != 4:
        raise ValueError(f"{path} must contain matched [H,R,K,D] K/V codebooks")
    key_assignment_whitener = None
    if key_override_dir is not None:
        override_path = key_override_dir / f"layer_{layer:02d}.pt"
        if not override_path.exists():
            raise FileNotFoundError(override_path)
        override = torch.load(
            override_path, map_location="cpu", weights_only=False
        )
        override_provenance = override.get("provenance", {})
        override_key = override.get("codebooks", {}).get("key")
        key_assignment_whitener = override.get("key_assignment_whiteners")
        checks = {
            "layer": int(override_provenance.get("layer", -1)) == layer,
            "model": str(Path(override_provenance.get("model", "")).resolve())
            == str(Path(provenance.get("model", "")).resolve()),
            "transform": override_provenance.get("key_transform")
            in {"m20_sqrt_fisher_fixed", "scf_spectral_fisher_fixed"},
            "weighted": bool(override_provenance.get("weighted")),
            "surface": normalize_key_surface(
                override_provenance.get("surface", "pre_rope")
            )
            == normalize_key_surface(provenance.get("surface", "pre_rope")),
            "key_shape": isinstance(override_key, torch.Tensor)
            and tuple(override_key.shape) == tuple(key_codebooks.shape),
            "whitener_shape": isinstance(key_assignment_whitener, torch.Tensor)
            and tuple(key_assignment_whitener.shape)
            == (
                key_codebooks.shape[0],
                key_codebooks.shape[-1],
                key_codebooks.shape[-1],
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{override_path} failed checks: {failed}")
        assert isinstance(override_key, torch.Tensor)
        assert isinstance(key_assignment_whitener, torch.Tensor)
        if not bool(torch.isfinite(override_key).all()) or not bool(
            torch.isfinite(key_assignment_whitener).all()
        ):
            raise ValueError(f"{override_path} contains non-finite tensors")
        key_codebooks = override_key.float()
        key_assignment_whitener = key_assignment_whitener.float()
    return _LoadedGSRQContext(
        key_codebooks=key_codebooks,
        value_codebooks=value_codebooks,
        key_surface=normalize_key_surface(provenance.get("surface", "pre_rope")),
        key_assignment_whitener=key_assignment_whitener,
    )


def quantized_cache_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, None]:
    if past_key_value is not None or use_cache:
        raise ValueError("compressed attention owns its cache and requires use_cache=False")
    context: QuantizedLayerContext = self._quantized_cache_context
    batch, query_length, _ = hidden_states.shape
    if batch != 1 or position_ids is None:
        raise ValueError("packed attention requires batch one and position_ids")
    expected = torch.arange(
        context.cache.length,
        context.cache.length + query_length,
        device=position_ids.device,
    )
    if tuple(position_ids.shape) != (1, query_length) or not torch.equal(
        position_ids[0], expected
    ):
        raise ValueError("position IDs do not match implicit cache append order")

    query_states = self.q_proj(hidden_states).view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    raw_key_states = self.k_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    if position_embeddings is None:
        current_cos, current_sin = self.rotary_emb(value_states, position_ids)
    else:
        current_cos, current_sin = position_embeddings
    query_states, _ = apply_rotary_pos_emb(
        query_states, raw_key_states, current_cos, current_sin
    )

    if context.key_assignment_whitener is None:
        key_indices = greedy_encode_parent(
            raw_key_states,
            context.key_codebooks,
            token_chunk_size=context.token_chunk_size,
        )
    else:
        key_indices = greedy_encode_with_assignment_whitener(
            raw_key_states,
            context.key_codebooks,
            context.key_assignment_whitener,
            token_chunk_size=context.token_chunk_size,
        )
    value_indices = greedy_encode_parent(
        value_states,
        context.value_codebooks,
        token_chunk_size=context.token_chunk_size,
    )
    context.cache.append(key_indices, value_indices)
    cached_keys, cached_values = context.cache.tensors()
    full_positions = torch.arange(
        context.cache.length, device=position_ids.device
    ).unsqueeze(0)
    full_cos, full_sin = self.rotary_emb(value_states, full_positions)
    output, probabilities = _explicit_attention(
        query_states,
        cached_keys.long(),
        cached_values.long(),
        context.key_codebooks,
        context.value_codebooks,
        full_cos,
        full_sin,
        attention_mask=attention_mask,
    )
    output = output.to(dtype=hidden_states.dtype)
    output = output.transpose(1, 2).contiguous().reshape(
        batch, query_length, self.hidden_size
    )
    return self.o_proj(output), probabilities if output_attentions else None, None


def build_gsrq_contexts(
    model: nn.Module,
    *,
    mode: str,
    codebook_dir: Path,
    parent_path: Path | None,
    rbg_path: Path | None,
    metric_path: Path | None,
    device: torch.device,
    token_chunk_size: int,
    key_override_dir: Path | None = None,
) -> tuple[list[QuantizedLayerContext], dict[str, Any]]:
    """Bind the parent or SCF-GSRQ packed runtime to a Llama decoder."""

    if mode != "gsrq":
        raise ValueError("the public SCF runtime supports mode='gsrq'")
    if any(path is not None for path in (parent_path, rbg_path, metric_path)):
        raise ValueError("GSRQ does not use FARQ/P3 artifacts")
    contexts: list[QuantizedLayerContext] = []
    expected_geometry: tuple[int, int, int, int] | None = None
    for layer, decoder_layer in enumerate(model.model.layers):
        loaded = _load_gsrq_context(
            codebook_dir,
            layer,
            key_override_dir=key_override_dir,
        )
        if normalize_key_surface(loaded.key_surface) != PRE_ROPE_KEY_SURFACE:
            raise ValueError("LongBench GSRQ/SCF requires pre-RoPE K codebooks")
        key_codebooks = loaded.key_codebooks.to(device=device, dtype=torch.float32)
        value_codebooks = loaded.value_codebooks.to(
            device=device, dtype=torch.float32
        )
        whiteners = (
            loaded.key_assignment_whitener.to(device=device, dtype=torch.float32)
            if loaded.key_assignment_whitener is not None
            else None
        )
        heads, stages, clusters, dimension = map(int, key_codebooks.shape)
        geometry = (heads, stages, clusters, dimension)
        if value_codebooks.shape != key_codebooks.shape:
            raise ValueError("GSRQ K/V codebook geometry differs")
        if expected_geometry is None:
            expected_geometry = geometry
        elif geometry != expected_geometry:
            raise ValueError("GSRQ layer geometries differ")
        expected_heads = int(model.config.num_key_value_heads)
        expected_dimension = int(
            model.config.hidden_size // model.config.num_attention_heads
        )
        if (
            heads != expected_heads
            or clusters != 256
            or dimension != expected_dimension
            or dimension != 128
            or stages not in (12, 16, 32)
        ):
            raise ValueError(
                "GSRQ geometry must match D=128, K=256, and R in {12,16,32}"
            )
        context = QuantizedLayerContext(
            layer=layer,
            key_codebooks=key_codebooks,
            value_codebooks=value_codebooks,
            token_chunk_size=token_chunk_size,
            cache=LayerIndexCache(),
            key_assignment_whitener=whiteners,
        )
        decoder_layer.self_attn._quantized_cache_context = context
        decoder_layer.self_attn.forward = types.MethodType(
            quantized_cache_attention_forward, decoder_layer.self_attn
        )
        contexts.append(context)

    if expected_geometry is None:
        raise ValueError("model has no decoder layers")
    heads, stages, clusters, dimension = expected_geometry
    rate = cache_bits_per_activation(
        stages=stages, clusters=clusters, dimension=dimension
    )
    if rate not in (0.75, 1.0, 2.0):
        raise ValueError("LongBench supports the frozen 0.75/1/2-BPA rates")
    ledger = {
        "layers": len(contexts),
        "heads": heads,
        "stages": stages,
        "clusters": clusters,
        "dimension": dimension,
        "cache_bits_per_activation": rate,
        "total_static_tensor_bytes": base_codebook_static_bytes(
            layers=len(contexts),
            heads=heads,
            stages=stages,
            clusters=clusters,
            dimension=dimension,
        ),
    }
    return contexts, ledger
