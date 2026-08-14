"""Canonical GSRQ key surfaces around Llama rotary position embeddings."""

from __future__ import annotations

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


PRE_ROPE_KEY_SURFACE = "pre_rope_key_and_value_projection"
POST_ROPE_KEY_SURFACE = "post_rope_key_and_value_projection"

_SURFACE_ALIASES = {
    "pre_rope": PRE_ROPE_KEY_SURFACE,
    "pre_rope_key_and_value": PRE_ROPE_KEY_SURFACE,
    PRE_ROPE_KEY_SURFACE: PRE_ROPE_KEY_SURFACE,
    "post_rope": POST_ROPE_KEY_SURFACE,
    "post_rope_key_and_value": POST_ROPE_KEY_SURFACE,
    POST_ROPE_KEY_SURFACE: POST_ROPE_KEY_SURFACE,
}


def normalize_key_surface(surface: str) -> str:
    """Return the canonical provenance label for a supported key surface."""
    try:
        return _SURFACE_ALIASES[surface]
    except KeyError as error:
        supported = ", ".join(sorted(_SURFACE_ALIASES))
        raise ValueError(
            f"unsupported GSRQ key surface {surface!r}; expected one of "
            f"{supported}"
        ) from error


def rotate_key_states(
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the model's exact RoPE transform to ``[B,H,T,D]`` keys."""
    _, rotated_keys = apply_rotary_pos_emb(
        key_states,
        key_states,
        cos,
        sin,
    )
    return rotated_keys


def key_quantization_input(
    raw_key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    surface: str,
) -> torch.Tensor:
    """Select the representation on which nearest-code assignment runs."""
    canonical = normalize_key_surface(surface)
    if canonical == POST_ROPE_KEY_SURFACE:
        return rotate_key_states(raw_key_states, cos, sin)
    return raw_key_states


def key_states_for_attention(
    quantized_key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    surface: str,
) -> torch.Tensor:
    """Map quantized keys from their fitted surface into attention space."""
    canonical = normalize_key_surface(surface)
    if canonical == PRE_ROPE_KEY_SURFACE:
        return rotate_key_states(quantized_key_states, cos, sin)
    return quantized_key_states
