#!/usr/bin/env python3
"""KVQuant-aligned perplexity simulation for the current FACQ design.

This is intentionally a standalone evaluator.  The local KVQuant checkout is
used only as a read-only reference because it contains unrelated user changes.

Comparable parts of the protocol:
* Llama-2-7B, eager FP16 attention, cache disabled;
* 4096-token non-overlapping WikiText-2 windows or the KVQuant C4 sampler;
* teacher-forced loss on every next-token target;
* the first token is always FP16.

FACQ-specific simulation:
* post-RoPE K is stored in 64-token anchored pages;
* the causal page containing a query remains FP16 for that query;
* completed K pages use the FACQ base, conservative directory, and optional
  fixed-top-k or full residual side reads;
* V uses either FP16 (diagnostic) or an explicitly provisional symmetric
  anchored residual codec.

The full-sequence activation-energy routing used here is an offline PPL
simulation convenience, not an online append implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from facq_certificate_backend_smoke import deterministic_rotation
from real_attention_turboquant_smoke import sphere_codebook


MODE_SPECS: dict[str, dict[str, Any]] = {
    "fp16": {
        "k_path": "fp16",
        "v_path": "fp16",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_base_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_top8_s1_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_top32_s1_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 1,
        "selection": "topk",
        "topk": 32,
    },
    "facq_full_s1_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 1,
        "selection": "full",
    },
    "facq_top8_s1_vfp16": {
        "v_path": "fp16",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_base_vfp16": {
        "v_path": "fp16",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_top8_s2_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 2,
        "selection": "topk",
        "topk": 8,
    },
    "facq_full_s2_vbase": {
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 2,
        "selection": "full",
    },
    "facq_kfp16_vbase": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "activation",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_kfp16_vresidual": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "residual",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_kfp16_vodiag": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "o_diag",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_kfp16_voeigen": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "o_eigen",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_kfp16_vosqrt": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "o_sqrt",
        "side_bits": 0,
        "selection": "none",
    },
    "facq_top8_s1_vresidual": {
        "v_path": "base",
        "v_variant": "residual",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_top8_s1_vodiag": {
        "v_path": "base",
        "v_variant": "o_diag",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_top8_s1_voeigen": {
        "v_path": "base",
        "v_variant": "o_eigen",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_top8_s1_vosqrt": {
        "v_path": "base",
        "v_variant": "o_sqrt",
        "side_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_base_v3p25": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 4,
        "v_regular_bits": 3,
        "side_bits": 0,
        "selection": "none",
    },
    "facq_base_v3p25_residual": {
        "v_path": "base",
        "v_variant": "residual",
        "v_high_bits": 4,
        "v_regular_bits": 3,
        "side_bits": 0,
        "selection": "none",
    },
    "facq_base_v3p25_odiag": {
        "v_path": "base",
        "v_variant": "o_diag",
        "v_high_bits": 4,
        "v_regular_bits": 3,
        "side_bits": 0,
        "selection": "none",
    },
    "facq_kfp16_v3p25": {
        "k_path": "fp16",
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 4,
        "v_regular_bits": 3,
        "side_bits": 0,
        "selection": "none",
    },
    "facq_g3_top8_s1_v2p5": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 4,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_groups": 3,
        "selection": "topk",
        "topk": 8,
    },
    "facq_g1_top8_s1_v3": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 3,
        "v_regular_bits": 3,
        "side_bits": 1,
        "side_groups": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t25_top8_s1_v3": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 3,
        "v_regular_bits": 3,
        "side_bits": 1,
        "side_token_fraction": 0.25,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_frozen": {
        "v_path": "base",
        "v_variant": "frozen_whiten",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_normq25": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_norm_mode": "q25",
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_normmid": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_norm_mode": "midpoint",
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_frozen_normq25": {
        "v_path": "base",
        "v_variant": "frozen_whiten",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_norm_mode": "q25",
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_frozen_normmid": {
        "v_path": "base",
        "v_variant": "frozen_whiten",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_norm_mode": "midpoint",
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_kunit": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vkunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_noread_s1_v2p75_vkunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "none",
    },
    "facq_t50_full_s1_v2p75_vkunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "full",
    },
    "facq_t50_top8_s1_v1p75_vkunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_high_fraction": 0.75,
        "v_high_bits": 2,
        "v_regular_bits": 1,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vlearn_mse_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "mse",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vlearn_out_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "output_aware",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vlearn_mix25_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "output_mix25",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vlearn_mix50_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "output_mix50",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vlearn_sym_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "output_symmetric",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vexact_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "exact_output",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vq2exact_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "exact_output_vector2",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vq2traj_kunit": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_codebook_objective": "trajectory_hidden_vector2",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_renorm": True,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vknormp50": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 0.5,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 0.5,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vknormp75": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 0.75,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 0.75,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vknormp125": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 1.25,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 1.25,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_vknormp150": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 1.5,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 1.5,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_v075k100": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 0.75,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 1.0,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_v125k100": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 1.25,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 1.0,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_v100k075": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 1.0,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 0.75,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_v100k125": {
        "v_path": "base",
        "v_variant": "unit_direction",
        "v_direction_norm_power": 1.0,
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_direction_norm_power": 1.25,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t75_top8_s1_v2p5": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_bits": 4,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.75,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50q_top8_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "side_token_score": "query_diag",
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top12_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 12,
    },
    "facq_t50_top16_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 16,
    },
    "facq_t50g75_top8_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_gain": 0.75,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50g875_top8_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_gain": 0.875,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50g75_top12_s1_v2p75": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_gain": 0.75,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 12,
    },
    "facq_t50_top8_s1_v2p75_residual": {
        "v_path": "base",
        "v_variant": "residual",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_odiag": {
        "v_path": "base",
        "v_variant": "o_diag",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_whiten": {
        "v_path": "base",
        "v_variant": "whiten",
        "v_high_fraction": 0.75,
        "v_high_bits": 3,
        "v_regular_bits": 2,
        "v_channel_scale_bits": 8,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
    "facq_t50_top8_s1_v2p75_h5": {
        "v_path": "base",
        "v_variant": "activation",
        "v_high_fraction": 0.25,
        "v_high_bits": 5,
        "v_regular_bits": 2,
        "side_bits": 1,
        "side_token_fraction": 0.5,
        "side_token_bitmap_bits": 1,
        "selection": "topk",
        "topk": 8,
    },
}


@dataclass
class ReadStats:
    query_heads: int = 0
    selected_side_vectors: int = 0
    feasible_side_vectors: int = 0
    max_selected_per_query_head: int = 0
    selected_count_histogram: dict[int, int] = field(default_factory=dict)

    def update(
        self,
        selected: torch.Tensor,
        feasible: torch.Tensor,
    ) -> None:
        selected_counts = selected.sum(dim=-1)
        self.query_heads += int(selected_counts.numel())
        self.selected_side_vectors += int(selected_counts.sum().item())
        self.feasible_side_vectors += int(feasible.sum().item())
        self.max_selected_per_query_head = max(
            self.max_selected_per_query_head,
            int(selected_counts.max().item()),
        )
        histogram = torch.bincount(
            selected_counts.detach().reshape(-1).cpu()
        )
        for count, frequency in enumerate(histogram.tolist()):
            if frequency:
                self.selected_count_histogram[count] = (
                    self.selected_count_histogram.get(count, 0)
                    + int(frequency)
                )


def read_count_percentile(
    histogram: dict[int, int], quantile: float
) -> int:
    if not histogram:
        return 0
    total = sum(histogram.values())
    target = max(1, math.ceil(total * quantile))
    cumulative = 0
    for count in sorted(histogram):
        cumulative += histogram[count]
        if cumulative >= target:
            return count
    return max(histogram)


@dataclass
class CodecContext:
    mode: str
    page_size: int
    groups: int
    quant_seed: int
    query_chunk: int
    first_token_fp16: bool
    codebooks: dict[tuple[int, int], torch.Tensor]
    rotations: dict[str, Any]
    read_stats: ReadStats
    value_calibration: torch.Tensor | None = None
    value_codebooks: dict[str, dict[str, torch.Tensor]] | None = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(root / "models" / "LLM-Research" / "llama-2-7b"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(root / "huggingface_cache_datasets" / "datasets"),
    )
    parser.add_argument(
        "--dataset", choices=["wikitext2", "c4"], default="wikitext2"
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_SPECS),
        default=["fp16", "facq_top8_s1_vbase"],
    )
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--quant-seed", type=int, default=41)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--value-calibration-file",
        help=(
            "Frozen per-layer/KV-head/channel V residual scales. Required by "
            "modes whose v_variant is frozen_whiten."
        ),
    )
    parser.add_argument(
        "--value-codebook-file",
        help=(
            "Frozen per-layer/head high/regular V scalar codebooks. "
            "Required by modes with v_codebook_objective."
        ),
    )
    parser.add_argument(
        "--no-first-token-fp16",
        action="store_true",
        help="Ablation only; the aligned default keeps token zero FP16.",
    )
    return parser.parse_args()


def local_arrow_path(dataset_root: Path, dataset: str, split: str) -> Path:
    if dataset == "wikitext2":
        return (
            dataset_root
            / "Salesforce___wikitext"
            / "wikitext-2-raw-v1"
            / "0.0.0"
            / "b08601e04326c79dfdd32d625aee71d232d685c3"
            / f"wikitext-{split}.arrow"
        )
    if dataset == "c4":
        return (
            dataset_root
            / "allenai___c4"
            / "default-c7bc8b0aefc5e48f"
            / "0.0.0"
            / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
            / "c4-validation.arrow"
        )
    raise ValueError(dataset)


def load_eval_tokens(
    tokenizer: Any,
    dataset_root: Path,
    dataset: str,
    sequence_length: int,
    max_windows: int,
) -> tuple[torch.Tensor, int]:
    """Reproduce the KVQuant WikiText-2 and C4 evaluation construction."""
    path = local_arrow_path(
        dataset_root, dataset, "test" if dataset == "wikitext2" else "validation"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    data = Dataset.from_file(str(path))
    if dataset == "wikitext2":
        encoded = tokenizer(
            "\n\n".join(data["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        available = encoded.numel() // sequence_length
        return encoded[:, : available * sequence_length].contiguous(), available

    target_windows = 256 if max_windows <= 0 else min(max_windows, 256)
    generator = random.Random(0)
    windows: list[torch.Tensor] = []
    while len(windows) < target_windows:
        row = generator.randint(0, len(data) - 1)
        tokenized = tokenizer(
            data[row]["text"],
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        if tokenized.shape[1] < sequence_length:
            continue
        start = generator.randint(
            0, max(0, tokenized.shape[1] - sequence_length - 1)
        )
        windows.append(tokenized[:, start : start + sequence_length])
    return torch.hstack(windows), 256


def make_codec_tables(
    head_dim: int,
    groups: int,
    quant_seed: int,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, Any]]:
    if head_dim % 4 or head_dim % groups:
        raise ValueError("head_dim must divide both the 3/2 split and groups")
    high_dim = head_dim // 4
    regular_dim = head_dim - high_dim
    group_dim = head_dim // groups
    dimensions = {high_dim, regular_dim, group_dim}
    codebooks = {
        (dimension, bits): torch.from_numpy(sphere_codebook(dimension, bits))
        for dimension in dimensions
        for bits in (1, 2, 3, 4, 5)
    }
    rotations = {
        "base_regular": torch.from_numpy(
            deterministic_rotation(regular_dim, quant_seed, 901)
        ),
        "base_high": torch.from_numpy(
            deterministic_rotation(high_dim, quant_seed, 902)
        ),
        "side": [
            torch.from_numpy(
                deterministic_rotation(group_dim, quant_seed, 910 + group)
            )
            for group in range(groups)
        ],
    }
    return codebooks, rotations


def move_codec_tables(
    codebooks: dict[tuple[int, int], torch.Tensor],
    rotations: dict[str, Any],
    device: torch.device,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, Any]]:
    moved_codebooks = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in codebooks.items()
    }
    moved_rotations = {
        "base_regular": rotations["base_regular"].to(
            device=device, dtype=torch.float32
        ),
        "base_high": rotations["base_high"].to(
            device=device, dtype=torch.float32
        ),
        "side": [
            value.to(device=device, dtype=torch.float32)
            for value in rotations["side"]
        ],
    }
    return moved_codebooks, moved_rotations


def scalar_quantize_torch(
    values: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    boundaries = 0.5 * (codebook[:-1] + codebook[1:])
    indices = torch.bucketize(values.contiguous(), boundaries)
    return codebook[indices]


def vector2_quantize_torch(
    values: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    if values.shape[-1] % 2:
        raise ValueError("2D vector quantization requires an even dimension")
    if codebook.ndim != 2 or codebook.shape[-1] != 2:
        raise ValueError("2D vector codebook must have shape [entries, 2]")
    pairs = values.reshape(*values.shape[:-1], -1, 2)
    distances = (
        pairs.unsqueeze(-2) - codebook
    ).square().sum(dim=-1)
    indices = torch.argmin(distances, dim=-1)
    return codebook[indices].reshape_as(values)


def log8_norm(norms: torch.Tensor) -> torch.Tensor:
    log_norms = torch.log2(torch.clamp(norms, min=2.0**-8))
    quantized = torch.clamp(torch.round(log_norms * 16.0), -128.0, 127.0)
    return torch.exp2(quantized / 16.0)


def quantize_direction(
    vectors: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    norm_power: float = 0.0,
) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    stored_norms = log8_norm(norms)
    units = vectors / torch.clamp(norms, min=1e-30)
    rotated = units @ rotation.T
    if codebook.ndim == 1:
        quantized = scalar_quantize_torch(rotated, codebook)
    elif codebook.ndim == 2 and codebook.shape[-1] == 2:
        quantized = vector2_quantize_torch(rotated, codebook)
    else:
        raise ValueError(
            "direction codebook must be scalar [entries] or vector "
            "[entries, 2]"
        )
    decoded_units = quantized @ rotation
    if norm_power:
        decoded_units = decoded_units / torch.clamp(
            torch.linalg.vector_norm(
                decoded_units, dim=-1, keepdim=True
            ).pow(norm_power),
            min=1e-30,
        )
    decoded = decoded_units * stored_norms
    return torch.where(norms > 1e-30, decoded, torch.zeros_like(decoded))


def facq_base_decode(
    states: torch.Tensor,
    context: CodecContext,
    *,
    routing: str = "activation",
    basis: torch.Tensor | None = None,
    inverse_basis: torch.Tensor | None = None,
    sensitivity: torch.Tensor | None = None,
    high_bits: int = 3,
    regular_bits: int = 2,
    high_fraction: float = 0.25,
    channel_whiten: bool = False,
    channel_scale_override: torch.Tensor | None = None,
    direction_norm_power: float = 0.0,
    codebook_override: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode the accepted FACQ 3/2-bit anchored base on [1,H,S,D]."""
    if states.shape[0] != 1:
        raise ValueError("aligned evaluator currently requires batch size one")
    _, heads, sequence, dimension = states.shape
    if sequence % context.page_size:
        raise ValueError("sequence length must be page aligned")
    high_dim = int(round(dimension * high_fraction))
    regular_dim = dimension - high_dim
    if {high_dim, regular_dim} != {
        dimension // 4,
        3 * dimension // 4,
    }:
        raise ValueError("high_fraction must select 1/4 or 3/4 dimensions")
    high_rotation = (
        context.rotations["base_high"]
        if high_dim == dimension // 4
        else context.rotations["base_regular"]
    )
    regular_rotation = (
        context.rotations["base_regular"]
        if regular_dim == 3 * dimension // 4
        else context.rotations["base_high"]
    )
    page_count = sequence // context.page_size
    original = states[0].float()
    source = original
    if basis is not None:
        source = torch.einsum("hsd,hde->hse", source, basis)
    decoded = torch.empty_like(source)
    for head in range(heads):
        vectors = source[head]
        pages = vectors.reshape(
            page_count, context.page_size, dimension
        )
        exact_anchor = pages.mean(dim=1, keepdim=True)
        stored_anchor = exact_anchor.to(torch.float16).float()
        residual = pages - exact_anchor
        if channel_whiten:
            if channel_scale_override is None:
                channel_scale = log8_norm(
                    torch.sqrt(
                        torch.mean(residual.square(), dim=(0, 1))
                    )
                )
            else:
                if channel_scale_override.shape != (heads, dimension):
                    raise ValueError(
                        "channel_scale_override must have shape [heads, dim]"
                    )
                channel_scale = channel_scale_override[head].to(
                    device=residual.device, dtype=torch.float32
                )
                if torch.any(channel_scale <= 0) or torch.any(
                    ~torch.isfinite(channel_scale)
                ):
                    raise ValueError(
                        "frozen channel scales must be finite and positive"
                    )
            quantized_residual = residual / torch.clamp(
                channel_scale, min=1e-30
            )
        else:
            channel_scale = None
            quantized_residual = residual
        if routing == "activation":
            score = torch.mean(vectors * vectors, dim=0)
        elif routing in {"residual", "sensitivity"}:
            score = torch.mean(residual * residual, dim=(0, 1))
            if routing == "sensitivity":
                if sensitivity is None:
                    raise ValueError("sensitivity routing needs weights")
                score = score * sensitivity[head]
        else:
            raise ValueError(f"unsupported routing: {routing}")
        high_indices = torch.topk(
            score, high_dim, sorted=False
        ).indices
        high_mask = torch.zeros(
            dimension, device=states.device, dtype=torch.bool
        )
        high_mask[high_indices] = True
        flat = quantized_residual.reshape(sequence, dimension)
        decoded_flat = torch.empty_like(flat)
        high_codebook = (
            codebook_override["high"][head]
            if codebook_override is not None
            else context.codebooks[(high_dim, high_bits)]
        )
        regular_codebook = (
            codebook_override["regular"][head]
            if codebook_override is not None
            else context.codebooks[(regular_dim, regular_bits)]
        )
        decoded_flat[:, high_mask] = quantize_direction(
            flat[:, high_mask],
            high_rotation,
            high_codebook,
            norm_power=direction_norm_power,
        )
        decoded_flat[:, ~high_mask] = quantize_direction(
            flat[:, ~high_mask],
            regular_rotation,
            regular_codebook,
            norm_power=direction_norm_power,
        )
        decoded_residual = decoded_flat.reshape_as(residual)
        if channel_scale is not None:
            decoded_residual = decoded_residual * channel_scale
        decoded_residual -= decoded_residual.mean(dim=1, keepdim=True)
        decoded[head] = (stored_anchor + decoded_residual).reshape(
            sequence, dimension
        )
    if inverse_basis is not None:
        decoded = torch.einsum(
            "hsd,hde->hse", decoded, inverse_basis
        )
    if context.first_token_fp16:
        decoded[:, 0] = original[:, 0]
    return decoded.unsqueeze(0).to(dtype=states.dtype)


def build_value_tables(
    attention: nn.Module,
    variant: str,
) -> dict[str, torch.Tensor]:
    """Build per-head model-side transforms from the frozen output projection."""
    if variant in {
        "activation",
        "residual",
        "whiten",
        "frozen_whiten",
        "unit_direction",
    }:
        return {}
    heads = attention.num_heads
    dimension = attention.head_dim
    weight = attention.o_proj.weight.detach().float()
    blocks = weight.view(
        weight.shape[0], heads, dimension
    ).permute(1, 0, 2)
    gram = torch.matmul(blocks.transpose(1, 2), blocks)
    if variant == "o_diag":
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        diagonal = diagonal / torch.clamp(
            diagonal.mean(dim=-1, keepdim=True), min=1e-12
        )
        return {"sensitivity": diagonal}

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    mean = torch.clamp(
        eigenvalues.mean(dim=-1, keepdim=True), min=1e-12
    )
    normalized = eigenvalues / mean
    normalized = torch.clamp(normalized, min=1e-4)
    if variant == "o_eigen":
        return {
            "basis": eigenvectors,
            "inverse_basis": eigenvectors.transpose(-2, -1),
            "sensitivity": normalized,
        }
    if variant == "o_sqrt":
        square_root = torch.sqrt(normalized)
        basis = eigenvectors * square_root.unsqueeze(-2)
        inverse = (
            eigenvectors / square_root.unsqueeze(-2)
        ).transpose(-2, -1)
        return {
            "basis": basis,
            "inverse_basis": inverse,
        }
    raise ValueError(f"unsupported value variant: {variant}")


def facq_value_decode(
    states: torch.Tensor,
    context: CodecContext,
    variant: str,
    tables: dict[str, torch.Tensor],
    high_bits: int = 3,
    regular_bits: int = 2,
    high_fraction: float = 0.25,
    direction_norm_power: float = 1.0,
) -> torch.Tensor:
    if variant == "activation":
        return facq_base_decode(
            states,
            context,
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
        )
    if variant == "whiten":
        return facq_base_decode(
            states,
            context,
            routing="residual",
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
            channel_whiten=True,
        )
    if variant == "frozen_whiten":
        return facq_base_decode(
            states,
            context,
            routing="activation",
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
            channel_whiten=True,
            channel_scale_override=tables["channel_scale"],
        )
    if variant == "unit_direction":
        codebook_override = None
        if "codebook_high" in tables:
            codebook_override = {
                "high": tables["codebook_high"],
                "regular": tables["codebook_regular"],
            }
        return facq_base_decode(
            states,
            context,
            routing="activation",
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
            direction_norm_power=direction_norm_power,
            codebook_override=codebook_override,
        )
    if variant == "residual":
        return facq_base_decode(
            states,
            context,
            routing="residual",
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
        )
    if variant == "o_diag":
        return facq_base_decode(
            states,
            context,
            routing="sensitivity",
            sensitivity=tables["sensitivity"],
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
        )
    if variant == "o_eigen":
        return facq_base_decode(
            states,
            context,
            routing="sensitivity",
            basis=tables["basis"],
            inverse_basis=tables["inverse_basis"],
            sensitivity=tables["sensitivity"],
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
        )
    if variant == "o_sqrt":
        return facq_base_decode(
            states,
            context,
            routing="residual",
            basis=tables["basis"],
            inverse_basis=tables["inverse_basis"],
            high_bits=high_bits,
            regular_bits=regular_bits,
            high_fraction=high_fraction,
        )
    raise ValueError(f"unsupported value variant: {variant}")


def upward_fp16_tensor(values: torch.Tensor) -> torch.Tensor:
    if torch.any(values < 0) or torch.any(~torch.isfinite(values)):
        raise ValueError("upward FP16 rounding expects finite nonnegative values")
    stored = values.to(torch.float16)
    needs_upward = stored.float() < values
    # All inputs are nonnegative.  For a finite positive IEEE binary16 value,
    # incrementing its unsigned bit pattern by one is exactly nextafter(+inf).
    # int16 arithmetic has the same low 16-bit result below +inf and works on
    # CUDA, where torch.nextafter is not implemented for Half.
    upward = (stored.view(torch.int16) + 1).view(torch.float16)
    return torch.where(needs_upward, upward, stored).float()


def conservative_group_directory(
    error: torch.Tensor,
    page_size: int,
    groups: int,
    first_token_fp16: bool,
) -> torch.Tensor:
    """Return decoded conservative radii on [1,H,S,G]."""
    batch, heads, sequence, dimension = error.shape
    if batch != 1 or sequence % page_size or dimension % groups:
        raise ValueError("invalid directory shape")
    page_count = sequence // page_size
    group_dim = dimension // groups
    exact = torch.linalg.vector_norm(
        error.float().reshape(
            heads, page_count, page_size, groups, group_dim
        ),
        dim=-1,
    )
    scale = upward_fp16_tensor(exact.amax(dim=2, keepdim=True))
    codes = torch.where(
        scale > 0,
        torch.ceil(exact / torch.clamp(scale, min=1e-30) * 15.0),
        torch.zeros_like(exact),
    ).clamp_(0, 15)
    decoded = torch.maximum(codes * (scale / 15.0), exact)
    decoded = decoded.reshape(1, heads, sequence, groups)
    if first_token_fp16:
        decoded[:, :, 0] = 0
    checked_exact = exact.reshape_as(decoded)
    if first_token_fp16:
        checked_exact = checked_exact.clone()
        checked_exact[:, :, 0] = 0
    if torch.any(decoded + 1e-7 < checked_exact):
        raise RuntimeError("group directory is not conservative")
    return decoded


def side_reconstruction_norms(
    directory: torch.Tensor,
    page_size: int,
    mode: str,
) -> torch.Tensor:
    """Derive reconstruction radii while retaining conservative selection."""
    if mode == "upper":
        return directory
    if mode not in {"q25", "midpoint"}:
        raise ValueError(f"unsupported side norm reconstruction: {mode}")
    batch, heads, sequence, groups = directory.shape
    if sequence % page_size:
        raise ValueError("directory sequence must be page aligned")
    pages = directory.reshape(
        batch, heads, sequence // page_size, page_size, groups
    )
    page_scale = pages.amax(dim=3, keepdim=True)
    fraction = 0.25 if mode == "q25" else 0.5
    interior = torch.clamp(
        pages - fraction * (page_scale / 15.0), min=0.0
    )
    interior = torch.where(pages > 0, interior, torch.zeros_like(interior))
    return interior.reshape_as(directory)


def residual_side_decode(
    error: torch.Tensor,
    reconstruction_norms: torch.Tensor,
    bits: int,
    context: CodecContext,
    direction_norm_power: float = 0.0,
) -> torch.Tensor:
    batch, heads, sequence, dimension = error.shape
    group_dim = dimension // context.groups
    decoded = torch.zeros_like(error, dtype=torch.float32)
    source = error.float()
    for group in range(context.groups):
        start = group * group_dim
        end = start + group_dim
        current = source[..., start:end]
        exact_norm = torch.linalg.vector_norm(
            current, dim=-1, keepdim=True
        )
        unit = current / torch.clamp(exact_norm, min=1e-30)
        rotation = context.rotations["side"][group]
        quantized = scalar_quantize_torch(
            unit @ rotation.T,
            context.codebooks[(group_dim, bits)],
        )
        decoded_direction = quantized @ rotation
        if direction_norm_power:
            decoded_direction = decoded_direction / torch.clamp(
                torch.linalg.vector_norm(
                    decoded_direction, dim=-1, keepdim=True
                ).pow(direction_norm_power),
                min=1e-30,
            )
        reconstructed = decoded_direction * reconstruction_norms[
            ..., group, None
        ]
        decoded[..., start:end] = torch.where(
            exact_norm > 1e-30,
            reconstructed,
            torch.zeros_like(reconstructed),
        )
    if context.first_token_fp16:
        decoded[:, :, 0] = 0
    return decoded.to(error.dtype)


def overwrite_same_page_logits(
    logits: torch.Tensor,
    query: torch.Tensor,
    exact_keys: torch.Tensor,
    query_start: int,
    page_size: int,
    scale: float,
) -> None:
    query_end = query_start + query.shape[-2]
    first_page = query_start // page_size
    last_page = (query_end - 1) // page_size
    for page in range(first_page, last_page + 1):
        global_start = page * page_size
        global_end = global_start + page_size
        local_start = max(global_start, query_start) - query_start
        local_end = min(global_end, query_end) - query_start
        if local_start >= local_end:
            continue
        exact = torch.matmul(
            query[:, :, local_start:local_end],
            exact_keys[:, :, global_start:global_end].transpose(2, 3),
        ) * scale
        logits[:, :, local_start:local_end, global_start:global_end] = exact


def exact_same_page_value_correction(
    output: torch.Tensor,
    weights: torch.Tensor,
    value_error: torch.Tensor,
    query_start: int,
    page_size: int,
) -> None:
    query_end = query_start + output.shape[-2]
    first_page = query_start // page_size
    last_page = (query_end - 1) // page_size
    for page in range(first_page, last_page + 1):
        global_start = page * page_size
        global_end = global_start + page_size
        local_start = max(global_start, query_start) - query_start
        local_end = min(global_end, query_end) - query_start
        if local_start >= local_end:
            continue
        correction = torch.matmul(
            weights[
                :,
                :,
                local_start:local_end,
                global_start:global_end,
            ],
            value_error[:, :, global_start:global_end],
        )
        output[:, :, local_start:local_end] += correction


def facq_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, Any]:
    context: CodecContext = self._facq_context
    spec = MODE_SPECS[context.mode]
    if past_key_value is not None or use_cache:
        raise ValueError("the aligned PPL simulator requires use_cache=False")
    batch, query_length, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, exact_keys = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if spec.get("k_path", "facq") == "fp16":
        base_keys = exact_keys
        corrected_keys = exact_keys
        directory = None
    else:
        base_keys = facq_base_decode(exact_keys, context)
        key_error = exact_keys - base_keys
        directory = conservative_group_directory(
            key_error,
            context.page_size,
            context.groups,
            context.first_token_fp16,
        )
        side_norms = side_reconstruction_norms(
            directory,
            context.page_size,
            str(spec.get("side_norm_mode", "upper")),
        )
        if spec["side_bits"]:
            side_token_fraction = float(
                spec.get("side_token_fraction", 1.0)
            )
            if not 0.0 < side_token_fraction <= 1.0:
                raise ValueError(
                    "side_token_fraction must be in (0, 1]"
                )
            if side_token_fraction < 1.0:
                page_count = query_length // context.page_size
                tokens_per_page = max(
                    1,
                    int(
                        round(
                            context.page_size
                            * side_token_fraction
                        )
                    ),
                )
                token_error_square = key_error.float().square()
                side_token_score = spec.get(
                    "side_token_score", "residual"
                )
                if side_token_score == "residual":
                    token_energy = torch.mean(
                        token_error_square, dim=-1
                    )
                elif side_token_score == "query_diag":
                    query_square = query_states.float().square()
                    query_diagonal = query_square.reshape(
                        batch,
                        self.num_key_value_heads,
                        self.num_key_value_groups,
                        query_length,
                        self.head_dim,
                    ).mean(dim=(2, 3))
                    token_energy = torch.mean(
                        token_error_square
                        * query_diagonal[:, :, None, :],
                        dim=-1,
                    )
                else:
                    raise ValueError(
                        f"unsupported side_token_score: "
                        f"{side_token_score}"
                    )
                token_energy = token_energy.reshape(
                    batch,
                    self.num_key_value_heads,
                    page_count,
                    context.page_size,
                )
                stored_token_indices = torch.topk(
                    token_energy,
                    tokens_per_page,
                    dim=-1,
                    sorted=False,
                ).indices
                stored_token_mask = torch.zeros_like(
                    token_energy, dtype=torch.bool
                )
                stored_token_mask.scatter_(
                    -1, stored_token_indices, True
                )
                directory = directory * stored_token_mask.reshape(
                    batch,
                    self.num_key_value_heads,
                    query_length,
                    1,
                )
                side_norms = side_norms * stored_token_mask.reshape(
                    batch,
                    self.num_key_value_heads,
                    query_length,
                    1,
                )
            side_groups = int(
                spec.get("side_groups", context.groups)
            )
            if not 0 < side_groups <= context.groups:
                raise ValueError("side_groups must be in [1, groups]")
            if side_groups < context.groups:
                group_dim = self.head_dim // context.groups
                group_energy = torch.mean(
                    key_error.float()
                    .reshape(
                        batch,
                        self.num_key_value_heads,
                        query_length,
                        context.groups,
                        group_dim,
                    )
                    .square(),
                    dim=(2, 4),
                )
                stored_group_indices = torch.topk(
                    group_energy, side_groups, dim=-1, sorted=False
                ).indices
                stored_group_mask = torch.zeros_like(
                    group_energy, dtype=torch.bool
                )
                stored_group_mask.scatter_(
                    -1, stored_group_indices, True
                )
                directory = directory * stored_group_mask[
                    :, :, None, :
                ]
                side_norms = side_norms * stored_group_mask[
                    :, :, None, :
                ]
            corrected_keys = base_keys + residual_side_decode(
                key_error,
                side_norms,
                spec["side_bits"],
                context,
                direction_norm_power=float(
                    spec.get(
                        "side_direction_norm_power",
                        int(
                            bool(
                                spec.get(
                                    "side_direction_renorm", False
                                )
                            )
                        ),
                    )
                ),
            ) * float(spec.get("side_gain", 1.0))
        else:
            corrected_keys = base_keys

    if spec["v_path"] == "base":
        variant = spec.get("v_variant", "activation")
        base_values = facq_value_decode(
            value_states,
            context,
            variant,
            self._facq_value_tables,
            high_bits=int(spec.get("v_high_bits", 3)),
            regular_bits=int(spec.get("v_regular_bits", 2)),
            high_fraction=float(spec.get("v_high_fraction", 0.25)),
            direction_norm_power=float(
                spec.get("v_direction_norm_power", 1.0)
            ),
        )
        value_error = value_states - base_values
    else:
        base_values = value_states
        value_error = None

    exact_keys = repeat_kv(exact_keys, self.num_key_value_groups)
    base_keys = repeat_kv(base_keys, self.num_key_value_groups)
    corrected_keys = repeat_kv(
        corrected_keys, self.num_key_value_groups
    )
    if directory is not None:
        directory = repeat_kv(
            directory.transpose(2, 3), self.num_key_value_groups
        )
        directory = directory.transpose(2, 3)
    base_values = repeat_kv(base_values, self.num_key_value_groups)
    if value_error is not None:
        value_error = repeat_kv(value_error, self.num_key_value_groups)

    scale = 1.0 / math.sqrt(self.head_dim)
    outputs: list[torch.Tensor] = []
    returned_weights: list[torch.Tensor] = []
    for query_start in range(0, query_length, context.query_chunk):
        query_end = min(query_start + context.query_chunk, query_length)
        query = query_states[:, :, query_start:query_end]
        logits = torch.matmul(query, base_keys.transpose(2, 3)) * scale
        overwrite_same_page_logits(
            logits,
            query,
            exact_keys,
            query_start,
            context.page_size,
            scale,
        )
        if attention_mask is not None:
            mask = attention_mask[
                :, :, query_start:query_end, : key_states.shape[-2]
            ]
        else:
            mask = None

        selection = spec["selection"]
        if selection == "full":
            has_side = directory.float().abs().sum(dim=-1) > 0
            query_pages = (
                torch.arange(
                    query_start,
                    query_end,
                    device=hidden_states.device,
                )
                // context.page_size
            )
            key_pages = (
                torch.arange(
                    query_length, device=hidden_states.device
                )
                // context.page_size
            )
            prior_page = key_pages[None, :] < query_pages[:, None]
            logical_full_reads = (
                has_side[:, :, None, :]
                & prior_page[None, None, :, :]
            )
            context.read_stats.update(
                logical_full_reads, logical_full_reads
            )
            logits = torch.matmul(
                query, corrected_keys.transpose(2, 3)
            ) * scale
            overwrite_same_page_logits(
                logits,
                query,
                exact_keys,
                query_start,
                context.page_size,
                scale,
            )
        elif selection == "topk":
            group_dim = self.head_dim // context.groups
            query_norms = torch.linalg.vector_norm(
                query.float().reshape(
                    batch,
                    self.num_heads,
                    query_end - query_start,
                    context.groups,
                    group_dim,
                ),
                dim=-1,
            )
            radius = torch.einsum(
                "bhqg,bhkg->bhqk", query_norms, directory.float()
            ) * scale
            query_pages = (
                torch.arange(
                    query_start,
                    query_end,
                    device=hidden_states.device,
                )
                // context.page_size
            )
            key_pages = (
                torch.arange(
                    query_length, device=hidden_states.device
                )
                // context.page_size
            )
            same_page = query_pages[:, None] == key_pages[None, :]
            radius.masked_fill_(same_page[None, None], 0.0)
            if context.first_token_fp16:
                radius[..., 0] = 0.0
            lower = logits.float() - radius
            upper = logits.float() + radius
            if mask is not None:
                lower = lower + mask.float()
                upper = upper + mask.float()
            threshold = lower.amax(dim=-1, keepdim=True)
            feasible = upper >= threshold
            needs_side = feasible & (radius > 0)
            topk = min(int(spec["topk"]), query_length)
            scores = upper.masked_fill(~needs_side, -torch.inf)
            _, indices = torch.topk(scores, topk, dim=-1)
            valid = torch.gather(needs_side, -1, indices)
            selected = torch.zeros_like(needs_side)
            selected.scatter_(-1, indices, valid)
            corrected_logits = torch.matmul(
                query, corrected_keys.transpose(2, 3)
            ) * scale
            logits = torch.where(
                selected,
                corrected_logits,
                logits,
            )
            context.read_stats.update(selected, needs_side)

        if mask is not None:
            logits = logits + mask
        weights = F.softmax(
            logits, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        weights = F.dropout(
            weights, p=self.attention_dropout, training=self.training
        )
        output = torch.matmul(weights, base_values)
        value_observer = getattr(self, "_facq_value_observer", None)
        if value_observer is not None and value_error is not None:
            value_observer.observe(
                attention=self,
                weights=weights,
                value_error=value_error,
                base_values=base_values,
                query_start=query_start,
            )
        value_controller = getattr(self, "_facq_value_controller", None)
        if value_controller is not None and value_error is not None:
            output += value_controller.correction(
                attention=self,
                weights=weights,
                value_error=value_error,
                base_values=base_values,
                query_start=query_start,
            )
        if value_error is not None:
            exact_same_page_value_correction(
                output,
                weights,
                value_error,
                query_start,
                context.page_size,
            )
        outputs.append(output)
        if output_attentions:
            returned_weights.append(weights)

    attention_output = torch.cat(outputs, dim=2)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(
        batch, query_length, self.hidden_size
    )
    attention_output = self.o_proj(attention_output)
    all_weights = (
        torch.cat(returned_weights, dim=2) if output_attentions else None
    )
    return attention_output, all_weights, past_key_value


def bind_attention(
    attention: nn.Module,
    mode: str,
    context: CodecContext | None,
) -> None:
    if mode == "fp16":
        attention.forward = types.MethodType(
            type(attention).forward, attention
        )
        if hasattr(attention, "_facq_context"):
            del attention._facq_context
        if hasattr(attention, "_facq_value_tables"):
            del attention._facq_value_tables
        return
    attention._facq_context = context
    spec = MODE_SPECS[mode]
    attention._facq_value_tables = (
        build_value_tables(
            attention, spec.get("v_variant", "activation")
        )
        if spec["v_path"] == "base"
        else {}
    )
    if spec.get("v_variant") == "frozen_whiten":
        if context is None or context.value_calibration is None:
            raise ValueError(
                f"{mode} requires --value-calibration-file"
            )
        layer_index = int(attention.layer_idx)
        attention._facq_value_tables["channel_scale"] = (
            context.value_calibration[layer_index].to(
                device=attention.o_proj.weight.device,
                dtype=torch.float32,
            )
        )
    objective = spec.get("v_codebook_objective")
    if objective is not None:
        if context is None or context.value_codebooks is None:
            raise ValueError(
                f"{mode} requires --value-codebook-file"
            )
        if objective not in context.value_codebooks:
            raise ValueError(
                f"value codebook artifact lacks objective {objective}"
            )
        layer_index = int(attention.layer_idx)
        learned = context.value_codebooks[objective]
        attention._facq_value_tables["codebook_high"] = learned[
            "high"
        ][layer_index].to(
            device=attention.o_proj.weight.device,
            dtype=torch.float32,
        )
        attention._facq_value_tables["codebook_regular"] = learned[
            "regular"
        ][layer_index].to(
            device=attention.o_proj.weight.device,
            dtype=torch.float32,
        )
    attention.forward = types.MethodType(
        facq_attention_forward, attention
    )


def capture_initial_hidden_states(
    model: nn.Module,
    token_ids: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    layers = model.model.layers
    windows = token_ids.numel() // sequence_length
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(model.parameters()).dtype
    inps = torch.empty(
        windows,
        sequence_length,
        model.config.hidden_size,
        dtype=dtype,
        device=device,
    )
    cache: dict[str, Any] = {"index": 0}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, hidden: torch.Tensor, **kwargs: Any) -> None:
            inps[cache["index"]].copy_(hidden[0])
            cache["index"] += 1
            for key in (
                "attention_mask",
                "position_ids",
                "position_embeddings",
                "cache_position",
            ):
                if key in kwargs:
                    cache[key] = kwargs[key]
            raise StopIteration

    layers[0] = Catcher(layers[0])
    for window in range(windows):
        batch = token_ids[
            :,
            window * sequence_length : (window + 1) * sequence_length,
        ].to(device)
        try:
            model(batch, use_cache=False)
        except StopIteration:
            pass
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    for key, value in list(cache.items()):
        if torch.is_tensor(value):
            cache[key] = value.cpu()
        elif isinstance(value, tuple):
            cache[key] = tuple(item.cpu() for item in value)
    return inps.cpu(), cache


def layer_kwargs(
    cache: dict[str, Any],
    model: nn.Module,
    hidden: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in ("attention_mask", "position_ids", "cache_position"):
        value = cache.get(key)
        if value is not None:
            kwargs[key] = value.to(device)
    position_embeddings = cache.get("position_embeddings")
    if position_embeddings is not None:
        kwargs["position_embeddings"] = tuple(
            item.to(device) for item in position_embeddings
        )
    elif "position_ids" in kwargs and hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        kwargs["position_embeddings"] = model.model.rotary_emb(
            hidden, kwargs["position_ids"]
        )
    return kwargs


def storage_accounting(
    sequence_length: int,
    head_dim: int,
    page_size: int,
    groups: int,
    mode: str,
    first_token_fp16: bool,
) -> dict[str, float]:
    spec = MODE_SPECS[mode]
    if mode == "fp16":
        return {
            "k_stored_bits_per_dimension": 16.0,
            "v_stored_bits_per_dimension": 16.0,
            "kv_stored_bits_per_dimension": 16.0,
        }
    completed_pages = max(sequence_length // page_size - 1, 0)
    fp16_tokens = page_size + int(first_token_fp16)
    low_tokens = sequence_length - page_size - int(first_token_fp16)
    if low_tokens < 0:
        raise ValueError("sequence is shorter than the active page")

    if spec.get("k_path", "facq") == "fp16":
        k_bits = 16.0
    else:
        k_bit_dimensions = float(fp16_tokens * 16 * head_dim)
        k_bit_dimensions += low_tokens * 2.25 * head_dim
        k_bit_dimensions += low_tokens * 16
        k_bit_dimensions += completed_pages * 16 * head_dim
        k_bit_dimensions += low_tokens * 4 * groups
        k_bit_dimensions += completed_pages * 16 * groups
        side_fraction = (
            float(spec.get("side_groups", groups)) / groups
        )
        side_fraction *= float(
            spec.get("side_token_fraction", 1.0)
        )
        k_bit_dimensions += (
            low_tokens
            * int(spec["side_bits"])
            * head_dim
            * side_fraction
        )
        k_bit_dimensions += (
            low_tokens
            * int(spec.get("side_token_bitmap_bits", 0))
        )
        k_bits = k_bit_dimensions / (sequence_length * head_dim)

    if spec["v_path"] == "fp16":
        v_bits = 16.0
    else:
        high_bits = int(spec.get("v_high_bits", 3))
        regular_bits = int(spec.get("v_regular_bits", 2))
        high_fraction = float(spec.get("v_high_fraction", 0.25))
        code_bits = (
            high_fraction * high_bits
            + (1.0 - high_fraction) * regular_bits
        )
        v_bit_dimensions = float(fp16_tokens * 16 * head_dim)
        v_bit_dimensions += low_tokens * code_bits * head_dim
        v_bit_dimensions += low_tokens * 16
        v_bit_dimensions += completed_pages * 16 * head_dim
        v_bit_dimensions += (
            int(spec.get("v_channel_scale_bits", 0)) * head_dim
        )
        v_bits = v_bit_dimensions / (sequence_length * head_dim)
    return {
        "k_stored_bits_per_dimension": k_bits,
        "v_stored_bits_per_dimension": v_bits,
        "kv_stored_bits_per_dimension": 0.5 * (k_bits + v_bits),
    }


@torch.no_grad()
def evaluate_mode(
    model: nn.Module,
    initial_inps: torch.Tensor,
    token_ids: torch.Tensor,
    cache: dict[str, Any],
    mode: str,
    context: CodecContext | None,
    device: torch.device,
    sequence_length: int,
) -> dict[str, Any]:
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inps = initial_inps.to(device)
    outs = torch.empty_like(inps)
    windows = inps.shape[0]
    layers = model.model.layers
    for layer_index in range(len(layers)):
        layer_start = time.time()
        layer = layers[layer_index].to(device)
        bind_attention(layer.self_attn, mode, context)
        for window in range(windows):
            hidden = inps[window].unsqueeze(0)
            outs[window].copy_(
                layer(
                    hidden,
                    **layer_kwargs(cache, model, hidden, device),
                    use_cache=False,
                )[0][0]
            )
        if hasattr(layer.self_attn, "_facq_value_tables"):
            del layer.self_attn._facq_value_tables
        layers[layer_index] = layer.cpu()
        del layer
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "mode": mode,
                    "layer": layer_index,
                    "seconds": round(time.time() - layer_start, 3),
                }
            ),
            flush=True,
        )

    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    labels = token_ids.to(device)
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    eval_tokens = 0
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    for window in range(windows):
        hidden = model.model.norm(inps[window].unsqueeze(0))
        logits = model.lm_head(hidden)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[
            :,
            window * sequence_length + 1 : (window + 1) * sequence_length,
        ]
        # FP16 reduction quantizes each 4095-token window loss to coarse
        # integer steps. Cast before CE so paired deltas and final PPL use
        # FP32 logits.
        nll_sum += loss_function(
            shift_logits.float().view(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
        ).double()
        eval_tokens += int(shift_labels.numel())
    nll_value = float(nll_sum.cpu())
    result = {
        "mode": mode,
        "ppl": float(math.exp(nll_value / eval_tokens)),
        "nll_sum": nll_value,
        "eval_tokens": eval_tokens,
        "evaluated_windows": windows,
        "elapsed_seconds": time.time() - start_time,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    if context is not None:
        stats = asdict(context.read_stats)
        histogram = stats.pop("selected_count_histogram")
        result["read_stats"] = stats
        if stats["query_heads"]:
            result["mean_side_vectors_read_per_query_head"] = (
                stats["selected_side_vectors"] / stats["query_heads"]
            )
            result["mean_feasible_side_vectors_per_query_head"] = (
                stats["feasible_side_vectors"] / stats["query_heads"]
            )
            result["p50_side_vectors_read_per_query_head"] = (
                read_count_percentile(histogram, 0.50)
            )
            result["p90_side_vectors_read_per_query_head"] = (
                read_count_percentile(histogram, 0.90)
            )
            result["p99_side_vectors_read_per_query_head"] = (
                read_count_percentile(histogram, 0.99)
            )
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    del inps, outs
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.sequence_length % args.page_size:
        raise ValueError("sequence length must be divisible by page size")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.quant_seed)
    np.random.seed(args.quant_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    load_max_windows = args.max_windows
    if (
        args.dataset == "c4"
        and args.max_windows > 0
        and args.window_start > 0
    ):
        load_max_windows = args.window_start + args.max_windows
    token_ids, available_windows = load_eval_tokens(
        tokenizer,
        Path(args.dataset_root),
        args.dataset,
        args.sequence_length,
        load_max_windows,
    )
    window_start = args.window_start
    window_end = available_windows
    if args.max_windows > 0:
        window_end = min(window_end, window_start + args.max_windows)
    if window_start < 0 or window_start >= window_end:
        raise ValueError("invalid selected window range")
    token_ids = token_ids[
        :,
        window_start
        * args.sequence_length : window_end
        * args.sequence_length,
    ].contiguous()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    if model.config.model_type != "llama":
        raise NotImplementedError(model.config.model_type)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    value_calibration = None
    calibration_provenance = None
    if args.value_calibration_file:
        calibration_path = Path(args.value_calibration_file).resolve()
        artifact = torch.load(
            calibration_path, map_location="cpu", weights_only=False
        )
        value_calibration = artifact["scales"].float()
        expected_shape = (
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            head_dim,
        )
        if tuple(value_calibration.shape) != expected_shape:
            raise ValueError(
                f"calibration shape {tuple(value_calibration.shape)} "
                f"does not match {expected_shape}"
            )
        if torch.any(value_calibration <= 0) or torch.any(
            ~torch.isfinite(value_calibration)
        ):
            raise ValueError("calibration scales must be finite and positive")
        calibration_provenance = {
            "path": str(calibration_path),
            "sha256": hashlib.sha256(
                calibration_path.read_bytes()
            ).hexdigest(),
            "metadata": artifact.get("metadata", {}),
        }
    frozen_requested = any(
        MODE_SPECS[mode].get("v_variant") == "frozen_whiten"
        for mode in args.modes
    )
    if frozen_requested and value_calibration is None:
        raise ValueError(
            "frozen_whiten modes require --value-calibration-file"
        )
    value_codebooks = None
    codebook_provenance = None
    if args.value_codebook_file:
        codebook_path = Path(args.value_codebook_file).resolve()
        codebook_artifact = torch.load(
            codebook_path, map_location="cpu", weights_only=False
        )
        value_codebooks = codebook_artifact["codebooks"]
        for objective, learned in value_codebooks.items():
            vector2 = objective.endswith("vector2")
            expected_shapes = {
                "high": (
                    model.config.num_hidden_layers,
                    model.config.num_key_value_heads,
                    64,
                    2,
                )
                if vector2
                else (
                    model.config.num_hidden_layers,
                    model.config.num_key_value_heads,
                    8,
                ),
                "regular": (
                    model.config.num_hidden_layers,
                    model.config.num_key_value_heads,
                    16,
                    2,
                )
                if vector2
                else (
                    model.config.num_hidden_layers,
                    model.config.num_key_value_heads,
                    4,
                ),
            }
            for tier, expected_shape in expected_shapes.items():
                table = learned[tier].float()
                if tuple(table.shape) != expected_shape:
                    raise ValueError(
                        f"{objective}/{tier} shape {tuple(table.shape)} "
                        f"does not match {expected_shape}"
                    )
                invalid = bool(torch.any(~torch.isfinite(table)))
                if not vector2:
                    invalid = invalid or bool(
                        torch.any(table[..., 1:] <= table[..., :-1])
                    )
                if invalid:
                    raise ValueError(
                        f"{objective}/{tier} must be finite"
                        + (" and ordered" if not vector2 else "")
                    )
                learned[tier] = table
        codebook_provenance = {
            "path": str(codebook_path),
            "sha256": hashlib.sha256(
                codebook_path.read_bytes()
            ).hexdigest(),
            "metadata": codebook_artifact.get("metadata", {}),
        }
    learned_requested = any(
        MODE_SPECS[mode].get("v_codebook_objective") is not None
        for mode in args.modes
    )
    if learned_requested and value_codebooks is None:
        raise ValueError(
            "learned V codebook modes require --value-codebook-file"
        )
    codebooks, rotations = make_codec_tables(
        head_dim, args.groups, args.quant_seed
    )
    codebooks, rotations = move_codec_tables(
        codebooks, rotations, device
    )

    print(
        json.dumps(
            {
                "event": "capture_start",
                "dataset": args.dataset,
                "available_windows": available_windows,
                "window_start": window_start,
                "window_end": window_end,
            }
        ),
        flush=True,
    )
    initial_inps, cache = capture_initial_hidden_states(
        model, token_ids, args.sequence_length, device
    )
    provenance = {
        "model": str(Path(args.model).resolve()),
        "dataset": args.dataset,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "sequence_length": args.sequence_length,
        "available_windows": available_windows,
        "window_start": window_start,
        "window_end": window_end,
        "evaluated_windows": int(initial_inps.shape[0]),
        "page_size": args.page_size,
        "groups": args.groups,
        "quant_seed": args.quant_seed,
        "query_chunk": args.query_chunk,
        "first_token_fp16": not args.no_first_token_fp16,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "facq_scope": (
            "post-RoPE K controller plus provisional symmetric V base proxy"
        ),
        "offline_simulation": True,
        "value_calibration": calibration_provenance,
        "value_codebooks": codebook_provenance,
    }
    (output_dir / "config.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    all_results: list[dict[str, Any]] = []
    for mode in args.modes:
        context = None
        if mode != "fp16":
            context = CodecContext(
                mode=mode,
                page_size=args.page_size,
                groups=args.groups,
                quant_seed=args.quant_seed,
                query_chunk=args.query_chunk,
                first_token_fp16=not args.no_first_token_fp16,
                codebooks=codebooks,
                rotations=rotations,
                read_stats=ReadStats(),
                value_calibration=value_calibration,
                value_codebooks=value_codebooks,
            )
        print(
            json.dumps({"event": "mode_start", "mode": mode}),
            flush=True,
        )
        result = evaluate_mode(
            model,
            initial_inps,
            token_ids,
            cache,
            mode,
            context,
            device,
            args.sequence_length,
        )
        result.update(
            storage_accounting(
                args.sequence_length,
                head_dim,
                args.page_size,
                args.groups,
                mode,
                not args.no_first_token_fp16,
            )
        )
        result["side_bits"] = MODE_SPECS[mode]["side_bits"]
        all_results.append(result)
        (output_dir / f"{mode}.json").write_text(
            json.dumps({**provenance, **result}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps({"event": "mode_complete", **result}),
            flush=True,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"provenance": provenance, "results": all_results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
