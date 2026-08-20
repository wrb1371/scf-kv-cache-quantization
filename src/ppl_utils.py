"""Shared dataset and layerwise-forward helpers for PPL experiments."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from datasets import Dataset
import torch
import torch.nn as nn


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
        dataset_root,
        dataset,
        "test" if dataset == "wikitext2" else "validation",
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
    inputs = torch.empty(
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
            inputs[cache["index"]].copy_(hidden[0])
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
    return inputs.cpu(), cache


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
