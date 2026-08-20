"""Deterministic C4-train window manifests for MPE-KV experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from datasets import Dataset


def load_text_dataset(path: Path) -> Dataset:
    """Load a local text dataset without changing its row order.

    Historical MPE manifests point at Hugging Face Arrow files.  The
    GSRQ-aligned LongBench calibration source is a pinned SlimPajama parquet
    shard, so support that container while preserving the same row-index
    manifest contract.
    """
    if path.suffix == ".parquet":
        return Dataset.from_parquet(str(path))
    return Dataset.from_file(str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def token_sha256(window: torch.Tensor) -> str:
    contiguous = window.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def tokenizer_provenance(model_path: Path) -> dict[str, Any]:
    candidates = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "config.json",
    )
    files = []
    for name in candidates:
        path = model_path / name
        if path.exists():
            files.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not files:
        raise FileNotFoundError("no tokenizer/model provenance files found")
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "model_path": str(model_path.resolve()),
        "files": files,
        "combined_sha256": combined,
    }


def intervals_overlap(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    if int(first["row"]) != int(second["row"]):
        return False
    first_start = int(first["token_start"])
    first_end = first_start + int(first["tokens"])
    second_start = int(second["token_start"])
    second_end = second_start + int(second["tokens"])
    return first_start < second_end and second_start < first_end


def collision_rows(
    first: Iterable[dict[str, Any]],
    second: Iterable[dict[str, Any]],
) -> list[dict[str, int]]:
    observed = []
    for first_index, first_window in enumerate(first):
        for second_index, second_window in enumerate(second):
            if intervals_overlap(first_window, second_window):
                observed.append(
                    {
                        "first_index": first_index,
                        "second_index": second_index,
                        "row": int(first_window["row"]),
                    }
                )
    return observed


def sample_windows(
    tokenizer: Any,
    arrow_path: Path,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
    excluded_rows: set[int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, int]]:
    if samples < 1 or sequence_length < 2:
        raise ValueError("samples must be positive and length >= 2")
    data = load_text_dataset(arrow_path)
    generator = random.Random(seed)
    excluded = set() if excluded_rows is None else set(excluded_rows)
    attempted: set[int] = set()
    windows: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    short_rows = 0
    excluded_draws = 0
    duplicate_draws = 0
    while len(windows) < samples:
        if len(attempted) + len(excluded) >= len(data):
            raise RuntimeError("dataset exhausted before enough windows")
        row = generator.randrange(len(data))
        if row in excluded:
            excluded_draws += 1
            continue
        if row in attempted:
            duplicate_draws += 1
            continue
        attempted.add(row)
        encoded = tokenizer(
            data[row]["text"],
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        if encoded.shape[1] < sequence_length:
            short_rows += 1
            continue
        max_start = encoded.shape[1] - sequence_length
        start = generator.randint(0, max_start)
        window = encoded[:, start : start + sequence_length].contiguous()
        windows.append(window)
        manifest.append(
            {
                "window": len(manifest),
                "row": row,
                "token_start": start,
                "tokens": sequence_length,
                "token_sha256": token_sha256(window),
            }
        )
        if len(windows) % 16 == 0 or len(windows) == samples:
            print(
                "__DS_PROGRESS__ "
                + json.dumps(
                    {
                        "event": "manifest_sampling",
                        "seed": seed,
                        "windows": len(windows),
                        "target": samples,
                        "attempted_unique_rows": len(attempted),
                    }
                ),
                flush=True,
            )
    return (
        torch.cat(windows, dim=0),
        manifest,
        {
            "attempted_unique_rows": len(attempted),
            "short_rows": short_rows,
            "excluded_draws": excluded_draws,
            "duplicate_draws": duplicate_draws,
        },
    )


def load_manifest_windows(
    tokenizer: Any,
    manifest_path: Path,
    *,
    start: int = 0,
    count: int | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrow_path = Path(payload["arrow_path"])
    if sha256_file(arrow_path) != payload["arrow_sha256"]:
        raise ValueError("arrow hash differs from manifest")
    rows = payload["windows"]
    end = len(rows) if count is None else min(len(rows), start + count)
    if not 0 <= start < end <= len(rows):
        raise ValueError("manifest slice is empty or out of range")
    selected = rows[start:end]
    data = load_text_dataset(arrow_path)
    contiguous_tokens = None
    if payload.get("container") == "contiguous_text":
        contiguous_tokens = tokenizer(
            "\n\n".join(data["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
    windows = []
    for expected in selected:
        if contiguous_tokens is None:
            encoded = tokenizer(
                data[int(expected["row"])]["text"],
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
        else:
            encoded = contiguous_tokens
        token_start = int(expected["token_start"])
        token_count = int(expected["tokens"])
        window = encoded[:, token_start : token_start + token_count].contiguous()
        if window.shape[1] != token_count:
            raise ValueError("manifest window is shorter than declared")
        observed_hash = token_sha256(window)
        if observed_hash != expected["token_sha256"]:
            raise ValueError("token hash differs from manifest")
        windows.append(window)
    return torch.cat(windows, dim=0), selected, payload
