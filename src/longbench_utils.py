"""Shared utilities for the paper-aligned LongBench runners.

The runners intentionally load the prompt templates, generation limits, and
metrics from a pinned checkout of the official LongBench repository.  Keeping
these helpers independent of any particular cache backend makes the public
artifact self-contained without importing unrelated experimental code.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "qmsum",
    "musique",
    "triviaqa",
    "samsum",
)
QA_TASKS = {
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "musique",
    "triviaqa",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_official_assets(
    source: Path,
) -> tuple[dict[str, str], dict[str, int], Any]:
    prompt_path = source / "config" / "dataset2prompt.json"
    maxlen_path = source / "config" / "dataset2maxlen.json"
    metric_path = source / "metrics.py"
    with prompt_path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    with maxlen_path.open("r", encoding="utf-8") as handle:
        max_lengths = json.load(handle)
    spec = importlib.util.spec_from_file_location(
        "paper_longbench_metrics", metric_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {metric_path}")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return prompts, max_lengths, metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_prompt(
    task: str,
    row: dict[str, Any],
    template: str,
    tokenizer: Any,
    *,
    max_input_length: int,
    prompt_wrapper: str,
) -> tuple[str, torch.Tensor, bool]:
    prompt = template.format(**row)
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    truncated = len(ids) > max_input_length
    if truncated:
        half = max_input_length // 2
        prompt = tokenizer.decode(
            ids[:half], skip_special_tokens=True
        ) + tokenizer.decode(ids[-half:], skip_special_tokens=True)
    use_chat = prompt_wrapper == "llama3_chat" or (
        prompt_wrapper == "llama3_official"
        and task not in {"triviaqa", "samsum"}
    )
    if use_chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    final_ids = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids
    return prompt, final_ids, truncated


def eos_ids(tokenizer: Any, task: str) -> list[int]:
    eos = [int(tokenizer.eos_token_id)]
    if task == "samsum":
        newline = int(tokenizer.encode("\n", add_special_tokens=False)[-1])
        if newline not in eos:
            eos.append(newline)
    return eos


def score_task(task: str, rows: list[dict[str, Any]], metrics: Any) -> float:
    if not rows:
        raise ValueError(f"cannot score empty task {task}")
    metric = metrics.qa_f1_score if task in QA_TASKS else metrics.rouge_score
    total = 0.0
    for row in rows:
        prediction = row["pred"]
        if task in {"triviaqa", "samsum"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        best = 0.0
        for answer in row["answers"]:
            best = max(
                best,
                metric(
                    prediction,
                    answer,
                    all_classes=row["all_classes"],
                ),
            )
        total += best
    return round(100.0 * total / len(rows), 2)
