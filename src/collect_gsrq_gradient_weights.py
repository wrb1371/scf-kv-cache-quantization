#!/usr/bin/env python3
"""Collect final-CE gradients for paper-style weighted GSRQ calibration."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpe_data import atomic_json_save, sha256_file, token_sha256  # noqa: E402


FORMAT = "gsrq_final_ce_projection_gradient_norms_v2"
COVARIANCE_FORMAT = "gsrq_final_ce_projection_gradient_statistics_v3"
ALIGNED_FORMAT = "gsrq_final_ce_projection_aligned_value_gradients_v4"
CQ_GROUP_FORMAT = "cq_final_ce_projection_group_statistics_v1"


def parse_layers(spec: str, total_layers: int) -> list[int]:
    if spec == "all":
        return list(range(total_layers))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    result = sorted(selected)
    if not result or result[0] < 0 or result[-1] >= total_layers:
        raise ValueError("selected layers are out of range")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=4096.0,
        help=(
            "Fixed backward loss scale used to prevent FP16 projection-gradient "
            "underflow; saved norms are divided by this value."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--save-value-covariance",
        action="store_true",
        help=(
            "Also save per-head sum_t (dCE/dv_t)(dCE/dv_t)^T sufficient "
            "statistics. This exact mode requires one selected layer so that "
            "other detached projection hooks cannot truncate its downstream "
            "gradient."
        ),
    )
    parser.add_argument(
        "--save-value-aligned-tensors",
        action="store_true",
        help=(
            "Also save token-aligned native FP16 V-projection activations and "
            "loss-scale-corrected FP32 final-CE gradients. Exact aligned mode "
            "requires exactly one selected layer."
        ),
    )
    parser.add_argument(
        "--save-cq-group-statistics",
        action="store_true",
        help=(
            "Save token-aligned per-group sums of squared K/V projection "
            "gradients and aggregate within-group empirical-Fisher matrices. "
            "This is the sufficient-statistics surface for paper-faithful CQ."
        ),
    )
    parser.add_argument(
        "--cq-group-size",
        type=int,
        default=8,
        help="Contiguous CQ channel-group width (CQ-8c8b uses 8).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-autograd-on-cpu",
        action="store_true",
        help=(
            "Store tensors saved for backward on CPU. This preserves the "
            "forward/backward arithmetic while reducing peak CUDA memory."
        ),
    )
    return parser.parse_args()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_manifest_windows(
    tokenizer: Any,
    manifest_path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(payload.get("calibration_arrow", payload.get("arrow_path", "")))
    source_hash = payload.get(
        "calibration_arrow_sha256", payload.get("arrow_sha256")
    )
    if not source_hash or sha256_file(source) != source_hash:
        raise ValueError("calibration source hash differs")
    data = (
        Dataset.from_parquet(str(source))
        if source.suffix == ".parquet"
        else Dataset.from_file(str(source))
    )
    rows = payload.get("sample_manifest", payload.get("windows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("calibration manifest has no windows")
    has_row = ["row" in expected for expected in rows]
    if any(has_row) and not all(has_row):
        raise ValueError("calibration manifest mixes row and stream windows")
    contiguous_tokens = None
    if not any(has_row):
        contiguous_tokens = tokenizer(
            "\n\n".join(data["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
    windows = []
    for expected in rows:
        if contiguous_tokens is None:
            encoded = tokenizer(
                data[int(expected["row"])] ["text"],
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids
        else:
            encoded = contiguous_tokens
        start = int(expected["token_start"])
        tokens = int(expected["tokens"])
        window = encoded[:, start : start + tokens].contiguous()
        if window.shape[1] != tokens:
            raise ValueError("calibration window is shorter than its manifest")
        if token_sha256(window) != expected["token_sha256"]:
            raise ValueError("calibration token hash differs")
        windows.append(window)
    normalized = {
        **payload,
        "calibration_arrow": str(source),
        "calibration_arrow_sha256": source_hash,
        "sample_manifest": rows,
    }
    return torch.cat(windows, dim=0), normalized


class ProjectionGradientCapture:
    """Inject leaf tensors at K/V projections without changing forward values."""

    def __init__(self, model: torch.nn.Module, layers: list[int]) -> None:
        self.model = model
        self.layers = layers
        self.key_leaves: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        self.value_leaves: dict[int, list[torch.Tensor]] = {
            layer: [] for layer in layers
        }
        self.handles: list[Any] = []

    @staticmethod
    def _leaf_hook(storage: list[torch.Tensor]):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            storage.append(leaf)
            return leaf

        return hook

    def attach(self) -> None:
        if self.handles:
            raise RuntimeError("gradient capture is already attached")
        for layer_index in self.layers:
            attention = self.model.model.layers[layer_index].self_attn
            self.handles.append(
                attention.k_proj.register_forward_hook(
                    self._leaf_hook(self.key_leaves[layer_index])
                )
            )
            self.handles.append(
                attention.v_proj.register_forward_hook(
                    self._leaf_hook(self.value_leaves[layer_index])
                )
            )

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        for storage in (self.key_leaves, self.value_leaves):
            for leaves in storage.values():
                leaves.clear()

    def pop_norms(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        loss_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kinds = []
        for storage in (self.key_leaves, self.value_leaves):
            rows = []
            for layer_index in self.layers:
                leaves = storage[layer_index]
                if len(leaves) != 1 or leaves[0].grad is None:
                    raise ValueError("projection leaf/gradient capture differs")
                # The projection leaf is FP16, so an unscaled mean-token CE
                # backward underflows many small K gradients to exact zero.
                # Fixed loss scaling preserves them through the FP16 backward;
                # divide before computing/storing the FP32 norm.
                gradient = leaves[0].grad.detach().float().div(loss_scale)
                batch, tokens, width = gradient.shape
                if batch != 1 or width != kv_heads * head_dim:
                    raise ValueError("projection gradient geometry differs")
                rows.append(
                    torch.linalg.vector_norm(
                        gradient.reshape(batch, tokens, kv_heads, head_dim),
                        dim=-1,
                    )[0].cpu()
                )
            kinds.append(torch.stack(rows))
        self.clear()
        return kinds[0], kinds[1]

    def pop_statistics(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        loss_scale: float,
        save_value_covariance: bool,
        save_value_aligned_tensors: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Return projection norms and optional full V-gradient covariance."""

        kinds = []
        value_covariances: list[torch.Tensor] = []
        value_activations: list[torch.Tensor] = []
        value_gradients: list[torch.Tensor] = []
        for kind_index, storage in enumerate((self.key_leaves, self.value_leaves)):
            rows = []
            for layer_index in self.layers:
                leaves = storage[layer_index]
                if len(leaves) != 1 or leaves[0].grad is None:
                    raise ValueError("projection leaf/gradient capture differs")
                gradient = leaves[0].grad.detach().float().div(loss_scale)
                batch, tokens, width = gradient.shape
                if batch != 1 or width != kv_heads * head_dim:
                    raise ValueError("projection gradient geometry differs")
                shaped = gradient.reshape(batch, tokens, kv_heads, head_dim)
                rows.append(torch.linalg.vector_norm(shaped, dim=-1)[0].cpu())
                if kind_index == 1 and save_value_covariance:
                    value_covariances.append(
                        torch.einsum("bthd,bthe->hde", shaped, shaped).cpu()
                    )
                if kind_index == 1 and save_value_aligned_tensors:
                    activation = leaves[0].detach().reshape(
                        batch, tokens, kv_heads, head_dim
                    )
                    value_activations.append(activation[0].half().cpu())
                    value_gradients.append(shaped[0].float().cpu())
            kinds.append(torch.stack(rows))
        covariance = (
            torch.stack(value_covariances) if save_value_covariance else None
        )
        aligned_activations = (
            torch.stack(value_activations)
            if save_value_aligned_tensors
            else None
        )
        aligned_gradients = (
            torch.stack(value_gradients)
            if save_value_aligned_tensors
            else None
        )
        self.clear()
        base = (kinds[0], kinds[1], covariance)
        if not save_value_aligned_tensors:
            return base
        return base + (aligned_activations, aligned_gradients)

    def pop_cq_group_statistics(
        self,
        *,
        kv_heads: int,
        head_dim: int,
        group_size: int,
        loss_scale: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return norms, group weights and group empirical-Fisher sums.

        For every token/group, CQ Equation (6) uses
        ``w_jg = ||dL/dA_jg||_2^2``.  The group outer sums are additionally
        retained so the paired M20 child can change only the within-group
        curvature spectrum without reconstructing or storing full gradients.
        """

        if group_size < 1 or head_dim % group_size:
            raise ValueError("CQ group size must divide the head dimension")
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        groups = head_dim // group_size
        for storage in (self.key_leaves, self.value_leaves):
            norm_rows = []
            weight_rows = []
            covariance_rows = []
            for layer_index in self.layers:
                leaves = storage[layer_index]
                if len(leaves) != 1 or leaves[0].grad is None:
                    raise ValueError("projection leaf/gradient capture differs")
                gradient = leaves[0].grad.detach().float().div(loss_scale)
                batch, tokens, width = gradient.shape
                if batch != 1 or width != kv_heads * head_dim:
                    raise ValueError("projection gradient geometry differs")
                shaped = gradient.reshape(
                    batch, tokens, kv_heads, groups, group_size
                )
                norm_rows.append(
                    torch.linalg.vector_norm(
                        shaped.reshape(batch, tokens, kv_heads, head_dim), dim=-1
                    )[0].cpu()
                )
                weight_rows.append(shaped.square().sum(dim=-1)[0].cpu())
                covariance_rows.append(
                    torch.einsum("bthgs,bthge->hgse", shaped, shaped).cpu()
                )
            outputs.append(
                (
                    torch.stack(norm_rows),
                    torch.stack(weight_rows),
                    torch.stack(covariance_rows),
                )
            )
        self.clear()
        key, value = outputs
        return key[0], value[0], key[1], value[1], key[2], value[2]


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir).resolve()
    window_dir = output_dir / "windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("gradient collection requires CUDA")
    if not torch.isfinite(torch.tensor(args.loss_scale)) or args.loss_scale < 1.0:
        raise ValueError("loss scale must be finite and at least one")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, local_files_only=True
    )
    token_matrix, calibration = load_manifest_windows(tokenizer, manifest_path)
    config = json.loads((model_path / "config.json").read_text())
    total_layers = int(config["num_hidden_layers"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["hidden_size"]) // int(config["num_attention_heads"])
    selected_layers = parse_layers(args.layers, total_layers)
    if (
        args.save_value_covariance or args.save_value_aligned_tensors
    ) and len(selected_layers) != 1:
        raise ValueError(
            "exact V-gradient covariance/aligned collection requires exactly one layer"
        )
    if args.save_cq_group_statistics and (
        args.save_value_covariance or args.save_value_aligned_tensors
    ):
        raise ValueError("CQ group statistics cannot be mixed with V-only modes")
    if args.cq_group_size < 1 or head_dim % args.cq_group_size:
        raise ValueError("CQ group size must divide the head dimension")
    artifact_format = (
        CQ_GROUP_FORMAT
        if args.save_cq_group_statistics
        else ALIGNED_FORMAT
        if args.save_value_aligned_tensors
        else COVARIANCE_FORMAT
        if args.save_value_covariance
        else FORMAT
    )
    start = args.start_window
    end = token_matrix.shape[0] if args.count is None else min(
        token_matrix.shape[0], start + args.count
    )
    if not 0 <= start < end <= token_matrix.shape[0]:
        raise ValueError("requested window range is empty")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    model.to(device)
    capture = ProjectionGradientCapture(model, selected_layers)

    first_ids = token_matrix[start : start + 1].to(device)
    with torch.no_grad():
        reference_logits = model(first_ids, use_cache=False).logits.float()
        capture.attach()
        captured_logits = model(first_ids, use_cache=False).logits.float()
        capture.clear()
    maximum_replay_error = float(
        (reference_logits - captured_logits).abs().max().cpu()
    )
    del first_ids, reference_logits, captured_logits
    if maximum_replay_error != 0.0:
        capture.detach()
        raise ValueError("projection hooks changed FP16 forward logits")

    window_metrics = []
    for window in range(start, end):
        output_path = window_dir / f"window_{window:03d}.pt"
        expected_row = calibration["sample_manifest"][window]
        if output_path.exists() and not args.overwrite:
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
            if (
                existing.get("format") != artifact_format
                or existing["token_sha256"] != expected_row["token_sha256"]
                or existing["selected_layers"] != selected_layers
            ):
                raise ValueError("existing gradient window provenance differs")
            print(
                "__DS_PROGRESS__ "
                + json.dumps({"event": "gradient_window_reused", "window": window}),
                flush=True,
            )
            continue
        ids = token_matrix[window : window + 1].to(device)
        autograd_storage = (
            torch.autograd.graph.save_on_cpu(pin_memory=False)
            if args.save_autograd_on_cpu
            else nullcontext()
        )
        with autograd_storage:
            outputs = model(ids, use_cache=False)
            labels = ids[:, 1:]
            loss_sum = F.cross_entropy(
                outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            mean_loss = loss_sum / labels.numel()
            scaled_mean_loss = mean_loss * args.loss_scale
            scaled_mean_loss.backward()
        if args.save_cq_group_statistics:
            (
                key_norms,
                value_norms,
                key_group_weights,
                value_group_weights,
                key_group_covariance,
                value_group_covariance,
            ) = capture.pop_cq_group_statistics(
                kv_heads=kv_heads,
                head_dim=head_dim,
                group_size=args.cq_group_size,
                loss_scale=args.loss_scale,
            )
            value_covariance = None
            value_activations = None
            value_gradients = None
        else:
            statistics = capture.pop_statistics(
                kv_heads=kv_heads,
                head_dim=head_dim,
                loss_scale=args.loss_scale,
                save_value_covariance=args.save_value_covariance,
                save_value_aligned_tensors=args.save_value_aligned_tensors,
            )
            key_group_weights = None
            value_group_weights = None
            key_group_covariance = None
            value_group_covariance = None
        if not args.save_cq_group_statistics and args.save_value_aligned_tensors:
            (
                key_norms,
                value_norms,
                value_covariance,
                value_activations,
                value_gradients,
            ) = statistics
        elif not args.save_cq_group_statistics:
            key_norms, value_norms, value_covariance = statistics
            value_activations = None
            value_gradients = None
        payload = {
            "format": artifact_format,
            "window": window,
            "token_sha256": expected_row["token_sha256"],
            "selected_layers": selected_layers,
            "key_gradient_norms": key_norms,
            "value_gradient_norms": value_norms,
            "mean_nll": float(mean_loss.detach().cpu()),
            "nll_sum": float(loss_sum.detach().cpu()),
            "eval_tokens": int(labels.numel()),
            "loss_scale": args.loss_scale,
        }
        if value_covariance is not None:
            payload["value_gradient_covariance"] = value_covariance
        if key_group_weights is not None:
            payload["key_group_gradient_squared_norms"] = key_group_weights
            payload["value_group_gradient_squared_norms"] = value_group_weights
            payload["key_group_gradient_outer_sum"] = key_group_covariance
            payload["value_group_gradient_outer_sum"] = value_group_covariance
        if value_activations is not None and value_gradients is not None:
            payload["value_projection_activations"] = value_activations
            payload["value_projection_gradients"] = value_gradients
        atomic_torch_save(payload, output_path)
        row = {
            "window": window,
            "mean_nll": payload["mean_nll"],
            "key_raw_median": float(key_norms.median()),
            "value_raw_median": float(value_norms.median()),
            "seconds": time.time() - started,
        }
        window_metrics.append(row)
        del (
            ids,
            outputs,
            labels,
            loss_sum,
            mean_loss,
            scaled_mean_loss,
            key_norms,
            value_norms,
            value_covariance,
            value_activations,
            value_gradients,
            key_group_weights,
            value_group_weights,
            key_group_covariance,
            value_group_covariance,
        )
        torch.cuda.empty_cache()
        print(
            "__DS_PROGRESS__ "
            + json.dumps({"event": "gradient_window_complete", **row}),
            flush=True,
        )

    capture.detach()
    manifest = {
        "format": artifact_format,
        "status": "complete" if start == 0 and end == token_matrix.shape[0] else "partial",
        "model": str(model_path),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "calibration_source_sha256": calibration["calibration_arrow_sha256"],
        "sample_manifest": calibration["sample_manifest"],
        "samples": int(token_matrix.shape[0]),
        "sequence_length": int(token_matrix.shape[1]),
        "selected_layers": selected_layers,
        "total_layers": total_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "loss": "mean next-token cross entropy over positions 0..T-2",
        "loss_scale": args.loss_scale,
        "saved_gradient_scale": 1.0,
        "gradient_surface": "native eager FP16 pre-RoPE k_proj/v_proj outputs",
        "value_covariance_saved": args.save_value_covariance,
        "value_aligned_tensors_saved": args.save_value_aligned_tensors,
        "cq_group_statistics_saved": args.save_cq_group_statistics,
        "cq_group_size": (
            args.cq_group_size if args.save_cq_group_statistics else None
        ),
        "cq_group_weight_formula": (
            "sum_over_group_channels_of_loss_scale_corrected_gradient_squared"
            if args.save_cq_group_statistics
            else None
        ),
        "cq_group_covariance_normalization": (
            "sum_over_token_of_gradient_outer_product"
            if args.save_cq_group_statistics
            else None
        ),
        "value_activation_dtype": (
            "float16" if args.save_value_aligned_tensors else None
        ),
        "value_gradient_dtype": (
            "float32" if args.save_value_aligned_tensors else None
        ),
        "value_covariance_normalization": (
            "sum_over_batch_and_token_of_outer_product_of_saved_scale_gradients"
            if args.save_value_covariance
            else None
        ),
        "layer_isolated_exact_gradient": (
            len(selected_layers) == 1
            if (args.save_value_covariance or args.save_value_aligned_tensors)
            else None
        ),
        "forward_replay_max_abs_logit_error": maximum_replay_error,
        "completed_range": [start, end],
        "window_directory": str(window_dir),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.time() - started,
        "autograd_saved_tensor_storage": (
            "cpu" if args.save_autograd_on_cpu else "cuda"
        ),
    }
    atomic_json_save(manifest, output_dir / "gradient_manifest.json")
    atomic_json_save(window_metrics, output_dir / "window_metrics.json")
    print("__DS_PROGRESS__ " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
