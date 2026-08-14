#!/usr/bin/env python3
"""Render the paper-facing SCF finite-rate motivation diagnostic."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "WSDM2027投稿"
FIG_DIR = PAPER / "figures"
SOURCE_DIR = FIG_DIR / "source_data"
METRIC_PATH = (
    ROOT
    / "FACQ_research_handoff_2026-07-28"
    / "artifacts/experiment/mpe_kv_m6_sf_pareto_20260731/metric/metric.pt"
)
MECHANISM_PATH = (
    ROOT
    / "FACQ_research_handoff_2026-07-28"
    / "artifacts/experiment/scfw_gsrq_m20_sqrt_20260806/diag/mechanism_gate.json"
)
# Editable text is mandatory for the primary SVG export.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


FULL = "#5B5F79"
SCF = "#2F8F83"
SCF_LIGHT = "#B9DDD7"
WORSE = "#B96B5C"
NEUTRAL = "#9A9A9A"
TEXT = "#303238"


def _load_spectra() -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(METRIC_PATH, map_location="cpu", weights_only=False)
    fisher = payload["metrics"]["fisher"].double().numpy()
    matrices = fisher.reshape(-1, fisher.shape[-1], fisher.shape[-1])
    eigenvalues = np.stack([np.linalg.eigvalsh(matrix) for matrix in matrices])
    eigenvalues = np.maximum(eigenvalues, 0.0)
    dimension = eigenvalues.shape[-1]

    full_component = eigenvalues * dimension / eigenvalues.sum(-1, keepdims=True)
    sqrt_component = np.sqrt(full_component)
    sqrt_component *= dimension / sqrt_component.sum(-1, keepdims=True)

    full_metric = 0.5 * (1.0 + full_component)
    scf_metric = 0.5 * (1.0 + sqrt_component)
    full_metric = np.sort(full_metric, axis=-1)[:, ::-1]
    scf_metric = np.sort(scf_metric, axis=-1)[:, ::-1]
    return full_metric, scf_metric


def _load_paper_source_data() -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the path-independent CSV/JSON snapshot shipped with the paper."""

    rows = list(csv.DictReader((SOURCE_DIR / "fig1_spectrum.csv").open()))
    if len(rows) != 128:
        raise ValueError("expected 128 spectrum points in paper source data")
    full_curve = np.asarray([float(row["full_mean"]) for row in rows])
    scf_curve = np.asarray([float(row["scf_mean"]) for row in rows])
    window_rows = list(csv.DictReader((SOURCE_DIR / "fig1_windows.csv").open()))
    mechanism = {
        "windows": [
            {
                "matched": {
                    "reference_forward_kl": float(row["full_attention_kl"]),
                    "reference_logit_mse": float(row["full_logit_mse"]),
                    "fisher_vector_mse": float(row["full_fisher_mse"]),
                    "euclidean_vector_mse": float(row["full_raw_mse"]),
                },
                "scfw": {
                    "reference_forward_kl": float(row["scf_attention_kl"]),
                    "reference_logit_mse": float(row["scf_logit_mse"]),
                    "fisher_vector_mse": float(row["scf_fisher_mse"]),
                    "euclidean_vector_mse": float(row["scf_raw_mse"]),
                },
            }
            for row in window_rows
        ]
    }
    # Plotting below needs only the aggregate curves. Keep a leading singleton
    # dimension so the same quantile code remains valid.
    return full_curve[None, :], scf_curve[None, :], mechanism


def _cumulative_budget(metric: np.ndarray) -> np.ndarray:
    return np.cumsum(metric, axis=-1) / metric.sum(-1, keepdims=True)


def _effective_rank(metric: np.ndarray) -> np.ndarray:
    return metric.sum(-1) ** 2 / np.square(metric).sum(-1)


def _write_source_data(
    full_metric: np.ndarray,
    scf_metric: np.ndarray,
    mechanism: dict,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    full_curve = _cumulative_budget(full_metric)
    scf_curve = _cumulative_budget(scf_metric)
    with (SOURCE_DIR / "fig1_spectrum.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "directions",
                "fraction_directions",
                "full_mean",
                "full_q25",
                "full_q75",
                "scf_mean",
                "scf_q25",
                "scf_q75",
            ]
        )
        for index in range(full_curve.shape[-1]):
            writer.writerow(
                [
                    index + 1,
                    (index + 1) / full_curve.shape[-1],
                    full_curve[:, index].mean(),
                    np.quantile(full_curve[:, index], 0.25),
                    np.quantile(full_curve[:, index], 0.75),
                    scf_curve[:, index].mean(),
                    np.quantile(scf_curve[:, index], 0.25),
                    np.quantile(scf_curve[:, index], 0.75),
                ]
            )

    with (SOURCE_DIR / "fig1_windows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "full_attention_kl",
                "scf_attention_kl",
                "full_logit_mse",
                "scf_logit_mse",
                "full_fisher_mse",
                "scf_fisher_mse",
                "full_raw_mse",
                "scf_raw_mse",
            ]
        )
        for index, row in enumerate(mechanism["windows"]):
            parent = row["matched"]
            child = row["scfw"]
            writer.writerow(
                [
                    index,
                    parent["reference_forward_kl"],
                    child["reference_forward_kl"],
                    parent["reference_logit_mse"],
                    child["reference_logit_mse"],
                    parent["fisher_vector_mse"],
                    child["fisher_vector_mse"],
                    parent["euclidean_vector_mse"],
                    child["euclidean_vector_mse"],
                ]
            )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.16,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.6,
            "legend.fontsize": 6.5,
        }
    )

    if METRIC_PATH.exists() and MECHANISM_PATH.exists():
        full_metric, scf_metric = _load_spectra()
        mechanism = json.loads(MECHANISM_PATH.read_text())
        _write_source_data(full_metric, scf_metric, mechanism)
        full_curve = _cumulative_budget(full_metric)
        scf_curve = _cumulative_budget(scf_metric)
        full_rank = float(_effective_rank(full_metric).mean())
        scf_rank = float(_effective_rank(scf_metric).mean())
        matrix_count = int(full_metric.shape[0])
        dimension = int(full_metric.shape[1])
    else:
        full_curve, scf_curve, mechanism = _load_paper_source_data()
        source_summary = json.loads((SOURCE_DIR / "fig1_summary.json").read_text())
        full_rank = float(source_summary["effective_dimensions_full"])
        scf_rank = float(source_summary["effective_dimensions_scf"])
        matrix_count = int(source_summary["layer_head_matrices"])
        dimension = int(source_summary["dimension"])
    fraction = np.arange(1, full_curve.shape[-1] + 1) / full_curve.shape[-1]
    full_mean = full_curve.mean(0)
    scf_mean = scf_curve.mean(0)

    fig = plt.figure(figsize=(7.2, 2.25), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        left=0.045,
        right=0.995,
        bottom=0.25,
        top=0.82,
        wspace=0.55,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[0, 3])

    # a: equal-rate schematic. The grids are explanatory, not data.
    ax_a.set_xlim(0, 2.12)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_title("Same rate, broader coverage", loc="left", pad=7)
    ax_a.axis("off")

    def draw_codebook(
        x0: float,
        x1: float,
        nx: int,
        ny: int,
        color: str,
        heading: str,
        note: str,
        shade_edges: bool,
    ) -> None:
        y0, y1 = 0.02, 0.70
        width, height = x1 - x0, y1 - y0
        frame = patches.FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#FAFAF8",
            edgecolor="#C9CACD",
            linewidth=0.7,
            zorder=0,
        )
        ax_a.add_patch(frame)
        if shade_edges:
            edge_width = 0.15 * width
            for edge_x in (x0, x1 - edge_width):
                ax_a.add_patch(
                    patches.Rectangle(
                        (edge_x, y0),
                        edge_width,
                        height,
                        facecolor=WORSE,
                        edgecolor="none",
                        alpha=0.10,
                        zorder=0.5,
                    )
                )
        for split in range(1, nx):
            xpos = x0 + split * width / nx
            ax_a.plot([xpos, xpos], [y0, y1], color="#DADBDD", lw=0.55, zorder=1)
        for split in range(1, ny):
            ypos = y0 + split * height / ny
            ax_a.plot([x0, x1], [ypos, ypos], color="#DADBDD", lw=0.55, zorder=1)
        xcoords = x0 + (np.arange(nx) + 0.5) * width / nx
        ycoords = y0 + (np.arange(ny) + 0.5) * height / ny
        xx, yy = np.meshgrid(xcoords, ycoords)
        ax_a.scatter(
            xx.ravel(),
            yy.ravel(),
            s=17,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax_a.text(
            (x0 + x1) / 2,
            0.75,
            heading,
            ha="center",
            va="bottom",
            color=color,
            fontsize=6.8,
            fontweight="bold",
        )
        ax_a.text(
            (x0 + x1) / 2,
            -0.17,
            note,
            ha="center",
            va="top",
            color=TEXT,
            fontsize=5.9,
            linespacing=1.15,
            clip_on=False,
        )

    draw_codebook(
        0.18,
        0.94,
        nx=2,
        ny=6,
        color=FULL,
        heading="Full + floor",
        note="2 $\\times$ 6 levels",
        shade_edges=True,
    )
    draw_codebook(
        1.18,
        1.94,
        nx=3,
        ny=4,
        color=SCF,
        heading="SCF",
        note="3 $\\times$ 4 levels",
        shade_edges=False,
    )
    ax_a.annotate(
        "",
        xy=(0.12, 0.68),
        xytext=(0.12, 0.05),
        arrowprops={"arrowstyle": "-|>", "color": "#777A80", "lw": 0.65},
    )
    ax_a.text(
        0.055,
        0.365,
        "sensitive",
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.9,
        color="#666970",
    )
    ax_a.text(
        0.99,
        0.95,
        "12 codewords each",
        ha="center",
        va="top",
        fontsize=5.8,
        color="#666970",
    )
    _panel_label(ax_a, "a")

    # b: empirical metric-budget concentration.
    ax_b.plot(fraction * 100, fraction * 100, color=NEUTRAL, ls="--", lw=1.0)
    ax_b.fill_between(
        fraction * 100,
        np.quantile(full_curve, 0.25, axis=0) * 100,
        np.quantile(full_curve, 0.75, axis=0) * 100,
        color=FULL,
        alpha=0.14,
        linewidth=0,
    )
    ax_b.fill_between(
        fraction * 100,
        np.quantile(scf_curve, 0.25, axis=0) * 100,
        np.quantile(scf_curve, 0.75, axis=0) * 100,
        color=SCF_LIGHT,
        alpha=0.50,
        linewidth=0,
    )
    ax_b.plot(fraction * 100, full_mean * 100, color=FULL)
    ax_b.plot(fraction * 100, scf_mean * 100, color=SCF)
    # Draw the keys explicitly so both samples have identical visible lengths.
    key_x0, key_x1, key_text_x = 0.46, 0.58, 0.61
    for key_y, key_label, key_color in (
        (0.215, "Full + floor", FULL),
        (0.125, "SCF", SCF),
    ):
        ax_b.plot(
            [key_x0, key_x1],
            [key_y, key_y],
            transform=ax_b.transAxes,
            color=key_color,
            lw=1.35,
            solid_capstyle="butt",
            clip_on=False,
            zorder=7,
        )
        ax_b.text(
            key_text_x,
            key_y,
            key_label,
            transform=ax_b.transAxes,
            ha="left",
            va="center",
            color=key_color,
            fontsize=6.1,
            fontweight="bold",
            zorder=7,
        )
    top = 13
    ax_b.scatter(
        [top / 128 * 100, top / 128 * 100],
        [full_mean[top - 1] * 100, scf_mean[top - 1] * 100],
        c=[FULL, SCF],
        s=17,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
    )
    ax_b.annotate(
        f"{full_mean[top - 1] * 100:.1f}%",
        (top / 128 * 100, full_mean[top - 1] * 100),
        xytext=(13, 9),
        textcoords="offset points",
        color=FULL,
        fontsize=6.4,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        arrowprops={"arrowstyle": "-", "color": FULL, "lw": 0.45, "shrinkA": 2, "shrinkB": 2},
        zorder=6,
    )
    ax_b.annotate(
        f"{scf_mean[top - 1] * 100:.1f}%",
        (top / 128 * 100, scf_mean[top - 1] * 100),
        xytext=(13, -10),
        textcoords="offset points",
        color=SCF,
        fontsize=6.4,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
        arrowprops={"arrowstyle": "-", "color": SCF, "lw": 0.45, "shrinkA": 2, "shrinkB": 2},
        zorder=6,
    )
    ax_b.text(
        0.03,
        0.96,
        f"$d_{{\\mathrm{{eff}}}}$: {full_rank:.1f} → {scf_rank:.1f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=TEXT,
    )
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel("Top eigen-directions (%)")
    ax_b.set_ylabel("Cumulative metric budget (%)")
    ax_b.set_title("SCF spreads spectral budget", loc="left", pad=7)
    _panel_label(ax_b, "b")

    # c: every held-out window has lower attention KL.
    parent_kl = np.array(
        [row["matched"]["reference_forward_kl"] for row in mechanism["windows"]]
    )
    child_kl = np.array(
        [row["scfw"]["reference_forward_kl"] for row in mechanism["windows"]]
    )
    jitter = np.linspace(-0.028, 0.028, num=len(parent_kl))
    for index, (parent, child) in enumerate(zip(parent_kl, child_kl)):
        ax_c.plot(
            [0 + jitter[index], 1 + jitter[index]],
            [parent, child],
            color="#B8BAC4",
            lw=0.8,
            zorder=1,
        )
    ax_c.scatter(jitter, parent_kl, color=FULL, s=17, zorder=2)
    ax_c.scatter(1 + jitter, child_kl, color=SCF, s=17, zorder=2)
    ax_c.set_xticks([0, 1], ["Full+floor", "SCF"])
    ax_c.set_xlim(-0.35, 1.35)
    ax_c.set_ylabel("Attention KL ↓")
    ax_c.set_title("Attention KL falls on 8/8", loc="left", pad=7)
    ax_c.text(
        0.98,
        0.95,
        f"mean −{(1 - child_kl.mean() / parent_kl.mean()) * 100:.1f}%",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        color=SCF,
        fontsize=6.5,
        fontweight="bold",
    )
    _panel_label(ax_c, "c")

    # d: aggregate diagnostic and honest Euclidean trade-off.
    overall = mechanism["halves"]["overall"]
    parent = overall["matched"]
    child = overall["scfw"]
    labels = ["$M_F$ error", "Logit MSE", "Attention KL", "Raw error"]
    parent_values = np.array(
        [
            parent["fisher_vector_mse"],
            parent["reference_logit_mse"],
            parent["reference_forward_kl"],
            parent["euclidean_vector_mse"],
        ]
    )
    child_values = np.array(
        [
            child["fisher_vector_mse"],
            child["reference_logit_mse"],
            child["reference_forward_kl"],
            child["euclidean_vector_mse"],
        ]
    )
    deltas = (child_values / parent_values - 1.0) * 100
    ypos = np.arange(len(labels))[::-1]
    colors = [SCF if value < 0 else WORSE for value in deltas]
    ax_d.set_axisbelow(True)
    ax_d.axvline(0, color=NEUTRAL, lw=0.9, zorder=1)
    ax_d.barh(ypos, deltas, color=colors, height=0.58, zorder=2)
    for y, value in zip(ypos, deltas):
        if value < 0:
            label_x, label_ha, label_color = -1.5, "right", "white"
        else:
            label_x, label_ha, label_color = value + 1.2, "left", TEXT
        ax_d.text(
            label_x,
            y,
            f"{value:+.1f}%",
            ha=label_ha,
            va="center",
            fontsize=6.5,
            color=label_color,
            fontweight="bold",
            zorder=3,
        )
    ax_d.set_yticks(ypos, labels)
    ax_d.set_xlim(-55, 12)
    ax_d.set_xlabel("SCF vs. full+floor (%)")
    ax_d.set_title("Downstream error trade-off", loc="left", pad=7)
    _panel_label(ax_d, "d")

    for ax in (ax_b, ax_c):
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax_d.grid(False)
    for ax in (ax_b, ax_c, ax_d):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    base = FIG_DIR / "scf_motivation"
    export_options = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(base.with_suffix(".svg"), **export_options)
    fig.savefig(base.with_suffix(".pdf"), **export_options)
    fig.savefig(base.with_suffix(".png"), dpi=600, **export_options)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, **export_options)
    plt.close(fig)

    summary = {
        "layer_head_matrices": matrix_count,
        "dimension": dimension,
        "top_13_full_percent": float(full_mean[12] * 100),
        "top_13_scf_percent": float(scf_mean[12] * 100),
        "effective_dimensions_full": float(full_rank),
        "effective_dimensions_scf": float(scf_rank),
        "heldout_windows": int(len(parent_kl)),
        "heldout_attention_kl_wins": int(np.sum(child_kl < parent_kl)),
        "schematic_codewords_per_geometry": 12,
        "schematic_full_grid": [2, 6],
        "schematic_scf_grid": [3, 4],
        "relative_deltas_percent": dict(zip(labels, map(float, deltas))),
    }
    (SOURCE_DIR / "fig1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
