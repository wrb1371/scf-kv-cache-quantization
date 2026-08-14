#!/usr/bin/env python3
"""Draw the compact SCF power/floor sensitivity figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TEAL = "#278F87"
INK = "#384052"
LIGHT = "#A7ADB8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "source_data" / "scf_sensitivity_summary.json",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=root / "scf_sensitivity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("sensitivity summary must be complete")
    rows = summary["rows"]
    power = sorted(
        (row for row in rows if np.isclose(row["fisher_mixture"], 0.5)),
        key=lambda row: row["spectral_power"],
    )
    mixture = sorted(
        (row for row in rows if np.isclose(row["spectral_power"], 0.5)),
        key=lambda row: row["fisher_mixture"],
    )
    if len(power) != 4 or len(mixture) != 4:
        raise ValueError("expected four points in each one-dimensional sweep")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.48), sharey=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.27, top=0.80, wspace=0.26)
    panels = (
        (axes[0], power, "spectral_power", r"$p$", r"$\alpha=0.5$"),
        (axes[1], mixture, "fisher_mixture", r"$\alpha$", r"$p=0.5$"),
    )
    all_y = [float(row["relative_ppl_reduction_percent"]) for row in rows]
    ymax = max(all_y) * 1.22
    ymin = min(0.0, min(all_y) * 1.12)
    for panel_index, (ax, panel_rows, xkey, xlabel, fixed) in enumerate(panels):
        x = np.asarray([float(row[xkey]) for row in panel_rows])
        y = np.asarray([float(row["relative_ppl_reduction_percent"]) for row in panel_rows])
        ax.plot(x, y, color=INK, linewidth=1.05, zorder=2)
        ax.scatter(x, y, s=20, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=3)
        default_index = int(np.argmin(np.abs(x - 0.5)))
        ax.scatter(
            [x[default_index]],
            [y[default_index]],
            s=30,
            marker="D",
            facecolor=TEAL,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.annotate(
            "SCF",
            (x[default_index], y[default_index]),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            color=TEAL,
            fontsize=6.4,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            zorder=5,
        )
        ax.axhline(0.0, color=LIGHT, linewidth=0.65, linestyle=(0, (2, 2)), zorder=0)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([".25", ".5", ".75", "1"])
        ax.set_xlim(0.19, 1.06)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel, labelpad=1.4)
        ax.set_title(fixed, fontsize=7.0, fontweight="bold", pad=2.0)
        ax.text(
            -0.12,
            1.06,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=7.6,
            fontweight="bold",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(LIGHT)
        ax.spines["bottom"].set_color(LIGHT)
        ax.tick_params(length=2.5, width=0.6, color=LIGHT, pad=1.5)
    axes[0].set_ylabel(r"PPL reduction (\%)", labelpad=1.5)

    output_stem = args.output_stem.resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        (".pdf", {"bbox_inches": "tight", "pad_inches": 0.01}),
        (".svg", {"bbox_inches": "tight", "pad_inches": 0.01}),
        (".png", {"dpi": 450, "bbox_inches": "tight", "pad_inches": 0.01}),
        (".tiff", {"dpi": 600, "bbox_inches": "tight", "pad_inches": 0.01}),
    ):
        fig.savefig(output_stem.with_suffix(suffix), **options)
    plt.close(fig)


if __name__ == "__main__":
    main()
