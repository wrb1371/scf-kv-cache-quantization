#!/usr/bin/env python3
"""Render the single-column rate-dependent geometry-attribution figure."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DATA_PATH = FIG_DIR / "source_data" / "geometry_attribution.csv"
OUTPUT_STEM = FIG_DIR / "geometry_attribution"

FULL = "#5B5F79"
SCF = "#2F8F83"
TEXT = "#2F3136"
AXIS = "#777A80"


def load_reductions() -> dict[str, dict[str, list[float]]]:
    rows: list[dict[str, str]] = []
    with DATA_PATH.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))

    result: dict[str, dict[str, list[float]]] = {}
    for dataset in ("WikiText-2", "C4"):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_rate: dict[float, dict[str, float]] = {}
        for row in dataset_rows:
            by_rate.setdefault(float(row["bpa"]), {})[row["geometry"]] = float(
                row["ppl"]
            )

        result[dataset] = {"Full Fisher": [], "SCF": []}
        for rate in (0.75, 1.0, 2.0):
            values = by_rate[rate]
            parent = values["Euclidean"]
            for geometry in ("Full Fisher", "SCF"):
                result[dataset][geometry].append(
                    100.0 * (1.0 - values[geometry] / parent)
                )
    return result


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.2,
            "axes.titlesize": 6.6,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "legend.fontsize": 5.9,
            "legend.frameon": False,
        }
    )

    reductions = load_reductions()
    positions = [0, 1, 2]
    rate_labels = ["0.75", "1", "2"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.35, 1.68),
        sharey=True,
        gridspec_kw={"wspace": 0.20},
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.25, top=0.74)

    for index, (ax, dataset) in enumerate(zip(axes, ("WikiText-2", "C4"))):
        ax.plot(
            positions,
            reductions[dataset]["Full Fisher"],
            color=FULL,
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.2,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=FULL,
            markeredgewidth=0.8,
            label="Full + floor",
            zorder=2,
        )
        ax.plot(
            positions,
            reductions[dataset]["SCF"],
            color=SCF,
            linestyle="-",
            linewidth=1.55,
            marker="D",
            markersize=3.3,
            markerfacecolor=SCF,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label="SCF",
            zorder=3,
        )
        ax.set_xlim(-0.18, 2.18)
        ax.set_ylim(0.0, 2.55)
        ax.set_xticks(positions, rate_labels)
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("Index rate (BPA)", labelpad=2.4)
        ax.set_title(dataset, loc="left", pad=4.5, color=TEXT, fontweight="bold")
        panel_x = -0.34 if index == 0 else -0.18
        ax.text(
            panel_x,
            1.20,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            ha="left",
            va="top",
            color=TEXT,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="both", colors=AXIS, width=0.55, length=2.2, pad=1.5)
        ax.axhline(0.0, color="#C8CACE", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("PPL reduction vs.\nEuclidean (\%, higher is better)", labelpad=4.5)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.985),
        ncol=2,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    export_args = {"bbox_inches": "tight", "pad_inches": 0.02, "facecolor": "white"}
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **export_args)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **export_args)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, **export_args)
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, **export_args)
    plt.close(fig)


if __name__ == "__main__":
    main()
