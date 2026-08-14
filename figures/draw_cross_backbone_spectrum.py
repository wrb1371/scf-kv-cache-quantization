#!/usr/bin/env python3
"""Draw the appendix cross-backbone selector-spectrum diagnostic."""

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
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "WSDM2027投稿" / "figures"
SOURCE_DIR = FIG_DIR / "source_data"
CAMPAIGN = (
    ROOT
    / "FACQ_research_handoff_2026-07-28"
    / "artifacts/experiment/scf_cross_backbone_spectrum_20260813"
)
CURVES = (
    CAMPAIGN / "spectrum_curves.csv"
    if (CAMPAIGN / "spectrum_curves.csv").exists()
    else SOURCE_DIR / "appendix_cross_backbone_spectrum.csv"
)
SUMMARY = (
    CAMPAIGN / "summary.json"
    if (CAMPAIGN / "summary.json").exists()
    else SOURCE_DIR / "appendix_cross_backbone_spectrum_summary.json"
)

FULL = "#5B5F79"
SCF = "#2F8F83"
UNIFORM = "#989A9E"
TEXT = "#303238"


def load_curves() -> dict[str, dict[str, np.ndarray]]:
    rows = list(csv.DictReader(CURVES.open()))
    if not rows:
        raise ValueError("spectrum source data are empty")
    fields = [
        "fraction_directions",
        "full_mean",
        "full_q25",
        "full_q75",
        "scf_mean",
        "scf_q25",
        "scf_q75",
    ]
    output: dict[str, dict[str, np.ndarray]] = {}
    for backbone in ("Meta-Llama-3-8B", "Mistral-7B-v0.1"):
        selected = [row for row in rows if row["backbone"] == backbone]
        if len(selected) != 128:
            raise ValueError(f"expected 128 directions for {backbone}")
        output[backbone] = {
            field: np.array([float(row[field]) for row in selected])
            for field in fields
        }
        if not all(np.isfinite(values).all() for values in output[backbone].values()):
            raise ValueError(f"nonfinite source data for {backbone}")
    return output


def prepend_origin(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], values))


def write_paper_source_data() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    curve_target = SOURCE_DIR / "appendix_cross_backbone_spectrum.csv"
    summary_target = SOURCE_DIR / "appendix_cross_backbone_spectrum_summary.json"
    if CURVES.resolve() != curve_target.resolve():
        curve_target.write_text(CURVES.read_text())
    if SUMMARY.resolve() != summary_target.resolve():
        summary_target.write_text(SUMMARY.read_text())


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.45,
            "legend.fontsize": 6.1,
        }
    )
    summary = json.loads(SUMMARY.read_text())
    if summary.get("status") != "complete":
        raise ValueError("cross-backbone analysis is not complete")
    data = load_curves()
    write_paper_source_data()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.46, 1.75),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.27, top=0.73, wspace=0.28)

    for panel, (axis, backbone) in enumerate(zip(axes, data)):
        values = data[backbone]
        x = 100 * prepend_origin(values["fraction_directions"])
        uniform = x.copy()
        full_mean = 100 * prepend_origin(values["full_mean"])
        full_q25 = 100 * prepend_origin(values["full_q25"])
        full_q75 = 100 * prepend_origin(values["full_q75"])
        scf_mean = 100 * prepend_origin(values["scf_mean"])
        scf_q25 = 100 * prepend_origin(values["scf_q25"])
        scf_q75 = 100 * prepend_origin(values["scf_q75"])

        axis.fill_between(x, full_q25, full_q75, color=FULL, alpha=0.13, linewidth=0)
        axis.fill_between(x, scf_q25, scf_q75, color=SCF, alpha=0.14, linewidth=0)
        axis.plot(x, full_mean, color=FULL, label="Full + floor")
        axis.plot(x, scf_mean, color=SCF, label="SCF")
        axis.plot(x, uniform, color=UNIFORM, linestyle=(0, (2.0, 2.0)), lw=0.8,
                  label="Uniform")
        axis.axvline(100 * 13 / 128, color="#C4C5C7", lw=0.65, linestyle=":")

        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 50, 100])
        axis.set_yticks([0, 50, 100])
        axis.grid(axis="y", color="#E6E6E4", linewidth=0.45, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.4, width=0.6, colors=TEXT)
        axis.set_title(backbone, loc="left", pad=4.2, color=TEXT)
        axis.text(
            -0.18,
            1.17,
            chr(ord("a") + panel),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=TEXT,
        )

    axes[0].set_ylabel("Cumulative metric budget (%)")
    fig.supxlabel("Leading eigen-directions (%)", x=0.565, y=0.055, fontsize=6.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.965),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.2,
    )

    output = FIG_DIR / "cross_backbone_spectrum"
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
