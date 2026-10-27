#!/usr/bin/env python3
"""Dual-panel forest plot of per-dataset OLS slopes (classification).

Reads all_datasets_statistical_summary.csv and writes:
  forest_plot_classification.{png,pdf}

Black-and-white, no interior grid lines, no panel borders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
AGREED = SCRIPT_DIR.parent  # .../SYNTH/Agreed analysis
sys.path.insert(0, str(AGREED))
from latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402

DPI = 600
INK = "#000000"
ZERO = "#000000"
FACE = "white"

# Display order (top → bottom), matching Figure 5
DATASET_ORDER = [
    "Cancer",
    "Alzheimer's",
    "Adult",
    "Forest Cover",
    "Bank Marketing",
    "Wine Quality",
    "CDC Diabetes",
    "Mushroom",
    "MAGIC Gamma",
]

FID_ANALYSIS = "fidelity_vs_accuracy_gap"
PRIV_ANALYSIS = "mia_vs_accuracy_gap"


def _load_panel(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    sub = df[df["Analysis"] == analysis].copy()
    sub["Dataset"] = sub["Dataset"].astype(str)
    sub = sub[sub["Dataset"].isin(DATASET_ORDER)]
    sub["Dataset"] = pd.Categorical(sub["Dataset"], categories=DATASET_ORDER, ordered=True)
    return sub.sort_values("Dataset").reset_index(drop=True)


def _fmt_ci(slope: float, lo: float, hi: float) -> str:
    return f"{slope:.2f} [{lo:.2f}, {hi:.2f}]"


def _draw_panel(ax, panel: pd.DataFrame, title: str) -> None:
    n = len(panel)
    y = np.arange(n)[::-1]

    ax.set_facecolor(FACE)
    ax.axvline(0.0, color=ZERO, linestyle="--", linewidth=1.15, zorder=1)

    xmax = float(panel["Slope_95CI_high"].max())
    xmin = float(panel["Slope_95CI_low"].min())
    span = max(xmax - xmin, 1.0)
    label_pad = 0.035 * span

    for yi, (_, row) in zip(y, panel.iterrows()):
        slope = float(row["Slope"])
        lo = float(row["Slope_95CI_low"])
        hi = float(row["Slope_95CI_high"])
        significant = (lo > 0) or (hi < 0)

        ax.plot([lo, hi], [yi, yi], color=INK, linewidth=1.8, solid_capstyle="round", zorder=2)
        ax.scatter(
            [slope],
            [yi],
            marker="D" if significant else "o",
            s=64 if significant else 54,
            facecolors=INK,
            edgecolors=FACE,
            linewidths=0.7,
            zorder=3,
        )
        ax.text(
            hi + label_pad,
            yi,
            _fmt_ci(slope, lo, hi),
            va="center",
            ha="left",
            fontsize=8.4,
            color=INK,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_xlabel("Slope", fontsize=11, color=INK)
    ax.set_title(title, fontsize=12, pad=10, color=INK)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(False)
    ax.tick_params(colors=INK, direction="out")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(xmin - 0.08 * span, xmax + 0.52 * span)


def main() -> None:
    configure_times_font()
    df = pd.read_csv(SCRIPT_DIR / "all_datasets_statistical_summary.csv")

    fid = _load_panel(df, FID_ANALYSIS)
    priv = _load_panel(df, PRIV_ANALYSIS)
    if len(fid) != len(DATASET_ORDER) or len(priv) != len(DATASET_ORDER):
        missing_f = set(DATASET_ORDER) - set(fid["Dataset"])
        missing_p = set(DATASET_ORDER) - set(priv["Dataset"])
        raise SystemExit(f"Missing datasets — fidelity: {missing_f}, privacy: {missing_p}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.8), sharey=False, facecolor=FACE)
    fig.suptitle(
        "Forest Plot — Classification\nRegression Slopes with 95% Confidence Intervals",
        fontsize=14,
        fontweight="bold",
        color=INK,
        y=0.98,
    )

    _draw_panel(axes[0], fid, r"Fidelity $\rightarrow$ Accuracy Gap")
    _draw_panel(axes[1], priv, r"Privacy $\rightarrow$ Accuracy Gap")

    y = np.arange(len(fid))[::-1]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(list(fid["Dataset"]), fontsize=10.5, color=INK)
    axes[0].tick_params(axis="y", pad=6, colors=INK)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", colors=INK)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=INK,
            markeredgecolor=FACE,
            markersize=8.5,
            label=r"Significant ($p < 0.05$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=INK,
            markeredgecolor=FACE,
            markersize=8.5,
            label="Non-significant",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.55, 0.015),
    )

    apply_font_to_figure(fig)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.86, bottom=0.12, wspace=0.14)

    for ext in ("png", "pdf"):
        out = SCRIPT_DIR / f"forest_plot_classification.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=FACE, pad_inches=0.2)
        print(f"Wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
