#!/usr/bin/env python3
"""Manuscript rank-table figures + generator-robustness chart (Agreed analysis style)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    CLS_GAP_METRICS,
    DISPLAY_GENERATOR,
    GENERATOR_COLORS,
    INK,
    NAVY,
    REG_GAP_LABELS,
    REG_GAP_METRICS,
)
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402

HEADER_BG = "#1f3a5f"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
BEST_BG = "#e6f2ea"
BEST_FG = "#1c5c37"
GRID = "#c5d0dc"

FIG_CLS = ANALYSIS / "figures" / "classification"
FIG_REG = ANALYSIS / "figures" / "regression"
TAB_CLS = ANALYSIS / "tables" / "main" / "classification_gaps"
TAB_REG = ANALYSIS / "tables" / "main" / "regression_gaps"


def render_rank_table(
    summary: pd.DataFrame,
    title: str,
    subtitle: str,
    footnote: str,
    out_stem: Path,
) -> None:
    font_name = configure_times_font()
    columns = ["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]
    col_w = [2.7, 1.55, 1.5, 1.6]
    total_w = sum(col_w)
    row_h, header_h, title_h = 0.46, 0.52, 0.78
    n_rows = len(summary)
    fig_h = title_h + header_h + n_rows * row_h + 0.35
    fig_w = total_w + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        total_w / 2,
        fig_h - 0.30,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        total_w / 2,
        fig_h - 0.58,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#5a5f66",
        style="italic",
    )

    y0 = fig_h - title_h - header_h
    edges = [0.0]
    for w in col_w:
        edges.append(edges[-1] + w)

    def draw_row(y, height, values, *, bg, fg, weight="normal", size=10.5):
        ax.add_patch(
            Rectangle(
                (0.02, y),
                total_w - 0.04,
                height,
                facecolor=bg,
                edgecolor="none",
                zorder=1,
            )
        )
        for i, val in enumerate(values):
            x = (edges[i] + edges[i + 1]) / 2
            ax.text(
                x,
                y + height / 2,
                str(val),
                ha="center",
                va="center",
                fontsize=size,
                fontweight=weight,
                color=fg,
                zorder=2,
            )
        for x in edges[1:-1]:
            ax.plot([x, x], [y, y + height], color=GRID, lw=0.6, zorder=3)
        ax.plot([0.02, total_w - 0.02], [y, y], color=GRID, lw=0.7, zorder=3)

    draw_row(y0, header_h, columns, bg=HEADER_BG, fg="white", weight="bold", size=11)
    ax.plot([0.02, total_w - 0.02], [y0 + header_h, y0 + header_h], color=NAVY, lw=1.4)

    for r, row in enumerate(summary.itertuples(index=False)):
        y = y0 - (r + 1) * row_h
        is_best = r == 0
        bg = BEST_BG if is_best else (ROW_A if r % 2 == 0 else ROW_B)
        fg = BEST_FG if is_best else INK
        weight = "bold" if is_best else "normal"
        vals = [
            row.Generator,
            f"{row.AverageRank:.2f}",
            f"{row.MedianRank:.2f}",
            f"{row.StdDevRank:.2f}",
        ]
        draw_row(y, row_h, vals, bg=bg, fg=fg, weight=weight, size=10.5)

    bottom = y0 - n_rows * row_h
    ax.add_patch(
        Rectangle(
            (0.02, bottom),
            total_w - 0.04,
            header_h + n_rows * row_h,
            facecolor="none",
            edgecolor=NAVY,
            linewidth=1.3,
            zorder=4,
        )
    )
    ax.text(0.02, bottom - 0.22, footnote, ha="left", va="top", fontsize=7.8, color="#5a5f66")

    apply_font_to_figure(fig, font_name)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(ext), dpi=300 if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(out_stem.with_name(out_stem.name + "_data.csv"), index=False)


def robustness_chart(summary: pd.DataFrame, title: str, out_stem: Path, n_datasets: int) -> None:
    font_name = configure_times_font()
    df = summary.sort_values("AverageRank", ascending=True).reset_index(drop=True)
    n = len(df)
    fig_h = max(4.8, 0.55 * n + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    y = np.arange(n)
    means = df["AverageRank"].to_numpy(dtype=float)
    stds = df["StdDevRank"].to_numpy(dtype=float)
    colors = []
    for g in df["Generator"]:
        key = "WGAN_GP" if g == "WGAN-GP" else g.replace("-", "_")
        colors.append(GENERATOR_COLORS.get(key, GENERATOR_COLORS.get(g, NAVY)))

    ax.barh(
        y,
        means,
        height=0.62,
        color=colors,
        edgecolor=NAVY,
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        means,
        y,
        xerr=stds,
        fmt="none",
        ecolor="#444444",
        elinewidth=1.4,
        capsize=4,
        capthick=1.3,
        zorder=4,
    )
    # Numeric labels: mean value to the right of each bar/error whisker
    for yi, m, s in zip(y, means, stds):
        ax.text(
            m + s + 0.12,
            yi,
            f"{m:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color=NAVY,
            fontweight="bold",
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["Generator"].tolist(), fontsize=11)
    ax.invert_yaxis()  # best (lowest rank) at top
    ax.set_xlabel(
        f"Average Rank across {n_datasets} datasets  (lower is better)",
        fontsize=11,
        color=NAVY,
    )
    ax.set_xlim(0, float(np.nanmax(means + stds)) + 1.4)
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="#c5d0dc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.tick_params(colors=NAVY)

    legend = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.4,
            marker="|",
            markersize=10,
            markeredgewidth=1.3,
            label="SD across datasets",
        )
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, fontsize=9, edgecolor="#c5d0dc")
    fig.tight_layout()
    apply_font_to_figure(fig, font_name)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(
            out_stem.with_suffix(ext),
            dpi=300 if ext == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.2,
        )
    plt.close(fig)


def _load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "generator" in cl:
            rename[c] = "Generator"
        elif "average" in cl or cl == "averagerank":
            rename[c] = "AverageRank"
        elif "median" in cl:
            rename[c] = "MedianRank"
        elif "std" in cl:
            rename[c] = "StdDevRank"
    df = df.rename(columns=rename)
    if "Generator" in df.columns:
        df["Generator"] = df["Generator"].map(lambda g: DISPLAY_GENERATOR.get(g, g))
    return df.sort_values("AverageRank", ignore_index=True)


def main() -> None:
    # Classification Accuracy gap rank table (Agreed primary table)
    acc = _load_summary(TAB_CLS / "accuracy_gap_friedman_average_ranks.csv")
    render_rank_table(
        acc,
        title="Generator Ranking by Accuracy Utility Gap",
        subtitle=(
            "Ranked per dataset by Accuracy Gap (TRTR − TSTR) using 10-seed means, "
            "averaged across 9 classification datasets"
        ),
        footnote="Rank 1 = smallest Accuracy Utility Gap (best) within each dataset. n = 10 generator-training seeds.",
        out_stem=FIG_CLS / "accuracy_gap_rank_table",
    )
    robustness_chart(
        acc,
        title="Generator Robustness — Average Accuracy-Gap Rank (10-seed means)",
        out_stem=FIG_CLS / "generator_robustness_average_rank",
        n_datasets=9,
    )

    # Utility-gap composite: average of Acc/Prec/Rec/F1 average ranks
    util_parts = []
    for m in CLS_GAP_METRICS:
        p = TAB_CLS / f"{m.lower()}_friedman_average_ranks.csv"
        if p.exists():
            s = _load_summary(p).set_index("Generator")["AverageRank"]
            util_parts.append(s)
    if util_parts:
        util = pd.concat(util_parts, axis=1).mean(axis=1).rename("AverageRank").to_frame()
        # approximate median/std from accuracy ranks file shape
        util["MedianRank"] = util["AverageRank"]
        util["StdDevRank"] = pd.concat(util_parts, axis=1).std(axis=1, ddof=1)
        util = util.reset_index().rename(columns={"index": "Generator"}).sort_values(
            "AverageRank", ignore_index=True
        )
        render_rank_table(
            util,
            title="Generator Ranking by Classification Utility Gaps",
            subtitle=(
                "Mean of average ranks across Accuracy/Precision/Recall/F1 gaps "
                "(10-seed means per dataset)"
            ),
            footnote="Lower average rank indicates smaller utility gaps across metrics and datasets.",
            out_stem=FIG_CLS / "utility_gap_rank_table",
        )

    # Regression R2 gap rank table
    r2 = _load_summary(TAB_REG / "r2_gap_friedman_average_ranks.csv")
    render_rank_table(
        r2,
        title="Generator Ranking by R² Utility Gap",
        subtitle=(
            "Ranked per dataset by R² Gap using 10-seed means, averaged across 6 regression datasets"
        ),
        footnote="Rank 1 = smallest R² Gap (best) within each dataset. n = 10 generator-training seeds.",
        out_stem=FIG_REG / "r2_gap_rank_table",
    )
    robustness_chart(
        r2,
        title="Generator Robustness — Average R²-Gap Rank (10-seed means)",
        out_stem=FIG_REG / "generator_robustness_average_rank",
        n_datasets=6,
    )

    print("Rank tables and robustness charts written.")


if __name__ == "__main__":
    main()
