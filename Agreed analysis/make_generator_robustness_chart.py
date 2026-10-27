"""Horizontal bar chart: generator robustness via average rank.

For each generator, rank is taken from OverallRank on each of the 9
classification datasets (no-leakage). The chart shows mean rank ± SD
across datasets, sorted best → worst (lower average rank is better).

Output:
    classification/generator_robustness_average_rank.png
    classification/generator_robustness_average_rank.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES = ROOT / "Results" / "Processed_Data" / "overall_scores.csv"

CLASSIFICATION_DATASETS = [
    "1. Cancer",
    "2. Alzhimers",
    "3. Adult",
    "4. Forest cover dataset",
    "5. Bank Markting",
    "6. Wine dataset",
    "7. CDC diabetes dataset",
    "8. Mushroom dataset",
    "9. MAGIC Gamma Telescope",
]

DISPLAY_NAMES = {
    "ForestDiffusion": "ForestDiffusion",
    "TVAE": "TVAE",
    "CTABGAN": "CTABGAN",
    "WGAN_GP": "WGAN-GP",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
}

# Distinct colour per generator (best → worst order in chart)
BAR_COLORS = [
    "#1b9e77",  # ForestDiffusion — teal
    "#d95f02",  # TVAE — orange
    "#7570b3",  # CTABGAN — purple
    "#e7298a",  # WGAN-GP — magenta
    "#66a61e",  # GaussianCopula — green
    "#e6ab02",  # CopulaGAN — gold
    "#a6761d",  # CTGAN — brown
    "#666666",  # TabDDPM — grey
]

NAVY = "#1f3a5f"
ERR = "#1c1f24"
GRID = "#d5dde6"


def compute_rank_stats() -> pd.DataFrame:
    df = pd.read_csv(SCORES)
    sub = df[
        df["Dataset"].isin(CLASSIFICATION_DATASETS) & (df["LeakageLevel"] == 0.0)
    ].copy()
    piv = sub.pivot_table(index="Generator", columns="Dataset", values="OverallRank")
    stats = pd.DataFrame({
        "AverageRank": piv.mean(axis=1),
        "RankStd": piv.std(axis=1, ddof=1),
        "N_Datasets": piv.notna().sum(axis=1),
    }).sort_values("AverageRank", ascending=True)
    stats["Display"] = [DISPLAY_NAMES.get(g, g) for g in stats.index]
    return stats


def render(stats: pd.DataFrame) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font

    font_name = configure_times_font()
    n = len(stats)
    fig_h = max(4.8, 0.55 * n + 1.6)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    y = np.arange(n)
    means = stats["AverageRank"].to_numpy()
    stds = stats["RankStd"].to_numpy()
    labels = stats["Display"].tolist()
    colors = BAR_COLORS[:n]

    ax.barh(
        y, means, height=0.62, color=colors, edgecolor=NAVY, linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        means, y, xerr=stds, fmt="none", ecolor=ERR, elinewidth=1.4,
        capsize=4, capthick=1.3, zorder=4,
    )

    for yi, m, s in zip(y, means, stds):
        ax.text(
            m + s + 0.12, yi, f"{m:.2f}",
            va="center", ha="left", fontsize=9.5, color=NAVY, fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # best (lowest rank) at top
    ax.set_xlabel("Average Rank  (lower is better)", fontsize=11, color=NAVY)
    ax.set_xlim(0, max(means + stds) + 1.2)
    ax.set_title(
        "Generator Robustness Across Datasets",
        fontsize=14, fontweight="bold", color=NAVY, pad=12,
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAVY)
    ax.spines["bottom"].set_color(NAVY)
    ax.tick_params(colors=NAVY)

    legend_handles = [
        Line2D(
            [0], [0], color=ERR, linewidth=1.4,
            marker="|", markersize=10, markeredgewidth=1.3,
            label="SD across datasets",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right", frameon=True, fontsize=9,
        edgecolor=GRID, fancybox=False,
    )

    fig.tight_layout()
    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    stats = compute_rank_stats()
    csv_path = OUT_DIR / "generator_robustness_average_rank.csv"
    stats.reset_index().to_csv(csv_path, index=False)

    fig = render(stats)
    png_path = OUT_DIR / "generator_robustness_average_rank.png"
    svg_path = OUT_DIR / "generator_robustness_average_rank.svg"

    prev = mpl.rcParams["svg.fonttype"]
    mpl.rcParams["svg.fonttype"] = "path"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(svg_path, format="svg", facecolor="white", bbox_inches="tight", pad_inches=0.2)
    mpl.rcParams["svg.fonttype"] = prev
    plt.close(fig)

    download = ROOT.parent / "generator_robustness"
    download.mkdir(parents=True, exist_ok=True)
    for src in (png_path, svg_path):
        (download / src.name).write_bytes(src.read_bytes())

    print(stats[["Display", "AverageRank", "RankStd", "N_Datasets"]].to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Copy:  {download / svg_path.name}")


if __name__ == "__main__":
    main()
