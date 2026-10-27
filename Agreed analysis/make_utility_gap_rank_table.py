"""Rank synthetic-data generators by Utility Gap, per dataset, and summarize.

Pipeline
--------
1. For each dataset, compute the Utility Gap (TRTR - TSTR, F1) for every
   generator.
2. Rank generators within that dataset from best (rank = 1, smallest gap)
   to worst (largest gap).
3. Repeat for all classification datasets.
4. Compute the Average Rank, Median Rank, and Std Dev of Rank across
   datasets for each generator.
5. Render a publication-quality table (sorted by Average Rank) and export
   it as PNG and CSV.

Outputs:
    Conor/utility_gap_rank_table.csv
    Conor/utility_gap_rank_table.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
GAPS_JSON = ROOT / "docs" / "data" / "utility_gaps.json"

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

DISPLAY = {
    "ForestDiffusion": "ForestDiffusion",
    "TVAE": "TVAE",
    "CTABGAN": "CTABGAN",
    "WGAN_GP": "WGAN-GP",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
}

GAP_METRIC = "F1_Gap"   # TRTR - TSTR (F1); lower is better

NAVY = "#1f3a5f"
HEADER_BG = "#1f3a5f"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
BEST_BG = "#e6f2ea"
BEST_FG = "#1c5c37"
GRID = "#c5d0dc"
INK = "#1c1f24"


# --------------------------------------------------------------------------
# Step 1-3: load utility gaps and rank generators within each dataset
# --------------------------------------------------------------------------
def load_utility_gaps() -> pd.DataFrame:
    """Step 1: Utility Gap per (Dataset, Generator)."""
    records = json.loads(GAPS_JSON.read_text())
    df = pd.DataFrame(records)
    df = df[
        (df["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (df["Metric"] == GAP_METRIC)
        & (df["TaskType"] == "classification")
    ].copy()
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    return df[["Dataset", "Generator", "UtilityGap"]]


def rank_within_datasets(gap_df: pd.DataFrame) -> pd.DataFrame:
    """Steps 2-3: rank generators within every dataset (rank 1 = smallest gap = best)."""
    ranked = gap_df.copy()
    ranked["Rank"] = ranked.groupby("Dataset")["UtilityGap"].rank(
        method="min", ascending=True
    )
    return ranked


# --------------------------------------------------------------------------
# Step 4: aggregate ranks across datasets
# --------------------------------------------------------------------------
def summarize_ranks(ranked: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ranked.groupby("Generator")["Rank"]
        .agg(
            AverageRank="mean",
            MedianRank="median",
            StdDevRank=lambda s: s.std(ddof=1),
            N_Datasets="count",
        )
        .reset_index()
        .sort_values("AverageRank", ascending=True, ignore_index=True)
    )
    return summary


# --------------------------------------------------------------------------
# Step 5: publication-quality table (PNG + CSV)
# --------------------------------------------------------------------------
def render_table(summary: pd.DataFrame) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font
    font_name = configure_times_font()

    columns = ["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]
    col_w = [2.7, 1.55, 1.5, 1.6]
    total_w = sum(col_w)
    row_h = 0.46
    header_h = 0.52
    title_h = 0.78
    n_rows = len(summary)
    fig_h = title_h + header_h + n_rows * row_h + 0.35
    fig_w = total_w + 0.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        total_w / 2, fig_h - 0.30,
        "Generator Ranking by Utility Gap",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY,
    )
    ax.text(
        total_w / 2, fig_h - 0.58,
        "Ranked per dataset by F1 Utility Gap (TRTR \u2212 TSTR), averaged across 9 classification datasets",
        ha="center", va="center", fontsize=8.8, color="#5a5f66", style="italic",
    )

    y0 = fig_h - title_h - header_h
    edges = [0.0]
    for w in col_w:
        edges.append(edges[-1] + w)

    def draw_row(y, height, values, *, bg, fg, weight="normal", size=10.5):
        ax.add_patch(Rectangle(
            (0.02, y), total_w - 0.04, height, facecolor=bg, edgecolor="none", zorder=1,
        ))
        for i, val in enumerate(values):
            x = (edges[i] + edges[i + 1]) / 2
            ax.text(
                x, y + height / 2, str(val), ha="center", va="center",
                fontsize=size, fontweight=weight, color=fg, zorder=2,
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
    ax.add_patch(Rectangle(
        (0.02, bottom), total_w - 0.04, header_h + n_rows * row_h,
        facecolor="none", edgecolor=NAVY, linewidth=1.3, zorder=4,
    ))

    ax.text(
        0.02, bottom - 0.22,
        "Rank 1 = smallest Utility Gap (best) within each dataset. Std Dev computed with N\u22121 (sample) denominator.",
        ha="left", va="top", fontsize=7.8, color="#5a5f66",
    )

    apply_font_to_figure(fig, font_name)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main() -> None:
    gap_df = load_utility_gaps()
    ranked = rank_within_datasets(gap_df)
    summary = summarize_ranks(ranked)

    ranked_path = OUT_DIR / "utility_gap_ranks_by_dataset.csv"
    ranked.to_csv(ranked_path, index=False)

    csv_path = OUT_DIR / "utility_gap_rank_table.csv"
    summary_out = summary.rename(columns={
        "AverageRank": "Average Rank",
        "MedianRank": "Median Rank",
        "StdDevRank": "Std Dev Rank",
    })[["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]]
    summary_out.to_csv(csv_path, index=False)

    fig = render_table(summary)
    png_path = OUT_DIR / "utility_gap_rank_table.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.22)

    print(summary_out.round(2).to_string(index=False))
    print(f"\nSaved: {ranked_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
