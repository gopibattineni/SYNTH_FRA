"""Rank generators by Accuracy Utility Gap (TRTR − TSTR) across datasets.

Same pipeline as the F1-gap ranking, but using Accuracy_Gap.

Outputs:
    Conor/accuracy_gap_by_dataset.csv
    Conor/accuracy_gap_ranks_by_dataset.csv
    Conor/accuracy_gap_rank_table.csv
    Conor/accuracy_gap_rank_table.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
from latex_fonts import apply_font_to_figure, configure_times_font

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

SHORT = {
    "1. Cancer": "Cancer",
    "2. Alzhimers": "Alzheimer's",
    "3. Adult": "Adult",
    "4. Forest cover dataset": "Forest Cover",
    "5. Bank Markting": "Bank Marketing",
    "6. Wine dataset": "Wine Quality",
    "7. CDC diabetes dataset": "CDC Diabetes",
    "8. Mushroom dataset": "Mushroom",
    "9. MAGIC Gamma Telescope": "MAGIC Gamma",
}

NAVY = "#1f3a5f"
HEADER_BG = "#1f3a5f"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
BEST_BG = "#e6f2ea"
BEST_FG = "#1c5c37"
GRID = "#c5d0dc"
INK = "#1c1f24"


def load_and_rank() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(json.loads(GAPS_JSON.read_text()))
    df = raw[
        (raw["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (raw["Metric"] == "Accuracy_Gap")
        & (raw["TaskType"] == "classification")
    ].copy()
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    df["DatasetShort"] = df["Dataset"].map(SHORT)

    gap_piv = df.pivot_table(index="DatasetShort", columns="Generator", values="UtilityGap")
    # preserve dataset order
    order = [SHORT[d] for d in CLASSIFICATION_DATASETS]
    gap_piv = gap_piv.reindex(order)

    ranked = df.copy()
    ranked["Rank"] = ranked.groupby("Dataset")["UtilityGap"].rank(
        method="average", ascending=True
    )

    summary = (
        ranked.groupby("Generator")["Rank"]
        .agg(
            AverageRank="mean",
            MedianRank="median",
            StdDevRank=lambda s: s.std(ddof=1),
        )
        .reset_index()
        .sort_values("AverageRank", ascending=True, ignore_index=True)
    )
    return gap_piv, ranked, summary


def render_table(summary: pd.DataFrame) -> plt.Figure:
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
        total_w / 2, fig_h - 0.30,
        "Generator Ranking by Accuracy Utility Gap",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY,
    )
    ax.text(
        total_w / 2, fig_h - 0.58,
        "Ranked per dataset by Accuracy Gap (TRTR \u2212 TSTR), averaged across 9 classification datasets",
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
        "Rank 1 = smallest Accuracy Utility Gap (best) within each dataset.",
        ha="left", va="top", fontsize=7.8, color="#5a5f66",
    )

    apply_font_to_figure(fig, font_name)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main() -> None:
    gap_piv, ranked, summary = load_and_rank()

    gap_piv.round(6).to_csv(OUT_DIR / "accuracy_gap_by_dataset.csv")
    ranked[["Dataset", "DatasetShort", "Generator", "UtilityGap", "Rank"]].sort_values(
        ["Dataset", "Rank"]
    ).to_csv(OUT_DIR / "accuracy_gap_ranks_by_dataset.csv", index=False)

    summary_out = summary.rename(columns={
        "AverageRank": "Average Rank",
        "MedianRank": "Median Rank",
        "StdDevRank": "Std Dev Rank",
    })[["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]]
    summary_out.to_csv(OUT_DIR / "accuracy_gap_rank_table.csv", index=False)

    fig = render_table(summary)
    png_path = OUT_DIR / "accuracy_gap_rank_table.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.22)

    print("Accuracy Utility Gap (TRTR − TSTR) — average ranks:")
    print(summary_out.round(2).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'accuracy_gap_by_dataset.csv'}")
    print(f"Saved: {OUT_DIR / 'accuracy_gap_ranks_by_dataset.csv'}")
    print(f"Saved: {OUT_DIR / 'accuracy_gap_rank_table.csv'}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
