"""Publication-quality summary table for the 9 classification datasets.

Samples = size of the real working set used in the benchmark
(after any stratified downsampling, before the fixed 80/20 split).
Feature counts exclude the target column.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from latex_fonts import apply_font_to_figure, configure_times_font
configure_times_font()

# Working-set sizes and feature typing as used in the classification pipeline.
ROWS = [
    # Dataset, Samples, Features, Numerical, Categorical, Classes
    ("Cancer",            569,   30, 30,  0, 2),
    ("Alzheimer's",       373,   10, 10,  0, 2),
    ("Adult Census",    1_000,   14,  6,  8, 2),
    ("Forest Cover",    1_000,   10, 10,  0, 7),
    ("Bank Marketing", 10_000,   13,  5,  8, 2),
    ("Wine Quality",    1_000,   11, 11,  0, 6),
    ("CDC Diabetes",    1_000,   21,  7, 14, 2),
    ("Mushroom",        1_000,   20,  3, 17, 2),
    ("MAGIC Gamma",    19_020,   10, 10,  0, 2),
]

COLUMNS = ["Dataset", "Samples", "Features", "Numerical", "Categorical", "Classes"]

NAVY = "#1f3a5f"
HEADER_BG = "#1f3a5f"
HEADER_FG = "#ffffff"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
GRID = "#c5d0dc"
INK = "#1c1f24"


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=COLUMNS)
    return df


def render_table(df: pd.DataFrame) -> plt.Figure:
    font_name = configure_times_font()
    n_rows, n_cols = df.shape
    # Column width weights (relative)
    col_w = [2.6, 1.15, 1.15, 1.25, 1.35, 1.1]
    total_w = sum(col_w)
    fig_w = 9.6
    row_h = 0.42
    header_h = 0.50
    title_h = 0.70
    fig_h = title_h + header_h + n_rows * row_h + 0.35

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        total_w / 2, fig_h - 0.32,
        "Dataset Summary \u2014 Classification Benchmark",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY,
    )
    ax.text(
        total_w / 2, fig_h - 0.58,
        "Working-set size after preprocessing / downsampling; feature counts exclude the target.",
        ha="center", va="center", fontsize=8.5, color="#5a5f66", style="italic",
    )

    y0 = fig_h - title_h - header_h

    def x_edges():
        edges = [0.0]
        for w in col_w:
            edges.append(edges[-1] + w)
        return edges

    edges = x_edges()

    def draw_row(y, height, values, *, bg, fg, weight="normal", size=10.5):
        ax.add_patch(FancyBboxPatch(
            (0.02, y), total_w - 0.04, height,
            boxstyle="square,pad=0", linewidth=0, facecolor=bg, zorder=1,
        ))
        for i, val in enumerate(values):
            x = (edges[i] + edges[i + 1]) / 2
            ax.text(
                x, y + height / 2, str(val),
                ha="center", va="center", fontsize=size,
                fontweight=weight, color=fg, zorder=2,
            )
        # vertical grid
        for x in edges[1:-1]:
            ax.plot([x, x], [y, y + height], color=GRID, lw=0.6, zorder=3)
        # bottom rule
        ax.plot([0.02, total_w - 0.02], [y, y], color=GRID, lw=0.7, zorder=3)

    # Header
    draw_row(
        y0, header_h, COLUMNS,
        bg=HEADER_BG, fg=HEADER_FG, weight="bold", size=11,
    )
    ax.plot([0.02, total_w - 0.02], [y0 + header_h, y0 + header_h], color=NAVY, lw=1.4)

    # Body
    for r, row in enumerate(df.itertuples(index=False)):
        y = y0 - (r + 1) * row_h
        bg = ROW_A if r % 2 == 0 else ROW_B
        vals = [
            row.Dataset,
            f"{row.Samples:,}",
            row.Features,
            row.Numerical,
            row.Categorical,
            row.Classes,
        ]
        draw_row(y, row_h, vals, bg=bg, fg=INK, size=10.5)

    # Outer border
    bottom = y0 - n_rows * row_h
    ax.add_patch(FancyBboxPatch(
        (0.02, bottom), total_w - 0.04, header_h + n_rows * row_h,
        boxstyle="square,pad=0", linewidth=1.3, edgecolor=NAVY, facecolor="none", zorder=4,
    ))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    df = build_dataframe()
    csv_path = OUT_DIR / "dataset_summary_table.csv"
    df.to_csv(csv_path, index=False)

    fig = render_table(df)
    png_path = OUT_DIR / "dataset_summary_table.png"
    pdf_path = OUT_DIR / "dataset_summary_table.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    print(df.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
