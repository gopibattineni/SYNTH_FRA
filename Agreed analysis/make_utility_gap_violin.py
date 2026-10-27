"""Publication-quality violin plots of Utility Gap by generator.

Utility Gap = TRTR − TSTR, one value per classification dataset.
Median is overlaid on each violin. Generators ordered best → worst
(lowest median gap).

Produces Accuracy, Precision, Recall, and F1 figures:

    classification/accuracy_gap_violin.png
    classification/precision_gap_violin.png
    classification/recall_gap_violin.png
    classification/utility_gap_violin.png   (F1)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# metric_key in utility_gaps.json → (ylabel short label, output stem)
METRICS = {
    "Accuracy_Gap": ("Accuracy", "accuracy_gap_violin"),
    "Precision_Gap": ("Precision", "precision_gap_violin"),
    "Recall_Gap": ("Recall", "recall_gap_violin"),
    "F1_Gap": ("F1", "utility_gap_violin"),
}

INK = "#1c1f24"
MEDIAN = "#c0392b"
POINT = "#1c1f24"
FACE = "#ffffff"
VIOLIN_FACE = "#f0f0f0"
ZERO = "#7a7a7a"


def load_gaps(metric_key: str) -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(GAPS_JSON.read_text()))
    df = raw[
        (raw["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (raw["Metric"] == metric_key)
        & (raw["TaskType"] == "classification")
    ].copy()
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    # Order by median gap (best = lowest)
    order = (
        df.groupby("Generator")["UtilityGap"]
        .median()
        .sort_values()
        .index.tolist()
    )
    df["Generator"] = pd.Categorical(df["Generator"], categories=order, ordered=True)
    return df.sort_values("Generator")


def render(df: pd.DataFrame, metric_label: str) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font, times_fontproperties

    # Seaborn resets rcParams — apply style first, then force Times-compatible font.
    sns.set_style("white", {"axes.edgecolor": INK})
    font_name = configure_times_font()
    font_prop = times_fontproperties()
    font_prop_italic = times_fontproperties(style="italic")

    order = list(df["Generator"].cat.categories)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=FACE)
    ax.set_facecolor(FACE)
    ax.grid(False)

    palette = [VIOLIN_FACE] * len(order)

    sns.violinplot(
        data=df,
        x="Generator",
        y="UtilityGap",
        hue="Generator",
        order=order,
        hue_order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.1,
        saturation=1.0,
        legend=False,
        ax=ax,
        zorder=2,
    )

    # Black-and-white violin faces with dark edges
    for coll in ax.collections:
        coll.set_alpha(1.0)
        coll.set_facecolor(VIOLIN_FACE)
        coll.set_edgecolor(INK)
        coll.set_linewidth(1.1)

    # Overlay individual dataset points
    sns.stripplot(
        data=df,
        x="Generator",
        y="UtilityGap",
        order=order,
        color=POINT,
        size=4.5,
        alpha=0.65,
        jitter=0.08,
        ax=ax,
        zorder=3,
        legend=False,
    )

    # Median overlay (horizontal ticks + connecting markers) — keep red
    medians = df.groupby("Generator", observed=True)["UtilityGap"].median().reindex(order)
    xs = np.arange(len(order))
    ax.scatter(
        xs, medians.values,
        s=70, color=MEDIAN, zorder=5, marker="D",
        edgecolors="white", linewidths=0.8, label="Median",
    )
    for x, m in zip(xs, medians.values):
        ax.hlines(
            m, x - 0.28, x + 0.28,
            colors=MEDIAN, linewidths=2.0, zorder=4,
        )

    ax.axhline(0, color=ZERO, linestyle="--", linewidth=1.0, zorder=1, alpha=0.85)

    ax.set_xlabel("")
    ax.set_ylabel(
        f"Utility Gap  (TRTR − TSTR, {metric_label})",
        fontsize=11.5, color=INK, fontproperties=font_prop,
    )

    ax.tick_params(axis="x", labelsize=10.5, colors=INK, rotation=18)
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(font_prop)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)

    ax.legend(
        loc="upper left", frameon=True, fontsize=9.5,
        edgecolor=INK, fancybox=False, prop=font_prop,
        facecolor=FACE,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.12, 0.02,
        "* One point per classification dataset (n = 9)  ·  lower gap is better  ·  dashed line = zero gap",
        fontsize=9, color="#5a5f66",
        va="bottom", ha="left", fontproperties=font_prop_italic,
    )
    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    import matplotlib as mpl
    import zipfile

    download_dir = ROOT / "utility_gap_violins"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Outline text as paths so Times-compatible glyphs render after download.
    prev_fonttype = mpl.rcParams["svg.fonttype"]
    mpl.rcParams["svg.fonttype"] = "path"

    try:
        for metric_key, (metric_label, stem) in METRICS.items():
            df = load_gaps(metric_key)
            csv_path = OUT_DIR / f"{stem}_data.csv"
            df.to_csv(csv_path, index=False)

            fig = render(df, metric_label)

            # Re-apply after render(): configure_times_font() may reset svg.fonttype.
            mpl.rcParams["svg.fonttype"] = "path"

            png_path = OUT_DIR / f"{stem}.png"
            svg_path = OUT_DIR / f"{stem}.svg"
            fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
            fig.savefig(
                svg_path,
                format="svg",
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.25,
            )
            plt.close(fig)

            for src in (png_path, svg_path):
                (download_dir / src.name).write_bytes(src.read_bytes())

            summary = (
                df.groupby("Generator", observed=True)["UtilityGap"]
                .agg(median="median", mean="mean", std="std", min="min", max="max")
                .round(4)
            )
            print(f"\n=== {metric_label} ===")
            print(summary.to_string())
            print(f"Saved: {csv_path}")
            print(f"Saved: {png_path}")
            print(f"Saved: {svg_path}")
    finally:
        mpl.rcParams["svg.fonttype"] = prev_fonttype

    zip_svg = ROOT / "utility_gap_violins_svg.zip"
    zip_all = ROOT / "utility_gap_violins.zip"
    with zipfile.ZipFile(zip_svg, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(download_dir.glob("*.svg")):
            zf.write(p, p.name)
    with zipfile.ZipFile(zip_all, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(download_dir.glob("*")):
            if p.suffix in {".png", ".svg"}:
                zf.write(p, p.name)
    print(f"\nDownload SVG zip: {zip_svg}")
    print(f"Download all zip:  {zip_all}")
    print(f"Folder:            {download_dir}")


if __name__ == "__main__":
    main()
