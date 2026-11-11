"""Dataset-specific 4-panel matching figures for the mapping manuscript.

Central story per dataset:
  Pairwise cosine ~ 0  →  Hungarian cosine saturates  →
  Cosine amplification Δ_C  →  Hungarian Mahalanobis restores discrimination.

Layout (identical for every dataset):
  (a) Pairwise cosine
  (b) Hungarian cosine
  (c) Cosine amplification Δ_C = C_H − C_P
  (d) Hungarian Mahalanobis

Main manuscript: Fig.1 methodology + Figs 2–6 (5 representative datasets)
Supplementary: Figs S1–S10 (remaining datasets)

15 datasets = 10 classification + 5 regression (Online Shopping treated as
classification for this manuscript organization).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AGREED = ROOT / "Agreed analysis"
OUT = Path(__file__).resolve().parent
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
CSV = AGREED / "mapping_cost_comparison.csv"

sys.path.insert(0, str(AGREED))
from latex_fonts import apply_font_to_figure, configure_times_font, times_fontproperties  # noqa: E402

# Fixed generator order across ALL dataset figures
GEN_ORDER = [
    "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
    "CTABGAN", "WGAN-GP", "TabDDPM", "ForestDiffusion",
]

GEN_COLORS = {
    "CTGAN": "#1f3a5f",
    "CopulaGAN": "#D55E00",
    "TVAE": "#E69F00",
    "GaussianCopula": "#CC79A7",
    "CTABGAN": "#56B4E9",
    "WGAN-GP": "#009E73",
    "TabDDPM": "#F0E442",
    "ForestDiffusion": "#0072B2",
}

# Colorblind-friendly story colors
INK = "#1c1f24"
FACE = "#ffffff"
BLUE = "#0072B2"
ORANGE = "#E69F00"
TEAL = "#009E73"
ACCENT = "#D55E00"
SOFT_BLUE = "#D6EAF8"
SOFT_ORANGE = "#FDEBD0"
SOFT_TEAL = "#D5F5E3"
SOFT_GREY = "#F4F6F7"
NAVY = "#1f3a5f"
GRID = "#c5cdd6"

# Dataset display names and keys in mapping_cost_comparison.csv
DATASETS = [
    # (csv Dataset_Short, display name, task)
    ("Cancer", "Cancer", "classification"),
    ("Alzhimers", "Alzheimer's", "classification"),
    ("Adult", "Adult", "classification"),
    ("Forest cover dataset", "Forest Cover", "classification"),
    ("Bank Markting", "Bank Marketing", "classification"),
    ("Wine dataset", "Wine", "classification"),
    ("CDC diabetes dataset", "CDC Diabetes", "classification"),
    ("Mushroom dataset", "Mushroom", "classification"),
    ("MAGIC Gamma Telescope", "MAGIC", "classification"),
    ("online shopping", "Online Shopping", "classification"),
    ("Metro interstate", "Metro Interstate", "regression"),
    ("Air Quality", "Air Quality", "regression"),
    ("Concrete Compressive Strength", "Concrete", "regression"),
    ("Energy Efficiency", "Energy Efficiency", "regression"),
    ("Real Estate Valuation", "Real Estate", "regression"),
]

# Main paper representative set (Fig 2–6)
MAIN_DATASETS = ["Cancer", "Adult", "MAGIC", "Metro Interstate", "Real Estate"]

# Datasets with no Hungarian MD in curated extracts
MD_UNAVAILABLE = {"Cancer", "Bank Marketing"}

# Datasets with numerical instability in MD (extreme values)
MD_UNSTABLE = {"Energy Efficiency", "Real Estate"}


def style():
    configure_times_font()
    plt.rcParams.update({
        "figure.facecolor": FACE,
        "axes.facecolor": FACE,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    })


def save(fig: plt.Figure, path: Path):
    apply_font_to_figure(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), facecolor=FACE)
    plt.close(fig)
    print(f"Saved {path.with_suffix('.png').name}")


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Generator"] = df["Generator"].replace({"WGAN_GP": "WGAN-GP"})
    df["Delta_C"] = df["Hungarian_Cosine"] - df["Pairwise_Cosine"]
    return df


def dataset_slice(df: pd.DataFrame, short: str) -> pd.DataFrame:
    sub = df[df["Dataset_Short"] == short].copy()
    # Reindex to fixed generator order
    sub = sub.set_index("Generator").reindex(GEN_ORDER).reset_index()
    sub["Generator"] = GEN_ORDER
    return sub


def _bar_panel(ax, gens, values, *, face, edge, ylabel, title, ylim=None,
               ref_line=None, ref_color=ACCENT, fmt="{:.3f}", empty_msg=None,
               annotate_unstable=None):
    fp = times_fontproperties()
    x = np.arange(len(gens))
    colors = [GEN_COLORS.get(g, "#56B4E9") for g in gens]

    if empty_msg:
        ax.set_xlim(-0.5, len(gens) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(gens, rotation=35, ha="right", fontsize=8)
        ax.text(
            0.5, 0.55, empty_msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#666",
            fontproperties=times_fontproperties(style="italic"),
            bbox=dict(boxstyle="round,pad=0.4", fc=SOFT_GREY, ec=GRID),
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, fontproperties=fp, color=edge, pad=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        return

    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    plot_vals = np.where(finite, vals, 0.0)
    bars = ax.bar(
        x, plot_vals, color=colors, edgecolor=INK, width=0.72, alpha=0.92, zorder=2,
    )
    # Grey-out missing generators
    for i, ok in enumerate(finite):
        if not ok:
            bars[i].set_facecolor("#dddddd")
            bars[i].set_alpha(0.45)
            bars[i].set_height(0)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif finite.any():
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        pad = max(0.05 * (hi - lo + 1e-9), 0.01)
        ax.set_ylim(min(0.0, lo) - pad * 0.2 if lo >= 0 else lo - pad, hi + pad)

    if ref_line is not None:
        ax.axhline(ref_line, color=ref_color, ls="--", lw=1.1, zorder=1)

    # Value labels
    y0, y1 = ax.get_ylim()
    for i, (v, ok) in enumerate(zip(vals, finite)):
        if not ok:
            ax.text(i, (y0 + y1) / 2, "n/a", ha="center", va="center",
                    fontsize=7.5, color="#888", rotation=90)
            continue
        # place label above or inside depending on sign/range
        if v >= 0:
            ypos = v + 0.02 * (y1 - y0)
            va = "bottom"
        else:
            ypos = v - 0.02 * (y1 - y0)
            va = "top"
        ax.text(i, ypos, fmt.format(v), ha="center", va=va, fontsize=6.8, color=INK)

    if annotate_unstable:
        ax.text(
            0.98, 0.95, annotate_unstable, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=ACCENT,
            fontproperties=times_fontproperties(style="italic"),
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec=ACCENT, alpha=0.95),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5, fontproperties=fp, color=edge, pad=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_dataset_figure(
    df: pd.DataFrame,
    short: str,
    display: str,
    task: str,
    out_path: Path,
    fig_label: str,
):
    sub = dataset_slice(df, short)
    gens = GEN_ORDER
    pair = sub["Pairwise_Cosine"].to_numpy(dtype=float)
    hcos = sub["Hungarian_Cosine"].to_numpy(dtype=float)
    delta = sub["Delta_C"].to_numpy(dtype=float)
    md = sub["Hungarian_Mahalanobis"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    fig.suptitle(
        f"{fig_label}. Real–synthetic matching results: {display}",
        fontsize=13, fontproperties=times_fontproperties(), y=0.995, color=INK,
    )
    fig.text(
        0.5, 0.955,
        f"{task.capitalize()} dataset  ·  same generator order in all panels  ·  "
        f"lower Hungarian Mahalanobis = closer matched records",
        ha="center", fontsize=8.5, color="#555",
        fontproperties=times_fontproperties(style="italic"),
    )

    # (a) Pairwise cosine — emphasize near-zero
    _bar_panel(
        axes[0, 0], gens, pair,
        face=SOFT_BLUE, edge=BLUE,
        ylabel="Mean pairwise cosine",
        title="(a) Pairwise cosine",
        ylim=(-0.08, 0.08),
        ref_line=0.0, ref_color=GRID,
        fmt="{:.3f}",
    )

    # (b) Hungarian cosine — ceiling
    _bar_panel(
        axes[0, 1], gens, hcos,
        face=SOFT_ORANGE, edge=ORANGE,
        ylabel="Mean Hungarian cosine",
        title="(b) Hungarian cosine",
        ylim=(0.80, 1.02),
        ref_line=0.95, ref_color=ACCENT,
        fmt="{:.3f}",
    )

    # (c) Amplification
    _bar_panel(
        axes[1, 0], gens, delta,
        face=SOFT_ORANGE, edge=ACCENT,
        ylabel=r"Amplification  $\Delta_C = C_H - C_P$",
        title="(c) Cosine amplification",
        ylim=(0.70, 1.10) if np.isfinite(delta).any() else (0, 1),
        ref_line=None,
        fmt="{:.3f}",
    )

    # (d) Hungarian Mahalanobis
    if display in MD_UNAVAILABLE or not np.isfinite(md).any():
        _bar_panel(
            axes[1, 1], gens, md,
            face=SOFT_TEAL, edge=TEAL,
            ylabel="Mean Hungarian Mahalanobis",
            title="(d) Hungarian Mahalanobis",
            empty_msg="Hungarian Mahalanobis:\nunavailable for this dataset\nin curated extracts",
        )
    elif display in MD_UNSTABLE:
        # Still plot values but annotate instability (use log-friendly note)
        finite = md[np.isfinite(md)]
        ymax = float(np.nanmax(finite)) * 1.15 if len(finite) else 1.0
        _bar_panel(
            axes[1, 1], gens, md,
            face=SOFT_TEAL, edge=TEAL,
            ylabel="Mean Hungarian Mahalanobis",
            title="(d) Hungarian Mahalanobis",
            ylim=(0, ymax),
            fmt="{:.1f}",
            annotate_unstable="Numerical instability\n(ill-conditioned Σ)",
        )
    else:
        finite = md[np.isfinite(md)]
        ymax = max(float(np.nanmax(finite)) * 1.2, 1.0) if len(finite) else 5.0
        _bar_panel(
            axes[1, 1], gens, md,
            face=SOFT_TEAL, edge=TEAL,
            ylabel="Mean Hungarian Mahalanobis",
            title="(d) Hungarian Mahalanobis",
            ylim=(0, ymax),
            fmt="{:.2f}",
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.28, hspace=0.42)
    fig.text(
        0.5, 0.005,
        "Generator order fixed: CTGAN, CopulaGAN, TVAE, GaussianCopula, CTABGAN, WGAN-GP, TabDDPM, ForestDiffusion.  "
        "Grey bars = missing metric.",
        ha="center", fontsize=7.5, color="#666",
        fontproperties=times_fontproperties(style="italic"),
    )
    save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 1 — Experimental framework (research-paper block diagram)
# ---------------------------------------------------------------------------
def fig1_framework(out_path: Path):
    """Formal block diagram: shared inputs → three parallel analyses A/B/C."""
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fp = times_fontproperties()

    def rect(x, y, w, h, fc, ec, lw=1.4):
        r = mpatches.Rectangle(
            (x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
            zorder=2, clip_on=False,
        )
        ax.add_patch(r)
        return r

    def text(x, y, s, *, fs=9.5, color=INK, weight=None, ha="center", va="center"):
        ax.text(
            x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontproperties=fp, zorder=3, linespacing=1.35,
        )

    def vline(x, y1, y2, color=NAVY, lw=1.2):
        ax.plot([x, x], [y1, y2], color=color, lw=lw, zorder=1, solid_capstyle="butt")

    def hline(x1, x2, y, color=NAVY, lw=1.2):
        ax.plot([x1, x2], [y, y], color=color, lw=lw, zorder=1, solid_capstyle="butt")

    # ---- Row 1: Experimental setting (full width) ----
    rect(4, 86, 92, 11, SOFT_GREY, NAVY, lw=1.6)
    text(50, 93.5, "Experimental setting", fs=11, color=NAVY)
    text(
        50, 89.0,
        "15 tabular datasets  (10 classification + 5 regression)   ·   8 generators\n"
        "Generators trained on real training data only  →  synthetic sample S",
        fs=8.5, color="#333",
    )

    # connector
    vline(50, 86, 80.5)

    # ---- Row 2: Shared controlled inputs ----
    rect(4, 66, 92, 14.5, SOFT_BLUE, BLUE, lw=1.6)
    text(50, 77.0, "Shared inputs  (held fixed across analyses)", fs=11, color=BLUE)
    # three equal sub-blocks inside
    gap, iw, ih = 2.0, 28.0, 7.5
    x0 = 8.0
    for i, (lab, sub) in enumerate([
        ("Real data  R", "training records"),
        ("Synthetic data  S", "generator output"),
        ("Feature matrix", "numeric; optional scaling"),
    ]):
        xi = x0 + i * (iw + gap)
        rect(xi, 67.5, iw, ih, "#ffffff", BLUE, lw=1.1)
        text(xi + iw / 2, 72.2, lab, fs=9, color=BLUE)
        text(xi + iw / 2, 69.2, sub, fs=7.5, color="#555")

    # connector fan-out to three columns
    vline(50, 66, 60)
    hline(17, 83, 60)
    vline(17, 60, 56.5)
    vline(50, 60, 56.5)
    vline(83, 60, 56.5)

    # ---- Row 3: Three parallel analysis blocks A / B / C ----
    cols = [
        (4, "A", "Pairwise cosine",
         "All-pairs similarity\n"
         r"$C_{ij}=r_i^{\top}s_j$" "\n"
         "Report mean  " r"$\bar C$",
         SOFT_BLUE, BLUE),
        (36, "B", "Hungarian + cosine",
         "Optimal assignment\n"
         "cost  =  1 " r"$-$" " cos\n"
         "Report mean matched cos",
         SOFT_ORANGE, ORANGE),
        (68, "C", "Hungarian + Mahalanobis",
         "Optimal assignment\n"
         "cost  =  Mahalanobis " r"$D$" "\n"
         r"$\Sigma$ estimated on real data",
         SOFT_TEAL, TEAL),
    ]
    bw, bh = 28, 28
    for x, letter, title, body, fc, ec in cols:
        rect(x, 28, bw, bh, fc, ec, lw=1.8)
        # letter badge
        rect(x + 1.2, 49.5, 4.2, 4.8, ec, ec, lw=0)
        text(x + 3.3, 51.9, letter, fs=11, color="white")
        text(x + bw / 2, 51.5, title, fs=10, color=ec)
        text(x + bw / 2, 39.5, body, fs=8.2, color="#222")

    # Labels under columns
    text(18, 25.5, "Analysis A", fs=8, color=BLUE)
    text(50, 25.5, "Analysis B", fs=8, color=ORANGE)
    text(82, 25.5, "Analysis C", fs=8, color=TEAL)

    # fan-in
    vline(17, 28, 23.5)
    vline(50, 28, 23.5)
    vline(83, 28, 23.5)
    hline(17, 83, 23.5)
    vline(50, 23.5, 19.5)

    # ---- Row 4: Output / interpretation ----
    rect(4, 5, 92, 14.5, "#FFF8E7", NAVY, lw=1.6)
    text(50, 16.0, "Record-level comparison", fs=11, color=NAVY)
    text(
        50, 10.5,
        "Same  R,  S,  and Hungarian solver;  only the matching cost changes.\n"
        "Each dataset is an independent case study (fixed generator order).\n"
        "Key contrast:  pairwise cosine  vs  Hungarian cosine  vs  Hungarian Mahalanobis.",
        fs=8.2, color="#333",
    )

    ax.set_title(
        "Figure 1. Block diagram of the experimental framework",
        fontsize=12.5, pad=10, fontproperties=fp, color=INK,
    )
    save(fig, out_path)


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("'", "")


def main():
    style()
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    df = load_df()

    # Fig 1
    fig1_framework(MAIN / "Fig1_experimental_framework")

    # Map display name -> meta
    meta = {d[1]: d for d in DATASETS}

    # Main Figs 2–6
    main_map = [
        (2, "Cancer"),
        (3, "Adult"),
        (4, "MAGIC"),
        (5, "Metro Interstate"),
        (6, "Real Estate"),
    ]
    for num, display in main_map:
        short, _, task = meta[display]
        plot_dataset_figure(
            df, short, display, task,
            MAIN / f"Fig{num}_{slug(display)}",
            fig_label=f"Figure {num}",
        )

    # Supplementary S1–S10 (remaining, classification then regression)
    remaining = [d[1] for d in DATASETS if d[1] not in MAIN_DATASETS]
    for i, display in enumerate(remaining, start=1):
        short, _, task = meta[display]
        plot_dataset_figure(
            df, short, display, task,
            SUPP / f"FigS{i}_{slug(display)}",
            fig_label=f"Figure S{i}",
        )

    # Also write a flat by_dataset copy for convenience
    by_ds = OUT / "by_dataset"
    by_ds.mkdir(exist_ok=True)
    for short, display, task in DATASETS:
        plot_dataset_figure(
            df, short, display, task,
            by_ds / f"{slug(display)}_matching",
            fig_label=display,
        )

    # README
    (OUT / "README.md").write_text(
        """# Mapping conference figures (dataset case studies)

**Story:** Pairwise cosine ≈ 0 → Hungarian cosine saturates → Δ_C amplification → Hungarian Mahalanobis restores discrimination.

**Design:** Each dataset is an independent experimental unit (8 generators, fixed order).

**Task split used here:** 10 classification + 5 regression (Online Shopping listed with classification for this manuscript organization).

## Main manuscript (`main/`)

| Figure | File | Content |
|------|------|---------|
| Fig. 1 | `Fig1_experimental_framework` | Methods / A–B–C cost control |
| Fig. 2 | `Fig2_Cancer` | Cancer 4-panel case study |
| Fig. 3 | `Fig3_Adult` | Adult 4-panel case study |
| Fig. 4 | `Fig4_MAGIC` | MAGIC 4-panel case study |
| Fig. 5 | `Fig5_Metro_Interstate` | Metro Interstate 4-panel case study |
| Fig. 6 | `Fig6_Real_Estate` | Real Estate 4-panel case study (MD instability noted) |

Each dataset figure panels:
**(a)** Pairwise cosine · **(b)** Hungarian cosine · **(c)** Cosine amplification Δ_C · **(d)** Hungarian Mahalanobis

## Supplementary (`supplementary/`)

| Figure | Dataset |
|------|---------|
| Fig. S1 | Alzheimer's |
| Fig. S2 | Forest Cover |
| Fig. S3 | Bank Marketing (MD unavailable) |
| Fig. S4 | Wine |
| Fig. S5 | CDC Diabetes |
| Fig. S6 | Mushroom |
| Fig. S7 | Online Shopping |
| Fig. S8 | Air Quality |
| Fig. S9 | Concrete |
| Fig. S10 | Energy Efficiency (MD instability noted) |

Flat copies of all 15 dataset figures also live in `by_dataset/`.

## Generator order (fixed)

1. CTGAN  2. CopulaGAN  3. TVAE  4. GaussianCopula
5. CTABGAN  6. WGAN-GP  7. TabDDPM  8. ForestDiffusion

## Regenerate

```bash
python Results/mapping_conference/make_dataset_case_figures.py
```
""",
        encoding="utf-8",
    )
    print(f"\nMain figures → {MAIN}")
    print(f"Supplementary → {SUPP}")
    print(f"All datasets → {by_ds}")


if __name__ == "__main__":
    main()
