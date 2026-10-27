"""Manuscript figures for Hungarian matching cost comparison.

Central story:
  Pairwise cosine ~ 0  →  Hungarian cosine saturates  →
  Hungarian Mahalanobis restores discrimination.

Outputs PNG + PDF under Results/mapping_conference/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
AGREED = ROOT / "Agreed analysis"
OUT = Path(__file__).resolve().parent
PRIV = ROOT / "excel sheets" / "2. Privacy"
CSV = AGREED / "mapping_cost_comparison.csv"

sys.path.insert(0, str(AGREED))
from latex_fonts import apply_font_to_figure, configure_times_font, times_fontproperties  # noqa: E402

CLASSIFICATION = {
    "Cancer", "Alzhimers", "Adult", "Forest cover dataset", "Bank Markting",
    "Wine dataset", "CDC diabetes dataset", "Mushroom dataset", "MAGIC Gamma Telescope",
}
REGRESSION = {
    "Metro interstate", "online shopping", "Air Quality",
    "Concrete Compressive Strength", "Energy Efficiency", "Real Estate Valuation",
}

GEN_ORDER = [
    "ForestDiffusion", "WGAN-GP", "TVAE", "CTABGAN",
    "GaussianCopula", "CopulaGAN", "TabDDPM", "CTGAN",
]
GEN_FAMILY = {
    "CTGAN": "SDV", "CopulaGAN": "SDV", "TVAE": "SDV", "GaussianCopula": "SDV",
    "CTABGAN": "Other GAN", "WGAN-GP": "Other GAN",
    "TabDDPM": "Diffusion", "ForestDiffusion": "Diffusion",
}

DS_ORDER = [
    "Cancer", "Alzhimers", "Adult", "Forest cover dataset", "Bank Markting",
    "Wine dataset", "CDC diabetes dataset", "Mushroom dataset", "MAGIC Gamma Telescope",
    "Metro interstate", "online shopping", "Air Quality",
    "Concrete Compressive Strength", "Energy Efficiency", "Real Estate Valuation",
]
DS_SHORT = {
    "Cancer": "Cancer", "Alzhimers": "Alzheimer's", "Adult": "Adult",
    "Forest cover dataset": "Forest Cover", "Bank Markting": "Bank Marketing",
    "Wine dataset": "Wine", "CDC diabetes dataset": "CDC Diabetes",
    "Mushroom dataset": "Mushroom", "MAGIC Gamma Telescope": "MAGIC",
    "Metro interstate": "Metro", "online shopping": "Online Shopping",
    "Air Quality": "Air Quality", "Concrete Compressive Strength": "Concrete",
    "Energy Efficiency": "Energy", "Real Estate Valuation": "Real Estate",
}

# Colorblind-friendly manuscript palette (Okabe–Ito inspired)
INK = "#1c1f24"
FACE = "#ffffff"
ACCENT = "#D55E00"       # vermillion — thresholds / medians
BLUE = "#0072B2"         # pairwise / analysis A
TEAL = "#009E73"         # Mahalanobis / analysis C
ORANGE = "#E69F00"       # Hungarian cosine / analysis B
SKY = "#56B4E9"
RED = "#CC79A7"
NAVY = "#1f3a5f"
SOFT_BLUE = "#D6EAF8"
SOFT_ORANGE = "#FDEBD0"
SOFT_TEAL = "#D5F5E3"
SOFT_GREY = "#F4F6F7"
GRID = "#c5cdd6"

GEN_COLORS = {
    "ForestDiffusion": "#0072B2",
    "WGAN-GP": "#009E73",
    "TVAE": "#E69F00",
    "CTABGAN": "#56B4E9",
    "GaussianCopula": "#CC79A7",
    "CopulaGAN": "#D55E00",
    "TabDDPM": "#F0E442",
    "CTGAN": "#1f3a5f",
}


def style():
    configure_times_font()
    sns.set_style("white")
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
        "savefig.pad_inches": 0.15,
    })


def save(fig: plt.Figure, stem: str):
    apply_font_to_figure(fig)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor=FACE)
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf")


def load_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["Generator"] = df["Generator"].replace({"WGAN_GP": "WGAN-GP"})
    df["Task"] = df["Dataset_Short"].map(
        lambda d: "Classification" if d in CLASSIFICATION else "Regression"
    )
    df["Family"] = df["Generator"].map(GEN_FAMILY)
    df["Delta_C"] = df["Hungarian_Cosine"] - df["Pairwise_Cosine"]
    df["Dataset_Label"] = df["Dataset_Short"].map(lambda d: DS_SHORT.get(d, d))
    return df


# ---------------------------------------------------------------------------
# Figure 1 — Experimental framework
# ---------------------------------------------------------------------------
def fig1_framework():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fp = times_fontproperties()
    fp_b = times_fontproperties()

    def box(x, y, w, h, text, fc="#f5f5f5", ec=INK, lw=1.2, fs=9.5, bold=False):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
        ax.add_patch(r)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontproperties=(fp_b if bold else fp), zorder=3,
            wrap=True,
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3),
            zorder=1,
        )

    # Top pipeline
    box(4.4, 8.7, 3.2, 0.85, "Real training data", fc=SOFT_BLUE, ec=BLUE, bold=True)
    arrow(6.0, 8.7, 6.0, 8.15)
    box(4.4, 7.3, 3.2, 0.85, "Train generator → Synthetic data", fc=SOFT_BLUE, ec=BLUE, bold=True)
    arrow(6.0, 7.3, 6.0, 6.75)
    box(4.1, 5.9, 3.8, 0.85, "Feature preprocessing\n(numeric; optional scaling)",
        fc=SOFT_GREY, ec=NAVY, bold=True)

    # Branch
    arrow(5.0, 5.9, 2.7, 5.15)
    arrow(7.0, 5.9, 9.3, 5.15)

    # Left: cosine path
    box(0.6, 4.35, 4.2, 0.9, "Cosine similarity matrix\n(direction-only, L2-normalized)",
        fc=SOFT_ORANGE, ec=ORANGE)
    arrow(2.7, 4.35, 2.7, 3.8)
    box(0.4, 2.55, 2.2, 1.2, "A\nPairwise cosine\n(mean over all pairs)",
        fc=SOFT_BLUE, ec=BLUE, lw=1.8, bold=True, fs=9)
    box(2.85, 2.55, 2.35, 1.2, "B\nHungarian + cosine\n(cost = 1 − cos)",
        fc=SOFT_ORANGE, ec=ORANGE, lw=1.8, bold=True, fs=9)

    # Right: MD path
    box(7.2, 4.35, 4.2, 0.9, "Mahalanobis distance matrix\n(Σ from real data)",
        fc=SOFT_TEAL, ec=TEAL)
    arrow(9.3, 4.35, 9.3, 3.8)
    box(7.5, 2.55, 3.6, 1.2, "C\nHungarian + Mahalanobis\n(cost = Mahalanobis D)",
        fc=SOFT_TEAL, ec=TEAL, lw=1.8, bold=True, fs=9)

    # Merge control statement
    box(1.5, 0.55, 9.0, 1.35,
        "Experimental control\n"
        "Same real data  ·  Same synthetic data  ·  Same Hungarian algorithm\n"
        "Only the matching cost changes  (cosine vs Mahalanobis)",
        fc="#FFF8E7", ec=NAVY, lw=1.5, bold=True, fs=10)

    arrow(2.7, 2.55, 4.5, 1.95)
    arrow(4.0, 2.55, 5.5, 1.95)
    arrow(9.3, 2.55, 7.5, 1.95)

    ax.set_title(
        "Figure 1. Experimental framework for record-level matching costs",
        fontsize=12, pad=8, fontproperties=fp_b, color=INK,
    )
    fig.text(
        0.5, 0.01,
        "A–C are evaluated on identical real–synthetic pairs; B and C share the Hungarian assignment solver.",
        ha="center", fontsize=8.5, fontproperties=times_fontproperties(style="italic"), color="#555",
    )
    save(fig, "Fig1_experimental_framework")


# ---------------------------------------------------------------------------
# Figure 2 — Ceiling-effect scatter
# ---------------------------------------------------------------------------
def fig2_ceiling(df: pd.DataFrame):
    sub = df.dropna(subset=["Pairwise_Cosine", "Hungarian_Cosine"]).copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    markers = {"Classification": "o", "Regression": "s"}
    colors = {"Classification": BLUE, "Regression": ORANGE}
    for task, m in markers.items():
        t = sub[sub["Task"] == task]
        ax.scatter(
            t["Pairwise_Cosine"], t["Hungarian_Cosine"],
            s=48, marker=m, c=colors[task],
            edgecolors="white", linewidths=0.5, alpha=0.88,
            label=task, zorder=3,
        )

    ax.axhline(0.95, color=ACCENT, ls="--", lw=1.2, zorder=1, label="y = 0.95")
    ax.axhline(0.0, color=GRID, ls=":", lw=0.9, zorder=0)
    ax.axvline(0.0, color=GRID, ls=":", lw=0.9, zorder=0)

    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Hungarian cosine similarity")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(0.75, 1.02)
    ax.legend(frameon=True, edgecolor=INK, fontsize=9, loc="lower right",
              facecolor="white", framealpha=0.95)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.text(
        0.02, 0.98,
        f"n = {len(sub)} dataset × generator cells\n"
        f"mean pairwise = {sub['Pairwise_Cosine'].mean():.3f}\n"
        f"mean Hungarian = {sub['Hungarian_Cosine'].mean():.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        fontproperties=times_fontproperties(),
        bbox=dict(boxstyle="round,pad=0.3", fc=SOFT_ORANGE, ec=ORANGE, alpha=0.95),
    )
    fig.suptitle(
        "Effect of optimal assignment on cosine similarity",
        fontsize=12, fontproperties=times_fontproperties(), y=0.98,
    )
    save(fig, "Fig2_pairwise_vs_hungarian_cosine")


# ---------------------------------------------------------------------------
# Figure 3 — Three-panel comparison
# ---------------------------------------------------------------------------
def fig3_threepanel(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    order = [g for g in GEN_ORDER if g in set(df["Generator"])]

    panels = [
        ("Pairwise_Cosine", "Pairwise cosine", (-0.08, 0.08), SOFT_BLUE, BLUE),
        ("Hungarian_Cosine", "Hungarian cosine", (0.75, 1.02), SOFT_ORANGE, ORANGE),
        ("Hungarian_Mahalanobis", "Hungarian Mahalanobis", None, SOFT_TEAL, TEAL),
    ]
    for ax, (col, title, ylim, face, edge) in zip(axes, panels):
        plot_df = df.dropna(subset=[col]).copy()
        plot_df = plot_df[plot_df["Generator"].isin(order)]
        sns.boxplot(
            data=plot_df, x="Generator", y=col, order=order,
            color=face, linewidth=1.0, fliersize=2.5,
            boxprops=dict(edgecolor=edge),
            medianprops=dict(color=ACCENT, linewidth=1.8),
            whiskerprops=dict(color=edge),
            capprops=dict(color=edge),
            flierprops=dict(marker="o", markerfacecolor=edge, markersize=3, alpha=0.55),
            ax=ax,
        )
        ax.set_title(title, fontsize=11, fontproperties=times_fontproperties(), color=edge)
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8.5)
        if ylim:
            ax.set_ylim(*ylim)
        if col == "Hungarian_Cosine":
            ax.axhline(0.95, color=ACCENT, ls="--", lw=0.9, alpha=0.85)
        if col == "Pairwise_Cosine":
            ax.axhline(0.0, color=GRID, ls=":", lw=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Cap extreme MD for readability in panel c (note outliers beyond)
    md = df["Hungarian_Mahalanobis"].dropna()
    if len(md):
        q99 = md.quantile(0.9)
        axes[2].set_ylim(0, max(8, min(q99 * 1.15, 40)))
        n_hi = (md > axes[2].get_ylim()[1]).sum()
        if n_hi:
            axes[2].text(
                0.98, 0.98, f"{n_hi} extreme values clipped\n(Energy / Real Estate)",
                transform=axes[2].transAxes, ha="right", va="top", fontsize=7.5,
                fontproperties=times_fontproperties(style="italic"), color="#555",
            )

    fig.suptitle(
        "Matching cost determines discrimination among generators",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    save(fig, "Fig3_three_panel_cost_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — MAGIC case study
# ---------------------------------------------------------------------------
def fig4_magic(df: pd.DataFrame):
    magic = df[df["Dataset_Short"] == "MAGIC Gamma Telescope"].dropna(
        subset=["Hungarian_Cosine", "Hungarian_Mahalanobis"]
    ).copy()
    # Order by Hungarian Mahalanobis ascending (closer = better overlap)
    magic = magic.sort_values("Hungarian_Mahalanobis")
    gens = magic["Generator"].tolist()
    y = np.arange(len(gens))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

    bar_colors = [GEN_COLORS.get(g, SKY) for g in gens]
    axes[0].barh(y, magic["Hungarian_Cosine"], color=bar_colors, edgecolor=INK,
                 height=0.65, alpha=0.9)
    axes[0].axvline(0.95, color=ACCENT, ls="--", lw=1.1)
    axes[0].set_xlim(0.90, 1.005)
    axes[0].set_xlabel("Hungarian cosine")
    axes[0].set_title("(a) Hungarian cosine", fontproperties=times_fontproperties(), color=ORANGE)
    for yi, v in zip(y, magic["Hungarian_Cosine"]):
        axes[0].text(v - 0.002, yi, f"{v:.3f}", va="center", ha="right", fontsize=8, color=INK)

    axes[1].barh(y, magic["Hungarian_Mahalanobis"], color=bar_colors, edgecolor=INK,
                 height=0.65, alpha=0.9)
    axes[1].set_xlabel("Hungarian Mahalanobis distance")
    axes[1].set_title("(b) Hungarian Mahalanobis", fontproperties=times_fontproperties(), color=TEAL)
    for yi, v in zip(y, magic["Hungarian_Mahalanobis"]):
        axes[1].text(v + 0.05, yi, f"{v:.2f}", va="center", ha="left", fontsize=8, color=INK)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(gens)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "MAGIC case study: same matching, different cost",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    save(fig, "Fig4_MAGIC_case_study")


# ---------------------------------------------------------------------------
# Figure 5 — Heatmaps
# ---------------------------------------------------------------------------
def fig5_heatmaps(df: pd.DataFrame):
    metrics = [
        ("Pairwise_Cosine", "Pairwise cosine", "RdBu_r", (-0.06, 0.06)),
        ("Hungarian_Cosine", "Hungarian cosine", "YlOrBr", (0.80, 1.00)),
        ("Hungarian_Mahalanobis", "Hungarian Mahalanobis", "YlGnBu", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.2))
    gens = [g for g in GEN_ORDER if g in set(df["Generator"])]
    ds_labels = [DS_SHORT[d] for d in DS_ORDER if d in set(df["Dataset_Short"])]

    for ax, (col, title, cmap, vlim) in zip(axes, metrics):
        piv = (
            df.pivot_table(index="Dataset_Short", columns="Generator", values=col, aggfunc="mean")
            .reindex(index=[d for d in DS_ORDER if d in df["Dataset_Short"].unique()], columns=gens)
        )
        piv.index = [DS_SHORT.get(i, i) for i in piv.index]
        # Clip MD display for heatmap readability
        data = piv.copy()
        if col == "Hungarian_Mahalanobis":
            data = data.clip(upper=20)
            vlim = (0, 20)
        sns.heatmap(
            data, ax=ax, cmap=cmap, vmin=vlim[0] if vlim else None, vmax=vlim[1] if vlim else None,
            linewidths=0.4, linecolor="white", cbar_kws={"shrink": 0.6},
            xticklabels=True, yticklabels=True,
        )
        ax.set_title(title, fontsize=11, fontproperties=times_fontproperties())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(
        "Benchmark-wide matching costs (15 datasets × 8 generators)",
        fontsize=12, fontproperties=times_fontproperties(), y=1.01,
    )
    fig.tight_layout()
    fig.text(
        0.98, 0.01,
        "Mahalanobis panel clipped at 20 for display; Energy/Real Estate exceed this range.",
        ha="right", fontsize=7.5, fontproperties=times_fontproperties(style="italic"), color="#555",
    )
    save(fig, "Fig5_benchmark_heatmaps")


# ---------------------------------------------------------------------------
# Figure 6 — Cosine amplification Δ_C
# ---------------------------------------------------------------------------
def fig6_amplification(df: pd.DataFrame):
    sub = df.dropna(subset=["Delta_C"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [1, 1.6]})

    # Box overall
    sns.boxplot(
        y=sub["Delta_C"], ax=axes[0], color=SOFT_ORANGE, width=0.45,
        medianprops=dict(color=ACCENT, linewidth=2.0),
        boxprops=dict(edgecolor=ORANGE), whiskerprops=dict(color=ORANGE),
        capprops=dict(color=ORANGE),
        flierprops=dict(marker="o", markerfacecolor=ORANGE, markersize=3, alpha=0.5),
    )
    axes[0].set_ylabel(r"Amplification  $\Delta_C = C_{\mathrm{Hungarian}} - C_{\mathrm{Pairwise}}$")
    axes[0].set_xlabel("")
    axes[0].set_title("(a) Overall", fontproperties=times_fontproperties(), color=ORANGE)
    axes[0].set_xticks([])
    for spine in ("top", "right", "bottom"):
        axes[0].spines[spine].set_visible(False)

    # Heatmap by dataset × generator
    piv = (
        sub.pivot_table(index="Dataset_Short", columns="Generator", values="Delta_C", aggfunc="mean")
        .reindex(index=[d for d in DS_ORDER if d in sub["Dataset_Short"].unique()],
                 columns=[g for g in GEN_ORDER if g in set(sub["Generator"])])
    )
    piv.index = [DS_SHORT.get(i, i) for i in piv.index]
    sns.heatmap(
        piv, ax=axes[1], cmap="YlOrRd", vmin=0.75, vmax=1.05,
        linewidths=0.35, linecolor="white", cbar_kws={"shrink": 0.7, "label": r"$\Delta_C$"},
    )
    axes[1].set_title("(b) By dataset × generator", fontproperties=times_fontproperties(), color=ACCENT)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Assignment-induced cosine amplification",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    save(fig, "Fig6_cosine_amplification")


# ---------------------------------------------------------------------------
# Figure 7 — Ranking consistency
# ---------------------------------------------------------------------------
def fig7_rankings(df: pd.DataFrame):
    # Rank within each dataset (higher cosine = rank 1; lower MD = rank 1)
    rows = []
    for ds, g in df.groupby("Dataset_Short"):
        g = g.copy()
        if g["Hungarian_Cosine"].notna().sum() >= 4:
            g["rank_hcos"] = g["Hungarian_Cosine"].rank(ascending=False, method="average")
        else:
            g["rank_hcos"] = np.nan
        if g["Pairwise_Cosine"].notna().sum() >= 4:
            g["rank_pair"] = g["Pairwise_Cosine"].rank(ascending=False, method="average")
        else:
            g["rank_pair"] = np.nan
        if g["Hungarian_Mahalanobis"].notna().sum() >= 4:
            g["rank_md"] = g["Hungarian_Mahalanobis"].rank(ascending=True, method="average")
        else:
            g["rank_md"] = np.nan
        rows.append(g)
    ranked = pd.concat(rows, ignore_index=True)

    # Mean ranks across datasets
    mean_rank = (
        ranked.groupby("Generator")[["rank_pair", "rank_hcos", "rank_md"]]
        .mean()
        .reindex(GEN_ORDER)
        .dropna(how="all")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Bump / parallel coordinates of mean ranks
    ax = axes[0]
    xlabels = ["Pairwise\ncosine", "Hungarian\ncosine", "Hungarian\nMahalanobis"]
    xs = [0, 1, 2]
    for gen, row in mean_rank.iterrows():
        ys = [row["rank_pair"], row["rank_hcos"], row["rank_md"]]
        if any(pd.isna(ys)):
            continue
        c = GEN_COLORS.get(gen, NAVY)
        ax.plot(xs, ys, "-o", color=c, lw=1.6, ms=6, alpha=0.9)
        ax.text(2.05, ys[2], gen, va="center", fontsize=8, color=c,
                fontproperties=times_fontproperties())
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Mean rank (1 = best)")
    ax.set_ylim(8.5, 0.5)
    ax.set_xlim(-0.2, 2.9)
    ax.set_title("(a) Mean generator ranks", fontproperties=times_fontproperties())
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Spearman correlations of ranks across datasets
    from scipy.stats import spearmanr
    pairs = [
        ("rank_pair", "rank_hcos", "Pairwise vs\nHung. cosine"),
        ("rank_pair", "rank_md", "Pairwise vs\nHung. MD"),
        ("rank_hcos", "rank_md", "Hung. cosine vs\nHung. MD"),
    ]
    corrs = []
    labels = []
    for a, b, lab in pairs:
        sub = ranked[[a, b]].dropna()
        if len(sub) < 5:
            rho = np.nan
        else:
            rho, _ = spearmanr(sub[a], sub[b])
        corrs.append(rho)
        labels.append(lab)

    ax = axes[1]
    bar_cols = [ACCENT if (pd.notna(c) and c < 0.5) else TEAL for c in corrs]
    bars = ax.bar(labels, corrs, color=bar_cols, edgecolor=INK, width=0.55, alpha=0.9)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel(r"Spearman $\rho$ of within-dataset ranks")
    ax.set_title("(b) Rank agreement", fontproperties=times_fontproperties())
    for b, c in zip(bars, corrs):
        if pd.notna(c):
            ax.text(b.get_x() + b.get_width() / 2, c + 0.03, f"{c:.2f}",
                    ha="center", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Does the matching cost change which generators look better?",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    save(fig, "Fig7_generator_ranking_consistency")


# ---------------------------------------------------------------------------
# Figure 8 — Computational scaling
# ---------------------------------------------------------------------------
def fig8_compute():
    rng = np.random.default_rng(42)
    ns = [100, 200, 400, 600, 800, 1000, 1200]
    build_t, assign_t, total_t, mem_mb = [], [], [], []

    for n in ns:
        X = rng.normal(size=(n, 20))
        Y = rng.normal(size=(n, 20))
        # normalize for cosine-like matrix
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

        t0 = time.perf_counter()
        C = X @ Y.T
        cost = 1.0 - C
        t1 = time.perf_counter()
        linear_sum_assignment(cost)
        t2 = time.perf_counter()

        build_t.append(t1 - t0)
        assign_t.append(t2 - t1)
        total_t.append(t2 - t0)
        mem_mb.append(cost.nbytes / (1024 ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    ax = axes[0]
    ax.plot(ns, build_t, "-o", color=SKY, label="Cost-matrix construction", ms=6, lw=1.8)
    ax.plot(ns, assign_t, "-s", color=ACCENT, label="Hungarian assignment", ms=6, lw=1.8)
    ax.plot(ns, total_t, "-^", color=BLUE, label="Total", ms=6, lw=1.8)
    ax.set_xlabel("Number of records  n")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("(a) Runtime vs n", fontproperties=times_fontproperties())
    ax.legend(frameon=True, edgecolor=INK, fontsize=8, facecolor="white")
    ax.text(0.05, 0.95, r"Assignment $\sim O(n^3)$", transform=ax.transAxes,
            va="top", fontsize=8.5, fontproperties=times_fontproperties(style="italic"), color=ACCENT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.plot(ns, mem_mb, "-o", color=TEAL, ms=6, lw=1.8)
    ax.fill_between(ns, mem_mb, color=SOFT_TEAL, alpha=0.7)
    ax.set_xlabel("Number of records  n")
    ax.set_ylabel("Cost-matrix memory (MB)")
    ax.set_title("(b) Memory vs n", fontproperties=times_fontproperties())
    ax.text(0.05, 0.95, r"Memory $\sim O(n^2)$", transform=ax.transAxes,
            va="top", fontsize=8.5, fontproperties=times_fontproperties(style="italic"), color=TEAL)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Computational cost of Hungarian record matching",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.02,
        "Timing on synthetic n×n cosine cost matrices (d = 20 features); justifies protocol subsampling on large tables.",
        ha="center", fontsize=8, fontproperties=times_fontproperties(style="italic"), color="#555",
    )
    save(fig, "Fig8_computational_scaling")


# ---------------------------------------------------------------------------
# Figure 9 — Covariance stability (ridge sweep on MAGIC-like / EE dims)
# ---------------------------------------------------------------------------
def fig9_covariance_stability():
    """Demonstrate ridge sensitivity of mean Hungarian MD for typical vs ill-conditioned Σ."""
    rng = np.random.default_rng(0)
    lambdas = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1])

    configs = [
        ("Well-conditioned (MAGIC-like, d=10)", 10, 1.0),
        ("Ill-conditioned (Energy-like, d=8)", 8, 1e-6),
        ("High-d collinear (RealEstate-like, d=6)", 6, 1e-8),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=False)

    panel_colors = [TEAL, ORANGE, ACCENT]
    for ax, (title, d, min_eig), pc in zip(axes, configs, panel_colors):
        # Build covariance with controlled min eigenvalue
        A = rng.normal(size=(d, d))
        cov = A @ A.T
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_eig)
        # stretch spectrum for ill-conditioned cases
        if min_eig < 1e-3:
            eigvals = np.linspace(min_eig, 1.0, d)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cond = np.linalg.cond(cov)

        mu = np.zeros(d)
        R = rng.multivariate_normal(mu, cov, size=400)
        S = rng.multivariate_normal(mu, cov * 1.1, size=400)  # slight shift in scale

        means = []
        for lam in lambdas:
            VI = np.linalg.pinv(cov + lam * np.eye(d))
            # sample subset for speed
            n = 200
            diff = R[:n, None, :] - S[None, :n, :]
            # mahalanobis squared approx via einsum
            # D_ij^2 = diff_ij VI diff_ij
            D2 = np.einsum("ijk,kl,ijl->ij", diff, VI, diff)
            D = np.sqrt(np.maximum(D2, 0))
            ri, ci = linear_sum_assignment(D)
            means.append(float(D[ri, ci].mean()))

        ax.plot(lambdas, means, "-o", color=pc, ms=6, lw=1.8)
        ax.fill_between(lambdas, means, color=pc, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel(r"Ridge $\lambda$")
        ax.set_ylabel("Mean Hungarian MD")
        ax.set_title(title, fontsize=9.5, fontproperties=times_fontproperties(), color=pc)
        ax.text(
            0.05, 0.95, f"cond(Σ) ≈ {cond:.1e}",
            transform=ax.transAxes, va="top", fontsize=8,
            fontproperties=times_fontproperties(style="italic"), color=pc,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Sensitivity of Hungarian Mahalanobis to covariance regularization",
        fontsize=12, fontproperties=times_fontproperties(), y=1.03,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.03,
        "Illustrative synthetic covariances; large condition numbers (Energy / Real Estate regime) amplify λ dependence.",
        ha="center", fontsize=8, fontproperties=times_fontproperties(style="italic"), color="#555",
    )
    save(fig, "Fig9_mahalanobis_covariance_stability")


# ---------------------------------------------------------------------------
# Figure 10 — ECDF of matched MD (MAGIC)
# ---------------------------------------------------------------------------
def fig10_ecdf_magic():
    path = PRIV / "9. MAGIC Gamma Telescope" / "Mahalanobis_Detail.xlsx"
    d = pd.read_excel(path)
    gcol = "Generator" if "Generator" in d.columns else "Model"
    d[gcol] = d[gcol].replace({"WGAN_GP": "WGAN-GP"})
    dist_col = "Mahalanobis_Distance"

    # Focus on four illustrative generators spanning the range
    focus = ["WGAN-GP", "ForestDiffusion", "CTGAN", "TabDDPM"]
    styles = {
        "WGAN-GP": ("-", GEN_COLORS["WGAN-GP"]),
        "ForestDiffusion": ("--", GEN_COLORS["ForestDiffusion"]),
        "CTGAN": ("-.", GEN_COLORS["CTGAN"]),
        "TabDDPM": (":", GEN_COLORS["TabDDPM"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    for g in focus:
        vals = np.sort(d.loc[d[gcol] == g, dist_col].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ls, c = styles[g]
        ax.plot(vals, y, ls=ls, color=c, lw=2.0, label=g)
    ax.set_xlabel("Matched Mahalanobis distance")
    ax.set_ylabel("ECDF")
    ax.set_title("(a) ECDF (selected generators)", fontproperties=times_fontproperties())
    ax.legend(frameon=True, edgecolor=INK, fontsize=8, facecolor="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    order = [g for g in GEN_ORDER if g in set(d[gcol])]
    plot_df = d[d[gcol].isin(order)][[gcol, dist_col]].rename(columns={gcol: "Generator", dist_col: "MD"})
    palette = {g: GEN_COLORS.get(g, SKY) for g in order}
    sns.boxplot(
        data=plot_df, x="Generator", y="MD", order=order, hue="Generator",
        palette=palette, legend=False,
        linewidth=1.0, fliersize=1.5,
        medianprops=dict(color=ACCENT, linewidth=1.6),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Matched Mahalanobis distance")
    ax.set_title("(b) Full MAGIC generator boxplots", fontproperties=times_fontproperties())
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Distribution of matched Mahalanobis distances (MAGIC)",
        fontsize=12, fontproperties=times_fontproperties(), y=1.02,
    )
    fig.tight_layout()
    save(fig, "Fig10_matched_MD_distributions_MAGIC")


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_df()
    print(f"Loaded {len(df)} rows from {CSV}")

    fig1_framework()
    fig2_ceiling(df)
    fig3_threepanel(df)
    fig4_magic(df)
    fig5_heatmaps(df)
    fig6_amplification(df)
    fig7_rankings(df)
    fig8_compute()
    fig9_covariance_stability()
    fig10_ecdf_magic()

    # README index
    readme = OUT / "README.md"
    readme.write_text(
        """# Mapping conference figures

Central manuscript story: **pairwise cosine ≈ 0 → Hungarian cosine saturates → Hungarian Mahalanobis restores discrimination.**

| Figure | File stem | Role |
|------|-----------|------|
| Fig. 1 | `Fig1_experimental_framework` | Methods framework (A/B/C) |
| Fig. 2 | `Fig2_pairwise_vs_hungarian_cosine` | Headline ceiling-effect scatter |
| Fig. 3 | `Fig3_three_panel_cost_comparison` | Pairwise / Hung. cosine / Hung. MD |
| Fig. 4 | `Fig4_MAGIC_case_study` | Eight-generator MAGIC case study |
| Fig. 5 | `Fig5_benchmark_heatmaps` | 15×8 heatmaps |
| Fig. 6 | `Fig6_cosine_amplification` | Δ_C amplification |
| Fig. 7 | `Fig7_generator_ranking_consistency` | Rank changes / Spearman |
| Fig. 8 | `Fig8_computational_scaling` | Runtime & memory vs n |
| Fig. 9 | `Fig9_mahalanobis_covariance_stability` | Ridge / condition-number sensitivity |
| Fig. 10 | `Fig10_matched_MD_distributions_MAGIC` | Matched-MD ECDF & boxplots |

Regenerate:

```bash
python Results/mapping_conference/make_mapping_figures.py
```
""",
        encoding="utf-8",
    )
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()
