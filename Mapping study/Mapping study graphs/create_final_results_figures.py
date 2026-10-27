#!/usr/bin/env python3
"""Final results: Greedy vs Hungarian Mahalanobis (9 classification + 6 regression).

Writes 5 focused figures + 3 tables under:
  mapping study graphs/final results/

Uses only paired generator×dataset units (both Greedy and Hungarian Mahalanobis present).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats

BASE = Path(__file__).resolve().parent
SRC = BASE / "extracted_results" / "hungarian_vs_greedy_summary.csv"
OUT = BASE / "final results"

# ---------------------------------------------------------------------------
# Dataset taxonomy + characteristics (working-set sizes used in this study)
# ---------------------------------------------------------------------------
CLASSIFICATION = [
    "Cancer",
    "Alzheimers",
    "Adult",
    "ForestCover",
    "Bank",
    "Wine",
    "CDC",
    "Mushroom",
    "MAGIC",
]
REGRESSION = [
    "Metro",
    "OnlineShop",
    "AirQuality",
    "Concrete",
    "Energy",
    "RealEstate",
]
DS_ORDER = CLASSIFICATION + REGRESSION

DS_META = {
    # Dataset: label, type, samples, features, classes (None for regression)
    "Cancer": ("Cancer", "Classification", 569, 30, 2),
    "Alzheimers": ("Alzheimer's", "Classification", 373, 10, 2),
    "Adult": ("Adult", "Classification", 1000, 14, 2),
    "ForestCover": ("Forest Cover", "Classification", 1000, 10, 7),
    "Bank": ("Bank Marketing", "Classification", 10000, 13, 2),
    "Wine": ("Wine Quality", "Classification", 1000, 11, 6),
    "CDC": ("CDC Diabetes", "Classification", 1000, 21, 2),
    "Mushroom": ("Mushroom", "Classification", 1000, 20, 2),
    "MAGIC": ("MAGIC Gamma", "Classification", 19020, 10, 2),
    "Metro": ("Metro Interstate", "Regression", 1000, 12, None),
    "OnlineShop": ("Online Shopping", "Regression", 1000, 12, None),
    "AirQuality": ("Air Quality", "Regression", 1000, 12, None),
    "Concrete": ("Concrete", "Regression", 1000, 8, None),
    "Energy": ("Energy Efficiency", "Regression", 768, 8, None),
    "RealEstate": ("Real Estate", "Regression", 414, 5, None),
}

SHORT_LABEL = {
    "Cancer": "Cancer",
    "Alzheimers": "Alzh.",
    "Adult": "Adult",
    "ForestCover": "Forest",
    "Bank": "Bank",
    "Wine": "Wine",
    "CDC": "CDC",
    "Mushroom": "Mush.",
    "MAGIC": "MAGIC",
    "Metro": "Metro",
    "OnlineShop": "Shop",
    "AirQuality": "AirQ.",
    "Concrete": "Concrete",
    "Energy": "Energy",
    "RealEstate": "RE",
}

# Okabe–Ito / colorblind-safe academic palette
COLOR_GREEDY = "#0072B2"  # blue
COLOR_HUNG = "#D55E00"  # vermillion
COLOR_POS = "#009E73"  # bluish green — Hungarian better
COLOR_NEG = "#CC79A7"  # reddish purple — Greedy better
COLOR_CLS = "#0072B2"
COLOR_REG = "#E69F00"  # orange
COLOR_INK = "#1a1a1a"
COLOR_MUTED = "#5a5a5a"
COLOR_GRID = "#d9d9d9"
COLOR_ZERO = "#333333"


def _register_project_fonts() -> None:
    """Register bundled Times-compatible fonts (Liberation Serif ≈ Times New Roman)."""
    fonts_dir = BASE / "fonts"
    if not fonts_dir.is_dir():
        return
    from matplotlib import font_manager as fm

    for ttf in sorted(fonts_dir.glob("LiberationSerif*.ttf")):
        try:
            fm.fontManager.addfont(str(ttf))
        except (OSError, RuntimeError, ValueError):
            pass
    for ttf in sorted(fonts_dir.glob("Times*.ttf")):
        try:
            fm.fontManager.addfont(str(ttf))
        except (OSError, RuntimeError, ValueError):
            pass


def setup_style() -> None:
    """Publication style: Times New Roman, clean spines, journal-ready DPI."""
    _register_project_fonts()
    serif_stack = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "TeX Gyre Termes",
        "DejaVu Serif",
    ]
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_stack,
            "mathtext.fontset": "stix",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": COLOR_INK,
            "axes.edgecolor": COLOR_INK,
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": COLOR_INK,
            "ytick.color": COLOR_INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": COLOR_INK,
        }
    )
    # Avoid seaborn theme overrides; keep ticks-only academic look.
    sns.set_theme(style="ticks", context="paper", font="serif", font_scale=1.0)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = serif_stack


def style_axes(ax: plt.Axes, grid: str | None = "y") -> None:
    """Apply consistent research-paper axis chrome."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_INK)
    ax.spines["bottom"].set_color(COLOR_INK)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(colors=COLOR_INK, width=0.8, length=3.5)
    if grid == "y":
        ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    elif grid == "x":
        ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    elif grid == "both":
        ax.grid(True, color=COLOR_GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str) -> None:
    """Nature/Science-style panel letter outside the axes."""
    ax.text(
        -0.12,
        1.05,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
        color=COLOR_INK,
        clip_on=False,
    )


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(
            stem.with_suffix(ext),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            dpi=600 if ext == ".png" else None,
        )
    plt.close(fig)


def load_paired() -> pd.DataFrame:
    raw = pd.read_csv(SRC)
    need = ["Dataset", "Generator", "Greedy_Mahalanobis", "Hungarian_Mahalanobis"]
    df = raw.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"]).copy()
    df = df[need].drop_duplicates(["Dataset", "Generator"], keep="first")
    df["Type"] = df["Dataset"].map(
        lambda d: "Classification" if d in CLASSIFICATION else "Regression"
    )
    df = df[df["Dataset"].isin(DS_ORDER)].copy()
    df["Greedy"] = df["Greedy_Mahalanobis"].astype(float)
    df["Hungarian"] = df["Hungarian_Mahalanobis"].astype(float)
    # Improvement %: positive ⇒ Hungarian better (lower distance)
    denom = df["Greedy"].abs()
    df["Improvement_Pct"] = np.where(
        denom > 1e-12, 100.0 * (df["Greedy"] - df["Hungarian"]) / denom, np.nan
    )
    return df


def dataset_summary(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DS_ORDER:
        sub = unit[unit["Dataset"] == ds]
        if len(sub) == 0:
            continue
        label, typ, n, p, k = DS_META[ds]
        g = float(sub["Greedy"].mean())
        h = float(sub["Hungarian"].mean())
        imp = 100.0 * (g - h) / g if abs(g) > 1e-12 else np.nan
        better = (
            "Hungarian"
            if imp > 0.05
            else ("Greedy" if imp < -0.05 else "Tie")
        )
        rows.append(
            {
                "Dataset": ds,
                "Dataset_Label": label,
                "Short": SHORT_LABEL[ds],
                "Type": typ,
                "Samples": n,
                "Features": p,
                "Classes": k if k is not None else "–",
                "N_Generators": int(len(sub)),
                "Greedy": g,
                "Hungarian": h,
                "Improvement_Pct": imp,
                "Better_Method": better,
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_block(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """Paired Wilcoxon: H1 that Greedy > Hungarian (Hungarian better)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(len(a))
    diff = a - b  # positive ⇒ Hungarian lower
    mean_diff = float(np.mean(diff)) if n else np.nan
    if n < 3 or np.allclose(a, b):
        return {
            "Comparison": name,
            "N": n,
            "Mean_Difference_Greedy_minus_Hungarian": mean_diff,
            "Effect_RankBiserial": np.nan,
            "Test": "Wilcoxon signed-rank",
            "p_value": np.nan,
            "Conclusion": "Insufficient variation / sample size",
        }
    try:
        res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        # Rank-biserial from Wilcoxon statistic
        # r = 2R / (n(n+1)) - 1 with R = statistic under greater
        s_max = n * (n + 1) / 2.0
        r_rb = (2.0 * float(res.statistic)) / s_max - 1.0 if s_max > 0 else np.nan
        p = float(res.pvalue)
    except ValueError:
        p, r_rb = np.nan, np.nan
    if np.isnan(p):
        concl = "Test not applicable"
    elif p < 0.05 and mean_diff > 0:
        concl = "Hungarian significantly better (lower distance)"
    elif p < 0.05 and mean_diff < 0:
        concl = "Greedy significantly better"
    else:
        concl = "No significant difference at α=0.05"
    return {
        "Comparison": name,
        "N": n,
        "Mean_Difference_Greedy_minus_Hungarian": mean_diff,
        "Effect_RankBiserial": r_rb,
        "Test": "Wilcoxon signed-rank (paired; H1: Greedy > Hungarian)",
        "p_value": p,
        "Conclusion": concl,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_overall(ds: pd.DataFrame) -> None:
    """Grouped bars: Greedy vs Hungarian, panels = Classification / Regression.

    Uses geometric mean across datasets (distances span orders of magnitude;
    arithmetic mean would be dominated by Energy / Real Estate / Cancer).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=False)
    letters = ("a", "b")
    for ax, typ, letter in zip(axes, ["Classification", "Regression"], letters):
        sub = ds[ds["Type"] == typ]
        gvals = sub["Greedy"].to_numpy(float)
        hvals = sub["Hungarian"].to_numpy(float)
        g_mean = float(np.exp(np.mean(np.log(gvals))))
        h_mean = float(np.exp(np.mean(np.log(hvals))))
        g_sem = g_mean * float(np.std(np.log(gvals), ddof=1) / np.sqrt(len(gvals)))
        h_sem = h_mean * float(np.std(np.log(hvals), ddof=1) / np.sqrt(len(hvals)))
        means = [g_mean, h_mean]
        sems = [g_sem, h_sem]
        x = np.arange(2)
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            color=[COLOR_GREEDY, COLOR_HUNG],
            width=0.58,
            capsize=3.5,
            error_kw={
                "ecolor": COLOR_MUTED,
                "elinewidth": 0.9,
                "capthick": 0.9,
                "zorder": 3,
            },
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        # Subtle hatch for B&W print readability
        bars[0].set_hatch("///")
        bars[0].set_edgecolor("white")
        bars[1].set_hatch("\\\\\\")
        bars[1].set_edgecolor("white")

        ax.set_xticks(x)
        ax.set_xticklabels(["Greedy", "Hungarian"])
        ax.set_title(f"{typ}  ($n={len(sub)}$ datasets)", pad=6)
        ax.set_ylabel("Geometric mean Mahalanobis distance" if ax is axes[0] else "")
        ax.set_yscale("log")
        # Headroom for error bars + labels
        ymax = max(m + s for m, s in zip(means, sems))
        ymin = min(max(m - s, m * 0.2) for m, s in zip(means, sems))
        ax.set_ylim(ymin * 0.75, ymax * 1.45)
        for b, v in zip(bars, means):
            ax.annotate(
                f"{v:.3g}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLOR_INK,
                xytext=(0, 4),
                textcoords="offset points",
                zorder=4,
            )
        style_axes(ax, grid="y")
        panel_label(ax, letter)
        ax.set_xlim(-0.55, 1.55)

    # Shared legend above panels
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=COLOR_GREEDY, edgecolor="white", hatch="///", label="One-to-one Greedy"),
        Patch(facecolor=COLOR_HUNG, edgecolor="white", hatch="\\\\\\", label="Hungarian"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Overall Mahalanobis matching cost by method\n"
        "Geometric mean $\\pm$ SEM across datasets (log scale)",
        y=1.14,
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_INK,
    )
    fig.tight_layout(w_pad=2.2)
    save_fig(fig, OUT / "figures" / "Fig01_overall_comparison")


def fig2_heatmap(ds: pd.DataFrame) -> None:
    """Dataset × {Greedy, Hungarian} absolute mean distances."""
    mat = ds.set_index("Short")[["Greedy", "Hungarian"]].copy()
    order = [SHORT_LABEL[d] for d in DS_ORDER if d in set(ds["Dataset"])]
    mat = mat.reindex(order)

    fig, ax = plt.subplots(figsize=(4.8, 6.6))
    vals = mat.to_numpy(dtype=float)
    vmin = max(np.nanmin(vals[vals > 0]), 1e-3)
    vmax = np.nanmax(vals)
    hm = sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".3g",
        cmap="cividis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidths=0.9,
        linecolor="white",
        cbar_kws={
            "label": "Mean Mahalanobis distance (log scale)",
            "shrink": 0.78,
            "pad": 0.04,
        },
        annot_kws={"size": 7.5, "fontweight": "medium"},
        square=False,
    )
    for text, val in zip(hm.texts, vals.ravel()):
        t = (np.log10(val) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin) + 1e-12)
        text.set_color("white" if t < 0.55 else COLOR_INK)
        text.set_fontsize(7.5)

    n_cls = sum(1 for d in DS_ORDER if d in CLASSIFICATION and d in set(ds["Dataset"]))
    ax.axhline(n_cls, color="white", lw=2.6)
    ax.axhline(n_cls, color=COLOR_INK, lw=1.1)

    # Left-side group labels (reliable, not clipped)
    ax.text(
        -0.42,
        n_cls / 2.0,
        "Classification",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=COLOR_CLS,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.text(
        -0.42,
        n_cls + (len(mat) - n_cls) / 2.0,
        "Regression",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=COLOR_REG,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xticklabels(["Greedy", "Hungarian"], fontsize=9)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7.5, width=0.7, length=3)
    cbar.outline.set_linewidth(0.6)

    ax.set_title(
        "Dataset-level Mahalanobis distance\nGreedy vs Hungarian (paired generators)",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    fig.subplots_adjust(left=0.22, right=0.92)
    save_fig(fig, OUT / "figures" / "Fig02_dataset_heatmap")


def fig3_improvement(ds: pd.DataFrame) -> None:
    """Diverging bars of Hungarian improvement % per dataset."""
    plot = ds.sort_values("Improvement_Pct").copy()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    y = np.arange(len(plot))
    colors = [
        COLOR_POS if v > 0.05 else COLOR_NEG if v < -0.05 else "#9ca3af"
        for v in plot["Improvement_Pct"]
    ]
    ax.barh(
        y,
        plot["Improvement_Pct"],
        color=colors,
        height=0.68,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax.axvline(0, color=COLOR_ZERO, lw=1.0, zorder=3)
    labels = [
        f"{r.Short}  ({'C' if r.Type == 'Classification' else 'R'})"
        for r in plot.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Improvement (%)  ·  positive = Hungarian better")
    ax.set_title(
        "Hungarian improvement over Greedy by dataset",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    style_axes(ax, grid="x")

    vmin = float(np.nanmin(plot["Improvement_Pct"]))
    vmax = float(np.nanmax(plot["Improvement_Pct"]))
    pad = max(vmax - vmin, 1.0) * 0.14
    ax.set_xlim(vmin - pad - 3.0, vmax + pad + 2.0)
    for yi, v in zip(y, plot["Improvement_Pct"]):
        ha = "left" if v >= 0 else "right"
        offset = 0.55 if v >= 0 else -0.55
        ax.text(
            v + offset,
            yi,
            f"{v:.2f}",
            va="center",
            ha=ha,
            fontsize=7.5,
            color=COLOR_MUTED,
        )

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=COLOR_POS, edgecolor="none", label="Hungarian better"),
            Patch(facecolor=COLOR_NEG, edgecolor="none", label="Greedy better"),
            Patch(facecolor="#9ca3af", edgecolor="none", label=r"Tie ($|\Delta|\leq 0.05\%$)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    ax.text(
        0.0,
        -0.13,
        r"Improvement $= (D_{\mathrm{Greedy}} - D_{\mathrm{Hungarian}}) / D_{\mathrm{Greedy}} \times 100$",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLOR_MUTED,
        ha="left",
        va="top",
    )
    save_fig(fig, OUT / "figures" / "Fig03_hungarian_improvement")


def fig4_distributions(unit: pd.DataFrame) -> None:
    """Violin + box + strip of log10 Mahalanobis by method × task type."""
    long = unit.melt(
        id_vars=["Dataset", "Type", "Generator"],
        value_vars=["Greedy", "Hungarian"],
        var_name="Method",
        value_name="Distance",
    )
    long = long[long["Distance"] > 0].copy()
    long["log10_Distance"] = np.log10(long["Distance"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    palette = {"Greedy": "#56B4E9", "Hungarian": "#E69F00"}  # light Okabe–Ito fills
    for ax, typ, letter in zip(axes, ["Classification", "Regression"], ("a", "b")):
        sub = long[long["Type"] == typ]
        n_units = sub["Dataset"].astype(str).str.cat(sub["Generator"].astype(str)).nunique()
        # n paired units = rows / 2 methods
        n_units = len(sub) // 2
        sns.violinplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            hue="Method",
            order=["Greedy", "Hungarian"],
            hue_order=["Greedy", "Hungarian"],
            palette=palette,
            inner=None,
            cut=0,
            ax=ax,
            saturation=0.85,
            linewidth=0.8,
            legend=False,
            zorder=1,
        )
        for coll in ax.collections:
            coll.set_alpha(0.40)
        sns.boxplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            order=["Greedy", "Hungarian"],
            width=0.22,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": COLOR_INK, "linewidth": 0.9, "alpha": 0.95},
            whiskerprops={"color": COLOR_INK, "linewidth": 0.9},
            capprops={"color": COLOR_INK, "linewidth": 0.9},
            medianprops={"color": COLOR_INK, "linewidth": 1.2},
            ax=ax,
            zorder=3,
        )
        sns.stripplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            order=["Greedy", "Hungarian"],
            color=COLOR_MUTED,
            size=2.4,
            alpha=0.35,
            jitter=0.10,
            ax=ax,
            zorder=2,
            legend=False,
        )
        ax.set_title(f"{typ}  ($n={n_units}$ units)", pad=6)
        ax.set_xlabel("")
        ax.set_ylabel(r"$\log_{10}$ Mahalanobis distance" if ax is axes[0] else "")
        style_axes(ax, grid="y")
        panel_label(ax, letter)

    fig.suptitle(
        "Distribution of Mahalanobis distances across dataset×generator units",
        y=1.06,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, OUT / "figures" / "Fig04_distance_distributions")


def fig5_characteristics(ds: pd.DataFrame) -> None:
    """Scatter: features vs Hungarian improvement %; size ~ samples."""
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    # Manual label offsets (points) to reduce overlap for dense clusters
    label_offset = {
        "AirQ.": (7, 5),
        "MAGIC": (7, -12),
        "Bank": (7, 5),
        "Wine": (7, 5),
        "Shop": (7, 5),
        "Metro": (7, -11),
        "Concrete": (7, 5),
        "Energy": (-38, -11),
        "RE": (7, 4),
        "Forest": (-40, -11),
        "Alzh.": (7, 5),
        "Adult": (7, -11),
        "Mush.": (7, 5),
        "CDC": (7, -11),
        "Cancer": (-48, 4),
    }
    for typ, color, marker in [
        ("Classification", COLOR_CLS, "o"),
        ("Regression", COLOR_REG, "s"),
    ]:
        sub = ds[ds["Type"] == typ]
        log_s = np.log10(sub["Samples"].astype(float))
        lo, hi = float(log_s.min()), float(log_s.max())
        sizes = 55 + 220 * (log_s - lo) / max(hi - lo, 1e-9)
        ax.scatter(
            sub["Features"],
            sub["Improvement_Pct"],
            s=sizes,
            c=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.7,
            marker=marker,
            label=typ,
            zorder=3,
        )
        for r in sub.itertuples():
            xytext = label_offset.get(r.Short, (6, 4))
            ax.annotate(
                f"{r.Short} {r.Improvement_Pct:.2f}%",
                (r.Features, r.Improvement_Pct),
                textcoords="offset points",
                xytext=xytext,
                fontsize=7.2,
                color=COLOR_INK,
                ha="left" if xytext[0] >= 0 else "right",
                va="center",
                zorder=4,
            )

    x = ds["Features"].to_numpy(float)
    y = ds["Improvement_Pct"].to_numpy(float)
    if len(ds) >= 3 and np.nanstd(x) > 0:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xx,
            intercept + slope * xx,
            color=COLOR_MUTED,
            ls="--",
            lw=1.1,
            label=f"Linear trend ($r={r:.2f}$, $p={p:.3f}$)",
            zorder=2,
        )

    ax.axhline(0, color=COLOR_ZERO, lw=0.95, zorder=1)
    ax.set_xlabel("Number of features (excluding target)")
    ax.set_ylabel("Hungarian improvement over Greedy (%)")
    ax.set_title(
        "Improvement vs dataset dimensionality\n(marker size $\\sim$ sample size)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    ax.set_ylim(ymin - 3.5, ymax + 3.5)
    ax.set_xlim(float(np.nanmin(x)) - 1.5, float(np.nanmax(x)) + 2.5)
    style_axes(ax, grid="both")
    ax.legend(loc="upper right", frameon=False, fontsize=8, markerscale=0.9)
    save_fig(fig, OUT / "figures" / "Fig05_improvement_vs_features")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_tables(ds: pd.DataFrame, unit: pd.DataFrame) -> None:
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # Table 1 — characteristics
    t1 = pd.DataFrame(
        [
            {
                "Dataset": DS_META[d][0],
                "Type": DS_META[d][1],
                "Samples": DS_META[d][2],
                "Features": DS_META[d][3],
                "Classes": DS_META[d][4] if DS_META[d][4] is not None else "–",
                "Task": DS_META[d][1],
            }
            for d in DS_ORDER
        ]
    )
    t1.to_csv(tdir / "Table1_dataset_characteristics.csv", index=False)

    # Table 2 — main quantitative results
    t2 = ds[
        [
            "Dataset_Label",
            "Type",
            "N_Generators",
            "Greedy",
            "Hungarian",
            "Improvement_Pct",
            "Better_Method",
        ]
    ].rename(
        columns={
            "Dataset_Label": "Dataset",
            "N_Generators": "N_paired_generators",
            "Greedy": "Greedy_Mahalanobis",
            "Hungarian": "Hungarian_Mahalanobis",
        }
    )
    t2.to_csv(tdir / "Table2_main_quantitative_results.csv", index=False)

    # Table 3 — statistical tests on dataset-level paired means
    blocks = []
    for typ in ["Classification", "Regression"]:
        sub = ds[ds["Type"] == typ]
        blocks.append(wilcoxon_block(sub["Greedy"].to_numpy(), sub["Hungarian"].to_numpy(), f"Hungarian vs Greedy – {typ}"))
    blocks.append(
        wilcoxon_block(ds["Greedy"].to_numpy(), ds["Hungarian"].to_numpy(), "Overall (15 datasets)")
    )
    # Also unit-level overall for power
    blocks.append(
        wilcoxon_block(
            unit["Greedy"].to_numpy(),
            unit["Hungarian"].to_numpy(),
            "Overall (dataset×generator units)",
        )
    )
    t3 = pd.DataFrame(blocks)
    t3.to_csv(tdir / "Table3_statistical_significance.csv", index=False)

    # Excel workbook
    with pd.ExcelWriter(tdir / "final_results_tables.xlsx") as writer:
        t1.to_excel(writer, sheet_name="Table1_characteristics", index=False)
        t2.to_excel(writer, sheet_name="Table2_main_results", index=False)
        t3.to_excel(writer, sheet_name="Table3_statistics", index=False)
        ds.to_excel(writer, sheet_name="dataset_level_detail", index=False)
        unit.to_excel(writer, sheet_name="unit_level_paired", index=False)

    # Pretty markdown
    md = []
    md.append("# Final results — Greedy vs Hungarian (Mahalanobis)\n")
    md.append("Paired units only (both methods present). Improvement % > 0 ⇒ Hungarian better.\n")
    md.append("## Table 1. Dataset characteristics\n")
    md.append(t1.to_markdown(index=False))
    md.append("\n## Table 2. Main quantitative results\n")
    show2 = t2.copy()
    show2["Greedy_Mahalanobis"] = show2["Greedy_Mahalanobis"].map(lambda v: f"{v:.4g}")
    show2["Hungarian_Mahalanobis"] = show2["Hungarian_Mahalanobis"].map(lambda v: f"{v:.4g}")
    show2["Improvement_Pct"] = show2["Improvement_Pct"].map(lambda v: f"{v:.2f}")
    md.append(show2.to_markdown(index=False))
    md.append("\n## Table 3. Statistical significance\n")
    show3 = t3.copy()
    show3["Mean_Difference_Greedy_minus_Hungarian"] = show3[
        "Mean_Difference_Greedy_minus_Hungarian"
    ].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    show3["Effect_RankBiserial"] = show3["Effect_RankBiserial"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else ""
    )
    show3["p_value"] = show3["p_value"].map(lambda v: f"{v:.3e}" if pd.notna(v) else "")
    md.append(show3.to_markdown(index=False))
    (tdir / "README_tables.md").write_text("\n".join(md), encoding="utf-8")


def write_readme(ds: pd.DataFrame, unit: pd.DataFrame) -> None:
    n_cls = int((ds["Type"] == "Classification").sum())
    n_reg = int((ds["Type"] == "Regression").sum())
    pct_h = 100.0 * (ds["Improvement_Pct"] > 0).mean()
    text = f"""# Final results — Greedy vs Hungarian Mahalanobis mapping

Focused publication set: **5 figures + 3 tables**.

## Scope
- **9 classification** + **6 regression** datasets ({n_cls}+{n_reg} with paired data)
- Metric: mean matched-pair **Mahalanobis distance** (lower is better)
- Methods: **One-to-One Greedy** vs **Hungarian**
- Only generator×dataset units where **both** methods exist (n={len(unit)} units)
- Dataset-level means average over paired generators for that dataset

## Improvement definition
```
Improvement(%) = (D_Greedy − D_Hungarian) / D_Greedy × 100
  > 0  → Hungarian better
  < 0  → Greedy better
```

## Figures
| File | Description |
|------|-------------|
| `figures/Fig01_overall_comparison` | Grouped bars, Classification vs Regression panels (log y) |
| `figures/Fig02_dataset_heatmap` | Dataset × {{Greedy, Hungarian}} absolute distances (log colour) |
| `figures/Fig03_hungarian_improvement` | Diverging improvement (%) for all 15 datasets |
| `figures/Fig04_distance_distributions` | Violin/box/strip of log₁₀ distances by method × task |
| `figures/Fig05_improvement_vs_features` | Improvement vs #features (size ∝ samples) |

## Tables
| File | Description |
|------|-------------|
| `tables/Table1_dataset_characteristics.csv` | Samples, features, classes, task |
| `tables/Table2_main_quantitative_results.csv` | Greedy, Hungarian, Improvement %, better method |
| `tables/Table3_statistical_significance.csv` | Wilcoxon signed-rank + rank-biserial |
| `tables/final_results_tables.xlsx` | All tables + detail sheets |

## Headline
- Datasets where Hungarian improves mean distance: **{pct_h:.0f}%** ({int((ds['Improvement_Pct']>0).sum())}/{len(ds)})
- Source summary: `{SRC}`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    unit = load_paired()
    ds = dataset_summary(unit)
    print(f"Paired units: {len(unit)}")
    print(f"Datasets: {len(ds)} (classification={sum(ds.Type=='Classification')}, regression={sum(ds.Type=='Regression')})")

    fig1_overall(ds)
    fig2_heatmap(ds)
    fig3_improvement(ds)
    fig4_distributions(unit)
    fig5_characteristics(ds)
    write_tables(ds, unit)
    write_readme(ds, unit)

    print(f"Wrote figures + tables under: {OUT}")


if __name__ == "__main__":
    main()
