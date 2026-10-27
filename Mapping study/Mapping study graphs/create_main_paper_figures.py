#!/usr/bin/env python3
"""Create 5–6 main-paper figures + supplementary material from existing extracted mapping results.

Does NOT recompute mappings or modify notebooks.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
EXT = ROOT / "extracted_results"
MAIN = ROOT / "main_paper"
SUPP = ROOT / "supplementary"
DPI = 300

GENERATOR_ORDER = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
    "TabDDPM",
    "ForestDiffusion",
]
METHOD_ORDER = ["Greedy", "One-to-One Greedy", "Hungarian"]
METHOD_COLORS = {
    "Greedy": "#4C78A8",
    "One-to-One Greedy": "#F58518",
    "Hungarian": "#54A24B",
    "Pairwise (no assignment)": "#B279A2",
}
DATASET_ORDER = [
    "1. Cancer",
    "2. Alzhimers",
    "3. Adult",
    "4. Forest cover dataset",
    "5. Bank Markting",
    "6. Wine dataset",
    "7. CDC diabetes dataset",
    "8. Mushroom dataset",
    "9. MAGIC Gamma Telescope",
    "10. Metro interstate",
    "11. online shopping",
    "12. Air Quality",
    "13. Concrete Compressive Strength",
    "14. Energy Efficiency",
    "15. Real Estate Valuation",
]
SHORT = {
    "1. Cancer": "Cancer",
    "2. Alzhimers": "Alzheimers",
    "3. Adult": "Adult",
    "4. Forest cover dataset": "ForestCover",
    "5. Bank Markting": "Bank",
    "6. Wine dataset": "Wine",
    "7. CDC diabetes dataset": "CDC",
    "8. Mushroom dataset": "Mushroom",
    "9. MAGIC Gamma Telescope": "MAGIC",
    "10. Metro interstate": "Metro",
    "11. online shopping": "OnlineShop",
    "12. Air Quality": "AirQuality",
    "13. Concrete Compressive Strength": "Concrete",
    "14. Energy Efficiency": "Energy",
    "15. Real Estate Valuation": "RealEstate",
}

# Prefer project-bundled Times/Liberation fonts when present.
_fonts_dir = Path(__file__).resolve().parent / "fonts"
if _fonts_dir.is_dir():
    from matplotlib import font_manager as fm

    for _ttf in _fonts_dir.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(_ttf))
        except (OSError, RuntimeError, ValueError):
            pass

sns.set_theme(style="ticks", context="paper", font_scale=1.15)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "svg.fonttype": "none",  # keep text as text in SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_mapping_units() -> pd.DataFrame:
    """One row = (Dataset, Generator, Mapping_Method, Metric) mean.

    Aggregation unit for main-paper figures:
      each existing (dataset × generator) mean for a method/metric.
    Pairwise cosine is excluded from mapping-method figures (not an assignment method).
    """
    df = pd.read_csv(EXT / "all_mapping_summary_15datasets.csv")
    df["Generator"] = df["Generator"].replace({"WGAN-GP": "WGAN_GP"})
    df["Dataset_Short"] = df["Dataset"].map(SHORT)
    df["Generator"] = pd.Categorical(df["Generator"], categories=GENERATOR_ORDER, ordered=True)
    return df


def mapping_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Mapping_Method"].isin(METHOD_ORDER)].copy()


# ---------------------------------------------------------------------------
# Figure 1 — Overall Cosine Similarity
# ---------------------------------------------------------------------------
def figure_1(df: pd.DataFrame) -> None:
    sub = mapping_only(df)
    sub = sub[sub["Metric"] == "Cosine Similarity"].copy()
    # Also keep pairwise as dashed reference for context? User asked for 3 methods.
    # Only plot methods that exist.
    methods = [m for m in METHOD_ORDER if m in set(sub["Mapping_Method"].astype(str))]
    folder = MAIN / "Figure_1_Overall_Cosine"

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

    # (a) Distribution of dataset×generator means by method
    ax0 = fig.add_subplot(gs[0])
    if len(sub):
        sns.boxplot(
            data=sub,
            x="Mapping_Method",
            y="Mean",
            order=methods,
            palette={m: METHOD_COLORS[m] for m in methods},
            ax=ax0,
            width=0.55,
            fliersize=2,
        )
        sns.stripplot(
            data=sub,
            x="Mapping_Method",
            y="Mean",
            order=methods,
            color="0.25",
            size=2.5,
            alpha=0.35,
            ax=ax0,
            jitter=0.18,
        )
    ax0.set_xlabel("")
    ax0.set_ylabel("Cosine similarity\n(higher = better match)")
    ax0.set_title("(a) Distribution over dataset × generator means")
    ax0.set_ylim(min(0.0, float(sub["Mean"].min()) - 0.05) if len(sub) else 0, 1.05)
    ax0.tick_params(axis="x", rotation=15)

    # (b) Mean ± 95% CI across dataset×generator units
    ax1 = fig.add_subplot(gs[1])
    rows = []
    for m in methods:
        vals = sub.loc[sub["Mapping_Method"] == m, "Mean"].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rows.append({"Mapping_Method": m, "Mean": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se, "n": len(vals)})
    agg = pd.DataFrame(rows)
    if len(agg):
        x = np.arange(len(agg))
        ax1.bar(x, agg["Mean"], color=[METHOD_COLORS[m] for m in agg["Mapping_Method"]], width=0.65, edgecolor="k", lw=0.4)
        ax1.errorbar(x, agg["Mean"], yerr=[agg["Mean"] - agg["lo"], agg["hi"] - agg["Mean"]], fmt="none", ecolor="k", capsize=4, lw=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(agg["Mapping_Method"], rotation=15)
        for i, r in agg.iterrows():
            ax1.text(i, min(r["hi"] + 0.02, 1.02), f"n={int(r['n'])}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Mean cosine similarity")
    ax1.set_title("(b) Overall mean ± 95% CI")
    ax1.set_ylim(0, 1.08)

    fig.suptitle(
        "Figure 1. Overall cosine similarity of sample-to-sample mappings\n"
        "Unit of analysis: each available (dataset × generator) mean; higher is better",
        y=1.05,
        fontsize=12,
    )
    save(fig, folder, "Figure_1_Overall_Cosine")

    # caption / aggregation note
    (folder / "AGGREGATION.txt").write_text(
        "Figure 1 aggregation\n"
        "--------------------\n"
        "Each point / box observation is the mean cosine similarity for one\n"
        "(Dataset, Generator, Mapping_Method) extracted from existing results.\n"
        "Panel (a): distribution of those means by mapping method.\n"
        "Panel (b): unweighted mean of those means ± 95% CI (normal approx.).\n"
        "No remapping was performed.\n"
        f"Methods present for cosine: {methods}\n"
        "Note: Greedy / One-to-One Greedy cosine aggregates were generally not\n"
        "persisted in notebooks; figure shows only methods with extracted means.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 2 — Overall Mahalanobis
# ---------------------------------------------------------------------------
def figure_2(df: pd.DataFrame) -> None:
    sub = mapping_only(df)
    sub = sub[sub["Metric"] == "Mahalanobis Distance"].copy()
    methods = [m for m in METHOD_ORDER if m in set(sub["Mapping_Method"].astype(str))]
    folder = MAIN / "Figure_2_Overall_Mahalanobis"

    fig = plt.figure(figsize=(10.5, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

    ax0 = fig.add_subplot(gs[0])
    if len(sub):
        # log scale helps when distances span orders of magnitude
        plot_df = sub.copy()
        plot_df["Mean_plot"] = plot_df["Mean"].clip(lower=1e-6)
        sns.boxplot(
            data=plot_df,
            x="Mapping_Method",
            y="Mean_plot",
            order=methods,
            palette={m: METHOD_COLORS[m] for m in methods},
            ax=ax0,
            width=0.55,
            fliersize=2,
        )
        sns.stripplot(
            data=plot_df,
            x="Mapping_Method",
            y="Mean_plot",
            order=methods,
            color="0.25",
            size=2.5,
            alpha=0.35,
            ax=ax0,
            jitter=0.18,
        )
        ax0.set_yscale("log")
    ax0.set_xlabel("")
    ax0.set_ylabel("Mahalanobis distance (log scale)\n(lower = better match)")
    ax0.set_title("(a) Distribution over dataset × generator means")
    ax0.tick_params(axis="x", rotation=15)

    ax1 = fig.add_subplot(gs[1])
    rows = []
    for m in methods:
        vals = sub.loc[sub["Mapping_Method"] == m, "Mean"].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            continue
        # Use median + IQR for skewed distances (more meaningful than mean±CI)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        rows.append({"Mapping_Method": m, "Median": med, "q1": q1, "q3": q3, "n": len(vals)})
    agg = pd.DataFrame(rows)
    if len(agg):
        x = np.arange(len(agg))
        ax1.bar(x, agg["Median"], color=[METHOD_COLORS[m] for m in agg["Mapping_Method"]], width=0.65, edgecolor="k", lw=0.4)
        ax1.errorbar(
            x,
            agg["Median"],
            yerr=[agg["Median"] - agg["q1"], agg["q3"] - agg["Median"]],
            fmt="none",
            ecolor="k",
            capsize=4,
            lw=1,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(agg["Mapping_Method"], rotation=15)
        for i, r in agg.iterrows():
            ax1.text(i, r["q3"] * 1.05 if r["q3"] > 0 else r["Median"] + 0.1, f"n={int(r['n'])}", ha="center", va="bottom", fontsize=8)
        ax1.set_yscale("log")
    ax1.set_ylabel("Median Mahalanobis distance")
    ax1.set_title("(b) Overall median ± IQR")

    fig.suptitle(
        "Figure 2. Overall Mahalanobis distance of sample-to-sample mappings\n"
        "Unit of analysis: each available (dataset × generator) mean; lower is better",
        y=1.05,
        fontsize=12,
    )
    save(fig, folder, "Figure_2_Overall_Mahalanobis")
    (folder / "AGGREGATION.txt").write_text(
        "Figure 2 aggregation\n"
        "--------------------\n"
        "Each observation is the mean matched Mahalanobis distance for one\n"
        "(Dataset, Generator, Mapping_Method) from existing results.\n"
        "Panel (a): distribution of those means (log y-axis).\n"
        "Panel (b): median across those means with IQR whiskers\n"
        "(preferred over mean±CI because distances are right-skewed).\n"
        f"Methods present: {methods}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 3 — Mapping method comparison (key figure)
# ---------------------------------------------------------------------------
def figure_3(df: pd.DataFrame) -> None:
    folder = MAIN / "Figure_3_Mapping_Method_Comparison"
    m = mapping_only(df)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"wspace": 0.28})

    # Cosine
    cos = m[m["Metric"] == "Cosine Similarity"]
    methods_c = [x for x in METHOD_ORDER if x in set(cos["Mapping_Method"].astype(str))]
    if len(cos):
        sns.violinplot(
            data=cos,
            x="Mapping_Method",
            y="Mean",
            order=methods_c,
            palette={k: METHOD_COLORS[k] for k in methods_c},
            inner="quartile",
            cut=0,
            ax=axes[0],
        )
        sns.stripplot(
            data=cos,
            x="Mapping_Method",
            y="Mean",
            order=methods_c,
            color="k",
            size=2.2,
            alpha=0.35,
            ax=axes[0],
            jitter=0.15,
        )
    axes[0].set_title("(a) Cosine similarity")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cosine similarity (higher = better)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0, 1.05)

    # Mahalanobis — paired O2O vs Hungarian where both exist
    mah = m[m["Metric"] == "Mahalanobis Distance"]
    methods_m = [x for x in METHOD_ORDER if x in set(mah["Mapping_Method"].astype(str))]
    if len(mah):
        sns.boxplot(
            data=mah,
            x="Mapping_Method",
            y="Mean",
            order=methods_m,
            palette={k: METHOD_COLORS[k] for k in methods_m},
            ax=axes[1],
            width=0.55,
            fliersize=2,
            showfliers=False,
        )
        sns.stripplot(
            data=mah,
            x="Mapping_Method",
            y="Mean",
            order=methods_m,
            color="k",
            size=2.2,
            alpha=0.35,
            ax=axes[1],
            jitter=0.15,
        )
        axes[1].set_yscale("log")
    axes[1].set_title("(b) Mahalanobis distance")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mahalanobis distance (lower = better)")
    axes[1].tick_params(axis="x", rotation=15)

    # Connected pairs for O2O vs Hungarian Mahalanobis
    wide = mah.pivot_table(index=["Dataset", "Generator"], columns="Mapping_Method", values="Mean")
    if {"One-to-One Greedy", "Hungarian"}.issubset(wide.columns):
        paired = wide.dropna(subset=["One-to-One Greedy", "Hungarian"])
        # small inset-style annotation
        n_pair = len(paired)
        better_h = int((paired["Hungarian"] <= paired["One-to-One Greedy"]).sum())
        axes[1].text(
            0.02,
            0.98,
            f"Paired units with both methods: n={n_pair}\n"
            f"Hungarian ≤ One-to-One Greedy: {better_h}/{n_pair}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )

    fig.suptitle(
        "Figure 3. Effect of mapping strategy on matching quality\n"
        "Each marker is one (dataset × generator) result extracted from completed experiments",
        y=1.04,
        fontsize=12,
    )
    save(fig, folder, "Figure_3_Mapping_Method_Comparison")

    # Extra paired difference plot for Mahalanobis (still one figure file set)
    if {"One-to-One Greedy", "Hungarian"}.issubset(wide.columns) and len(wide.dropna(subset=["One-to-One Greedy", "Hungarian"])):
        paired = wide.dropna(subset=["One-to-One Greedy", "Hungarian"]).reset_index()
        paired["Delta"] = paired["One-to-One Greedy"] - paired["Hungarian"]
        fig2, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.axhline(0, color="0.4", lw=0.8)
        sns.histplot(paired["Delta"], bins=20, color=METHOD_COLORS["One-to-One Greedy"], edgecolor="k", ax=ax)
        ax.set_xlabel("Δ Mahalanobis = One-to-One Greedy − Hungarian\n(positive ⇒ Hungarian better)")
        ax.set_ylabel("Count of dataset × generator units")
        ax.set_title("Paired method gap (Mahalanobis)")
        fig2.tight_layout()
        save(fig2, folder, "Figure_3_supplement_paired_delta_Mahalanobis")

    (folder / "AGGREGATION.txt").write_text(
        "Figure 3 aggregation\n"
        "--------------------\n"
        "No cross-dataset averaging in the primary panels: each point is one\n"
        "(Dataset × Generator) mean for a mapping method.\n"
        "Cosine panel: violin + strip of available methods.\n"
        "Mahalanobis panel: box + strip; annotation counts paired units where\n"
        "both One-to-One Greedy and Hungarian exist.\n"
        "Optional paired Δ histogram saved alongside the main figure.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 4 — Generator robustness
# ---------------------------------------------------------------------------
def figure_4(df: pd.DataFrame) -> None:
    folder = MAIN / "Figure_4_Generator_Robustness"
    m = mapping_only(df)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), gridspec_kw={"hspace": 0.35})

    cos = m[m["Metric"] == "Cosine Similarity"].copy()
    mah = m[m["Metric"] == "Mahalanobis Distance"].copy()

    if len(cos):
        sns.boxplot(
            data=cos,
            x="Generator",
            y="Mean",
            hue="Mapping_Method",
            order=GENERATOR_ORDER,
            hue_order=[h for h in METHOD_ORDER if h in set(cos["Mapping_Method"].astype(str))],
            palette=METHOD_COLORS,
            ax=axes[0],
            fliersize=1.5,
            linewidth=0.8,
        )
    axes[0].set_title("(a) Cosine similarity by generator")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cosine similarity (higher = better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Mapping method", loc="lower right", frameon=True)
    axes[0].set_ylim(0, 1.05)

    if len(mah):
        sns.boxplot(
            data=mah,
            x="Generator",
            y="Mean",
            hue="Mapping_Method",
            order=GENERATOR_ORDER,
            hue_order=[h for h in METHOD_ORDER if h in set(mah["Mapping_Method"].astype(str))],
            palette=METHOD_COLORS,
            ax=axes[1],
            fliersize=1.5,
            linewidth=0.8,
        )
        axes[1].set_yscale("log")
    axes[1].set_title("(b) Mahalanobis distance by generator")
    axes[1].set_xlabel("Generator")
    axes[1].set_ylabel("Mahalanobis distance (lower = better)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Mapping method", loc="upper right", frameon=True)

    fig.suptitle(
        "Figure 4. Generator robustness across the 15-dataset experiment\n"
        "Boxes summarise available dataset-level means for each generator × mapping method",
        y=0.98,
        fontsize=12,
    )
    save(fig, folder, "Figure_4_Generator_Robustness")
    (folder / "AGGREGATION.txt").write_text(
        "Figure 4 aggregation\n"
        "--------------------\n"
        "For each generator, the boxplot pools all available dataset means\n"
        "for that generator under each mapping method.\n"
        "Thus variability reflects cross-dataset consistency of a generator.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 5 — Dataset heatmap
# ---------------------------------------------------------------------------
def figure_5(df: pd.DataFrame) -> None:
    folder = MAIN / "Figure_5_Dataset_Heatmap"
    m = mapping_only(df)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.0), gridspec_kw={"wspace": 0.35})

    def panel(ax, metric, higher_better, title):
        sub = m[m["Metric"] == metric].copy()
        # Aggregate over generators: mean of available generator means per dataset×method
        g = sub.groupby(["Dataset", "Mapping_Method"], as_index=False)["Mean"].mean()
        mat = g.pivot(index="Dataset", columns="Mapping_Method", values="Mean")
        mat = mat.reindex(index=DATASET_ORDER)
        cols = [c for c in METHOD_ORDER if c in mat.columns]
        mat = mat[cols]
        mat.index = [SHORT[i] for i in mat.index]
        cmap = "YlGn" if higher_better else "YlOrRd_r"
        sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap, ax=ax, linewidths=0.4, cbar_kws={"shrink": 0.7})
        ax.set_title(title)
        ax.set_xlabel("Mapping method")
        ax.set_ylabel("Dataset")

    panel(axes[0], "Cosine Similarity", True, "(a) Cosine similarity\n(mean over generators; higher = better)")
    panel(axes[1], "Mahalanobis Distance", False, "(b) Mahalanobis distance\n(mean over generators; lower = better)")

    fig.suptitle(
        "Figure 5. Dataset-level mapping performance\n"
        "Cell = mean over available generators for that dataset × method",
        y=0.98,
        fontsize=12,
    )
    save(fig, folder, "Figure_5_Dataset_Heatmap")
    (folder / "AGGREGATION.txt").write_text(
        "Figure 5 aggregation\n"
        "--------------------\n"
        "For each (Dataset, Mapping_Method), take the unweighted mean of\n"
        "available generator-level means. Missing generators are omitted\n"
        "(not imputed). Two panels: cosine and Mahalanobis.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 6 — Distribution / robustness
# ---------------------------------------------------------------------------
def figure_6(df: pd.DataFrame) -> None:
    folder = MAIN / "Figure_6_Distribution_Robustness"
    m = mapping_only(df)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"wspace": 0.28})

    cos = m[m["Metric"] == "Cosine Similarity"]
    mah = m[m["Metric"] == "Mahalanobis Distance"]

    # ECDF cosine by method
    for method in [x for x in METHOD_ORDER if x in set(cos["Mapping_Method"].astype(str))]:
        vals = np.sort(cos.loc[cos["Mapping_Method"] == method, "Mean"].astype(float).dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[0].step(vals, y, where="post", label=method, color=METHOD_COLORS[method], lw=2)
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_title("(a) ECDF of cosine similarity")
    axes[0].legend(frameon=False)
    axes[0].set_xlim(0, 1.02)

    # ECDF mahalanobis by method (log x)
    for method in [x for x in METHOD_ORDER if x in set(mah["Mapping_Method"].astype(str))]:
        vals = np.sort(mah.loc[mah["Mapping_Method"] == method, "Mean"].astype(float).dropna().to_numpy())
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[1].step(vals, y, where="post", label=method, color=METHOD_COLORS[method], lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mahalanobis distance")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("(b) ECDF of Mahalanobis distance")
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Figure 6. Distributional robustness of mapping results\n"
        "ECDFs over all available (dataset × generator) means — not a single average",
        y=1.04,
        fontsize=12,
    )
    save(fig, folder, "Figure_6_Distribution_Robustness")
    (folder / "AGGREGATION.txt").write_text(
        "Figure 6 aggregation\n"
        "--------------------\n"
        "Empirical CDFs of the same (Dataset × Generator) means used in Figs 1–4.\n"
        "No further averaging: the curve shows the full distribution of outcomes.\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Supplementary material
# ---------------------------------------------------------------------------
def write_supplementary(df: pd.DataFrame) -> dict:
    m = mapping_only(df)
    tables = SUPP / "detailed_tables"
    ds_dir = SUPP / "dataset_level"
    gen_dir = SUPP / "generator_level"
    tables.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Full long table
    m.to_csv(tables / "all_mapping_units_long.csv", index=False)

    # Wide tables
    for metric, fname in [
        ("Cosine Similarity", "cosine_wide_dataset_generator_method.csv"),
        ("Mahalanobis Distance", "mahalanobis_wide_dataset_generator_method.csv"),
    ]:
        sub = m[m["Metric"] == metric]
        wide = sub.pivot_table(index=["Dataset", "Generator"], columns="Mapping_Method", values="Mean")
        wide = wide.reset_index()
        wide.to_csv(tables / fname, index=False)

    # Coverage
    rows = []
    for ds in DATASET_ORDER:
        for gen in GENERATOR_ORDER:
            for method in METHOD_ORDER:
                for metric in ["Cosine Similarity", "Mahalanobis Distance"]:
                    hit = m[
                        (m["Dataset"] == ds)
                        & (m["Generator"] == gen)
                        & (m["Mapping_Method"] == method)
                        & (m["Metric"] == metric)
                    ]
                    rows.append(
                        {
                            "Dataset": ds,
                            "Generator": gen,
                            "Mapping_Method": method,
                            "Metric": metric,
                            "Available": int(len(hit) > 0),
                            "Mean": float(hit["Mean"].iloc[0]) if len(hit) else np.nan,
                        }
                    )
    cov = pd.DataFrame(rows)
    cov.to_csv(tables / "condition_coverage_15x8x3x2.csv", index=False)
    cov.groupby("Dataset")["Available"].sum().rename("n_of_48").to_csv(tables / "coverage_by_dataset.csv")

    n_figs = 0
    # Dataset-level: one compact bar figure per dataset (supplementary only)
    for ds in DATASET_ORDER:
        short = SHORT[ds]
        sub = m[m["Dataset"] == ds]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.3})
        for ax, metric, ylab in [
            (axes[0], "Cosine Similarity", "Cosine (higher better)"),
            (axes[1], "Mahalanobis Distance", "Mahalanobis (lower better)"),
        ]:
            s = sub[sub["Metric"] == metric]
            if s.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=s,
                x="Generator",
                y="Mean",
                hue="Mapping_Method",
                order=[g for g in GENERATOR_ORDER if g in set(s["Generator"].astype(str))],
                hue_order=[h for h in METHOD_ORDER if h in set(s["Mapping_Method"].astype(str))],
                palette=METHOD_COLORS,
                ax=ax,
            )
            if metric == "Mahalanobis Distance":
                ax.set_yscale("log")
            ax.set_title(metric)
            ax.set_xlabel("")
            ax.set_ylabel(ylab)
            ax.tick_params(axis="x", rotation=25)
            ax.legend(fontsize=7, title="")
        fig.suptitle(f"Supplementary — {short}", fontsize=11)
        fig.tight_layout()
        save(fig, ds_dir, f"{short}_mapping_summary")
        n_figs += 1

    # Generator-level: one figure per generator across datasets
    for gen in GENERATOR_ORDER:
        sub = m[m["Generator"] == gen]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"wspace": 0.3})
        for ax, metric, ylab in [
            (axes[0], "Cosine Similarity", "Cosine (higher better)"),
            (axes[1], "Mahalanobis Distance", "Mahalanobis (lower better)"),
        ]:
            s = sub[sub["Metric"] == metric].copy()
            if s.empty:
                ax.set_visible(False)
                continue
            s["Short"] = s["Dataset"].map(SHORT)
            sns.boxplot(
                data=s,
                x="Short",
                y="Mean",
                hue="Mapping_Method",
                hue_order=[h for h in METHOD_ORDER if h in set(s["Mapping_Method"].astype(str))],
                palette=METHOD_COLORS,
                ax=ax,
                fliersize=1.5,
            )
            if metric == "Mahalanobis Distance":
                ax.set_yscale("log")
            ax.set_title(metric)
            ax.set_xlabel("Dataset")
            ax.set_ylabel(ylab)
            ax.tick_params(axis="x", rotation=40)
            ax.legend(fontsize=7, title="")
        fig.suptitle(f"Supplementary — generator {gen}", fontsize=11)
        fig.tight_layout()
        save(fig, gen_dir, f"{gen}_across_datasets")
        n_figs += 1

    # Summary stats table for main methods
    stats = (
        m.groupby(["Metric", "Mapping_Method"])["Mean"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    stats.to_csv(tables / "method_metric_summary_stats.csv", index=False)

    return {"n_supp_figures": n_figs, "n_tables": len(list(tables.glob("*.csv")))}


def write_readme(supp_info: dict, df: pd.DataFrame) -> None:
    m = mapping_only(df)
    lines = [
        "# Mapping study graphs — main paper (5–6 figures) + supplementary",
        "",
        "This folder contains visualizations extracted from already-completed",
        "mapping experiments. No mapping experiments were rerun.",
        "",
        "## Main paper figures",
        "",
        "| Figure | Folder | File stem |",
        "|--------|--------|-----------|",
        "| Figure 1 — Overall Cosine Similarity | `main_paper/Figure_1_Overall_Cosine/` | `Figure_1_Overall_Cosine` |",
        "| Figure 2 — Overall Mahalanobis Distance | `main_paper/Figure_2_Overall_Mahalanobis/` | `Figure_2_Overall_Mahalanobis` |",
        "| Figure 3 — Mapping Method Comparison | `main_paper/Figure_3_Mapping_Method_Comparison/` | `Figure_3_Mapping_Method_Comparison` |",
        "| Figure 4 — Generator Robustness | `main_paper/Figure_4_Generator_Robustness/` | `Figure_4_Generator_Robustness` |",
        "| Figure 5 — Dataset-Level Heatmap | `main_paper/Figure_5_Dataset_Heatmap/` | `Figure_5_Dataset_Heatmap` |",
        "| Figure 6 — Distribution / Robustness | `main_paper/Figure_6_Distribution_Robustness/` | `Figure_6_Distribution_Robustness` |",
        "",
        "Each folder contains PNG (300 DPI), SVG (vector), and `AGGREGATION.txt`.",
        "Typography: Times New Roman / Times-compatible serif (`font.family=serif`).",
        "",
        "## Aggregation (shared definition)",
        "",
        "Primary analysis unit for main figures:",
        "",
        "```text",
        "(Dataset × Generator × Mapping_Method × Metric) mean",
        "```",
        "",
        "extracted from existing notebook/Excel/curated CSV results.",
        "Main figures summarise the distribution of these units; they do not",
        "recompute mappings.",
        "",
        "## Data availability (important)",
        "",
        "From extracted sources:",
        "",
        f"- Hungarian × Cosine: {len(m[(m.Mapping_Method=='Hungarian')&(m.Metric=='Cosine Similarity')])} units",
        f"- Hungarian × Mahalanobis: {len(m[(m.Mapping_Method=='Hungarian')&(m.Metric=='Mahalanobis Distance')])} units",
        f"- One-to-One Greedy × Mahalanobis: {len(m[(m.Mapping_Method=='One-to-One Greedy')&(m.Metric=='Mahalanobis Distance')])} units",
        f"- Greedy (any metric): {len(m[m.Mapping_Method=='Greedy'])} units",
        "",
        "Greedy many-to-one cosine was computed in notebooks via `argmax` but",
        "full aggregates were generally not persisted (top-10 prints only).",
        "One-to-One Greedy cosine was not implemented in the notebooks.",
        "Figures show only methods with extracted means; missing methods are",
        "documented rather than imputed.",
        "",
        "## Supplementary",
        "",
        f"- Detailed figures: **{supp_info['n_supp_figures']}** (PNG+SVG each)",
        f"- Summary/detail tables: **{supp_info['n_tables']}**",
        "- Locations:",
        "  - `supplementary/dataset_level/`",
        "  - `supplementary/generator_level/`",
        "  - `supplementary/detailed_tables/`",
        "",
        "## Generators",
        "",
        ", ".join(GENERATOR_ORDER),
        "",
        "## Reproduce figures only",
        "",
        "```bash",
        "cd \"SYNTH/mapping study graphs\"",
        "python create_main_paper_figures.py",
        "```",
        "",
    ]
    (ROOT / "MAIN_PAPER_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_mapping_units()
    figure_1(df)
    figure_2(df)
    figure_3(df)
    figure_4(df)
    figure_5(df)
    figure_6(df)
    supp_info = write_supplementary(df)
    write_readme(supp_info, df)
    print("Main paper figures written under:", MAIN)
    print("Supplementary:", supp_info)


if __name__ == "__main__":
    main()
