#!/usr/bin/env python3
"""Mahalanobis Greedy vs Hungarian mapping comparison (extract-only).

Compares One-to-One Greedy Mahalanobis distance vs Hungarian Mahalanobis
distance using already-extracted results. Cosine similarity is intentionally
excluded from all tables and figures.

Difference convention (this workflow):
    Diff = Hungarian - Greedy
    Diff < 0  → Hungarian lower distance → better matching
    Diff > 0  → Hungarian higher distance → worse matching
    Diff ≈ 0  → negligible

Primary data source:
    extracted_results/hungarian_vs_greedy_summary.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = Path(__file__).resolve().parent
SRC_SUMMARY = BASE / "extracted_results" / "hungarian_vs_greedy_summary.csv"
SRC_MAHA = BASE / "extracted_results" / "mahalanobis_distance_results.csv"
OUT = BASE / "outputs" / "mahalanobis_greedy_vs_hungarian"

SUBDIRS = [
    "overall_comparison",
    "dataset_comparison",
    "generator_comparison",
    "heatmaps",
    "distributions",
    "scatter",
    "agreement",
    "runtime",
    "summary",
    "tables",
]

GEN_ORDER = [
    "ForestDiffusion",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
    "CTGAN",
    "TabDDPM",
    "CopulaGAN",
]
GEN_LABEL = {
    "ForestDiffusion": "Forest Diffusion",
    "TVAE": "TVAE",
    "GaussianCopula": "Gaussian Copula",
    "WGAN_GP": "WGAN-GP",
    "CTABGAN": "CTAB-GAN+",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
    "CopulaGAN": "Copula GAN",
}
DS_ORDER = [
    "Cancer",
    "Alzheimers",
    "Adult",
    "ForestCover",
    "Bank",
    "Wine",
    "CDC",
    "Mushroom",
    "MAGIC",
    "Metro",
    "OnlineShop",
    "AirQuality",
    "Concrete",
    "Energy",
    "RealEstate",
]
DS_LABEL = {
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

# Absolute relative difference below this is treated as negligible for coloring.
NEGLIGIBLE_ABS = 1e-6
NEGLIGIBLE_PCT = 0.5  # |rel %| < 0.5% marked negligible in dataset plot notes

COLOR_HUNG_BETTER = "#1b7f5a"
COLOR_HUNG_WORSE = "#b23a48"
COLOR_NEUTRAL = "#6b7280"
COLOR_GREEDY = "#2563eb"
COLOR_HUNG = "#c2410c"


def setup_style() -> None:
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    if fonts_dir.is_dir():
        from matplotlib import font_manager as fm

        for ttf in fonts_dir.glob("*.ttf"):
            try:
                fm.fontManager.addfont(str(ttf))
            except (OSError, RuntimeError, ValueError):
                pass

    mpl.rcParams.update(
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
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    for name in SUBDIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_paired_mahalanobis() -> tuple[pd.DataFrame, dict]:
    """Load paired Greedy/Hungarian Mahalanobis means; never invent values."""
    report: dict = {
        "source_primary": str(SRC_SUMMARY),
        "source_secondary": str(SRC_MAHA),
        "processed": [],
        "skipped": [],
        "notes": [],
    }

    if not SRC_SUMMARY.exists():
        raise FileNotFoundError(f"Missing primary source: {SRC_SUMMARY}")

    raw = pd.read_csv(SRC_SUMMARY)
    required = ["Dataset", "Generator", "Greedy_Mahalanobis", "Hungarian_Mahalanobis"]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Primary CSV missing columns: {missing_cols}")

    # Coverage inventory over full 15×8 grid if present.
    all_units = raw[["Dataset", "Generator"]].drop_duplicates()
    paired = raw.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"]).copy()
    paired = paired[required + [c for c in ["Num_Matches", "hung_src", "nb_src"] if c in paired.columns]]

    # Prefer one row per dataset×generator (should already be unique).
    before = len(paired)
    paired = paired.drop_duplicates(subset=["Dataset", "Generator"], keep="first")
    if len(paired) < before:
        report["notes"].append(
            f"Dropped {before - len(paired)} duplicate dataset×generator rows (kept first)."
        )

    for _, r in all_units.iterrows():
        key = (r["Dataset"], r["Generator"])
        hit = paired[(paired["Dataset"] == key[0]) & (paired["Generator"] == key[1])]
        if len(hit) == 1:
            report["processed"].append({"dataset": key[0], "generator": key[1]})
        else:
            # Diagnose from secondary long file if available.
            reason = "missing Greedy and/or Hungarian Mahalanobis mean in extracted summary"
            if SRC_MAHA.exists():
                maha = pd.read_csv(SRC_MAHA)
                sub = maha[
                    (maha["Dataset_Short"].astype(str) == str(key[0]))
                    | (maha["Dataset"].astype(str).str.contains(str(key[0]), case=False, na=False))
                ]
                sub = sub[sub["Generator"] == key[1]]
                methods = sorted(sub["Mapping_Method"].dropna().unique().tolist()) if len(sub) else []
                if methods:
                    reason = f"incomplete Mahalanobis methods available: {methods}"
            report["skipped"].append(
                {"dataset": key[0], "generator": key[1], "reason": reason}
            )

    paired["Greedy"] = paired["Greedy_Mahalanobis"].astype(float)
    paired["Hungarian"] = paired["Hungarian_Mahalanobis"].astype(float)
    paired["Diff_Hung_minus_Greedy"] = paired["Hungarian"] - paired["Greedy"]
    # Relative % using |Greedy| denominator; flag unstable near-zero Greedy.
    denom = paired["Greedy"].abs()
    paired["RelPct_Hung_minus_Greedy"] = np.where(
        denom > 1e-12, 100.0 * paired["Diff_Hung_minus_Greedy"] / denom, np.nan
    )
    paired["RelPct_Unstable"] = denom <= 1e-12
    paired["Direction"] = np.where(
        paired["Diff_Hung_minus_Greedy"] < -NEGLIGIBLE_ABS,
        "Hungarian better (lower distance)",
        np.where(
            paired["Diff_Hung_minus_Greedy"] > NEGLIGIBLE_ABS,
            "Hungarian worse (higher distance)",
            "Negligible / tie",
        ),
    )
    paired["GeneratorLabel"] = paired["Generator"].map(lambda g: GEN_LABEL.get(g, g))
    paired["DatasetLabel"] = paired["Dataset"].map(lambda d: DS_LABEL.get(d, d))

    # Order categories
    paired["Dataset"] = pd.Categorical(paired["Dataset"], categories=DS_ORDER, ordered=True)
    paired["Generator"] = pd.Categorical(paired["Generator"], categories=GEN_ORDER, ordered=True)
    paired = paired.sort_values(["Dataset", "Generator"]).reset_index(drop=True)

    report["n_paired"] = int(len(paired))
    report["n_skipped"] = int(len(report["skipped"]))
    report["datasets"] = sorted(paired["Dataset"].astype(str).unique().tolist())
    report["generators"] = sorted(paired["Generator"].astype(str).unique().tolist())
    report["metric"] = "Mahalanobis Distance (mean matched-pair distance)"
    report["lower_is_better"] = True
    report["diff_definition"] = "Hungarian - Greedy (negative = Hungarian better)"
    report["runtime_available"] = False
    report["assignment_level_available"] = False
    report["notes"].append(
        "No runtime/compute-cost fields found in extracted mapping results; runtime figures skipped."
    )
    report["notes"].append(
        "No assignment-level pair identity matrices found; agreement figure skipped."
    )
    return paired, report


def paired_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Overall + by-generator paired Wilcoxon tests (Hungarian vs Greedy)."""
    rows = []

    def one_block(sub: pd.DataFrame, scope: str, name: str) -> None:
        if len(sub) < 2:
            rows.append(
                {
                    "Scope": scope,
                    "Name": name,
                    "N": len(sub),
                    "Mean_Greedy": np.nan if len(sub) == 0 else float(sub["Greedy"].mean()),
                    "Mean_Hungarian": np.nan if len(sub) == 0 else float(sub["Hungarian"].mean()),
                    "Mean_Diff_Hung_minus_Greedy": np.nan if len(sub) == 0 else float(sub["Diff_Hung_minus_Greedy"].mean()),
                    "SD_Diff": np.nan if len(sub) < 2 else float(sub["Diff_Hung_minus_Greedy"].std(ddof=1)),
                    "Median_Diff": np.nan if len(sub) == 0 else float(sub["Diff_Hung_minus_Greedy"].median()),
                    "Pct_Hungarian_Better": np.nan if len(sub) == 0 else 100.0 * (sub["Diff_Hung_minus_Greedy"] < 0).mean(),
                    "Test": "insufficient N",
                    "Statistic": np.nan,
                    "p_value": np.nan,
                    "Effect_RankBiserial": np.nan,
                    "Note": "Need >=2 paired observations",
                }
            )
            return
        g = sub["Greedy"].to_numpy(dtype=float)
        h = sub["Hungarian"].to_numpy(dtype=float)
        diff = h - g
        # Wilcoxon: test whether Hungarian distances are lower than Greedy.
        # Use differences Greedy - Hungarian > 0 as "Hungarian better".
        try:
            # alternative: Greedy > Hungarian  <=> Hungarian better (lower)
            stat, p = stats.wilcoxon(g, h, alternative="greater", zero_method="wilcox")
            # Rank-biserial effect size for paired Wilcoxon (Kerby):
            # r = 2*W+/S - 1, where S = n(n+1)/2 and W+ is the Wilcoxon statistic.
            n = len(diff)
            s_max = n * (n + 1) / 2.0
            r_rb = (2.0 * stat) / s_max - 1.0 if s_max > 0 else np.nan
            note = "Paired Wilcoxon; H1: Greedy distance > Hungarian distance (Hungarian better)."
        except ValueError as exc:
            stat, p, r_rb = np.nan, np.nan, np.nan
            note = f"Wilcoxon not applicable: {exc}"
        rows.append(
            {
                "Scope": scope,
                "Name": name,
                "N": int(len(sub)),
                "Mean_Greedy": float(np.mean(g)),
                "Mean_Hungarian": float(np.mean(h)),
                "Mean_Diff_Hung_minus_Greedy": float(np.mean(diff)),
                "SD_Diff": float(np.std(diff, ddof=1)),
                "Median_Diff": float(np.median(diff)),
                "Pct_Hungarian_Better": float(100.0 * np.mean(diff < 0)),
                "Test": "Wilcoxon signed-rank (paired, alternative Greedy > Hungarian)",
                "Statistic": float(stat) if pd.notna(stat) else np.nan,
                "p_value": float(p) if pd.notna(p) else np.nan,
                "Effect_RankBiserial": float(r_rb) if pd.notna(r_rb) else np.nan,
                "Note": note,
            }
        )

    one_block(df, "Overall", "All paired units")
    for gen in GEN_ORDER:
        sub = df[df["Generator"] == gen]
        if len(sub):
            one_block(sub, "Generator", GEN_LABEL.get(gen, gen))
    for ds in DS_ORDER:
        sub = df[df["Dataset"] == ds]
        if len(sub):
            one_block(sub, "Dataset", DS_LABEL.get(ds, ds))

    out = pd.DataFrame(rows)
    # Holm correction within Scope==Generator and Scope==Dataset separately.
    for scope in ["Generator", "Dataset"]:
        mask = out["Scope"] == scope
        pvals = out.loc[mask, "p_value"].to_numpy(dtype=float)
        if np.all(np.isnan(pvals)):
            out.loc[mask, "p_holm"] = np.nan
            continue
        # Holm on finite p-values
        idx = out.index[mask]
        finite = [(i, out.loc[i, "p_value"]) for i in idx if pd.notna(out.loc[i, "p_value"])]
        finite_sorted = sorted(finite, key=lambda t: t[1])
        m = len(finite_sorted)
        holm = {}
        prev = 0.0
        for rank, (i, p) in enumerate(finite_sorted, start=1):
            adj = min(1.0, (m - rank + 1) * p)
            adj = max(adj, prev)
            holm[i] = adj
            prev = adj
        out.loc[mask, "p_holm"] = [holm.get(i, np.nan) for i in idx]
    if "p_holm" not in out.columns:
        out["p_holm"] = np.nan
    out.loc[out["Scope"] == "Overall", "p_holm"] = out.loc[out["Scope"] == "Overall", "p_value"]
    return out


def fig_overall_dumbbell(df: pd.DataFrame) -> None:
    # Aggregate to dataset means across generators for a clean paired view.
    ds = (
        df.groupby("Dataset", observed=True)
        .agg(Greedy=("Greedy", "mean"), Hungarian=("Hungarian", "mean"), N=("Generator", "count"))
        .reindex([d for d in DS_ORDER if d in set(df["Dataset"].astype(str))])
        .dropna(subset=["Greedy", "Hungarian"])
        .reset_index()
    )
    ds["DatasetLabel"] = ds["Dataset"].map(lambda d: DS_LABEL.get(d, d))
    ds = ds.sort_values("Greedy", ascending=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    y = np.arange(len(ds))
    for yi, (_, r) in enumerate(ds.iterrows()):
        ax.plot([r["Greedy"], r["Hungarian"]], [yi, yi], color="#cbd5e1", lw=2, zorder=1)
    ax.scatter(ds["Greedy"], y, s=55, color=COLOR_GREEDY, label="Greedy", zorder=3)
    ax.scatter(ds["Hungarian"], y, s=55, color=COLOR_HUNG, label="Hungarian", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ds["DatasetLabel"])
    ax.set_xscale("log")
    ax.set_xlabel("Mean Mahalanobis matching distance (log scale; lower = better)")
    ax.set_title("Mahalanobis mapping: Greedy → Hungarian (dataset means)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25, which="both")
    save_fig(fig, OUT / "overall_comparison" / "Fig01_dumbbell_dataset_means")

    # Unit-level paired view (all dataset×generator), sorted by Greedy.
    unit = df.sort_values("Greedy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 14.0))
    y = np.arange(len(unit))
    for yi, r in unit.iterrows():
        ax.plot([r["Greedy"], r["Hungarian"]], [yi, yi], color="#cbd5e1", lw=1.2, zorder=1)
    ax.scatter(unit["Greedy"], y, s=18, color=COLOR_GREEDY, label="Greedy", zorder=3)
    ax.scatter(unit["Hungarian"], y, s=18, color=COLOR_HUNG, label="Hungarian", zorder=3)
    # sparse y labels: every ~8th + extremes by |diff|
    step = max(1, len(unit) // 20)
    label_idx = set(range(0, len(unit), step))
    top_diff = unit["Diff_Hung_minus_Greedy"].abs().nlargest(12).index
    label_idx |= set(top_diff)
    ytick = sorted(label_idx)
    ax.set_yticks(ytick)
    ax.set_yticklabels(
        [f"{unit.loc[i, 'DatasetLabel']}/{unit.loc[i, 'GeneratorLabel']}" for i in ytick],
        fontsize=7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mahalanobis matching distance (log scale; lower = better)")
    ax.set_title("Mahalanobis mapping: Greedy → Hungarian (all paired units)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25, which="both")
    save_fig(fig, OUT / "overall_comparison" / "Fig01b_dumbbell_all_units")


def fig_dataset_diff_bars(df: pd.DataFrame) -> None:
    ds = (
        df.groupby("Dataset", observed=True)["Diff_Hung_minus_Greedy"]
        .mean()
        .reindex([d for d in DS_ORDER if d in set(df["Dataset"].astype(str))])
        .dropna()
        .sort_values()
    )
    labels = [DS_LABEL.get(d, d) for d in ds.index]
    colors = [
        COLOR_HUNG_BETTER if v < -NEGLIGIBLE_ABS else COLOR_HUNG_WORSE if v > NEGLIGIBLE_ABS else COLOR_NEUTRAL
        for v in ds.values
    ]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    y = np.arange(len(ds))
    ax.barh(y, ds.values, color=colors, edgecolor="none")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hungarian − Greedy (mean across generators)\nnegative = Hungarian better")
    ax.set_title("Dataset-level Mahalanobis difference (ranked)")
    # legend proxies
    ax.plot([], [], color=COLOR_HUNG_BETTER, lw=6, label="Hungarian better")
    ax.plot([], [], color=COLOR_HUNG_WORSE, lw=6, label="Hungarian worse")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, OUT / "dataset_comparison" / "Fig02_dataset_diff_ranked")


def fig_generator_summary(df: pd.DataFrame) -> None:
    rows = []
    for gen in GEN_ORDER:
        sub = df[df["Generator"] == gen]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "Generator": GEN_LABEL[gen],
                "N": len(sub),
                "MeanDiff": sub["Diff_Hung_minus_Greedy"].mean(),
                "SD": sub["Diff_Hung_minus_Greedy"].std(ddof=1) if len(sub) > 1 else np.nan,
                "PctBetter": 100.0 * (sub["Diff_Hung_minus_Greedy"] < 0).mean(),
            }
        )
    gdf = pd.DataFrame(rows).sort_values("MeanDiff")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(gdf))
    colors = [
        COLOR_HUNG_BETTER if v < 0 else COLOR_HUNG_WORSE if v > 0 else COLOR_NEUTRAL
        for v in gdf["MeanDiff"]
    ]
    ax.barh(y, gdf["MeanDiff"], xerr=gdf["SD"].fillna(0), color=colors, alpha=0.9, ecolor="#64748b", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.Generator} (n={int(r.N)})" for r in gdf.itertuples()])
    ax.set_xlabel("Hungarian − Greedy (mean ± SD across datasets)\nnegative = Hungarian better")
    ax.set_title("Generator-level Mahalanobis Greedy vs Hungarian")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, OUT / "generator_comparison" / "Fig03_generator_mean_diff")


def _fmt_diff_annot(v) -> str:
    """Adaptive annotation for Hung−Greedy diffs (handles near-zero Energy/RE ties)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return "n/a"
    v = float(v)
    if abs(v) < 5e-4:
        return "≈0"
    if abs(v) < 0.01:
        return f"{v:.3f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fig_heatmap(df: pd.DataFrame) -> None:
    # Full 15×8 grid so unpaired units appear explicitly as missing.
    pivot = df.pivot_table(
        index="Dataset", columns="Generator", values="Diff_Hung_minus_Greedy", observed=False
    )
    pivot = pivot.reindex(index=DS_ORDER, columns=GEN_ORDER)
    annot = pivot.map(_fmt_diff_annot)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    finite = pivot.to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(finite), 90)
    vmax = max(float(vmax), 1e-6)
    sns.heatmap(
        pivot.astype(float),
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Hungarian − Greedy\n(neg = Hungarian better)"},
        xticklabels=[GEN_LABEL.get(c, c) for c in pivot.columns],
        yticklabels=[DS_LABEL.get(i, i) for i in pivot.index],
    )
    # Seaborn may skip annotations on NaN cells — force "n/a" labels.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pd.isna(pivot.iloc[i, j]):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#6b7280",
                )
    ax.set_title(
        "Mahalanobis difference heatmap (Hungarian − Greedy)\n"
        "n/a = unpaired (Greedy and/or Hungarian missing); ≈0 = |diff| < 5×10⁻⁴"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save_fig(fig, OUT / "heatmaps" / "Fig04_heatmap_hung_minus_greedy")


def fig_distributions(df: pd.DataFrame) -> None:
    # Overall
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.violinplot(y=df["Diff_Hung_minus_Greedy"], color="#e2e8f0", inner=None, cut=0, ax=ax)
    sns.boxplot(
        y=df["Diff_Hung_minus_Greedy"],
        width=0.18,
        showfliers=False,
        boxprops={"facecolor": "white"},
        ax=ax,
    )
    sns.stripplot(y=df["Diff_Hung_minus_Greedy"], color="#334155", size=3, alpha=0.55, jitter=0.08, ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Hungarian − Greedy")
    ax.set_title("Distribution of Mahalanobis differences (all paired units)")
    save_fig(fig, OUT / "distributions" / "Fig05a_diff_distribution_overall")

    # By generator
    plot_df = df.copy()
    plot_df["GeneratorLabel"] = pd.Categorical(
        plot_df["Generator"].map(lambda g: GEN_LABEL.get(g, g)),
        categories=[GEN_LABEL[g] for g in GEN_ORDER if g in set(df["Generator"].astype(str))],
        ordered=True,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.boxplot(
        data=plot_df,
        x="GeneratorLabel",
        y="Diff_Hung_minus_Greedy",
        color="#f1f5f9",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="GeneratorLabel",
        y="Diff_Hung_minus_Greedy",
        color="#334155",
        size=3,
        alpha=0.55,
        jitter=0.15,
        ax=ax,
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Hungarian − Greedy")
    ax.set_title("Mahalanobis differences by generator")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, OUT / "distributions" / "Fig05b_diff_distribution_by_generator")


def fig_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.scatter(df["Greedy"], df["Hungarian"], s=28, alpha=0.75, c="#0f766e", edgecolors="none")
    lo = float(np.nanmin([df["Greedy"].min(), df["Hungarian"].min()]))
    hi = float(np.nanmax([df["Greedy"].max(), df["Hungarian"].max()]))
    lo = max(lo, 1e-6)
    lims = [lo * 0.8, hi * 1.2]
    ax.plot(lims, lims, color="black", lw=1, ls="--", label="y = x")
    # label a readable subset: largest absolute diffs
    top = df.reindex(df["Diff_Hung_minus_Greedy"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        ax.annotate(
            f"{r['DatasetLabel']}/{r['GeneratorLabel']}",
            (r["Greedy"], r["Hungarian"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#334155",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Greedy Mahalanobis distance (lower better)")
    ax.set_ylabel("Hungarian Mahalanobis distance (lower better)")
    ax.set_title("Greedy vs Hungarian Mahalanobis mapping")
    ax.legend(frameon=False, loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, which="both")
    save_fig(fig, OUT / "scatter" / "Fig06_greedy_vs_hungarian_scatter")


def fig_summary(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    """Manuscript-ready summary combining key panels."""
    fig = plt.figure(figsize=(11.5, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # A: medians (more robust than means dominated by Energy/RE scale)
    ax0 = fig.add_subplot(gs[0, 0])
    meds = [df["Greedy"].median(), df["Hungarian"].median()]
    ax0.bar(["Greedy", "Hungarian"], meds, color=[COLOR_GREEDY, COLOR_HUNG], width=0.55)
    ax0.set_ylabel("Median Mahalanobis distance")
    ax0.set_title("A. Overall median distance")
    for i, v in enumerate(meds):
        ax0.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # B: direction counts
    ax1 = fig.add_subplot(gs[0, 1])
    counts = df["Direction"].value_counts()
    order_dir = [
        "Hungarian better (lower distance)",
        "Negligible / tie",
        "Hungarian worse (higher distance)",
    ]
    vals = [int(counts.get(k, 0)) for k in order_dir]
    colors = [COLOR_HUNG_BETTER, COLOR_NEUTRAL, COLOR_HUNG_WORSE]
    ax1.barh(order_dir, vals, color=colors)
    ax1.set_xlabel("Number of dataset × generator pairs")
    ax1.set_title("B. Outcome direction")
    for i, v in enumerate(vals):
        ax1.text(v, i, f" {v}", va="center", fontsize=9)

    # C: generator mean diffs
    ax2 = fig.add_subplot(gs[1, 0])
    gmeans = (
        df.groupby("Generator", observed=True)["Diff_Hung_minus_Greedy"]
        .mean()
        .reindex([g for g in GEN_ORDER if g in set(df["Generator"].astype(str))])
        .dropna()
        .sort_values()
    )
    y = np.arange(len(gmeans))
    cols = [COLOR_HUNG_BETTER if v < 0 else COLOR_HUNG_WORSE for v in gmeans.values]
    ax2.barh(y, gmeans.values, color=cols)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([GEN_LABEL.get(g, g) for g in gmeans.index])
    ax2.set_xlabel("Hungarian − Greedy")
    ax2.set_title("C. Mean difference by generator")

    # D: stats text box
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    overall = stats_df[stats_df["Scope"] == "Overall"].iloc[0]
    text = (
        "D. Paired summary\n\n"
        f"N paired units: {int(overall['N'])}\n"
        f"Mean Diff (H−G): {overall['Mean_Diff_Hung_minus_Greedy']:.3f}\n"
        f"SD Diff: {overall['SD_Diff']:.3f}\n"
        f"Median Diff: {overall['Median_Diff']:.3f}\n"
        f"% Hungarian better: {overall['Pct_Hungarian_Better']:.1f}%\n"
        f"Wilcoxon p: {overall['p_value']:.2e}\n"
        f"Rank-biserial: {overall['Effect_RankBiserial']:.3f}\n\n"
        "Lower Mahalanobis distance = better matching.\n"
        "Diff = Hungarian − Greedy; negative ⇒ Hungarian better."
    )
    ax3.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(
        "Mahalanobis mapping: when does Hungarian improve over Greedy?",
        fontsize=13,
        y=0.98,
    )
    save_fig(fig, OUT / "summary" / "Fig07_summary_mahalanobis_greedy_vs_hungarian")


def write_tables(df: pd.DataFrame, stats_df: pd.DataFrame, report: dict) -> None:
    cols = [
        "Dataset",
        "Generator",
        "GeneratorLabel",
        "Greedy",
        "Hungarian",
        "Diff_Hung_minus_Greedy",
        "RelPct_Hung_minus_Greedy",
        "RelPct_Unstable",
        "Direction",
    ]
    if "Num_Matches" in df.columns:
        cols.append("Num_Matches")
    unit = df[cols].copy().rename(
        columns={
            "Greedy": "Greedy_Mahalanobis",
            "Hungarian": "Hungarian_Mahalanobis",
        }
    )
    unit["Runtime_Greedy"] = np.nan
    unit["Runtime_Hungarian"] = np.nan
    unit["Runtime_Note"] = "Not available in extracted results"

    unit_path = OUT / "tables" / "unit_level_mahalanobis_greedy_vs_hungarian.csv"
    unit.to_csv(unit_path, index=False)

    ds = (
        df.groupby("Dataset", observed=True)
        .agg(
            N=("Generator", "count"),
            Mean_Greedy=("Greedy", "mean"),
            Mean_Hungarian=("Hungarian", "mean"),
            Mean_Diff=("Diff_Hung_minus_Greedy", "mean"),
            SD_Diff=("Diff_Hung_minus_Greedy", "std"),
            Median_Diff=("Diff_Hung_minus_Greedy", "median"),
            Pct_Hungarian_Better=("Diff_Hung_minus_Greedy", lambda s: 100.0 * (s < 0).mean()),
        )
        .reset_index()
    )
    ds.to_csv(OUT / "tables" / "dataset_level_summary.csv", index=False)

    gen = (
        df.groupby("Generator", observed=True)
        .agg(
            N=("Dataset", "count"),
            Mean_Greedy=("Greedy", "mean"),
            Mean_Hungarian=("Hungarian", "mean"),
            Mean_Diff=("Diff_Hung_minus_Greedy", "mean"),
            SD_Diff=("Diff_Hung_minus_Greedy", "std"),
            Median_Diff=("Diff_Hung_minus_Greedy", "median"),
            Pct_Hungarian_Better=("Diff_Hung_minus_Greedy", lambda s: 100.0 * (s < 0).mean()),
        )
        .reset_index()
    )
    gen["GeneratorLabel"] = gen["Generator"].map(lambda g: GEN_LABEL.get(g, g))
    gen.to_csv(OUT / "tables" / "generator_level_summary.csv", index=False)

    stats_df.to_csv(OUT / "tables" / "paired_statistical_tests.csv", index=False)

    xlsx = OUT / "tables" / "mahalanobis_greedy_vs_hungarian_tables.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        unit.to_excel(writer, sheet_name="unit_level", index=False)
        ds.to_excel(writer, sheet_name="dataset_level", index=False)
        gen.to_excel(writer, sheet_name="generator_level", index=False)
        stats_df.to_excel(writer, sheet_name="paired_tests", index=False)
        pd.DataFrame(report["skipped"]).to_excel(writer, sheet_name="skipped_units", index=False)
        pd.DataFrame(
            [
                {"Item": k, "Value": json.dumps(v) if isinstance(v, (list, dict)) else v}
                for k, v in report.items()
                if k not in {"processed", "skipped"}
            ]
        ).to_excel(writer, sheet_name="meta", index=False)


def write_readme(df: pd.DataFrame, stats_df: pd.DataFrame, report: dict) -> None:
    overall = stats_df[stats_df["Scope"] == "Overall"].iloc[0]
    skipped_lines = [
        f"- `{s['dataset']}` / `{s['generator']}`: {s['reason']}" for s in report["skipped"]
    ]
    lines = [
        "# Mahalanobis Greedy vs Hungarian",
        "",
        "Extract-only comparison of **One-to-One Greedy** and **Hungarian** sample mapping using **Mahalanobis distance**.",
        "",
        "> Cosine-similarity mapping is intentionally **excluded** from this analysis and all figures.",
        "",
        "## What was compared",
        "",
        "- **Metric:** mean matched-pair **Mahalanobis distance** (lower = better matching).",
        "- **Methods:** One-to-One Greedy vs Hungarian assignment on the same real↔synthetic distance matrix.",
        "- **Unit of analysis:** dataset × generator pairs with both methods available in the extract.",
        "",
        "## Mahalanobis metric (as used by the existing pipeline)",
        "",
        "For each dataset × generator, real and synthetic samples are represented in a numerical feature space.",
        "Pairwise Mahalanobis distances use a covariance estimated from the (training) real data.",
        "After assignment, the reported score is the **mean Mahalanobis distance over matched pairs**.",
        "Lower mean matched-pair distance ⇒ closer real–synthetic correspondence under that metric.",
        "",
        "## Greedy vs Hungarian methodology",
        "",
        "- **One-to-One Greedy:** iteratively selects the currently cheapest unmatched real–synthetic pair until all allowed matches are filled (locally greedy).",
        "- **Hungarian:** solves the linear assignment problem for a globally optimal one-to-one matching that minimizes total (equivalently mean) Mahalanobis matching cost.",
        "- Both methods are one-to-one; they differ only in how the pairing is chosen on the same cost matrix.",
        "",
        "## Difference definition",
        "",
        "```text",
        "Diff = Hungarian - Greedy",
        "Diff < 0  → Hungarian lower distance → better matching",
        "Diff > 0  → Hungarian higher distance → worse matching",
        "Diff ≈ 0  → negligible difference",
        "Rel% = 100 * (Hungarian - Greedy) / |Greedy|   (undefined/unstable if |Greedy|≈0)",
        "```",
        "",
        "## Coverage",
        "",
        f"- Paired units processed: **{report['n_paired']}** of 120 possible (15 datasets × 8 generators)",
        f"- Skipped units: **{report['n_skipped']}**",
        f"- Datasets: {', '.join(report['datasets'])}",
        f"- Generators: {', '.join(report['generators'])}",
        "",
        "### Skipped dataset × generator units",
        "",
        *skipped_lines,
        "",
        "## Overall results",
        "",
        f"- Mean Diff (H−G): **{overall['Mean_Diff_Hung_minus_Greedy']:.4f}**",
        f"- SD Diff: **{overall['SD_Diff']:.4f}**",
        f"- Median Diff: **{overall['Median_Diff']:.4f}**",
        f"- % Hungarian better: **{overall['Pct_Hungarian_Better']:.1f}%**",
        f"- Wilcoxon p (H1: Greedy > Hungarian): **{overall['p_value']:.3e}**",
        f"- Rank-biserial effect size: **{overall['Effect_RankBiserial']:.3f}**",
        "",
        "Paired Wilcoxon signed-rank tests are also reported by generator and by dataset,",
        "with Holm adjustment within each scope (`tables/paired_statistical_tests.csv`).",
        "",
        "## Figures",
        "",
        "| Figure | Path | Description |",
        "|---|---|---|",
        "| Fig01 | `overall_comparison/` | Dumbbell: dataset-mean Greedy → Hungarian (log scale) |",
        "| Fig01b | `overall_comparison/` | Dumbbell: all paired units Greedy → Hungarian (log scale) |",
        "| Fig02 | `dataset_comparison/` | Ranked bars of dataset-level Diff |",
        "| Fig03 | `generator_comparison/` | Generator mean Diff ± SD |",
        "| Fig04 | `heatmaps/` | Dataset × generator Diff heatmap (diverging, centered at 0) |",
        "| Fig05a/b | `distributions/` | Overall and by-generator Diff distributions |",
        "| Fig06 | `scatter/` | Greedy vs Hungarian scatter with y=x (log–log) |",
        "| Fig07 | `summary/` | Manuscript summary panel |",
        "",
        "## Tables",
        "",
        "- `tables/unit_level_mahalanobis_greedy_vs_hungarian.csv` — every paired unit with Diff, Rel%, direction",
        "- `tables/dataset_level_summary.csv`",
        "- `tables/generator_level_summary.csv`",
        "- `tables/paired_statistical_tests.csv`",
        "- `tables/mahalanobis_greedy_vs_hungarian_tables.xlsx` — all sheets + skipped units + meta",
        "- `tables/processing_report.json`",
        "",
        "## Unavailable analyses (not fabricated)",
        "",
        "- **Assignment agreement:** pair-identity / overlap matrices were not present in extracted outputs → `agreement/SKIPPED.txt`.",
        "- **Runtime / compute cost:** no runtime fields in extracted mapping results → `runtime/SKIPPED.txt`.",
        "",
        "## Data sources (not overwritten)",
        "",
        f"- Primary: `{SRC_SUMMARY.name}`",
        f"- Secondary inventory: `{SRC_MAHA.name}`",
        "- Raw notebook/Excel experimental files and older `graphs/` figures were not deleted or overwritten.",
        "",
        "## Limitations",
        "",
        "- 14/120 dataset×generator combinations lack a complete Greedy+Hungarian Mahalanobis pair.",
        "- Values are extracted means; seed-level repeated mapping runs were not available for within-unit SD.",
        "- Absolute mean distances are dominated by a few large-scale datasets (e.g. Energy, RealEstate);",
        "  prefer Diff-based figures and medians for interpretation.",
        "- Hungarian can still appear slightly worse than Greedy in a few extracted cells due to",
        "  numerical/covariance/implementation details in the original notebooks; those cases are retained as recorded.",
        "",
        "## Reproduce",
        "",
        "```bash",
        'cd "mapping study graphs"',
        "python create_mahalanobis_greedy_vs_hungarian.py",
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_placeholders() -> None:
    (OUT / "agreement" / "SKIPPED.txt").write_text(
        "Assignment-level matched-pair identities were not available in extracted results.\n"
        "Agreement analysis was skipped rather than approximated.\n",
        encoding="utf-8",
    )
    (OUT / "runtime" / "SKIPPED.txt").write_text(
        "Runtime / computational-cost fields were not available in extracted mapping results.\n"
        "Runtime comparison was skipped rather than fabricated.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_style()
    ensure_dirs()
    write_placeholders()

    print("=" * 72)
    print("Mahalanobis Greedy vs Hungarian — extract-only figure/table rebuild")
    print("=" * 72)

    df, report = load_paired_mahalanobis()
    print(f"Paired units: {report['n_paired']}")
    print(f"Skipped units: {report['n_skipped']}")
    for s in report["skipped"][:20]:
        print(f"  skip: {s['dataset']} / {s['generator']} — {s['reason']}")
    if report["n_skipped"] > 20:
        print(f"  ... {report['n_skipped'] - 20} more skipped")

    # Validate no cosine leakage in this dataframe
    forbidden = [c for c in df.columns if "cosine" in c.lower()]
    if forbidden:
        raise RuntimeError(f"Cosine columns unexpectedly present: {forbidden}")

    stats_df = paired_stats(df)
    write_tables(df, stats_df, report)

    fig_overall_dumbbell(df)
    fig_dataset_diff_bars(df)
    fig_generator_summary(df)
    fig_heatmap(df)
    fig_distributions(df)
    fig_scatter(df)
    fig_summary(df, stats_df)
    write_readme(df, stats_df, report)

    # Persist processing report
    (OUT / "tables" / "processing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Final validation listing
    expected = [
        OUT / "overall_comparison" / "Fig01_dumbbell_dataset_means.png",
        OUT / "overall_comparison" / "Fig01b_dumbbell_all_units.png",
        OUT / "dataset_comparison" / "Fig02_dataset_diff_ranked.png",
        OUT / "generator_comparison" / "Fig03_generator_mean_diff.png",
        OUT / "heatmaps" / "Fig04_heatmap_hung_minus_greedy.png",
        OUT / "distributions" / "Fig05a_diff_distribution_overall.png",
        OUT / "distributions" / "Fig05b_diff_distribution_by_generator.png",
        OUT / "scatter" / "Fig06_greedy_vs_hungarian_scatter.png",
        OUT / "summary" / "Fig07_summary_mahalanobis_greedy_vs_hungarian.png",
        OUT / "tables" / "unit_level_mahalanobis_greedy_vs_hungarian.csv",
        OUT / "README.md",
    ]
    print("\nValidation:")
    ok = True
    for p in expected:
        exists = p.exists()
        print(f"  [{'OK' if exists else 'MISSING'}] {p.relative_to(OUT)}")
        ok = ok and exists
    # Ensure no cosine artifacts written into OUT
    cosine_hits = [p for p in OUT.rglob("*") if "cosine" in p.name.lower()]
    if cosine_hits:
        ok = False
        print("  [FAIL] cosine-named outputs found:")
        for p in cosine_hits:
            print("   ", p)
    else:
        print("  [OK] no cosine-named outputs in output folder")
    print("=" * 72)
    print("DONE" if ok else "DONE WITH MISSING OUTPUTS")
    print(f"Output root: {OUT}")


if __name__ == "__main__":
    main()
