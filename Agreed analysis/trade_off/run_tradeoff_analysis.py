#!/usr/bin/env python3
"""Per-dataset fidelity/privacy vs utility trade-off analysis.

IMPORTANT CONSTRAINTS
---------------------
* Never average or combine results across datasets.
* Average ONLY across the 10 downstream classifiers (classification)
  or 10 regression models (regression) within the same dataset.
* Each scatter has one point per generator (up to 8).

Outputs (under trade_off/<DatasetFolder>/):
  fidelity_vs_accuracy_gap.{png,pdf}   (classification)
  mia_vs_accuracy_gap.{png,pdf}        (classification)
  fidelity_vs_r2_gap.{png,pdf}         (regression)
  mia_vs_r2_gap.{png,pdf}              (regression)
  summary_table.csv
  statistical_summary.csv
  plus dataset-prefixed copies of each figure.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MASTER = ROOT / "Results" / "MasterData"
PAPER_RESULTS = ROOT / "paper results"
sys.path.insert(0, str(ROOT / "Agreed analysis"))
from latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402

DPI = 600
ALPHA = 0.05
NAVY = "#1f3a5f"
INK = "#1c1f24"
GRID = "#d5dde6"
CI_FILL = "#9bb7d4"

GENERATORS = [
    "ForestDiffusion",
    "TVAE",
    "CTABGAN",
    "WGAN_GP",
    "GaussianCopula",
    "CopulaGAN",
    "CTGAN",
    "TabDDPM",
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

# Consistent colours across every figure
GENERATOR_COLORS = {
    "ForestDiffusion": "#1b9e77",
    "TVAE": "#d95f02",
    "CTABGAN": "#7570b3",
    "WGAN_GP": "#e7298a",
    "GaussianCopula": "#66a61e",
    "CopulaGAN": "#e6ab02",
    "CTGAN": "#a6761d",
    "TabDDPM": "#666666",
}

# Canonical classifier aliases (Wine notebooks use mixed names)
CLASSIFIER_ALIASES = {
    "LogisticRegression": "Logistic Regression",
    "GaussianNB": "Naive Bayes",
    "SVC-RBF": "SVM-RBF",
}

CLASSIFICATION_DATASETS = [
    {
        "raw": "1. Cancer",
        "folder": "Cancer",
        "title": "Cancer",
        "slug": "cancer",
    },
    {
        "raw": "2. Alzhimers",
        "folder": "Alzheimer's",
        "title": "Alzheimer's",
        "slug": "alzheimers",
    },
    {
        "raw": "3. Adult",
        "folder": "Adult",
        "title": "Adult",
        "slug": "adult",
    },
    {
        "raw": "4. Forest cover dataset",
        "folder": "ForestCover",
        "title": "Forest Cover",
        "slug": "forestcover",
    },
    {
        "raw": "5. Bank Markting",
        "folder": "BankMarketing",
        "title": "Bank Marketing",
        "slug": "bankmarketing",
    },
    {
        "raw": "6. Wine dataset",
        "folder": "WineQuality",
        "title": "Wine Quality",
        "slug": "winequality",
    },
    {
        "raw": "7. CDC diabetes dataset",
        "folder": "CDCDiabetes",
        "title": "CDC Diabetes",
        "slug": "cdcdiabetes",
    },
    {
        "raw": "8. Mushroom dataset",
        "folder": "Mushroom",
        "title": "Mushroom",
        "slug": "mushroom",
    },
    {
        "raw": "9. MAGIC Gamma Telescope",
        "folder": "MAGICGamma",
        "title": "MAGIC Gamma",
        "slug": "magicgamma",
    },
]

REGRESSION_DATASETS = [
    {
        "raw": "10. Metro interstate",
        "folder": "MetroInterstate",
        "title": "Metro Interstate",
        "slug": "metrointerstate",
    },
    {
        "raw": "11. online shopping",
        "folder": "OnlineShopping",
        "title": "Online Shopping",
        "slug": "onlineshopping",
    },
    {
        "raw": "12. Air Quality",
        "folder": "AirQuality",
        "title": "Air Quality",
        "slug": "airquality",
    },
    {
        "raw": "13. Concrete Compressive Strength",
        "folder": "Concrete",
        "title": "Concrete",
        "slug": "concrete",
    },
    {
        "raw": "14. Energy Efficiency",
        "folder": "EnergyEfficiency",
        "title": "Energy Efficiency",
        "slug": "energyefficiency",
    },
    {
        "raw": "15. Real Estate Valuation",
        "folder": "RealEstate",
        "title": "Real Estate",
        "slug": "realestate",
    },
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _metric_value(df: pd.DataFrame) -> pd.Series:
    """Prefer MetricValue, then Mean, then Value."""
    out = df["MetricValue"].copy()
    if "Mean" in df.columns:
        out = out.fillna(df["Mean"])
    if "Value" in df.columns:
        out = out.fillna(df["Value"])
    return pd.to_numeric(out, errors="coerce")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    util = pd.read_csv(MASTER / "utility_long.csv", low_memory=False)
    fid = pd.read_csv(MASTER / "fidelity_long.csv", low_memory=False)
    priv = pd.read_csv(MASTER / "privacy_long.csv", low_memory=False)

    # Keep no-leakage rows (and rows with missing leakage treated as 0)
    for df in (util, fid, priv):
        if "LeakageLevel" in df.columns:
            ll = pd.to_numeric(df["LeakageLevel"], errors="coerce")
            df.drop(index=df.index[(ll.notna()) & (ll != 0.0)], inplace=True)

    return util, fid, priv


def fidelity_by_generator(fid: pd.DataFrame, dataset: str) -> pd.Series:
    sub = fid[(fid["Dataset"] == dataset) & (fid["Metric"] == "Quality_Score")].copy()
    sub["_v"] = _metric_value(sub)
    # One Quality_Score per generator; if duplicates, take mean within dataset
    return sub.groupby("Generator", sort=False)["_v"].mean()


def _mia_from_paper_results(dataset: str) -> pd.Series:
    """Load MIA AUC from paper-results Privacy_Summary / MIA sheet (all generators)."""
    xlsx = PAPER_RESULTS / dataset / "fidelity_privacy_metrics.xlsx"
    if not xlsx.exists():
        return pd.Series(dtype=float)

    # Prefer dedicated MIA sheet; fall back to Privacy_Summary.AUC
    for sheet in ("MIA", "Privacy_Summary"):
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet)
        except Exception:
            continue
        if "Generator" not in df.columns:
            continue
        if "AUC" not in df.columns:
            continue
        tmp = df[["Generator", "AUC"]].copy()
        tmp["AUC"] = pd.to_numeric(tmp["AUC"], errors="coerce")
        tmp = tmp.dropna(subset=["AUC"])
        if tmp.empty:
            continue
        return tmp.groupby("Generator", sort=False)["AUC"].mean()
    return pd.Series(dtype=float)


def mia_by_generator(priv: pd.DataFrame, dataset: str) -> pd.Series:
    """MIA AUC per generator for one dataset.

    MasterData is incomplete for Adult (SDV generators missing). When that
    happens, use the paper-results workbook so all 8 generators appear.
    Prefer the more complete single source rather than mixing incompatible runs.
    """
    sub = priv[(priv["Dataset"] == dataset) & (priv["Metric"] == "MIA_AUC")].copy()
    sub["_v"] = _metric_value(sub)
    master = sub.groupby("Generator", sort=False)["_v"].mean()

    paper = _mia_from_paper_results(dataset)
    if paper.empty:
        return master

    master_n = int(master.reindex(GENERATORS).notna().sum())
    paper_n = int(paper.reindex(GENERATORS).notna().sum())
    if paper_n > master_n:
        return paper
    return master


def _normalize_classifier(name: str) -> str:
    if pd.isna(name):
        return name
    return CLASSIFIER_ALIASES.get(str(name), str(name))


def mean_utility_by_generator(
    util: pd.DataFrame,
    dataset: str,
    metric: str,
    evaluation: str,
    *,
    task: str,
) -> pd.Series:
    """Average metric across classifiers / regressors for one dataset.

    Never averages across datasets. Within a dataset, averages only across
    the downstream models for that generator.
    """
    sub = util[
        (util["Dataset"] == dataset)
        & (util["Metric"] == metric)
        & (util["EvaluationType"] == evaluation)
    ].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    sub["_v"] = _metric_value(sub)

    if task == "classification":
        sub["_model"] = sub["Classifier"].map(_normalize_classifier)
    else:
        model = sub["RegressionModel"].where(
            sub["RegressionModel"].notna() & (sub["RegressionModel"].astype(str) != ""),
            sub["Regressor"],
        )
        sub["_model"] = model

    # Collapse duplicate model aliases first, then average across models
    per_model = (
        sub.dropna(subset=["_model", "_v"])
        .groupby(["Generator", "_model"], sort=False)["_v"]
        .mean()
        .reset_index()
    )
    return per_model.groupby("Generator", sort=False)["_v"].mean()


def build_dataset_frame(
    util: pd.DataFrame,
    fid: pd.DataFrame,
    priv: pd.DataFrame,
    dataset: str,
    *,
    task: str,
) -> pd.DataFrame:
    """One row per generator for a single dataset (no cross-dataset ops)."""
    if task == "classification":
        util_metric = "Accuracy"
        tstr_col = "Mean_TSTR_Accuracy"
        trtr_col = "TRTR_Accuracy"
        gap_col = "Utility_Gap_Accuracy"
    else:
        util_metric = "R2"
        tstr_col = "Mean_TSTR_R2"
        trtr_col = "TRTR_R2"
        gap_col = "Utility_Gap_R2"

    fidelity = fidelity_by_generator(fid, dataset)
    mia = mia_by_generator(priv, dataset)
    tstr = mean_utility_by_generator(
        util, dataset, util_metric, "TSTR", task=task
    )
    trtr = mean_utility_by_generator(
        util, dataset, util_metric, "TRTR", task=task
    )

    rows = []
    for gen in GENERATORS:
        trtr_v = float(trtr[gen]) if gen in trtr.index and pd.notna(trtr[gen]) else np.nan
        tstr_v = float(tstr[gen]) if gen in tstr.index and pd.notna(tstr[gen]) else np.nan
        fid_v = float(fidelity[gen]) if gen in fidelity.index and pd.notna(fidelity[gen]) else np.nan
        mia_v = float(mia[gen]) if gen in mia.index and pd.notna(mia[gen]) else np.nan
        gap_v = trtr_v - tstr_v if pd.notna(trtr_v) and pd.notna(tstr_v) else np.nan
        rows.append(
            {
                "Generator": gen,
                "Display": DISPLAY[gen],
                "Fidelity_SDMetrics": fid_v,
                "MIA": mia_v,
                tstr_col: tstr_v,
                trtr_col: trtr_v,
                gap_col: gap_v,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson, Spearman, OLS, R², 95% CI for slope & intercept (n points)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask].astype(float)
    y = np.asarray(y)[mask].astype(float)
    n = int(x.size)

    empty = {
        "n": n,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r2": np.nan,
        "slope_ci_low": np.nan,
        "slope_ci_high": np.nan,
        "intercept_ci_low": np.nan,
        "intercept_ci_high": np.nan,
        "stderr_slope": np.nan,
    }
    if n < 3:
        return empty

    # Constant input → correlations undefined
    if np.nanstd(x) < 1e-15 or np.nanstd(y) < 1e-15:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, intercept, r_val, p_val, se = stats.linregress(x, y)
        return {
            **empty,
            "n": n,
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r_val**2) if np.isfinite(r_val) else np.nan,
            "stderr_slope": float(se) if np.isfinite(se) else np.nan,
        }

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    slope, intercept, r_val, p_lin, se = stats.linregress(x, y)
    r2 = float(r_val**2)

    # 95% CI for slope and intercept via Student-t
    df = n - 2
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, df))
    slope_ci = (slope - tcrit * se, slope + tcrit * se)

    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)
    y_hat = intercept + slope * x
    sse = np.sum((y - y_hat) ** 2)
    mse = sse / df
    se_intercept = float(np.sqrt(mse * (1.0 / n + x_mean**2 / ssx)))
    intercept_ci = (intercept - tcrit * se_intercept, intercept + tcrit * se_intercept)

    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "slope_ci_low": float(slope_ci[0]),
        "slope_ci_high": float(slope_ci[1]),
        "intercept_ci_low": float(intercept_ci[0]),
        "intercept_ci_high": float(intercept_ci[1]),
        "stderr_slope": float(se),
    }


def regression_band(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS mean prediction and 95% confidence band on x_grid."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask].astype(float)
    y = np.asarray(y)[mask].astype(float)
    n = x.size
    if n < 3 or np.nanstd(x) < 1e-15:
        y_hat = np.full_like(x_grid, np.nan, dtype=float)
        return y_hat, y_hat, y_hat

    slope, intercept, *_ = stats.linregress(x, y)
    y_fit = intercept + slope * x_grid
    df = n - 2
    tcrit = float(stats.t.ppf(1 - ALPHA / 2, df))
    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)
    resid = y - (intercept + slope * x)
    mse = float(np.sum(resid**2) / df)
    se_fit = np.sqrt(mse * (1.0 / n + (x_grid - x_mean) ** 2 / ssx))
    return y_fit, y_fit - tcrit * se_fit, y_fit + tcrit * se_fit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8a93a0")
    ax.spines["bottom"].set_color("#8a93a0")
    ax.tick_params(colors=INK, labelsize=11)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def _stats_annotation(st: dict) -> str:
    def fmt(v: float, digits: int = 3) -> str:
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"

    def fmt_p(p: float) -> str:
        if not np.isfinite(p):
            return "n/a"
        if p < 1e-4:
            return f"{p:.2e}"
        return f"{p:.4f}"

    return (
        f"Pearson $r$ = {fmt(st['pearson_r'])}  ($p$ = {fmt_p(st['pearson_p'])})\n"
        f"Spearman $\\rho$ = {fmt(st['spearman_rho'])}  ($p$ = {fmt_p(st['spearman_p'])})\n"
        f"$R^2$ = {fmt(st['r2'])}   slope = {fmt(st['slope'])}"
    )


def plot_tradeoff(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_stem: Path,
    font_name: str,
) -> dict:
    """Publication scatter with OLS line, 95% CI band, labels, stats box."""
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)
    st = fit_stats(x, y)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    _style_axes(ax)

    if len(plot_df) >= 3 and np.nanstd(x) > 1e-15:
        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_fit, y_lo, y_hi = regression_band(x, y, x_grid)
        ax.fill_between(x_grid, y_lo, y_hi, color=CI_FILL, alpha=0.35, linewidth=0, zorder=1)
        ax.plot(x_grid, y_fit, color=NAVY, linewidth=2.0, zorder=2, label="OLS fit")

    # Fixed staggered offsets keep all 8 labels inside the axes
    offsets = [
        (10, 10), (10, -18), (-10, 10), (-10, -18),
        (12, 2), (-12, 2), (10, 22), (-10, 22),
    ]
    # Sort points so nearby clusters get different offset slots
    ordered = plot_df.sort_values([x_col, y_col]).reset_index(drop=True)
    for i, row in ordered.iterrows():
        gen = row["Generator"]
        color = GENERATOR_COLORS.get(gen, "#333333")
        xv = float(row[x_col])
        yv = float(row[y_col])
        ax.scatter(
            xv,
            yv,
            s=110,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )
        label = f"{row['Display']}\n({xv:.3f}, {yv:.3f})"
        dx, dy = offsets[int(i) % len(offsets)]
        ax.annotate(
            label,
            (xv, yv),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.2,
            color=INK,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            linespacing=1.1,
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="#d5dde6",
                alpha=0.9,
                linewidth=0.6,
            ),
        )

    ax.set_xlabel(xlabel, fontsize=12.5, color=NAVY, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12.5, color=NAVY, fontweight="bold")
    ax.set_title(title, fontsize=13.5, color=NAVY, fontweight="bold", pad=10)

    # Extra padding so edge labels are not clipped
    if len(plot_df):
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        xpad = max((xmax - xmin) * 0.12, 0.02)
        ypad = max((ymax - ymin) * 0.18, 0.02)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    # Legend: generators + fit
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GENERATOR_COLORS[g],
            markeredgecolor="white",
            markersize=9,
            label=DISPLAY[g],
        )
        for g in GENERATORS
        if g in set(plot_df["Generator"])
    ]
    if len(plot_df) >= 3 and np.nanstd(x) > 1e-15:
        handles.append(
            Line2D([0], [0], color=NAVY, linewidth=2.0, label="OLS + 95% CI")
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=8.5,
        title="Generator",
        title_fontsize=9,
        edgecolor="#c5d0dc",
        fancybox=False,
        borderaxespad=0.0,
    )

    ax.text(
        0.02,
        0.02,
        _stats_annotation(st),
        transform=ax.transAxes,
        fontsize=9,
        color=INK,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.45",
            facecolor="white",
            edgecolor="#c5d0dc",
            alpha=0.92,
        ),
        zorder=6,
    )

    apply_font_to_figure(fig, font_name)
    fig.tight_layout()

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    fig.savefig(pdf, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return st


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def write_summary_table(
    df: pd.DataFrame,
    path: Path,
    *,
    task: str,
    fidelity_stats: dict,
    mia_stats: dict,
) -> None:
    if task == "classification":
        tstr_col, trtr_col, gap_col = (
            "Mean_TSTR_Accuracy",
            "TRTR_Accuracy",
            "Utility_Gap_Accuracy",
        )
        out = pd.DataFrame(
            {
                "Generator": df["Display"],
                "Fidelity_SDMetrics": df["Fidelity_SDMetrics"],
                "MIA": df["MIA"],
                "Mean_TSTR_Accuracy": df[tstr_col],
                "TRTR_Accuracy": df[trtr_col],
                "Utility_Gap": df[gap_col],
                "Pearson_Fidelity_vs_Gap": fidelity_stats["pearson_r"],
                "Spearman_Fidelity_vs_Gap": fidelity_stats["spearman_rho"],
                "Pearson_MIA_vs_Gap": mia_stats["pearson_r"],
                "Spearman_MIA_vs_Gap": mia_stats["spearman_rho"],
            }
        )
    else:
        tstr_col, trtr_col, gap_col = "Mean_TSTR_R2", "TRTR_R2", "Utility_Gap_R2"
        out = pd.DataFrame(
            {
                "Generator": df["Display"],
                "Fidelity_SDMetrics": df["Fidelity_SDMetrics"],
                "MIA": df["MIA"],
                "Mean_TSTR_R2": df[tstr_col],
                "TRTR_R2": df[trtr_col],
                "Utility_Gap": df[gap_col],
                "Pearson_Fidelity_vs_Gap": fidelity_stats["pearson_r"],
                "Spearman_Fidelity_vs_Gap": fidelity_stats["spearman_rho"],
                "Pearson_MIA_vs_Gap": mia_stats["pearson_r"],
                "Spearman_MIA_vs_Gap": mia_stats["spearman_rho"],
            }
        )
    out.to_csv(path, index=False, float_format="%.6f")


def write_statistical_summary(
    path: Path,
    *,
    dataset_title: str,
    rows: list[dict],
) -> None:
    frame = pd.DataFrame(rows)
    frame.insert(0, "Dataset", dataset_title)
    cols = [
        "Dataset",
        "Analysis",
        "X_Metric",
        "Y_Metric",
        "n_generators",
        "Pearson_r",
        "Pearson_p",
        "Spearman_rho",
        "Spearman_p",
        "Slope",
        "Intercept",
        "R2",
        "Slope_95CI_low",
        "Slope_95CI_high",
        "Intercept_95CI_low",
        "Intercept_95CI_high",
    ]
    frame[cols].to_csv(path, index=False, float_format="%.6g")


def process_dataset(
    cfg: dict,
    util: pd.DataFrame,
    fid: pd.DataFrame,
    priv: pd.DataFrame,
    *,
    task: str,
    font_name: str,
) -> None:
    out_dir = SCRIPT_DIR / cfg["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset_frame(util, fid, priv, cfg["raw"], task=task)
    df.to_csv(out_dir / "generator_metrics_raw.csv", index=False, float_format="%.6f")

    stats_rows: list[dict] = []

    if task == "classification":
        gap_col = "Utility_Gap_Accuracy"
        y_label = "Utility Gap (Accuracy)\nTRTR Acc. − Mean TSTR Acc."
        fid_stem = "fidelity_vs_accuracy_gap"
        mia_stem = "mia_vs_accuracy_gap"
        fid_title = f"{cfg['title']}: Fidelity vs Utility Gap (Accuracy)"
        mia_title = f"{cfg['title']}: MIA vs Utility Gap (Accuracy)"
        y_metric_name = "Utility Gap (Accuracy)"
    else:
        gap_col = "Utility_Gap_R2"
        y_label = "Utility Gap ($R^2$)\nTRTR $R^2$ − Mean TSTR $R^2$"
        fid_stem = "fidelity_vs_r2_gap"
        mia_stem = "mia_vs_r2_gap"
        fid_title = f"{cfg['title']}: Fidelity vs Utility Gap ($R^2$)"
        mia_title = f"{cfg['title']}: MIA vs Utility Gap ($R^2$)"
        y_metric_name = "Utility Gap (R2)"

    # --- Fidelity vs Utility Gap ---
    fid_st = plot_tradeoff(
        df,
        x_col="Fidelity_SDMetrics",
        y_col=gap_col,
        xlabel="SDMetrics Overall Quality Score (Fidelity)",
        ylabel=y_label,
        title=fid_title,
        out_stem=out_dir / fid_stem,
        font_name=font_name,
    )
    # Dataset-prefixed copy
    for ext in (".png", ".pdf"):
        src = out_dir / f"{fid_stem}{ext}"
        dst = out_dir / f"{cfg['slug']}_{fid_stem}{ext}"
        if src.exists():
            dst.write_bytes(src.read_bytes())

    stats_rows.append(
        {
            "Analysis": fid_stem,
            "X_Metric": "SDMetrics Quality Score",
            "Y_Metric": y_metric_name,
            "n_generators": fid_st["n"],
            "Pearson_r": fid_st["pearson_r"],
            "Pearson_p": fid_st["pearson_p"],
            "Spearman_rho": fid_st["spearman_rho"],
            "Spearman_p": fid_st["spearman_p"],
            "Slope": fid_st["slope"],
            "Intercept": fid_st["intercept"],
            "R2": fid_st["r2"],
            "Slope_95CI_low": fid_st["slope_ci_low"],
            "Slope_95CI_high": fid_st["slope_ci_high"],
            "Intercept_95CI_low": fid_st["intercept_ci_low"],
            "Intercept_95CI_high": fid_st["intercept_ci_high"],
        }
    )

    # --- MIA vs Utility Gap ---
    mia_st = plot_tradeoff(
        df,
        x_col="MIA",
        y_col=gap_col,
        xlabel="Membership Inference Attack (MIA AUC)",
        ylabel=y_label,
        title=mia_title,
        out_stem=out_dir / mia_stem,
        font_name=font_name,
    )
    for ext in (".png", ".pdf"):
        src = out_dir / f"{mia_stem}{ext}"
        dst = out_dir / f"{cfg['slug']}_{mia_stem}{ext}"
        if src.exists():
            dst.write_bytes(src.read_bytes())

    stats_rows.append(
        {
            "Analysis": mia_stem,
            "X_Metric": "MIA AUC",
            "Y_Metric": y_metric_name,
            "n_generators": mia_st["n"],
            "Pearson_r": mia_st["pearson_r"],
            "Pearson_p": mia_st["pearson_p"],
            "Spearman_rho": mia_st["spearman_rho"],
            "Spearman_p": mia_st["spearman_p"],
            "Slope": mia_st["slope"],
            "Intercept": mia_st["intercept"],
            "R2": mia_st["r2"],
            "Slope_95CI_low": mia_st["slope_ci_low"],
            "Slope_95CI_high": mia_st["slope_ci_high"],
            "Intercept_95CI_low": mia_st["intercept_ci_low"],
            "Intercept_95CI_high": mia_st["intercept_ci_high"],
        }
    )

    write_summary_table(
        df,
        out_dir / "summary_table.csv",
        task=task,
        fidelity_stats=fid_st,
        mia_stats=mia_st,
    )
    write_statistical_summary(
        out_dir / "statistical_summary.csv",
        dataset_title=cfg["title"],
        rows=stats_rows,
    )

    n_fid = int(df["Fidelity_SDMetrics"].notna().sum())
    n_mia = int(df["MIA"].notna().sum())
    n_gap = int(df[gap_col].notna().sum())
    print(
        f"  [{cfg['folder']}] generators with fidelity={n_fid}, "
        f"MIA={n_mia}, utility gap={n_gap}"
    )


def main() -> None:
    font_name = configure_times_font()
    print(f"Font: {font_name}")
    print(f"Loading master tables from {MASTER}")
    util, fid, priv = load_tables()

    print("\n=== Classification datasets (Accuracy gap) ===")
    for cfg in CLASSIFICATION_DATASETS:
        process_dataset(cfg, util, fid, priv, task="classification", font_name=font_name)

    print("\n=== Regression datasets (R² gap) ===")
    for cfg in REGRESSION_DATASETS:
        process_dataset(cfg, util, fid, priv, task="regression", font_name=font_name)

    # Master index across datasets (no averaging — just a catalogue)
    index_rows = []
    for cfg in CLASSIFICATION_DATASETS + REGRESSION_DATASETS:
        folder = SCRIPT_DIR / cfg["folder"]
        stats_path = folder / "statistical_summary.csv"
        if stats_path.exists():
            s = pd.read_csv(stats_path)
            index_rows.append(s)
    if index_rows:
        catalogue = pd.concat(index_rows, ignore_index=True)
        catalogue.to_csv(SCRIPT_DIR / "all_datasets_statistical_summary.csv", index=False)
        print(f"\nWrote {SCRIPT_DIR / 'all_datasets_statistical_summary.csv'}")

    print("\nDone. Each dataset folder is independent (no cross-dataset averaging).")


if __name__ == "__main__":
    main()
