#!/usr/bin/env python3
"""Friedman → Nemenyi CD diagrams + rank tables for classification and regression gaps."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    ALPHA,
    CLS_GAP_METRICS,
    DISPLAY_DATASET,
    DISPLAY_GENERATOR,
    NAVY,
    REG_GAP_LABELS,
    REG_GAP_METRICS,
)
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402
from src.load_results import aggregate_10_seed  # noqa: E402

FIG_CLS = ANALYSIS / "figures" / "classification"
FIG_REG = ANALYSIS / "figures" / "regression"
TAB_MAIN = ANALYSIS / "tables" / "main"
TAB_SUPP = ANALYSIS / "tables" / "supplementary"


def gap_matrix(agg: pd.DataFrame, metric: str, task: str) -> pd.DataFrame:
    sub = agg[
        (agg["problem_type"] == task)
        & (agg["metric_name"] == metric)
        & (agg["n_seeds"] > 0)
    ].copy()
    sub["Generator"] = sub["generator"].map(lambda g: DISPLAY_GENERATOR.get(g, g))
    sub["Dataset"] = sub["dataset_id"].map(lambda d: DISPLAY_DATASET.get(d, d))
    wide = sub.pivot_table(index="Dataset", columns="Generator", values="mean")
    return wide


def analyze_metric(wide: pd.DataFrame, metric: str, label: str, out_dir: Path, fig_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = metric.lower()

    # Lower gap = better → rank ascending
    ranks = wide.rank(axis=1, method="average", ascending=True)
    summary = (
        ranks.mean()
        .rename("AverageRank")
        .to_frame()
        .assign(
            MedianRank=ranks.median(),
            StdDevRank=ranks.std(ddof=1),
        )
        .reset_index()
        .rename(columns={"index": "Generator"})
        .sort_values("AverageRank", ignore_index=True)
    )
    # fix column name if Generator already from reset
    if "Generator" not in summary.columns:
        summary = summary.rename(columns={summary.columns[0]: "Generator"})

    # Friedman on negated gaps (higher better)
    wide_perf = -wide
    order = summary["Generator"].tolist()
    arrays = [wide_perf[g].to_numpy() for g in wide_perf.columns]
    chi2, p = friedmanchisquare(*arrays)
    friedman = {
        "chi2": float(chi2),
        "df": int(wide.shape[1] - 1),
        "p_value": float(p),
        "n_datasets": int(wide.shape[0]),
        "n_generators": int(wide.shape[1]),
        "significant": bool(p < ALPHA),
        "metric": metric,
        "unit_of_analysis": "dataset (mean over 10 generator-training seeds)",
        "sample_size_note": "N datasets × K generators; seeds aggregated before ranking",
    }

    nemenyi = sp.posthoc_nemenyi_friedman(wide_perf[order])
    nemenyi = nemenyi.loc[order, order]
    holm = sp.posthoc_wilcoxon(wide_perf[order].to_numpy().T, p_adjust="holm")
    holm.index = order
    holm.columns = order

    avg_ranks = summary.set_index("Generator")["AverageRank"]

    def pairs(matrix: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for a, b in combinations(order, 2):
            rows.append(
                {
                    "Generator_A": a,
                    "Generator_B": b,
                    "AvgRank_A": float(avg_ranks[a]),
                    "AvgRank_B": float(avg_ranks[b]),
                    "p_value": float(matrix.loc[a, b]),
                    "Significant_alpha_0.05": bool(float(matrix.loc[a, b]) < ALPHA),
                }
            )
        return pd.DataFrame(rows).sort_values("p_value", ignore_index=True)

    nemenyi_pairs = pairs(nemenyi)
    holm_pairs = pairs(holm)

    # Save tables
    wide.round(6).to_csv(out_dir / f"{prefix}_by_dataset.csv")
    ranks.round(4).to_csv(out_dir / f"{prefix}_ranks_by_dataset.csv")
    summary.round(6).to_csv(out_dir / f"{prefix}_friedman_average_ranks.csv", index=False)
    pd.DataFrame([friedman]).to_csv(out_dir / f"{prefix}_friedman_test.csv", index=False)
    nemenyi.round(6).to_csv(out_dir / f"{prefix}_nemenyi_pvalues.csv")
    nemenyi_pairs.to_csv(out_dir / f"{prefix}_nemenyi_pairwise.csv", index=False)
    holm.round(6).to_csv(out_dir / f"{prefix}_holm_wilcoxon_pvalues.csv")
    holm_pairs.to_csv(out_dir / f"{prefix}_holm_pairwise.csv", index=False)

    # CD diagram via scikit-posthocs (Agreed analysis style)
    font_name = configure_times_font()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=nemenyi,
        alpha=ALPHA,
        ax=ax,
        marker_props={"s": 60, "zorder": 4},
        elbow_props={"linewidth": 1.25},
        crossbar_props={"color": "#c0392b", "linewidth": 2.8, "marker": "o", "markersize": 4},
        label_props={"fontsize": 10.5, "fontfamily": "serif"},
        text_h_margin=0.02,
    )
    ax.set_title(
        f"Critical-Difference Diagram — {label} Utility Gap Ranking",
        fontsize=14.5,
        fontweight="bold",
        color=NAVY,
        pad=36,
        fontfamily="serif",
        fontname=font_name,
    )
    p_str = "< 0.0001" if friedman["p_value"] < 0.0001 else f"{friedman['p_value']:.4f}"
    fig.text(
        0.5,
        0.965,
        f"Friedman: $\\chi^2$ = {friedman['chi2']:.2f}, df = {friedman['df']}, "
        f"p = {p_str}   |   Nemenyi post-hoc, α = {ALPHA}   |   "
        "generators joined by a red bar are not significantly different   |   "
        "ranks from 10-seed means",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#5a5f66",
        fontfamily="serif",
        fontname=font_name,
    )
    apply_font_to_figure(fig, font_name)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(fig_dir / f"{prefix}_critical_difference_diagram.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Excel workbook
    xlsx = out_dir / f"{prefix}_analysis.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"Item": "Metric", "Description": f"{metric} (TRTR−TSTR); lower gap better"},
                {"Item": "Seeds", "Description": "Mean over 10 generator-training seeds before ranking"},
                {"Item": "Unit of analysis", "Description": friedman["unit_of_analysis"]},
                {"Item": "Omnibus", "Description": f"Friedman α={ALPHA}"},
                {"Item": "Post-hoc", "Description": "Nemenyi + Holm-Wilcoxon"},
            ]
        ).to_excel(writer, sheet_name="00_README", index=False)
        wide.round(6).to_excel(writer, sheet_name="01_Gap_by_Dataset")
        ranks.round(4).to_excel(writer, sheet_name="02_Ranks_by_Dataset")
        summary.round(6).to_excel(writer, sheet_name="03_Average_Ranks", index=False)
        pd.DataFrame([friedman]).to_excel(writer, sheet_name="04_Friedman_Test", index=False)
        nemenyi.round(6).to_excel(writer, sheet_name="05_Nemenyi_Pvalues")
        nemenyi_pairs.to_excel(writer, sheet_name="06_Nemenyi_Pairwise", index=False)
        holm.round(6).to_excel(writer, sheet_name="08_Holm_Pvalues")
        holm_pairs.to_excel(writer, sheet_name="09_Holm_Pairwise", index=False)

    print(f"  {metric}: chi2={chi2:.2f} p={p:.4g} -> {fig_dir / (prefix + '_critical_difference_diagram.png')}")
    return summary, friedman


def main():
    print("Loading 10-seed aggregate…")
    agg = aggregate_10_seed()
    TAB_MAIN.mkdir(parents=True, exist_ok=True)
    TAB_SUPP.mkdir(parents=True, exist_ok=True)

    print("Classification utility-gap CD analysis")
    cls_out = TAB_MAIN / "classification_gaps"
    for metric in CLS_GAP_METRICS:
        label = metric.replace("_Gap", "")
        wide = gap_matrix(agg, metric, "classification")
        analyze_metric(wide, metric, label, cls_out, FIG_CLS)

    print("Regression utility-gap CD analysis")
    reg_out = TAB_MAIN / "regression_gaps"
    for metric in REG_GAP_METRICS:
        wide = gap_matrix(agg, metric, "regression")
        analyze_metric(wide, metric, REG_GAP_LABELS[metric], reg_out, FIG_REG)

    print("done")


if __name__ == "__main__":
    main()
