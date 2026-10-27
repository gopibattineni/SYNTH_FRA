"""Regression utility-gap analysis for datasets 10–15 (same style as classification).

For each gap metric (default: R2_Gap, RMSE_Gap, MAE_Gap):
  per-dataset ranking → Friedman → Nemenyi / Holm → CD diagram + rank table + Excel.

Outputs (Conor/regression/):
    {prefix}_by_dataset.csv
    {prefix}_ranks_by_dataset.csv
    {prefix}_friedman_average_ranks.csv
    {prefix}_friedman_test.csv
    {prefix}_nemenyi_pvalues.csv / _nemenyi_pairwise.csv
    {prefix}_holm_wilcoxon_pvalues.csv / _holm_pairwise.csv
    {prefix}_critical_difference_diagram.png
    {prefix}_rank_table.csv / _rank_table.png
    {prefix}_analysis.xlsx
    regression_utility_gap_cd_analysis.xlsx   (combined overview)
    regression_gap_by_dataset_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.patches import Rectangle
from scipy.stats import friedmanchisquare

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "regression"
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
from latex_fonts import apply_font_to_figure, configure_times_font

GAPS_JSON = ROOT / "docs" / "data" / "utility_gaps.json"
ALPHA = 0.05
NAVY = "#1f3a5f"
HEADER_BG = "#1f3a5f"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
BEST_BG = "#e6f2ea"
BEST_FG = "#1c5c37"
GRID = "#c5d0dc"
INK = "#1c1f24"

REGRESSION_DATASETS = [
    "10. Metro interstate",
    "11. online shopping",
    "12. Air Quality",
    "13. Concrete Compressive Strength",
    "14. Energy Efficiency",
    "15. Real Estate Valuation",
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
    "10. Metro interstate": "Metro Interstate",
    "11. online shopping": "Online Shopping",
    "12. Air Quality": "Air Quality",
    "13. Concrete Compressive Strength": "Concrete",
    "14. Energy Efficiency": "Energy Efficiency",
    "15. Real Estate Valuation": "Real Estate",
}

METRIC_LABELS = {
    "R2_Gap": "R²",
    "RMSE_Gap": "RMSE",
    "MAE_Gap": "MAE",
    "MSE_Gap": "MSE",
}

DEFAULT_METRICS = ("R2_Gap", "RMSE_Gap", "MAE_Gap")

GAP_DEFINITIONS = {
    "R2_Gap": "R² Drop (TRTR − TSTR); lower gap = better (less utility loss)",
    "RMSE_Gap": "RMSE Increase (same as RMSE_Increase); lower gap = better",
    "MAE_Gap": "MAE Increase (same as MAE_Increase); lower gap = better",
    "MSE_Gap": "MSE Increase (same as MSE_Increase); lower gap = better",
}


def metric_prefix(metric: str) -> str:
    return metric.lower()


def load_gap(metric: str) -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(GAPS_JSON.read_text()))
    df = raw[
        (raw["Dataset"].isin(REGRESSION_DATASETS))
        & (raw["Metric"] == metric)
        & (raw["TaskType"] == "regression")
    ].copy()
    if df.empty:
        raise ValueError(f"No rows for metric={metric!r} in {GAPS_JSON}")
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    df["DatasetShort"] = df["Dataset"].map(SHORT)
    return df[["Dataset", "DatasetShort", "Generator", "UtilityGap"]]


def rank_within_datasets(gap_df: pd.DataFrame) -> pd.DataFrame:
    ranked = gap_df.copy()
    # Lower gap is better for R2_Gap / RMSE_Gap / MAE_Gap.
    ranked["Rank"] = ranked.groupby("Dataset")["UtilityGap"].rank(
        method="average", ascending=True
    )
    return ranked


def summarize_ranks(ranked: pd.DataFrame) -> pd.DataFrame:
    return (
        ranked.groupby("Generator")["Rank"]
        .agg(AverageRank="mean", MedianRank="median", StdDevRank=lambda s: s.std(ddof=1))
        .reset_index()
        .sort_values("AverageRank", ascending=True, ignore_index=True)
    )


def friedman_on_gaps(gap_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    wide_gap = gap_df.pivot_table(index="Dataset", columns="Generator", values="UtilityGap")
    wide_perf = -wide_gap  # higher = better for Friedman / post-hoc
    arrays = [wide_perf[g].to_numpy() for g in wide_perf.columns]
    chi2, p = friedmanchisquare(*arrays)
    k = wide_perf.shape[1]
    n = wide_perf.shape[0]
    return {
        "chi2": float(chi2),
        "df": int(k - 1),
        "p_value": float(p),
        "n_datasets": int(n),
        "n_generators": int(k),
        "significant": bool(p < ALPHA),
    }, wide_perf


def posthoc_tests(wide_perf: pd.DataFrame, order: list[str]):
    nemenyi = sp.posthoc_nemenyi_friedman(wide_perf[order])
    nemenyi = nemenyi.loc[order, order]
    holm = sp.posthoc_wilcoxon(wide_perf[order].to_numpy().T, p_adjust="holm")
    holm.index = order
    holm.columns = order
    return nemenyi, holm


def pairwise_long(matrix: pd.DataFrame, avg_ranks: pd.Series) -> pd.DataFrame:
    rows = []
    for a, b in combinations(avg_ranks.index.tolist(), 2):
        rows.append({
            "Generator_A": a,
            "Generator_B": b,
            "AvgRank_A": float(avg_ranks[a]),
            "AvgRank_B": float(avg_ranks[b]),
            "p_value": float(matrix.loc[a, b]),
            "Significant_alpha_0.05": bool(float(matrix.loc[a, b]) < ALPHA),
        })
    return pd.DataFrame(rows).sort_values("p_value", ascending=True, ignore_index=True)


def render_cd_diagram(
    summary: pd.DataFrame,
    nemenyi: pd.DataFrame,
    friedman_res: dict,
    metric_label: str,
) -> plt.Figure:
    font_name = configure_times_font()
    ranks = summary.set_index("Generator")["AverageRank"]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    sp.critical_difference_diagram(
        ranks=ranks,
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
        f"Critical-Difference Diagram \u2014 {metric_label} Utility Gap Ranking (Regression)",
        fontsize=13.5, fontweight="bold", color=NAVY, pad=36,
        fontfamily="serif", fontname=font_name,
    )
    p_str = "< 0.0001" if friedman_res["p_value"] < 0.0001 else f"{friedman_res['p_value']:.4f}"
    fig.text(
        0.5, 0.965,
        f"Friedman: $\\chi^2$ = {friedman_res['chi2']:.2f}, df = {friedman_res['df']}, "
        f"p = {p_str}   |   Nemenyi post-hoc, \u03b1 = {ALPHA}   |   "
        f"N = {friedman_res['n_datasets']} regression datasets   |   "
        "generators joined by a red bar are not significantly different",
        ha="center", va="top", fontsize=8.5, color="#5a5f66",
        fontfamily="serif", fontname=font_name,
    )

    apply_font_to_figure(fig, font_name)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def render_rank_table(summary: pd.DataFrame, metric_label: str) -> plt.Figure:
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
        f"Generator Ranking by {metric_label} Utility Gap (Regression)",
        ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY,
    )
    ax.text(
        total_w / 2, fig_h - 0.58,
        f"Ranked per dataset by {metric_label} Gap, averaged across "
        f"{len(REGRESSION_DATASETS)} regression datasets (10\u201315)",
        ha="center", va="center", fontsize=8.6, color="#5a5f66", style="italic",
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
        f"Rank 1 = smallest {metric_label} Utility Gap (best) within each dataset.",
        ha="left", va="top", fontsize=7.8, color="#5a5f66",
    )

    apply_font_to_figure(fig, font_name)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def render_gap_heatmap(gap_piv: pd.DataFrame, metric_label: str, prefix: str) -> Path:
    font_name = configure_times_font()
    # Order generators by mean gap (best left)
    order = gap_piv.mean(axis=0).sort_values().index.tolist()
    mat = gap_piv[order]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    data = mat.to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(data), 90)
    vmax = max(vmax, 1e-6)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(
        f"{metric_label} Utility Gap by Dataset (Regression)",
        fontsize=13.5, fontweight="bold", color=NAVY, pad=12,
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color="#111",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"{metric_label} Gap (lower better)", fontsize=9)
    apply_font_to_figure(fig, font_name)
    fig.tight_layout()
    path = OUT_DIR / f"{prefix}_by_dataset_heatmap.png"
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def _readme_sheet(metric: str, label: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"Item": "Analysis", "Description": f"{label} Utility Gap CD analysis (regression)"},
        {"Item": "Metric", "Description": GAP_DEFINITIONS.get(metric, metric)},
        {"Item": "Datasets", "Description": "6 regression datasets (IDs 10–15)"},
        {"Item": "Generators", "Description": "8 synthetic data generators"},
        {"Item": "Ranking rule", "Description": "Rank 1 = smallest utility gap within each dataset"},
        {"Item": "Omnibus test", "Description": f"Friedman test (α = {ALPHA})"},
        {"Item": "Post-hoc", "Description": "Nemenyi (CD diagram) + Holm-corrected Wilcoxon"},
        {"Item": "Sheet: 01_Gap_by_Dataset", "Description": f"{label} gap for each generator × dataset"},
        {"Item": "Sheet: 02_Ranks_by_Dataset", "Description": "Per-dataset ranks (long form)"},
        {"Item": "Sheet: 03_Average_Ranks", "Description": "Average / median / std rank (best → worst)"},
        {"Item": "Sheet: 04_Friedman_Test", "Description": "Friedman χ², df, p-value"},
        {"Item": "Sheet: 05_Nemenyi_Pvalues", "Description": "Nemenyi p-value matrix"},
        {"Item": "Sheet: 06_Nemenyi_Pairwise", "Description": "All Nemenyi pairs + significance"},
        {"Item": "Sheet: 07_Nemenyi_Significant", "Description": "Significant Nemenyi pairs only"},
        {"Item": "Sheet: 08_Holm_Pvalues", "Description": "Holm-Wilcoxon p-value matrix"},
        {"Item": "Sheet: 09_Holm_Pairwise", "Description": "All Holm pairs + significance"},
        {"Item": "Sheet: 10_Holm_Significant", "Description": "Significant Holm pairs only"},
        {"Item": "Figures", "Description": (
            f"{metric_prefix(metric)}_critical_difference_diagram.png, "
            f"{metric_prefix(metric)}_rank_table.png, "
            f"{metric_prefix(metric)}_by_dataset_heatmap.png"
        )},
    ])


def write_analysis_excel(
    *,
    metric: str,
    label: str,
    prefix: str,
    gap_piv: pd.DataFrame,
    ranked: pd.DataFrame,
    summary: pd.DataFrame,
    friedman_row: pd.DataFrame,
    nemenyi: pd.DataFrame,
    holm: pd.DataFrame,
    nemenyi_pairs: pd.DataFrame,
    holm_pairs: pd.DataFrame,
) -> Path:
    xlsx_path = OUT_DIR / f"{prefix}_analysis.xlsx"
    avg = summary.rename(columns={
        "AverageRank": "Average Rank",
        "MedianRank": "Median Rank",
        "StdDevRank": "Std Dev Rank",
    })[["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]]
    nemenyi_sig = nemenyi_pairs[nemenyi_pairs["Significant_alpha_0.05"]].copy()
    holm_sig = holm_pairs[holm_pairs["Significant_alpha_0.05"]].copy()

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        _readme_sheet(metric, label).to_excel(writer, sheet_name="00_README", index=False)
        gap_piv.round(6).to_excel(writer, sheet_name="01_Gap_by_Dataset")
        ranked.sort_values(["Dataset", "Rank"]).to_excel(
            writer, sheet_name="02_Ranks_by_Dataset", index=False
        )
        avg.round(6).to_excel(writer, sheet_name="03_Average_Ranks", index=False)
        friedman_row.to_excel(writer, sheet_name="04_Friedman_Test", index=False)
        nemenyi.round(6).to_excel(writer, sheet_name="05_Nemenyi_Pvalues")
        nemenyi_pairs.to_excel(writer, sheet_name="06_Nemenyi_Pairwise", index=False)
        nemenyi_sig.to_excel(writer, sheet_name="07_Nemenyi_Significant", index=False)
        holm.round(6).to_excel(writer, sheet_name="08_Holm_Pvalues")
        holm_pairs.to_excel(writer, sheet_name="09_Holm_Pairwise", index=False)
        holm_sig.to_excel(writer, sheet_name="10_Holm_Significant", index=False)
    return xlsx_path


def write_combined_excel(metric_summaries: list[dict]) -> Path:
    xlsx_path = OUT_DIR / "regression_utility_gap_cd_analysis.xlsx"
    overview_rows = []
    rank_frames = []
    for item in metric_summaries:
        fr = item["friedman"].iloc[0].to_dict()
        overview_rows.append({
            "Metric": item["label"],
            "Gap_Metric": item["metric"],
            "Friedman_chi2": fr["Statistic_chi2"],
            "df": fr["DegreesOfFreedom"],
            "p_value": fr["p_value"],
            "Significant": fr["Significant"],
            "Nemenyi_sig_pairs": fr["Nemenyi_sig_pairs"],
            "Holm_sig_pairs": fr["Holm_sig_pairs"],
            "Best_Generator": item["summary"].iloc[0]["Generator"],
            "Best_Avg_Rank": item["summary"].iloc[0]["AverageRank"],
            "Worst_Generator": item["summary"].iloc[-1]["Generator"],
            "Worst_Avg_Rank": item["summary"].iloc[-1]["AverageRank"],
            "Excel_Workbook": f"{item['prefix']}_analysis.xlsx",
            "CD_Figure": f"{item['prefix']}_critical_difference_diagram.png",
            "Rank_Table": f"{item['prefix']}_rank_table.png",
        })
        tmp = item["summary"][["Generator", "AverageRank"]].rename(
            columns={"AverageRank": item["label"]}
        )
        rank_frames.append(tmp.set_index("Generator"))

    overview = pd.DataFrame(overview_rows)
    ranks_wide = pd.concat(rank_frames, axis=1).reset_index()
    ranks_wide["_mean"] = ranks_wide.drop(columns=["Generator"]).mean(axis=1)
    ranks_wide = ranks_wide.sort_values("_mean").drop(columns="_mean").reset_index(drop=True)

    dataset_list = pd.DataFrame([
        {"Dataset_ID": d.split(".", 1)[0].strip(), "Dataset": d, "Short_Name": SHORT[d]}
        for d in REGRESSION_DATASETS
    ])
    readme = pd.DataFrame([
        {"Item": "Purpose", "Description": "Combined CD analysis for R² / RMSE / MAE utility gaps (regression)"},
        {"Item": "Datasets", "Description": "Benchmark datasets 10–15 (6 regression tasks)"},
        {"Item": "Gap definition", "Description": "Lower utility gap is better"},
        {"Item": "Sheet: Overview", "Description": "Friedman / post-hoc summary per metric"},
        {"Item": "Sheet: Average_Ranks_Compare", "Description": "Average ranks side-by-side"},
        {"Item": "Sheet: Datasets", "Description": "Regression dataset list"},
    ])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="00_README", index=False)
        overview.to_excel(writer, sheet_name="Overview", index=False)
        ranks_wide.round(4).to_excel(writer, sheet_name="Average_Ranks_Compare", index=False)
        dataset_list.to_excel(writer, sheet_name="Datasets", index=False)
        for item in metric_summaries:
            sheet = f"{item['label']}_Nemenyi_Sig"[:31]
            sig = item["nemenyi_pairs"][item["nemenyi_pairs"]["Significant_alpha_0.05"]]
            sig.to_excel(writer, sheet_name=sheet, index=False)
    return xlsx_path


def run_metric(metric: str) -> dict:
    label = METRIC_LABELS.get(metric, metric.replace("_Gap", ""))
    prefix = metric_prefix(metric)
    print(f"\n======== {label} Utility Gap ({metric}) — Regression ========")

    gap_df = load_gap(metric)
    ranked = rank_within_datasets(gap_df)
    summary = summarize_ranks(ranked)
    order = summary["Generator"].tolist()
    avg_ranks = summary.set_index("Generator")["AverageRank"]

    gap_piv = gap_df.pivot_table(index="DatasetShort", columns="Generator", values="UtilityGap")
    gap_piv = gap_piv.reindex([SHORT[d] for d in REGRESSION_DATASETS])
    gap_piv.round(6).to_csv(OUT_DIR / f"{prefix}_by_dataset.csv")

    ranked.sort_values(["Dataset", "Rank"]).to_csv(
        OUT_DIR / f"{prefix}_ranks_by_dataset.csv", index=False
    )
    avg_ranks.rename("Average Rank").reset_index().to_csv(
        OUT_DIR / f"{prefix}_friedman_average_ranks.csv", index=False
    )

    summary_out = summary.rename(columns={
        "AverageRank": "Average Rank",
        "MedianRank": "Median Rank",
        "StdDevRank": "Std Dev Rank",
    })[["Generator", "Average Rank", "Median Rank", "Std Dev Rank"]]
    summary_out.to_csv(OUT_DIR / f"{prefix}_rank_table.csv", index=False)

    friedman_res, wide_perf = friedman_on_gaps(gap_df)
    nemenyi, holm = posthoc_tests(wide_perf, order)
    nemenyi_pairs = pairwise_long(nemenyi, avg_ranks)
    holm_pairs = pairwise_long(holm, avg_ranks)
    nemenyi_sig = int(nemenyi_pairs["Significant_alpha_0.05"].sum())
    holm_sig = int(holm_pairs["Significant_alpha_0.05"].sum())

    friedman_row = pd.DataFrame([{
        "Test": "Friedman",
        "Metric": metric,
        "TaskType": "regression",
        "Statistic_chi2": friedman_res["chi2"],
        "DegreesOfFreedom": friedman_res["df"],
        "p_value": friedman_res["p_value"],
        "N_Datasets": friedman_res["n_datasets"],
        "N_Generators": friedman_res["n_generators"],
        "Alpha": ALPHA,
        "Significant": friedman_res["significant"],
        "Posthoc": "Nemenyi + Holm-Wilcoxon",
        "Nemenyi_sig_pairs": nemenyi_sig,
        "Holm_sig_pairs": holm_sig,
    }])
    friedman_row.to_csv(OUT_DIR / f"{prefix}_friedman_test.csv", index=False)
    nemenyi.to_csv(OUT_DIR / f"{prefix}_nemenyi_pvalues.csv")
    holm.to_csv(OUT_DIR / f"{prefix}_holm_wilcoxon_pvalues.csv")
    nemenyi_pairs.to_csv(OUT_DIR / f"{prefix}_nemenyi_pairwise.csv", index=False)
    holm_pairs.to_csv(OUT_DIR / f"{prefix}_holm_pairwise.csv", index=False)

    xlsx_path = write_analysis_excel(
        metric=metric,
        label=label,
        prefix=prefix,
        gap_piv=gap_piv,
        ranked=ranked,
        summary=summary,
        friedman_row=friedman_row,
        nemenyi=nemenyi,
        holm=holm,
        nemenyi_pairs=nemenyi_pairs,
        holm_pairs=holm_pairs,
    )

    print(summary_out.round(3).to_string(index=False))
    print(
        f"\nFriedman: chi2={friedman_res['chi2']:.4f}, df={friedman_res['df']}, "
        f"p={friedman_res['p_value']:.6g} → "
        f"{'SIGNIFICANT' if friedman_res['significant'] else 'not significant'} "
        f"(α={ALPHA}, N={friedman_res['n_datasets']})"
    )
    print(f"Nemenyi significant pairs: {nemenyi_sig}")
    print(f"Holm-Wilcoxon significant pairs: {holm_sig}")

    fig_cd = render_cd_diagram(summary, nemenyi, friedman_res, label)
    cd_path = OUT_DIR / f"{prefix}_critical_difference_diagram.png"
    fig_cd.savefig(cd_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig_cd)

    fig_tbl = render_rank_table(summary, label)
    tbl_path = OUT_DIR / f"{prefix}_rank_table.png"
    fig_tbl.savefig(tbl_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.22)
    plt.close(fig_tbl)

    heat_path = render_gap_heatmap(gap_piv, label, prefix)

    print(f"Saved: {cd_path}")
    print(f"Saved: {tbl_path}")
    print(f"Saved: {heat_path}")
    print(f"Saved: {xlsx_path}")

    return {
        "metric": metric,
        "label": label,
        "prefix": prefix,
        "summary": summary,
        "friedman": friedman_row,
        "nemenyi_pairs": nemenyi_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Gap metrics (default: R2_Gap RMSE_Gap MAE_Gap)",
    )
    args = parser.parse_args()
    summaries = [run_metric(m) for m in args.metrics]
    combined = write_combined_excel(summaries)
    print(f"\nSaved combined workbook: {combined}")
    print(f"All regression outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
