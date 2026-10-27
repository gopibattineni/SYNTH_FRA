"""Critical-difference diagrams for classification utility gaps.

Same pipeline used for Accuracy Utility Gap:
  per-dataset ranking → Friedman → Nemenyi / Holm-Wilcoxon → CD diagram.

Default metrics: Accuracy_Gap, Precision_Gap, Recall_Gap, F1_Gap (TRTR − TSTR).

Outputs per metric prefix (e.g. precision_gap_ / recall_gap_):
    {prefix}_by_dataset.csv
    {prefix}_ranks_by_dataset.csv
    {prefix}_friedman_average_ranks.csv
    {prefix}_friedman_test.csv
    {prefix}_nemenyi_pvalues.csv
    {prefix}_nemenyi_pairwise.csv
    {prefix}_holm_wilcoxon_pvalues.csv
    {prefix}_holm_pairwise.csv
    {prefix}_critical_difference_diagram.png
    {prefix}_critical_difference_diagram.svg
    {prefix}_analysis.xlsx               clear multi-sheet Excel workbook
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
OUT_DIR.mkdir(parents=True, exist_ok=True)
from latex_fonts import apply_font_to_figure, configure_times_font

GAPS_JSON = ROOT / "docs" / "data" / "utility_gaps.json"
ALPHA = 0.05
NAVY = "#1f3a5f"

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

SHORT = {
    "1. Cancer": "Cancer",
    "2. Alzhimers": "Alzheimer's",
    "3. Adult": "Adult",
    "4. Forest cover dataset": "Forest Cover",
    "5. Bank Markting": "Bank Marketing",
    "6. Wine dataset": "Wine Quality",
    "7. CDC diabetes dataset": "CDC Diabetes",
    "8. Mushroom dataset": "Mushroom",
    "9. MAGIC Gamma Telescope": "MAGIC Gamma",
}

METRIC_LABELS = {
    "Accuracy_Gap": "Accuracy",
    "Precision_Gap": "Precision",
    "Recall_Gap": "Recall",
    "F1_Gap": "F1",
}

DEFAULT_METRICS = ("Accuracy_Gap", "Precision_Gap", "Recall_Gap", "F1_Gap")


def metric_prefix(metric: str) -> str:
    return metric.lower()


def load_gap(metric: str) -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(GAPS_JSON.read_text()))
    df = raw[
        (raw["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (raw["Metric"] == metric)
        & (raw["TaskType"] == "classification")
    ].copy()
    if df.empty:
        raise ValueError(f"No rows for metric={metric!r} in {GAPS_JSON}")
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    df["DatasetShort"] = df["Dataset"].map(SHORT)
    return df[["Dataset", "DatasetShort", "Generator", "UtilityGap"]]


def rank_within_datasets(gap_df: pd.DataFrame) -> pd.DataFrame:
    ranked = gap_df.copy()
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
    # Negate so higher = better for Friedman / post-hoc conventions.
    wide_gap = gap_df.pivot_table(index="Dataset", columns="Generator", values="UtilityGap")
    wide_perf = -wide_gap
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
    out = pd.DataFrame(rows).sort_values("p_value", ascending=True, ignore_index=True)
    return out


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
        f"Critical-Difference Diagram \u2014 {metric_label} Utility Gap Ranking",
        fontsize=14.5, fontweight="bold", color=NAVY, pad=36,
        fontfamily="serif", fontname=font_name,
    )
    p_str = "< 0.0001" if friedman_res["p_value"] < 0.0001 else f"{friedman_res['p_value']:.4f}"
    fig.text(
        0.5, 0.965,
        f"Friedman: $\\chi^2$ = {friedman_res['chi2']:.2f}, df = {friedman_res['df']}, "
        f"p = {p_str}   |   Nemenyi post-hoc, \u03b1 = {ALPHA}   |   "
        "generators joined by a red bar are not significantly different",
        ha="center", va="top", fontsize=9, color="#5a5f66",
        fontfamily="serif", fontname=font_name,
    )

    apply_font_to_figure(fig, font_name)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    print(f"CD diagram font: {font_name}")
    return fig


def _readme_sheet(metric: str, label: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"Item": "Analysis", "Description": f"{label} Utility Gap critical-difference analysis"},
        {"Item": "Metric", "Description": f"{metric} = TRTR − TSTR ({label}); lower gap is better"},
        {"Item": "Datasets", "Description": "9 classification datasets"},
        {"Item": "Generators", "Description": "8 synthetic data generators"},
        {"Item": "Ranking rule", "Description": "Rank 1 = smallest utility gap within each dataset (average-rank ties)"},
        {"Item": "Omnibus test", "Description": f"Friedman test on gaps across datasets (α = {ALPHA})"},
        {"Item": "Post-hoc", "Description": "Nemenyi (CD diagram) + Holm-corrected Wilcoxon signed-rank"},
        {"Item": "Sheet: 01_Gap_by_Dataset", "Description": f"{label} gap (TRTR−TSTR) for each generator × dataset"},
        {"Item": "Sheet: 02_Ranks_by_Dataset", "Description": "Per-dataset ranks (long form)"},
        {"Item": "Sheet: 03_Average_Ranks", "Description": "Average / median / std rank across datasets (best → worst)"},
        {"Item": "Sheet: 04_Friedman_Test", "Description": "Friedman χ², df, p-value, significance"},
        {"Item": "Sheet: 05_Nemenyi_Pvalues", "Description": "All-pairs Nemenyi p-value matrix"},
        {"Item": "Sheet: 06_Nemenyi_Pairwise", "Description": "Nemenyi pairs sorted by p-value, with Significant flag"},
        {"Item": "Sheet: 07_Nemenyi_Significant", "Description": "Only Nemenyi pairs significant at α = 0.05 (clear takeaway)"},
        {"Item": "Sheet: 08_Holm_Pvalues", "Description": "Holm-Wilcoxon p-value matrix"},
        {"Item": "Sheet: 09_Holm_Pairwise", "Description": "Holm-Wilcoxon pairs sorted by p-value, with Significant flag"},
        {"Item": "Sheet: 10_Holm_Significant", "Description": "Only Holm pairs significant at α = 0.05"},
        {"Item": "Figure", "Description": f"{metric_prefix(metric)}_critical_difference_diagram.png/.svg"},
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
    """One workbook comparing Accuracy / Precision / Recall side by side."""
    xlsx_path = OUT_DIR / "classification_utility_gap_cd_analysis.xlsx"
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
            "CD_Figure": f"{item['prefix']}_critical_difference_diagram.png/.svg",
        })
        tmp = item["summary"][["Generator", "AverageRank"]].rename(
            columns={"AverageRank": item["label"]}
        )
        rank_frames.append(tmp.set_index("Generator"))

    overview = pd.DataFrame(overview_rows)
    ranks_wide = pd.concat(rank_frames, axis=1).reset_index()
    # Order rows by mean rank across metrics (best first)
    ranks_wide["_mean"] = ranks_wide.drop(columns=["Generator"]).mean(axis=1)
    ranks_wide = ranks_wide.sort_values("_mean").drop(columns="_mean").reset_index(drop=True)

    readme = pd.DataFrame([
        {"Item": "Purpose", "Description": "Combined CD analysis for Accuracy / Precision / Recall utility gaps"},
        {"Item": "Gap definition", "Description": "Utility Gap = TRTR − TSTR; lower is better"},
        {"Item": "Sheet: Overview", "Description": "Friedman / post-hoc summary for each metric"},
        {"Item": "Sheet: Average_Ranks_Compare", "Description": "Average ranks side-by-side across metrics"},
        {"Item": "Per-metric workbooks", "Description": "See precision_gap_analysis.xlsx and recall_gap_analysis.xlsx (and accuracy_gap_analysis.xlsx)"},
    ])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="00_README", index=False)
        overview.to_excel(writer, sheet_name="Overview", index=False)
        ranks_wide.round(4).to_excel(writer, sheet_name="Average_Ranks_Compare", index=False)
        for item in metric_summaries:
            sheet = f"{item['label']}_Nemenyi_Sig"[:31]
            sig = item["nemenyi_pairs"][item["nemenyi_pairs"]["Significant_alpha_0.05"]]
            sig.to_excel(writer, sheet_name=sheet, index=False)
    return xlsx_path


def run_metric(metric: str) -> dict:
    label = METRIC_LABELS.get(metric, metric.replace("_Gap", ""))
    prefix = metric_prefix(metric)
    print(f"\n======== {label} Utility Gap ({metric}) ========")

    gap_df = load_gap(metric)
    ranked = rank_within_datasets(gap_df)
    summary = summarize_ranks(ranked)
    order = summary["Generator"].tolist()
    avg_ranks = summary.set_index("Generator")["AverageRank"]

    gap_piv = gap_df.pivot_table(index="DatasetShort", columns="Generator", values="UtilityGap")
    gap_piv = gap_piv.reindex([SHORT[d] for d in CLASSIFICATION_DATASETS])
    gap_piv.round(6).to_csv(OUT_DIR / f"{prefix}_by_dataset.csv")

    ranked.sort_values(["Dataset", "Rank"]).to_csv(
        OUT_DIR / f"{prefix}_ranks_by_dataset.csv", index=False
    )
    avg_ranks.rename("Average Rank").reset_index().to_csv(
        OUT_DIR / f"{prefix}_friedman_average_ranks.csv", index=False
    )

    friedman_res, wide_perf = friedman_on_gaps(gap_df)
    nemenyi, holm = posthoc_tests(wide_perf, order)

    nemenyi_pairs = pairwise_long(nemenyi, avg_ranks)
    holm_pairs = pairwise_long(holm, avg_ranks)
    nemenyi_sig = int(nemenyi_pairs["Significant_alpha_0.05"].sum())
    holm_sig = int(holm_pairs["Significant_alpha_0.05"].sum())
    friedman_row = pd.DataFrame([{
        "Test": "Friedman",
        "Metric": metric,
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

    print("Average ranks (best → worst):")
    print(summary.rename(columns={
        "AverageRank": "Average Rank",
        "MedianRank": "Median Rank",
        "StdDevRank": "Std Dev Rank",
    }).round(3).to_string(index=False))
    print(
        f"\nFriedman: chi2={friedman_res['chi2']:.4f}, df={friedman_res['df']}, "
        f"p={friedman_res['p_value']:.6g} → "
        f"{'SIGNIFICANT' if friedman_res['significant'] else 'not significant'} "
        f"(α={ALPHA})"
    )
    print(f"Nemenyi significant pairs: {nemenyi_sig}")
    print(f"Holm-Wilcoxon significant pairs: {holm_sig}")

    fig = render_cd_diagram(summary, nemenyi, friedman_res, label)
    png_path = OUT_DIR / f"{prefix}_critical_difference_diagram.png"
    svg_path = OUT_DIR / f"{prefix}_critical_difference_diagram.svg"
    prev_fonttype = mpl.rcParams["svg.fonttype"]
    mpl.rcParams["svg.fonttype"] = "path"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    mpl.rcParams["svg.fonttype"] = prev_fonttype
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
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
        help="Gap metrics to process (default: Accuracy Precision Recall F1)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write classification_utility_gap_cd_analysis.xlsx comparing metrics",
    )
    args = parser.parse_args()
    summaries = [run_metric(metric) for metric in args.metrics]
    if args.combined or len(summaries) > 1:
        combined = write_combined_excel(summaries)
        print(f"\nSaved combined workbook: {combined}")


if __name__ == "__main__":
    main()
