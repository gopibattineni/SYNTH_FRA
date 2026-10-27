"""Full statistical pipeline: per-dataset Utility Gap ranking, Friedman
test, Nemenyi / Holm post-hoc comparisons, and a critical-difference (CD)
diagram (Dem\u0161ar, 2006).

Pipeline
--------
1. For each dataset, compute each generator's Utility Gap (TRTR - TSTR, F1)
   relative to TRTR.
2. Rank the 8 generators independently within that dataset (rank 1 = best,
   i.e. smallest gap). Ties use the average-rank convention.
3. Repeat for every classification dataset in the benchmark.
4. Compute each generator's average rank across all datasets.
5. Apply the Friedman test to the per-dataset ranks (omnibus test for at
   least one generator differing).
6. Because Friedman is significant, run both post-hoc procedures:
     a. Nemenyi test (all-pairs), used directly for the CD diagram.
     b. Holm-corrected pairwise Wilcoxon signed-rank tests (alternative /
        confirmatory view).
7. Render a critical-difference diagram: generators connected by a
   crossbar cannot be statistically distinguished (Nemenyi, alpha=0.05).

Outputs (Conor/):
    utility_gap_ranks_full.csv          per-dataset ranks (long form)
    utility_gap_average_rank.csv        avg/median/std rank per generator
    friedman_test_result.csv            chi2, df, p-value
    nemenyi_pvalues.csv                 pairwise Nemenyi p-value matrix
    holm_wilcoxon_pvalues.csv           pairwise Holm-adjusted p-value matrix
    critical_difference_diagram.png     CD diagram (Nemenyi, alpha=0.05)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

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

GAP_METRIC = "F1_Gap"   # TRTR - TSTR (F1); lower is better
ALPHA = 0.05

NAVY = "#1f3a5f"


# --------------------------------------------------------------------------
# Steps 1-3: load Utility Gap per (dataset, generator) and rank within
# each dataset.
# --------------------------------------------------------------------------
def load_utility_gaps() -> pd.DataFrame:
    records = json.loads(GAPS_JSON.read_text())
    df = pd.DataFrame(records)
    df = df[
        (df["Dataset"].isin(CLASSIFICATION_DATASETS))
        & (df["Metric"] == GAP_METRIC)
        & (df["TaskType"] == "classification")
    ].copy()
    df = df.rename(columns={"Mean": "UtilityGap"})
    df["Generator"] = df["Generator"].map(lambda g: DISPLAY.get(g, g))
    return df[["Dataset", "Generator", "UtilityGap"]]


def rank_within_datasets(gap_df: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = smallest Utility Gap (best) within each dataset; average-rank ties."""
    ranked = gap_df.copy()
    ranked["Rank"] = ranked.groupby("Dataset")["UtilityGap"].rank(
        method="average", ascending=True
    )
    return ranked


# --------------------------------------------------------------------------
# Step 4: average rank per generator
# --------------------------------------------------------------------------
def summarize_ranks(ranked: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ranked.groupby("Generator")["Rank"]
        .agg(AverageRank="mean", MedianRank="median", StdDevRank=lambda s: s.std(ddof=1))
        .reset_index()
        .sort_values("AverageRank", ascending=True, ignore_index=True)
    )
    return summary


# --------------------------------------------------------------------------
# Step 5: Friedman omnibus test on per-dataset Utility Gap values
# --------------------------------------------------------------------------
def friedman_test(gap_df: pd.DataFrame):
    wide_gap = gap_df.pivot_table(index="Dataset", columns="Generator", values="UtilityGap")
    generators = wide_gap.columns.tolist()
    arrays = [wide_gap[g].to_numpy() for g in generators]
    chi2, p = friedmanchisquare(*arrays)
    k = wide_gap.shape[1]
    n = wide_gap.shape[0]
    return {
        "chi2": float(chi2),
        "df": int(k - 1),
        "p_value": float(p),
        "n_datasets": int(n),
        "n_generators": int(k),
        "significant": bool(p < ALPHA),
    }, wide_gap


# --------------------------------------------------------------------------
# Step 6: post-hoc comparisons (Nemenyi + Holm-corrected Wilcoxon)
# --------------------------------------------------------------------------
def posthoc_tests(wide_gap: pd.DataFrame, order: list[str]):
    # Nemenyi: all-pairs post-hoc for Friedman (Demšar 2006)
    nemenyi = sp.posthoc_nemenyi_friedman(wide_gap[order])
    nemenyi = nemenyi.loc[order, order]

    # Holm-corrected Wilcoxon signed-rank (treatments as rows for array API)
    holm = sp.posthoc_wilcoxon(wide_gap[order].to_numpy().T, p_adjust="holm")
    holm.index = order
    holm.columns = order
    return nemenyi, holm


# --------------------------------------------------------------------------
# Step 7: critical-difference diagram
# --------------------------------------------------------------------------
def render_cd_diagram(summary: pd.DataFrame, nemenyi: pd.DataFrame, friedman_res: dict) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font

    font_name = configure_times_font()
    ranks = summary.set_index("Generator")["AverageRank"]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    # Do not pass color in label_props — scikit_posthocs already sets it.
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
        "Critical-Difference Diagram \u2014 Utility Gap Ranking",
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


def main() -> None:
    # Steps 1-3
    gap_df = load_utility_gaps()
    ranked = rank_within_datasets(gap_df)
    ranked.to_csv(OUT_DIR / "utility_gap_ranks_full.csv", index=False)

    # Step 4
    summary = summarize_ranks(ranked)
    summary_out = summary.rename(columns={
        "AverageRank": "Average Rank", "MedianRank": "Median Rank", "StdDevRank": "Std Dev Rank",
    })
    summary_out.to_csv(OUT_DIR / "utility_gap_average_rank.csv", index=False)
    order = summary["Generator"].tolist()

    # Step 5
    friedman_res, wide_gap = friedman_test(gap_df)
    pd.DataFrame([friedman_res]).to_csv(OUT_DIR / "friedman_test_result.csv", index=False)

    print("=== Average Rank (best \u2192 worst) ===")
    print(summary_out.round(3).to_string(index=False))
    print(
        f"\n=== Friedman test ===\n"
        f"chi2 = {friedman_res['chi2']:.4f}, df = {friedman_res['df']}, "
        f"p = {friedman_res['p_value']:.6g}  ->  "
        f"{'SIGNIFICANT' if friedman_res['significant'] else 'not significant'} at alpha={ALPHA}"
    )

    # Step 6 (only meaningful post-hoc step follows a significant Friedman result)
    nemenyi, holm = posthoc_tests(wide_gap, order)
    nemenyi.to_csv(OUT_DIR / "nemenyi_pvalues.csv")
    holm.to_csv(OUT_DIR / "holm_wilcoxon_pvalues.csv")

    print("\n=== Significant Nemenyi pairs (p < 0.05) ===")
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            p = float(nemenyi.loc[a, b])
            if p < ALPHA:
                print(f"  {a} vs {b}: Nemenyi p = {p:.4g}")

    print("\n=== Significant Holm-Wilcoxon pairs (p < 0.05) ===")
    holm_sig = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            p = float(holm.loc[a, b])
            if p < ALPHA:
                holm_sig.append((a, b, p))
                print(f"  {a} vs {b}: Holm p = {p:.4g}")
    if not holm_sig:
        print("  (none — Holm correction is conservative with N = 9 datasets)")

    # Step 7
    fig = render_cd_diagram(summary, nemenyi, friedman_res)
    png_path = OUT_DIR / "critical_difference_diagram.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"\nSaved: {OUT_DIR / 'utility_gap_ranks_full.csv'}")
    print(f"Saved: {OUT_DIR / 'utility_gap_average_rank.csv'}")
    print(f"Saved: {OUT_DIR / 'friedman_test_result.csv'}")
    print(f"Saved: {OUT_DIR / 'nemenyi_pvalues.csv'}")
    print(f"Saved: {OUT_DIR / 'holm_wilcoxon_pvalues.csv'}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
