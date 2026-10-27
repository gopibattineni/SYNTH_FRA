"""Table 8: Nemenyi pairwise p-values for regression OverallScore (10-seed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parents[1]
EXCEL = (
    ROOT
    / "multi seed generators"
    / "merged 10 seeds"
    / "regression"
    / "results"
    / "multi_seed_results.xlsx"
)
OUT_DIR = ROOT / "multi seed generators" / "merged 10 seeds"
OUT_XLSX = OUT_DIR / "table8_regression_nemenyi_overallscore_10seed.xlsx"
OUT_MD = OUT_DIR / "table8_regression_nemenyi_overallscore_10seed.md"

SEED_COLS = [
    "seed_42",
    "seed_123",
    "seed_2024",
    "seed_68",
    "seed_91",
    "seed_2025",
    "seed_55",
    "seed_155",
    "seed_255",
    "seed_355",
]
DS_ORDER = ["Metro", "OnlineShop", "AirQuality", "Concrete", "Energy", "RealEstate"]
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
ROW_LABEL = {
    "ForestDiffusion": "ForestDiffusion (FD)",
    "TVAE": "TVAE",
    "GaussianCopula": "Gaussian Copula (GC)",
    "WGAN_GP": "WGAN-GP (WGAN)",
    "CTABGAN": "CTAB-GAN+ (CTAB)",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM (DDPM)",
    "CopulaGAN": "Copula GAN (CGAN)",
}
SHORT = {
    "ForestDiffusion": "FD",
    "TVAE": "TVAE",
    "GaussianCopula": "GC",
    "WGAN_GP": "WGAN",
    "CTABGAN": "CTAB",
    "CTGAN": "CTGAN",
    "TabDDPM": "DDPM",
    "CopulaGAN": "CGAN",
}
METRICS = ["R2_Gap", "RMSE_Increase", "MAE_Increase"]


def fmt_p(pval: float) -> str:
    star = "*" if pval < 0.05 else ""
    if pval < 0.0001:
        return f"<0.0001{star}"
    if pval < 0.001:
        return f"{pval:.4f}{star}"
    return f"{pval:.3f}{star}"


def main() -> None:
    df = pd.read_excel(EXCEL, sheet_name="Utility")
    mats: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        sub = df[df["metric_name"] == metric]
        wide = pd.DataFrame(index=DS_ORDER, columns=GEN_ORDER, dtype=float)
        for _, r in sub.iterrows():
            if r["dataset"] in DS_ORDER and r["generator"] in GEN_ORDER:
                vals = r[SEED_COLS].astype(float).dropna().to_numpy(dtype=float)
                wide.loc[r["dataset"], r["generator"]] = float(np.mean(vals))
        mats[metric] = wide

    # Lower gap better. OverallScore composite = mean of 3 oriented gaps.
    composite_gap = sum(mats[m] for m in METRICS) / float(len(METRICS))
    composite_score = -composite_gap  # higher better for tests
    composite_ranks = composite_gap.rank(axis=1, method="average", ascending=True)
    avg_rank = composite_ranks.mean(axis=0).sort_values()
    order = avg_rank.index.tolist()

    arrays = [composite_score[g].to_numpy(dtype=float) for g in order]
    chi2, p = friedmanchisquare(*arrays)
    k = len(order)
    n = len(DS_ORDER)
    df_f = k - 1

    nemenyi = sp.posthoc_nemenyi_friedman(composite_score[order].to_numpy())
    nemenyi = pd.DataFrame(np.asarray(nemenyi), index=order, columns=order)

    shorts = [SHORT[g] for g in order]
    lines = [
        f"Friedman χ² = {chi2:.3f}, df = {df_f}, p = {p:.6g}; N = {n} datasets, k = {k} generators",
        "",
        "| Generator | " + " | ".join(shorts) + " |",
        "| --- | " + " | ".join(["---:"] * k) + " |",
        "| Avg. rank | " + " | ".join(f"{avg_rank[g]:.2f}" for g in order) + " |",
    ]
    for g in order:
        cells = []
        for h in order:
            if g == h:
                cells.append("—")
            else:
                cells.append(fmt_p(float(nemenyi.loc[g, h])))
        lines.append("| " + ROW_LABEL[g] + " | " + " | ".join(cells) + " |")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Display sheet with avg rank row
    disp = pd.DataFrame(index=[ROW_LABEL[g] for g in order], columns=shorts, dtype=object)
    for i, g in enumerate(order):
        for j, h in enumerate(order):
            if g == h:
                disp.iloc[i, j] = "—"
            else:
                disp.iloc[i, j] = fmt_p(float(nemenyi.loc[g, h]))
    avg_row = pd.DataFrame(
        [[round(avg_rank[g], 2) for g in order]],
        columns=shorts,
        index=["Avg. rank"],
    )
    disp_full = pd.concat([avg_row, disp])

    num = nemenyi.copy()
    num.index = [ROW_LABEL[g] for g in order]
    num.columns = shorts

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Source", str(EXCEL)],
                ["OverallScore", "Mean of R2_Gap, RMSE_Increase, MAE_Increase (10-seed means)"],
                ["R2 Gap", "TRTR - TSTR"],
                ["RMSE/MAE Gap", "TSTR - TRTR"],
                ["Rank", "1 = smallest composite gap within dataset"],
                ["Test", "Friedman omnibus + Nemenyi post-hoc on -composite_gap"],
                ["Friedman chi2", f"{chi2:.6f}"],
                ["Friedman df", df_f],
                ["Friedman p", float(p)],
                ["N datasets", n],
                ["k generators", k],
                ["Alpha", 0.05],
                ["Significant pairs (p<0.05)", int((nemenyi.values < 0.05).sum() // 2)],
            ],
            columns=["Item", "Value"],
        ).to_excel(writer, sheet_name="00_Meta", index=False)
        pd.DataFrame(
            [
                {
                    "chi2": chi2,
                    "df": df_f,
                    "p": float(p),
                    "N_datasets": n,
                    "k_generators": k,
                    "significant_omnibus_0.05": bool(p < 0.05),
                }
            ]
        ).to_excel(writer, sheet_name="Friedman_test", index=False)
        disp_full.to_excel(writer, sheet_name="Table8_Nemenyi_display")
        num.to_excel(writer, sheet_name="Table8_Nemenyi_pvalues")
        named = composite_ranks.copy()
        named.columns = [ROW_LABEL[c] for c in named.columns]
        named.index.name = "Dataset"
        named.to_excel(writer, sheet_name="OverallScore_ranks_by_dataset")

    printable = "\n".join(lines).replace("\u03c7", "chi").replace("\u2014", "-").replace("\u00b1", "+/-")
    print(printable)
    print(f"WROTE {OUT_XLSX}")
    print(f"WROTE {OUT_MD}")


if __name__ == "__main__":
    main()
