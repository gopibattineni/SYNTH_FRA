"""Table 4 analogue for regression: Friedman average ranks from 10-seed gaps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
OUT_XLSX = OUT_DIR / "table4_regression_friedman_ranks_10seed.xlsx"
OUT_MD = OUT_DIR / "table4_regression_friedman_ranks_10seed.md"

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
GEN_DISP = {
    "ForestDiffusion": "Forest Diffusion",
    "TVAE": "TVAE",
    "GaussianCopula": "Gaussian Copula",
    "WGAN_GP": "WGAN-GP",
    "CTABGAN": "CTAB-GAN+",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
    "CopulaGAN": "Copula GAN",
}
METRICS = ["R2_Gap", "RMSE_Increase", "MAE_Increase"]


def main() -> None:
    df = pd.read_excel(EXCEL, sheet_name="Utility")
    mats: dict[str, pd.DataFrame] = {}
    rank_mats: dict[str, pd.DataFrame] = {}
    avg_ranks: dict[str, pd.Series] = {}

    for metric in METRICS:
        sub = df[df["metric_name"] == metric]
        wide = pd.DataFrame(index=DS_ORDER, columns=GEN_ORDER, dtype=float)
        for _, r in sub.iterrows():
            if r["dataset"] in DS_ORDER and r["generator"] in GEN_ORDER:
                vals = r[SEED_COLS].astype(float).dropna().to_numpy(dtype=float)
                wide.loc[r["dataset"], r["generator"]] = float(np.mean(vals))
        mats[metric] = wide
        ranks = wide.rank(axis=1, method="average", ascending=True)
        rank_mats[metric] = ranks
        avg_ranks[metric] = ranks.mean(axis=0)

    composite_gap = sum(mats[m] for m in METRICS) / float(len(METRICS))
    composite_ranks = composite_gap.rank(axis=1, method="average", ascending=True)
    overall_mean = composite_ranks.mean(axis=0)
    overall_sd = composite_ranks.std(axis=0, ddof=1)

    order = avg_ranks["R2_Gap"].sort_values().index.tolist()

    rows = []
    lines = [
        "| Generator | R2 gap* | RMSE gap | MAE gap | OverallScore† |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for g in order:
        rows.append(
            {
                "Generator": GEN_DISP[g],
                "R2 gap*": round(avg_ranks["R2_Gap"][g], 2),
                "RMSE gap": round(avg_ranks["RMSE_Increase"][g], 2),
                "MAE gap": round(avg_ranks["MAE_Increase"][g], 2),
                "OverallScore mean": round(overall_mean[g], 2),
                "OverallScore SD": round(overall_sd[g], 2),
                "OverallScore": f"{overall_mean[g]:.2f} ± {overall_sd[g]:.2f}",
                "R2_full": avg_ranks["R2_Gap"][g],
                "RMSE_full": avg_ranks["RMSE_Increase"][g],
                "MAE_full": avg_ranks["MAE_Increase"][g],
                "Overall_mean_full": overall_mean[g],
                "Overall_sd_full": overall_sd[g],
            }
        )
        lines.append(
            "| {name} | {r2:.2f} | {rmse:.2f} | {mae:.2f} | {om:.2f} ± {osd:.2f} |".format(
                name=GEN_DISP[g],
                r2=avg_ranks["R2_Gap"][g],
                rmse=avg_ranks["RMSE_Increase"][g],
                mae=avg_ranks["MAE_Increase"][g],
                om=overall_mean[g],
                osd=overall_sd[g],
            )
        )

    t4 = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def named(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        out.columns = [GEN_DISP[c] for c in out.columns]
        out.index.name = "Dataset"
        return out

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Source", str(EXCEL)],
                ["N datasets", 6],
                ["k generators", 8],
                ["Seeds", "10 generator-training seeds; cell = mean gap over seeds"],
                ["R2 Gap", "TRTR - TSTR (lower better)"],
                ["RMSE/MAE Gap", "TSTR - TRTR / Increase (lower better)"],
                ["Rank", "1 = smallest gap within dataset"],
                ["Metric columns", "Average rank across 6 regression datasets"],
                [
                    "OverallScore",
                    "Mean of R2/RMSE/MAE gaps per dataset -> rank -> mean +/- SD across datasets",
                ],
                ["Sort", "Ascending R2 gap* average rank (primary)"],
            ],
            columns=["Item", "Value"],
        ).to_excel(writer, sheet_name="00_Meta", index=False)
        t4[
            [
                "Generator",
                "R2 gap*",
                "RMSE gap",
                "MAE gap",
                "OverallScore mean",
                "OverallScore SD",
                "OverallScore",
            ]
        ].to_excel(writer, sheet_name="Table4_Friedman_ranks", index=False)
        t4[
            [
                "Generator",
                "R2_full",
                "RMSE_full",
                "MAE_full",
                "Overall_mean_full",
                "Overall_sd_full",
            ]
        ].to_excel(writer, sheet_name="Table4_full_precision", index=False)
        named(rank_mats["R2_Gap"]).to_excel(writer, sheet_name="Ranks_R2_gap")
        named(rank_mats["RMSE_Increase"]).to_excel(writer, sheet_name="Ranks_RMSE_gap")
        named(rank_mats["MAE_Increase"]).to_excel(writer, sheet_name="Ranks_MAE_gap")
        named(composite_ranks).to_excel(
            writer, sheet_name="OverallScore_ranks_by_dataset"
        )

    print("\n".join(lines))
    print(f"WROTE {OUT_XLSX}")
    print(f"WROTE {OUT_MD}")


if __name__ == "__main__":
    main()
