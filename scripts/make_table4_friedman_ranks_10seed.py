"""Table 4: Friedman average ranks from 10-seed classification utility gaps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXCEL = (
    ROOT
    / "multi seed generators"
    / "merged 10 seeds"
    / "classification"
    / "results"
    / "multi_seed_results.xlsx"
)
OUT = ROOT / "paper results" / "table4_friedman_ranks_10seed.md"

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
METRICS = ["Accuracy_Gap", "Precision_Gap", "Recall_Gap", "F1_Gap"]


def seed_mean(row: pd.Series) -> float:
    vals = row[SEED_COLS].astype(float).dropna().to_numpy(dtype=float)
    return float(np.mean(vals))


def main() -> None:
    df = pd.read_excel(EXCEL, sheet_name="Utility")
    mats: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        sub = df[df["metric_name"] == metric]
        wide = pd.DataFrame(index=DS_ORDER, columns=GEN_ORDER, dtype=float)
        for _, r in sub.iterrows():
            if r["dataset"] in DS_ORDER and r["generator"] in GEN_ORDER:
                wide.loc[r["dataset"], r["generator"]] = seed_mean(r)
        mats[metric] = wide

    avg_ranks: dict[str, pd.Series] = {}
    rank_mats: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        ranks = mats[metric].rank(axis=1, method="average", ascending=True)
        rank_mats[metric] = ranks
        avg_ranks[metric] = ranks.mean(axis=0)

    # Utility composite: mean of Acc/Prec/Rec/F1 gaps per dataset, then rank.
    # OverallScore = Mean +/- SD of those ranks across 9 datasets.
    composite_gap = sum(mats[m] for m in METRICS) / 4.0
    composite_ranks = composite_gap.rank(axis=1, method="average", ascending=True)
    overall_mean = composite_ranks.mean(axis=0)
    overall_sd = composite_ranks.std(axis=0, ddof=1)

    order = avg_ranks["Accuracy_Gap"].sort_values().index.tolist()

    print("Accuracy mean gaps:")
    print(mats["Accuracy_Gap"].round(4).to_string())
    print("\nAccuracy ranks:")
    print(rank_mats["Accuracy_Gap"].round(2).to_string())
    print("\nAverage Friedman ranks:")
    for m in METRICS:
        print(m)
        print(avg_ranks[m].sort_values().round(4).to_string())
    print("\nOverallScore (composite of 4 gaps -> rank -> mean+/-SD across datasets):")
    print(
        pd.DataFrame({"mean": overall_mean, "sd": overall_sd})
        .sort_values("mean")
        .round(4)
        .to_string()
    )

    lines = [
        "| Generator | Accuracy gap* | Precision gap | Recall gap | F1 gap | OverallScore† |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in order:
        lines.append(
            "| {name} | {acc:.2f} | {prec:.2f} | {rec:.2f} | {f1:.2f} | {om:.2f} ± {osd:.2f} |".format(
                name=GEN_DISP[g],
                acc=avg_ranks["Accuracy_Gap"][g],
                prec=avg_ranks["Precision_Gap"][g],
                rec=avg_ranks["Recall_Gap"][g],
                f1=avg_ranks["F1_Gap"][g],
                om=overall_mean[g],
                osd=overall_sd[g],
            )
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
