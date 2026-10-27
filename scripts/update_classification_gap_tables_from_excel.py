"""Rebuild classification utility-gap tables from merged 10-seed Excel."""
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
AGREED_DIR = ROOT / "Agreed analysis" / "classification"
PAPER_DIR = ROOT / "paper results"
MULTI_DIR = (
    ROOT
    / "multi seed generators"
    / "analysis"
    / "tables"
    / "main"
    / "classification_gaps"
)

GEN_CSV = [
    "CTABGAN",
    "CTGAN",
    "CopulaGAN",
    "ForestDiffusion",
    "GaussianCopula",
    "TVAE",
    "TabDDPM",
    "WGAN-GP",
]
GEN_EXCEL_TO_CSV = {
    "CTABGAN": "CTABGAN",
    "CTGAN": "CTGAN",
    "CopulaGAN": "CopulaGAN",
    "ForestDiffusion": "ForestDiffusion",
    "GaussianCopula": "GaussianCopula",
    "TVAE": "TVAE",
    "TabDDPM": "TabDDPM",
    "WGAN_GP": "WGAN-GP",
    "WGAN-GP": "WGAN-GP",
}
DS_CSV_ORDER = [
    "Cancer",
    "Alzheimer's",
    "Adult",
    "Forest Cover",
    "Bank Marketing",
    "Wine Quality",
    "CDC Diabetes",
    "Mushroom",
    "MAGIC Gamma",
]
SHORT = {
    "Cancer": "Cancer",
    "Alzheimer's": "Alzh.",
    "Adult": "Adult",
    "Forest Cover": "Forest",
    "Bank Marketing": "Bank",
    "Wine Quality": "Wine",
    "CDC Diabetes": "CDC",
    "Mushroom": "Mush.",
    "MAGIC Gamma": "MAGIC",
}
DISP = {
    "ForestDiffusion": "Forest Diffusion",
    "TVAE": "TVAE",
    "GaussianCopula": "Gaussian Copula",
    "WGAN-GP": "WGAN-GP",
    "CTABGAN": "CTAB-GAN+",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
    "CopulaGAN": "Copula GAN",
}
METRIC_FILES = {
    "Accuracy_Gap": "accuracy_gap_by_dataset.csv",
    "Precision_Gap": "precision_gap_by_dataset.csv",
    "Recall_Gap": "recall_gap_by_dataset.csv",
    "F1_Gap": "f1_gap_by_dataset.csv",
}
METRIC_TITLES = {
    "Accuracy_Gap": "Accuracy Utility Gap",
    "Precision_Gap": "Precision Utility Gap",
    "Recall_Gap": "Recall Utility Gap",
    "F1_Gap": "F1-score Utility Gap",
}


def normalize_dataset(raw: str) -> str:
    low = str(raw).strip().lower()
    if "alzh" in low:
        return "Alzheimer's"
    if "cancer" in low:
        return "Cancer"
    if "adult" in low:
        return "Adult"
    if "forest" in low:
        return "Forest Cover"
    if "bank" in low:
        return "Bank Marketing"
    if "wine" in low:
        return "Wine Quality"
    if "cdc" in low or "diabetes" in low:
        return "CDC Diabetes"
    if "mush" in low:
        return "Mushroom"
    if "magic" in low:
        return "MAGIC Gamma"
    return str(raw).strip()


def fmt3(x: float) -> str:
    if pd.isna(x):
        return ""
    s = f"{x:.3f}"
    if s.startswith("-"):
        return "−" + s[1:]
    return s


def main() -> None:
    xl = pd.ExcelFile(EXCEL)
    util_sheet = next(
        s for s in xl.sheet_names if s.startswith("Utility") and "mean" not in s.lower()
    )
    df = pd.read_excel(EXCEL, sheet_name=util_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    AGREED_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_DIR.mkdir(parents=True, exist_ok=True)

    paper_blocks: list[str] = []
    for metric, filename in METRIC_FILES.items():
        sub = df[df["metric_name"] == metric].copy()
        rows = []
        for _, r in sub.iterrows():
            rows.append(
                {
                    "DatasetShort": normalize_dataset(r["dataset"]),
                    "Generator": GEN_EXCEL_TO_CSV.get(
                        str(r["generator"]).strip(), str(r["generator"]).strip()
                    ),
                    "value": float(r["mean"]),
                }
            )
        long = pd.DataFrame(rows)
        wide = long.pivot_table(
            index="DatasetShort", columns="Generator", values="value", aggfunc="mean"
        )
        for g in GEN_CSV:
            if g not in wide.columns:
                wide[g] = np.nan
        wide = wide[GEN_CSV].reindex(DS_CSV_ORDER)
        wide.index.name = "DatasetShort"

        agreed_path = AGREED_DIR / filename
        wide.to_csv(agreed_path, float_format="%.6f")
        print(f"wrote {agreed_path}")

        # multi-seed analysis uses Dataset as index name / display names
        multi = wide.copy()
        multi.index.name = "Dataset"
        multi_path = MULTI_DIR / filename
        multi.to_csv(multi_path, float_format="%.6f")
        print(f"wrote {multi_path}")

        gen_means = wide.mean(axis=0, skipna=True).sort_values()
        lines = [METRIC_TITLES[metric]]
        lines.append("Generator\t" + "\t".join(SHORT[d] for d in DS_CSV_ORDER))
        for g in gen_means.index:
            vals = [fmt3(wide.loc[d, g]) for d in DS_CSV_ORDER]
            lines.append(DISP[g] + "\t" + "\t".join(vals))
        ds_means = wide.mean(axis=1)
        lines.append("Mean\t" + "\t".join(fmt3(ds_means.loc[d]) for d in DS_CSV_ORDER))
        paper_blocks.append("\n".join(lines))

    paper_path = PAPER_DIR / "classification_utility_gaps_10seed_from_excel.tsv"
    paper_path.write_text("\n\n".join(paper_blocks), encoding="utf-8")
    print(f"wrote {paper_path}")
    # Avoid UnicodeEncodeError on Windows cp1252 consoles (minus sign U+2212).
    printable = "\n\n".join(paper_blocks).replace("\u2212", "-")
    print()
    print(printable)


if __name__ == "__main__":
    main()
