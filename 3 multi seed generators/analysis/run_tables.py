#!/usr/bin/env python3
"""Manuscript tables: Mean±SD main tables, full seed supplementary, captions."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    DISPLAY_DATASET,
    DISPLAY_GENERATOR,
    SEEDS,
)
from src.load_results import aggregate_10_seed  # noqa: E402

TAB_MAIN = ANALYSIS / "tables" / "main"
TAB_SUPP = ANALYSIS / "tables" / "supplementary"
REPORTS = ANALYSIS / "reports"

CLS_CORE = {
    "Fidelity": ["Quality_Score", "KS_Complement", "MMD", "Wasserstein_Distance"],
    "Utility": [
        "Accuracy_TSTR",
        "Accuracy_TRTR",
        "Accuracy_Gap",
        "F1_TSTR",
        "F1_Gap",
        "Precision_TSTR",
        "Precision_Gap",
        "Recall_TSTR",
        "Recall_Gap",
        "ROC_AUC_TSTR",
        "ROC_AUC_Gap",
    ],
    "Privacy": ["MIA_AUC", "NNDR", "Mean_Distance", "Median_Distance", "Mahalanobis_Distance"],
}

REG_CORE = {
    "Fidelity": ["Quality_Score", "KS_Complement", "MMD", "Wasserstein_Distance"],
    "Utility": [
        "R2_TSTR",
        "R2_TRTR",
        "R2_Gap",
        "RMSE_TSTR",
        "RMSE_TRTR",
        "RMSE_Increase",
        "MAE_TSTR",
        "MAE_TRTR",
        "MAE_Increase",
    ],
    "Privacy": ["MIA_AUC", "NNDR", "Mean_Distance", "Median_Distance", "Mahalanobis_Distance"],
}


def _pretty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Dataset"] = out["dataset_id"].map(lambda d: DISPLAY_DATASET.get(d, d))
    out["Generator"] = out["generator"].map(lambda g: DISPLAY_GENERATOR.get(g, g))
    return out


def wide_category(agg: pd.DataFrame, task: str, metrics: list[str]) -> pd.DataFrame:
    sub = agg[(agg["problem_type"] == task) & (agg["metric_name"].isin(metrics))].copy()
    sub = _pretty(sub)
    piv = sub.pivot_table(
        index=["Dataset", "Generator"],
        columns="metric_name",
        values="mean_sd",
        aggfunc="first",
    ).reset_index()
    # order columns
    cols = ["Dataset", "Generator"] + [m for m in metrics if m in piv.columns]
    return piv[cols]


def seed_level_table(agg: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = agg[agg["problem_type"] == task].copy()
    sub = _pretty(sub)
    seed_cols = [f"seed_{s}" for s in SEEDS]
    cols = [
        "Dataset",
        "Generator",
        "metric_category",
        "metric_name",
        *seed_cols,
        "mean",
        "sd",
        "n_seeds",
    ]
    return sub[cols].sort_values(["Dataset", "Generator", "metric_category", "metric_name"])


def main_summary_table(agg: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compact Dataset × Generator with Fidelity / key Utility / Privacy Mean±SD."""
    if task == "classification":
        want = {
            "Quality_Score": "Fidelity (Quality)",
            "Accuracy_TSTR": "Accuracy TSTR",
            "Accuracy_Gap": "Accuracy Gap",
            "F1_TSTR": "F1 TSTR",
            "F1_Gap": "F1 Gap",
            "MIA_AUC": "MIA AUC",
        }
    else:
        want = {
            "Quality_Score": "Fidelity (Quality)",
            "R2_TSTR": "R² TSTR",
            "R2_Gap": "R² Gap",
            "RMSE_TSTR": "RMSE TSTR",
            "RMSE_Increase": "RMSE Gap",
            "MIA_AUC": "MIA AUC",
        }
    sub = agg[(agg["problem_type"] == task) & (agg["metric_name"].isin(want))].copy()
    sub = _pretty(sub)
    piv = sub.pivot_table(
        index=["Dataset", "Generator"], columns="metric_name", values="mean_sd", aggfunc="first"
    )
    piv = piv.rename(columns=want)
    order = [want[k] for k in want if want[k] in piv.columns]
    return piv[order].reset_index()


def write_captions():
    text = """# Manuscript Captions — 10-Seed Evaluation

## General phrasing

Results are reported as mean ± SD across **10 independent generator-training seeds** using the same leakage-safe train/test partition (split seed = 42).

The 10 generator-training seeds were executed in three computational batches: three seeds in Batch 1 (42, 123, 2024), three seeds in Batch 2 (68, 91, 2025), and four seeds in Batch 3 (55, 155, 255, 355). Batches are execution groups only; primary results aggregate all 10 seed-level observations.

## Critical-difference diagrams

Critical-difference diagram of generator rankings by utility gap (TRTR − TSTR). Within each dataset, generators were ranked by the 10-seed mean gap (lower is better). Friedman omnibus and Nemenyi post-hoc tests (α = 0.05) were applied across datasets; generators joined by a red bar are not significantly different.

## Seed-stability figures

Each distribution (or error bar) represents results from **10 independent generator-training seeds** for the same dataset × generator configuration (not 10 datasets or 10 participants).

## Trade-off figures

Points show the 10-seed mean for each generator. Where shown, error bars are the sample SD (ddof = 1) across the 10 seeds.

## Supplementary complete-seed table

Per-metric values for all 10 seeds plus mean and sample SD, enabling full reproducibility of the aggregated Mean ± SD results.
"""
    (REPORTS / "figure_table_captions.md").write_text(text, encoding="utf-8")


def main():
    agg = aggregate_10_seed()
    TAB_MAIN.mkdir(parents=True, exist_ok=True)
    TAB_SUPP.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    for task, core in (("classification", CLS_CORE), ("regression", REG_CORE)):
        # main compact
        main_tbl = main_summary_table(agg, task)
        main_tbl.to_excel(TAB_MAIN / f"{task}_mean_sd_summary.xlsx", index=False)
        main_tbl.to_csv(TAB_MAIN / f"{task}_mean_sd_summary.csv", index=False)

        # category workbooks
        with pd.ExcelWriter(TAB_MAIN / f"{task}_mean_sd_by_category.xlsx", engine="openpyxl") as w:
            for cat, metrics in core.items():
                wide_category(agg, task, metrics).to_excel(w, sheet_name=cat, index=False)

        # full seed supplementary
        seeds = seed_level_table(agg, task)
        seeds.to_excel(TAB_SUPP / f"{task}_all_10_seeds.xlsx", index=False)
        seeds.to_csv(TAB_SUPP / f"{task}_all_10_seeds.csv", index=False)

        # stability summary (mean, sd, median, min, max, CV where |mean|>eps)
        stab = agg[agg["problem_type"] == task].copy()
        stab = _pretty(stab)
        stab["CV"] = stab.apply(
            lambda r: (r["sd"] / abs(r["mean"]))
            if pd.notna(r["sd"]) and pd.notna(r["mean"]) and abs(r["mean"]) > 1e-8
            else float("nan"),
            axis=1,
        )
        cols = [
            "Dataset",
            "Generator",
            "metric_category",
            "metric_name",
            "mean",
            "sd",
            "median",
            "min",
            "max",
            "CV",
            "n_seeds",
            "mean_sd",
        ]
        stab[cols].to_excel(TAB_SUPP / f"{task}_seed_stability.xlsx", index=False)

        print(f"{task}: wrote mean±SD + seed tables ({len(main_tbl)} config rows in summary)")

    write_captions()
    print("captions -> reports/figure_table_captions.md")


if __name__ == "__main__":
    main()
