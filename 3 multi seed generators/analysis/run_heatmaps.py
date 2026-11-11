#!/usr/bin/env python3
"""Utility-gap heatmaps (dataset × generator) for classification and regression."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import DISPLAY_DATASET, DISPLAY_GENERATOR, NAVY  # noqa: E402
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402
from src.load_results import aggregate_10_seed  # noqa: E402

FIG_CLS = ANALYSIS / "figures" / "classification"
FIG_REG = ANALYSIS / "figures" / "regression"


def heatmap(agg: pd.DataFrame, task: str, metric: str, label: str, out_dir: Path):
    font = configure_times_font()
    sub = agg[(agg["problem_type"] == task) & (agg["metric_name"] == metric)].copy()
    sub["Dataset"] = sub["dataset_id"].map(lambda d: DISPLAY_DATASET.get(d, d))
    sub["Generator"] = sub["generator"].map(lambda g: DISPLAY_GENERATOR.get(g, g))
    wide = sub.pivot(index="Dataset", columns="Generator", values="mean")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.heatmap(wide, annot=True, fmt=".3f", cmap="RdYlGn_r", linewidths=0.4, ax=ax, cbar_kws={"label": label})
    ax.set_title(f"{label} by dataset × generator (10-seed mean)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("")
    ax.set_ylabel("")
    apply_font_to_figure(fig, font)
    fig.tight_layout()
    stem = f"{metric.lower()}_by_dataset_heatmap"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    wide.to_csv(out_dir / f"{stem}.csv")
    print("wrote", stem)


def main():
    agg = aggregate_10_seed()
    FIG_CLS.mkdir(parents=True, exist_ok=True)
    FIG_REG.mkdir(parents=True, exist_ok=True)
    for m, lab in [
        ("Accuracy_Gap", "Accuracy Gap"),
        ("F1_Gap", "F1 Gap"),
        ("Precision_Gap", "Precision Gap"),
        ("Recall_Gap", "Recall Gap"),
        ("Quality_Score", "Quality Score"),
        ("MIA_AUC", "MIA AUC"),
    ]:
        heatmap(agg, "classification", m, lab, FIG_CLS)
    for m, lab in [
        ("R2_Gap", "R² Gap"),
        ("RMSE_Increase", "RMSE Gap"),
        ("MAE_Increase", "MAE Gap"),
        ("Quality_Score", "Quality Score"),
        ("MIA_AUC", "MIA AUC"),
    ]:
        heatmap(agg, "regression", m, lab, FIG_REG)


if __name__ == "__main__":
    main()
