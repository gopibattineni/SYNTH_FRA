#!/usr/bin/env python3
"""Seed-stability figures (n=10 generator-training seeds) in Agreed-analysis style."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    DISPLAY_DATASET,
    DISPLAY_GENERATOR,
    GENERATOR_COLORS,
    NAVY,
    SEEDS,
)
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402
from src.load_results import aggregate_10_seed  # noqa: E402

OUT = ANALYSIS / "figures" / "seed_stability"


def violin_gaps(agg: pd.DataFrame, task: str, metrics: list[tuple[str, str]]):
    """One violin panel set: generators on x, gap values as the 10 seed means... 

    Actually for seed stability we want the distribution of the 10 seed values
    pooled across datasets OR per metric as strip of seed values by generator.
    Agreed analysis violins used n=9 datasets (already averaged). Here we show
    seed-level values: for each generator, all dataset×seed gap observations,
    annotated that points are seed-level.
    """
    font = configure_times_font()
    OUT.mkdir(parents=True, exist_ok=True)

    for metric, label in metrics:
        rows = []
        sub = agg[(agg["problem_type"] == task) & (agg["metric_name"] == metric)]
        for _, r in sub.iterrows():
            gen = DISPLAY_GENERATOR.get(r["generator"], r["generator"])
            ds = DISPLAY_DATASET.get(r["dataset_id"], r["dataset_id"])
            for s in SEEDS:
                v = r.get(f"seed_{s}")
                if pd.isna(v):
                    continue
                rows.append({"Generator": gen, "Dataset": ds, "Seed": s, "Value": float(v)})
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        order = [
            DISPLAY_GENERATOR.get(g, g)
            for g in [
                "ForestDiffusion",
                "TVAE",
                "CTABGAN",
                "WGAN_GP",
                "GaussianCopula",
                "CopulaGAN",
                "CTGAN",
                "TabDDPM",
            ]
        ]
        palette = {g: GENERATOR_COLORS.get(g, "#333") for g in order}

        fig, ax = plt.subplots(figsize=(11, 4.8))
        sns.violinplot(
            data=df,
            x="Generator",
            y="Value",
            order=order,
            palette=palette,
            cut=0,
            inner=None,
            ax=ax,
            saturation=0.85,
        )
        sns.stripplot(
            data=df,
            x="Generator",
            y="Value",
            order=order,
            color="#222",
            size=2.4,
            alpha=0.35,
            ax=ax,
            jitter=0.18,
        )
        ax.set_title(
            f"{label} — seed-level distribution (n = 10 seeds × datasets)",
            fontsize=13,
            fontweight="bold",
            color=NAVY,
            fontfamily="serif",
        )
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} (seed-level)", fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color="#c5d0dc", linewidth=0.7)
        ax.set_axisbelow(True)
        fig.text(
            0.5,
            0.01,
            "Each point is one generator-training seed for one dataset (same leakage-safe split).",
            ha="center",
            fontsize=8.5,
            color="#5a5f66",
        )
        apply_font_to_figure(fig, font)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        stem = f"{task}_{metric.lower()}_seed_violin"
        for ext in ("png", "svg", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", stem)


def mean_sd_bars(agg: pd.DataFrame, task: str, metric: str, label: str):
    """Per-dataset grouped bars: 8 generators with mean±SD error bars (10 seeds)."""
    font = configure_times_font()
    ds_dir = ANALYSIS / "figures" / "dataset_level" / task
    ds_dir.mkdir(parents=True, exist_ok=True)
    sub = agg[(agg["problem_type"] == task) & (agg["metric_name"] == metric)].copy()
    gens = [
        "ForestDiffusion",
        "TVAE",
        "CTABGAN",
        "WGAN_GP",
        "GaussianCopula",
        "CopulaGAN",
        "CTGAN",
        "TabDDPM",
    ]
    for ds_id, g in sub.groupby("dataset_id"):
        g = g.set_index("generator").reindex(gens)
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        x = np.arange(len(gens))
        means = g["mean"].to_numpy(dtype=float)
        sds = g["sd"].fillna(0).to_numpy(dtype=float)
        colors = [GENERATOR_COLORS.get(gen, "#333") for gen in gens]
        ax.bar(x, means, yerr=sds, color=colors, edgecolor="#222", linewidth=0.5, capsize=3, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_GENERATOR.get(gen, gen) for gen in gens], rotation=25, ha="right")
        ax.set_title(
            f"{DISPLAY_DATASET.get(ds_id, ds_id)} — {label} (mean ± SD, 10 seeds)",
            fontsize=12.5,
            fontweight="bold",
            color=NAVY,
        )
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#c5d0dc", linewidth=0.7)
        ax.set_axisbelow(True)
        apply_font_to_figure(fig, font)
        fig.tight_layout()
        stem = f"{ds_id}_{metric.lower()}_mean_sd"
        for ext in ("png", "pdf"):
            fig.savefig(ds_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"dataset-level {task} {metric}: {sub['dataset_id'].nunique()} figures")


def main():
    agg = aggregate_10_seed()
    violin_gaps(
        agg,
        "classification",
        [
            ("Accuracy_Gap", "Accuracy Gap"),
            ("Precision_Gap", "Precision Gap"),
            ("Recall_Gap", "Recall Gap"),
            ("F1_Gap", "F1 Gap"),
            ("Quality_Score", "Quality Score"),
            ("MIA_AUC", "MIA AUC"),
        ],
    )
    violin_gaps(
        agg,
        "regression",
        [
            ("R2_Gap", "R² Gap"),
            ("RMSE_Increase", "RMSE Gap"),
            ("MAE_Increase", "MAE Gap"),
            ("Quality_Score", "Quality Score"),
            ("MIA_AUC", "MIA AUC"),
        ],
    )
    # Representative dataset-level mean±SD bars
    mean_sd_bars(agg, "classification", "Accuracy_Gap", "Accuracy Gap")
    mean_sd_bars(agg, "classification", "Quality_Score", "Quality Score")
    mean_sd_bars(agg, "regression", "R2_Gap", "R² Gap")
    mean_sd_bars(agg, "regression", "Quality_Score", "Quality Score")


if __name__ == "__main__":
    main()
