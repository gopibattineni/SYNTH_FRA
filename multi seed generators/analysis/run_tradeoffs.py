#!/usr/bin/env python3
"""Trade-off figures (Fidelity / Utility / Privacy) using 10-seed means ± SD."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    DISPLAY_DATASET,
    DISPLAY_GENERATOR,
    GENERATOR_COLORS,
    NAVY,
)
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402
from src.load_results import aggregate_10_seed  # noqa: E402

OUT = ANALYSIS / "figures" / "tradeoffs"
OUT_IND = OUT / "individual"
OUT.mkdir(parents=True, exist_ok=True)
OUT_IND.mkdir(parents=True, exist_ok=True)


def metric_frame(agg: pd.DataFrame, task: str) -> pd.DataFrame:
    util = "Accuracy_TSTR" if task == "classification" else "R2_TSTR"
    util_gap = "Accuracy_Gap" if task == "classification" else "R2_Gap"
    need = ["Quality_Score", util, util_gap, "MIA_AUC"]
    sub = agg[(agg["problem_type"] == task) & (agg["metric_name"].isin(need))].copy()
    rows = []
    for (ds, gen), g in sub.groupby(["dataset_id", "generator"]):
        rec = {
            "dataset_id": ds,
            "Dataset": DISPLAY_DATASET.get(ds, ds),
            "generator": gen,
            "Generator": DISPLAY_GENERATOR.get(gen, gen),
        }
        for _, r in g.iterrows():
            rec[r["metric_name"]] = r["mean"]
            rec[r["metric_name"] + "_sd"] = r["sd"]
        rows.append(rec)
    return pd.DataFrame(rows)


def scatter_panels(df: pd.DataFrame, task: str, xcol: str, ycol: str, xlabel: str, ylabel: str, stem: str):
    font = configure_times_font()
    datasets = sorted(df["Dataset"].unique())
    n = len(datasets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.6 * nrows), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["Dataset"] == ds]
        for _, r in sub.iterrows():
            c = GENERATOR_COLORS.get(r["generator"], GENERATOR_COLORS.get(r["Generator"], "#333"))
            ax.errorbar(
                r[xcol],
                r[ycol],
                xerr=r.get(f"{xcol}_sd", 0) if pd.notna(r.get(f"{xcol}_sd")) else None,
                yerr=r.get(f"{ycol}_sd", 0) if pd.notna(r.get(f"{ycol}_sd")) else None,
                fmt="o",
                color=c,
                ecolor="#888",
                elinewidth=0.8,
                capsize=2,
                markersize=7,
                label=r["Generator"],
            )
        ax.set_title(ds, fontsize=11, fontweight="bold", color=NAVY)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(color="#c5d0dc", linewidth=0.6)
        ax.set_axisbelow(True)
        # individual dataset OLS file
        if sub[xcol].notna().sum() >= 3:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                sub[xcol].astype(float), sub[ycol].astype(float)
            )
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
            ax.plot(xs, intercept + slope * xs, color="#555", lw=1.0, ls="--", alpha=0.7)

        # save individual
        fig_i, ax_i = plt.subplots(figsize=(5.2, 4.2))
        for _, r in sub.iterrows():
            c = GENERATOR_COLORS.get(r["generator"], "#333")
            ax_i.errorbar(
                r[xcol],
                r[ycol],
                xerr=r.get(f"{xcol}_sd") if pd.notna(r.get(f"{xcol}_sd")) else None,
                yerr=r.get(f"{ycol}_sd") if pd.notna(r.get(f"{ycol}_sd")) else None,
                fmt="o",
                color=c,
                ecolor="#888",
                elinewidth=0.8,
                capsize=2,
                markersize=8,
            )
            ax_i.annotate(
                r["Generator"],
                (r[xcol], r[ycol]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7.5,
            )
        ax_i.set_title(f"{ds}: {xlabel} vs {ylabel}", fontsize=11, fontweight="bold", color=NAVY)
        ax_i.set_xlabel(xlabel)
        ax_i.set_ylabel(ylabel)
        ax_i.grid(color="#c5d0dc", linewidth=0.6)
        apply_font_to_figure(fig_i, font)
        fig_i.tight_layout()
        slug = ds.lower().replace(" ", "_").replace("'", "")
        for ext in ("png", "pdf"):
            fig_i.savefig(OUT_IND / f"{task}_{slug}_{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig_i)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    # rebuild unique legend from first panel gens
    by_label = {}
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            by_label[ll] = hh
    if by_label:
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            ncol=4,
            fontsize=9,
            frameon=False,
        )
    fig.suptitle(
        f"{task.title()}: {xlabel} vs {ylabel} (10-seed mean ± SD)",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
        y=1.01,
    )
    apply_font_to_figure(fig, font)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{task}_{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem, task)


def bubble(df: pd.DataFrame, task: str, util: str):
    font = configure_times_font()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for _, r in df.iterrows():
        c = GENERATOR_COLORS.get(r["generator"], "#333")
        size = 40 + 400 * float(r["MIA_AUC"]) if pd.notna(r["MIA_AUC"]) else 60
        ax.scatter(r["Quality_Score"], r[util], s=size, c=c, alpha=0.75, edgecolors="#222", linewidths=0.4)
    # average per generator across datasets for legend clarity
    for gen, g in df.groupby("generator"):
        ax.scatter([], [], c=GENERATOR_COLORS.get(gen, "#333"), s=80, label=DISPLAY_GENERATOR.get(gen, gen))
    ax.set_xlabel("SDV Quality Score (Fidelity)")
    ax.set_ylabel("Accuracy (Utility)" if util == "Accuracy_TSTR" else r"$R^{2}$ (Utility)")
    ax.set_title(f"{task.title()} bubble trade-off (size ∝ MIA AUC)", fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(color="#c5d0dc", linewidth=0.6)
    apply_font_to_figure(fig, font)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{task}_Fig4_Bubble_Tradeoff.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    agg = aggregate_10_seed()
    for task in ("classification", "regression"):
        df = metric_frame(agg, task)
        df.to_csv(ANALYSIS / "data" / f"tradeoff_points_{task}.csv", index=False)
        util = "Accuracy_TSTR" if task == "classification" else "R2_TSTR"
        util_lab = "Accuracy (Utility)" if task == "classification" else r"$R^{2}$ (Utility)"
        gap = "Accuracy_Gap" if task == "classification" else "R2_Gap"
        gap_lab = "Accuracy Gap" if task == "classification" else r"$R^{2}$ Gap"

        scatter_panels(
            df, task, "Quality_Score", util, "SDV Quality Score (Fidelity)", util_lab, "Fig1_Fidelity_vs_Utility"
        )
        scatter_panels(
            df, task, util, "MIA_AUC", util_lab, "MIA (AUC)", "Fig2_Utility_vs_Privacy"
        )
        scatter_panels(
            df, task, "Quality_Score", "MIA_AUC", "SDV Quality Score (Fidelity)", "MIA (AUC)", "Fig3_Fidelity_vs_Privacy"
        )
        scatter_panels(
            df, task, "Quality_Score", gap, "SDV Quality Score (Fidelity)", gap_lab, "Fidelity_vs_UtilityGap"
        )
        bubble(df, task, util)
    print("trade-offs done")


if __name__ == "__main__":
    main()
