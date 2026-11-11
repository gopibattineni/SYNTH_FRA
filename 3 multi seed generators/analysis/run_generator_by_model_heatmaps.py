#!/usr/bin/env python3
"""Generator × downstream-model heatmaps from per_model_utility.csv files."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS = Path(__file__).resolve().parent
MULTI = ANALYSIS.parent
OUT = ANALYSIS / "figures" / "generator_by_model"
OUT.mkdir(parents=True, exist_ok=True)

BATCHES = ["1st batch", "second batch", "third batch"]


def _load_all() -> pd.DataFrame:
    frames = []
    for batch in BATCHES:
        for path in (MULTI / batch).rglob("per_model_utility.csv"):
            df = pd.read_csv(path)
            df["batch"] = batch
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No per_model_utility.csv found — run rerun_utility_10models.py first")
    return pd.concat(frames, ignore_index=True)


def _heatmap(df: pd.DataFrame, metric: str, path_label: str, title: str, outfile: Path) -> None:
    sub = df[(df["metric"] == metric) & (df["path"] == path_label)]
    if sub.empty:
        return
    # Mean over datasets × generator seeds × utility seeds
    pivot = (
        sub.groupby(["generator", "model"], as_index=False)["value"]
        .mean()
        .pivot(index="generator", columns="model", values="value")
    )
    plt.figure(figsize=(12, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile.with_suffix(".png"), dpi=200)
    plt.savefig(outfile.with_suffix(".svg"))
    plt.close()
    pivot.to_csv(outfile.with_suffix(".csv"))


def main() -> None:
    df = _load_all()
    print(f"Loaded {len(df)} per-model rows; models={sorted(df['model'].unique())}")
    _heatmap(
        df,
        "Accuracy",
        "TSTR",
        "Generator × Classifier — mean TSTR Accuracy",
        OUT / "cls_generator_by_model_TSTR_Accuracy",
    )
    _heatmap(
        df,
        "Accuracy",
        "TRTR",
        "Generator × Classifier — mean TRTR Accuracy",
        OUT / "cls_generator_by_model_TRTR_Accuracy",
    )
    # Gap = TRTR - TSTR at row level then mean
    acc = df[df["metric"] == "Accuracy"].copy()
    if not acc.empty:
        wide = acc.pivot_table(
            index=["dataset", "generator", "generator_seed", "model", "utility_seed"],
            columns="path",
            values="value",
            aggfunc="mean",
        ).reset_index()
        if {"TRTR", "TSTR"}.issubset(wide.columns):
            wide["Gap"] = wide["TRTR"] - wide["TSTR"]
            gap = wide.rename(columns={"Gap": "value"})
            gap["metric"] = "Accuracy_Gap"
            gap["path"] = "GAP"
            _heatmap(
                gap,
                "Accuracy_Gap",
                "GAP",
                "Generator × Classifier — mean Accuracy Gap (TRTR−TSTR)",
                OUT / "cls_generator_by_model_Accuracy_Gap",
            )
    _heatmap(
        df,
        "R2",
        "TSTR",
        "Generator × Regressor — mean TSTR R²",
        OUT / "reg_generator_by_model_TSTR_R2",
    )
    print(f"Wrote heatmaps under {OUT}")


if __name__ == "__main__":
    main()
