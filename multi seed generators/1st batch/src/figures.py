"""Reusable figure helpers for Batch 1 (mean ± SD)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import ensure_task_dirs, task_paths
from .table_io import read_results


def load_agg(task: str = "classification") -> pd.DataFrame:
    p = ensure_task_dirs(task)["aggregated"] / "all_metrics_mean_sd.xlsx"
    if not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    if not p.exists():
        raise FileNotFoundError(p)
    return read_results(p)


def plot_generator_comparison(
    dataset: str,
    metric_name: str,
    metric_category: str | None = None,
    task: str = "classification",
    out_path: Path | None = None,
) -> Path:
    df = load_agg(task)
    sub = df[(df["dataset"] == dataset) & (df["metric_name"] == metric_name)]
    if metric_category:
        sub = sub[sub["metric_category"] == metric_category]
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset} metric={metric_name}")
    sub = sub.sort_values("generator")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"], yerr=sub["sd"], capsize=4, color="#4C78A8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["generator"], rotation=35, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{dataset}: {metric_name} (mean ± SD across 3 seeds)")
    ax.axhline(0, color="black", lw=0.6)
    fig.tight_layout()
    figures = ensure_task_dirs(task)["figures"]
    out_path = out_path or figures / f"{dataset}_{metric_name}_generators.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_seed_stability(
    dataset: str,
    metric_name: str,
    task: str = "classification",
    out_path: Path | None = None,
) -> Path:
    df = load_agg(task)
    sub = df[(df["dataset"] == dataset) & (df["metric_name"] == metric_name)].copy()
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset} metric={metric_name}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(sub))
    w = 0.25
    for i, (seed, col) in enumerate([(42, "seed_42"), (123, "seed_123"), (2024, "seed_2024")]):
        ax.bar(x + (i - 1) * w, sub[col], width=w, label=f"seed {seed}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["generator"], rotation=35, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{dataset}: seed stability — {metric_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    figures = ensure_task_dirs(task)["figures"]
    out_path = out_path or figures / f"{dataset}_{metric_name}_seed_stability.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
