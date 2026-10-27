"""Load multi-seed Mean ± SD results from merged 10-seed Excel workbooks."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import MERGED_CLASSIFICATION_XLSX, MERGED_REGRESSION_XLSX

GENERATOR_ORDER = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "CTABGAN",
    "WGAN_GP",
    "TabDDPM",
    "ForestDiffusion",
]

SEED_LIST = [42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]

# Preferred metrics for rankings / heatmaps (lower or higher better).
METRIC_META: dict[str, dict[str, str]] = {
    # Utility — classification
    "Accuracy_Gap": {"category": "Utility", "label": "Accuracy gap (TRTR−TSTR)", "better": "lower"},
    "F1_Gap": {"category": "Utility", "label": "F1 gap", "better": "lower"},
    "Precision_Gap": {"category": "Utility", "label": "Precision gap", "better": "lower"},
    "Recall_Gap": {"category": "Utility", "label": "Recall gap", "better": "lower"},
    "ROC_AUC_Gap": {"category": "Utility", "label": "ROC-AUC gap", "better": "lower"},
    "Accuracy_TSTR": {"category": "Utility", "label": "Accuracy TSTR", "better": "higher"},
    "F1_TSTR": {"category": "Utility", "label": "F1 TSTR", "better": "higher"},
    "Precision_TSTR": {"category": "Utility", "label": "Precision TSTR", "better": "higher"},
    "Recall_TSTR": {"category": "Utility", "label": "Recall TSTR", "better": "higher"},
    "ROC_AUC_TSTR": {"category": "Utility", "label": "ROC-AUC TSTR", "better": "higher"},
    "Accuracy_TRTR": {"category": "Utility", "label": "Accuracy TRTR", "better": "higher"},
    "F1_TRTR": {"category": "Utility", "label": "F1 TRTR", "better": "higher"},
    # Utility — regression
    "R2_Gap": {"category": "Utility", "label": "R² gap (TRTR−TSTR)", "better": "lower"},
    "RMSE_Increase": {"category": "Utility", "label": "RMSE increase", "better": "lower"},
    "MAE_Increase": {"category": "Utility", "label": "MAE increase", "better": "lower"},
    "R2_TSTR": {"category": "Utility", "label": "R² TSTR", "better": "higher"},
    "RMSE_TSTR": {"category": "Utility", "label": "RMSE TSTR", "better": "lower"},
    "MAE_TSTR": {"category": "Utility", "label": "MAE TSTR", "better": "lower"},
    "R2_TRTR": {"category": "Utility", "label": "R² TRTR", "better": "higher"},
    # Fidelity
    "Quality_Score": {"category": "Fidelity", "label": "SDV Quality Score", "better": "higher"},
    "KS_Complement": {"category": "Fidelity", "label": "KS Complement", "better": "higher"},
    "MMD": {"category": "Fidelity", "label": "MMD", "better": "lower"},
    "Wasserstein_Distance": {"category": "Fidelity", "label": "Wasserstein distance", "better": "lower"},
    # Privacy
    "MIA_AUC": {"category": "Privacy", "label": "MIA AUC", "better": "lower"},
    "NNDR": {"category": "Privacy", "label": "NNDR", "better": "higher"},
    "Mahalanobis_Distance": {"category": "Privacy", "label": "Mahalanobis distance", "better": "higher"},
    "Mean_Distance": {"category": "Privacy", "label": "Mean distance", "better": "higher"},
    "Median_Distance": {"category": "Privacy", "label": "Median distance", "better": "higher"},
}

_MEAN_SD_RE = re.compile(
    r"^\s*(?P<mean>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*[±]\s*"
    r"(?P<sd>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def _parse_mean_sd(value: Any) -> tuple[float | None, float | None]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return None, None
    match = _MEAN_SD_RE.match(text)
    if match:
        return float(match.group("mean")), float(match.group("sd"))
    try:
        return float(text), None
    except ValueError:
        return None, None


def _workbook_path(task: str):
    if task == "classification":
        return MERGED_CLASSIFICATION_XLSX
    if task == "regression":
        return MERGED_REGRESSION_XLSX
    raise ValueError(f"Unknown task: {task}")


@lru_cache(maxsize=4)
def _load_long(task: str) -> pd.DataFrame:
    path = _workbook_path(task)
    if not path.is_file():
        raise FileNotFoundError(f"Merged results not found: {path}")
    df = pd.read_excel(path, sheet_name="All metrics")
    required = {"dataset", "generator", "metric_category", "metric_name", "mean", "sd", "mean_sd", "n_seeds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    out = df.copy()
    out["task"] = task
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["sd"] = pd.to_numeric(out["sd"], errors="coerce")
    out["n_seeds"] = pd.to_numeric(out["n_seeds"], errors="coerce").fillna(0).astype(int)
    return out


def available_tasks() -> list[str]:
    tasks = []
    if MERGED_CLASSIFICATION_XLSX.is_file():
        tasks.append("classification")
    if MERGED_REGRESSION_XLSX.is_file():
        tasks.append("regression")
    return tasks


def get_benchmark_overview() -> dict[str, Any]:
    tasks = available_tasks()
    datasets: list[str] = []
    generators: list[str] = []
    n_rows = 0
    categories: set[str] = set()
    for task in tasks:
        df = _load_long(task)
        n_rows += len(df)
        datasets.extend(df["dataset"].dropna().unique().tolist())
        generators.extend(df["generator"].dropna().unique().tolist())
        categories.update(df["metric_category"].dropna().unique().tolist())

    gen_sorted = [g for g in GENERATOR_ORDER if g in set(generators)]
    for g in sorted(set(generators)):
        if g not in gen_sorted:
            gen_sorted.append(g)

    return {
        "source": "multi seed generators/merged 10 seeds",
        "n_generator_seeds": len(SEED_LIST),
        "generator_seeds": SEED_LIST,
        "split_seed": 42,
        "aggregation": "Mean ± SD across 10 generator-training seeds (ddof=1)",
        "tasks": tasks,
        "n_datasets": len(set(datasets)),
        "datasets": sorted(set(datasets)),
        "n_generators": len(gen_sorted),
        "generators": gen_sorted,
        "categories": sorted(categories),
        "n_metric_rows": n_rows,
        "n_runs": 15 * 8 * 10 if len(set(datasets)) >= 15 else None,
        "files": {
            "classification": str(MERGED_CLASSIFICATION_XLSX) if MERGED_CLASSIFICATION_XLSX.is_file() else None,
            "regression": str(MERGED_REGRESSION_XLSX) if MERGED_REGRESSION_XLSX.is_file() else None,
        },
    }


def list_metrics(task: str | None = None, category: str | None = None) -> list[dict[str, str]]:
    tasks = [task] if task else available_tasks()
    names: set[str] = set()
    for t in tasks:
        df = _load_long(t)
        if category:
            df = df[df["metric_category"] == category]
        names.update(df["metric_name"].dropna().astype(str).tolist())

    items = []
    for name in sorted(names):
        meta = METRIC_META.get(name, {})
        items.append(
            {
                "id": name,
                "label": meta.get("label", name.replace("_", " ")),
                "category": meta.get("category", "Other"),
                "better": meta.get("better", "higher"),
            }
        )
    # Put well-known metrics first within category
    preferred = list(METRIC_META.keys())
    items.sort(key=lambda x: (x["category"], preferred.index(x["id"]) if x["id"] in preferred else 999, x["id"]))
    return items


def query_metrics(
    task: str | None = None,
    category: str | None = None,
    dataset: str | None = None,
    generator: str | None = None,
    metric: str | None = None,
) -> list[dict[str, Any]]:
    tasks = [task] if task else available_tasks()
    rows: list[dict[str, Any]] = []
    for t in tasks:
        df = _load_long(t)
        if category:
            df = df[df["metric_category"] == category]
        if dataset:
            df = df[df["dataset"] == dataset]
        if generator:
            df = df[df["generator"] == generator]
        if metric:
            df = df[df["metric_name"] == metric]
        for _, r in df.iterrows():
            mean = None if pd.isna(r["mean"]) else float(r["mean"])
            sd = None if pd.isna(r["sd"]) else float(r["sd"])
            mean_sd = r.get("mean_sd")
            if pd.isna(mean_sd):
                mean_sd = None
            elif mean is not None and sd is not None and (mean_sd is None or str(mean_sd).strip() == ""):
                mean_sd = f"{mean} ± {sd}"
            rows.append(
                {
                    "task": t,
                    "dataset": str(r["dataset"]),
                    "generator": str(r["generator"]),
                    "category": str(r["metric_category"]),
                    "metric": str(r["metric_name"]),
                    "mean": mean,
                    "sd": sd,
                    "mean_sd": None if mean_sd is None else str(mean_sd),
                    "n_seeds": int(r["n_seeds"]),
                }
            )
    return rows


def metric_matrix(task: str, metric: str) -> dict[str, Any]:
    df = _load_long(task)
    sub = df[df["metric_name"] == metric].copy()
    if sub.empty:
        raise KeyError(f"Metric not found for {task}: {metric}")

    datasets = sorted(sub["dataset"].unique().tolist())
    generators = [g for g in GENERATOR_ORDER if g in set(sub["generator"])]
    for g in sorted(sub["generator"].unique()):
        if g not in generators:
            generators.append(g)

    cells: list[dict[str, Any]] = []
    values: list[list[float | None]] = []
    display: list[list[str | None]] = []
    for gen in generators:
        row_vals: list[float | None] = []
        row_disp: list[str | None] = []
        for ds in datasets:
            hit = sub[(sub["generator"] == gen) & (sub["dataset"] == ds)]
            if hit.empty:
                row_vals.append(None)
                row_disp.append(None)
                cells.append({"dataset": ds, "generator": gen, "mean": None, "sd": None, "mean_sd": None})
                continue
            r = hit.iloc[0]
            mean = None if pd.isna(r["mean"]) else float(r["mean"])
            sd = None if pd.isna(r["sd"]) else float(r["sd"])
            mean_sd = None if pd.isna(r.get("mean_sd")) else str(r["mean_sd"])
            row_vals.append(mean)
            row_disp.append(mean_sd)
            cells.append(
                {
                    "dataset": ds,
                    "generator": gen,
                    "mean": mean,
                    "sd": sd,
                    "mean_sd": mean_sd,
                    "n_seeds": int(r["n_seeds"]),
                }
            )
        values.append(row_vals)
        display.append(row_disp)

    meta = METRIC_META.get(metric, {})
    return {
        "task": task,
        "metric": metric,
        "label": meta.get("label", metric.replace("_", " ")),
        "better": meta.get("better", "higher"),
        "category": meta.get("category", sub.iloc[0]["metric_category"] if len(sub) else "Other"),
        "datasets": datasets,
        "generators": generators,
        "values": values,
        "display": display,
        "cells": cells,
    }


def generator_ranking(task: str, metric: str) -> list[dict[str, Any]]:
    matrix = metric_matrix(task, metric)
    better = matrix["better"]
    ranked: list[dict[str, Any]] = []
    for i, gen in enumerate(matrix["generators"]):
        vals = [v for v in matrix["values"][i] if v is not None]
        if not vals:
            continue
        mean_over_ds = float(sum(vals) / len(vals))
        ranked.append(
            {
                "generator": gen,
                "mean_over_datasets": mean_over_ds,
                "n_datasets": len(vals),
                "better": better,
            }
        )
    reverse = better == "higher"
    ranked.sort(key=lambda x: x["mean_over_datasets"], reverse=reverse)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def wide_category_table(task: str, category: str, dataset: str | None = None) -> dict[str, Any]:
    """Wide Mean±SD table: one row per dataset×generator, columns = metrics."""
    df = _load_long(task)
    sub = df[df["metric_category"] == category].copy()
    if dataset:
        sub = sub[sub["dataset"] == dataset]
    if sub.empty:
        return {"task": task, "category": category, "dataset": dataset, "columns": [], "rows": []}

    metrics = sorted(sub["metric_name"].unique().tolist())
    preferred = [m for m in METRIC_META if m in metrics]
    metrics = preferred + [m for m in metrics if m not in preferred]

    rows = []
    for (ds, gen), group in sub.groupby(["dataset", "generator"], sort=True):
        row: dict[str, Any] = {"dataset": ds, "generator": gen}
        for m in metrics:
            hit = group[group["metric_name"] == m]
            if hit.empty:
                row[m] = None
                row[f"{m}__mean"] = None
                row[f"{m}__sd"] = None
            else:
                r = hit.iloc[0]
                mean_sd = None if pd.isna(r.get("mean_sd")) else str(r["mean_sd"])
                mean = None if pd.isna(r["mean"]) else float(r["mean"])
                sd = None if pd.isna(r["sd"]) else float(r["sd"])
                row[m] = mean_sd
                row[f"{m}__mean"] = mean
                row[f"{m}__sd"] = sd
        rows.append(row)

    # Stable generator order within each dataset
    order_index = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    rows.sort(key=lambda r: (r["dataset"], order_index.get(r["generator"], 999), r["generator"]))

    return {
        "task": task,
        "category": category,
        "dataset": dataset,
        "columns": ["dataset", "generator", *metrics],
        "metric_meta": [
            {
                "id": m,
                "label": METRIC_META.get(m, {}).get("label", m.replace("_", " ")),
                "better": METRIC_META.get(m, {}).get("better", "higher"),
            }
            for m in metrics
        ],
        "rows": rows,
    }


def clear_cache() -> None:
    _load_long.cache_clear()
