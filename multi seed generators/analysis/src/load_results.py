"""Load seed-level metrics from all three batches with batch provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import (
    ALL_DATASETS,
    BATCHES,
    CLASSIFICATION_DATASETS,
    GENERATORS,
    MULTI_SEED_ROOT,
    REGRESSION_DATASETS,
    SEED_TO_BATCH,
    SEEDS,
)


def _task_for_dataset(dataset_id: str) -> str:
    if dataset_id in CLASSIFICATION_DATASETS:
        return "classification"
    if dataset_id in REGRESSION_DATASETS:
        return "regression"
    raise KeyError(dataset_id)


def load_raw_long(force_rebuild: bool = False) -> pd.DataFrame:
    """Load every metrics.xlsx across batches into one long table with batch labels.

    Columns:
      dataset, dataset_id, problem_type, generator, seed, batch, metric_category,
      metric_name, metric_value, plus compute/meta fields when present.
    """
    out_path = Path(__file__).resolve().parents[1] / "data" / "seed_level_long.csv"
    if out_path.exists() and not force_rebuild:
        return pd.read_csv(out_path)

    frames: list[pd.DataFrame] = []
    for batch_id, meta in BATCHES.items():
        batch_dir = MULTI_SEED_ROOT / meta["folder"]
        for seed in meta["seeds"]:
            for dataset_id in ALL_DATASETS:
                task = _task_for_dataset(dataset_id)
                for gen in GENERATORS:
                    mx = (
                        batch_dir
                        / task
                        / "results"
                        / "raw"
                        / f"seed_{seed}"
                        / dataset_id
                        / gen
                        / "metrics.xlsx"
                    )
                    if not mx.exists() or mx.stat().st_size <= 0:
                        continue
                    try:
                        df = pd.read_excel(mx)
                    except Exception:
                        continue
                    if df.empty:
                        continue
                    df = df.copy()
                    df["batch"] = batch_id
                    df["batch_label"] = meta["label"]
                    if "dataset_id" not in df.columns:
                        df["dataset_id"] = dataset_id
                    if "task" not in df.columns:
                        df["task"] = task
                    df["problem_type"] = df["task"]
                    frames.append(df)

    if not frames:
        raise RuntimeError("No metrics.xlsx files found across batches")

    raw = pd.concat(frames, ignore_index=True)
    # Canonical columns
    if "dataset" not in raw.columns and "dataset_id" in raw.columns:
        raw["dataset"] = raw["dataset_id"]
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce").astype("Int64")
    raw["metric_value"] = pd.to_numeric(raw["metric_value"], errors="coerce")
    # Prefer batch from seed map if missing
    raw["batch"] = raw["seed"].map(SEED_TO_BATCH).fillna(raw.get("batch"))

    keep_order = [
        "dataset",
        "dataset_id",
        "problem_type",
        "generator",
        "seed",
        "batch",
        "batch_label",
        "metric_category",
        "metric_name",
        "metric_value",
        "split_seed",
        "n_synthetic_samples",
        "n_train",
        "n_test",
        "training_time_seconds",
        "generation_time_seconds",
        "evaluation_time_seconds",
        "total_time_seconds",
    ]
    cols = [c for c in keep_order if c in raw.columns] + [
        c for c in raw.columns if c not in keep_order
    ]
    raw = raw[cols]

    # Deduplicate identical dataset×generator×seed×metric rows (keep last)
    dedupe = [
        c
        for c in (
            "dataset_id",
            "generator",
            "seed",
            "metric_category",
            "metric_name",
        )
        if c in raw.columns
    ]
    raw = raw.drop_duplicates(subset=dedupe, keep="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_path, index=False)
    return raw


def load_run_index(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per completed generator run (dataset × generator × seed)."""
    if raw is None:
        raw = load_raw_long()
    cols = ["dataset_id", "problem_type", "generator", "seed", "batch", "batch_label"]
    for c in cols:
        if c not in raw.columns:
            raise KeyError(c)
    runs = raw[cols].drop_duplicates()
    return runs.sort_values(["batch", "seed", "problem_type", "dataset_id", "generator"]).reset_index(
        drop=True
    )


def aggregate_10_seed(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean ± SD across the 10 individual seed observations (ddof=1).

    Critical: aggregates seed-level values directly — never mean-of-batch-means.
    """
    if raw is None:
        raw = load_raw_long()
    key = ["dataset", "dataset_id", "problem_type", "generator", "metric_category", "metric_name"]
    rows = []
    for keys, g in raw.groupby(key, dropna=False):
        by_seed = {}
        for seed, sub in g.groupby("seed"):
            vals = pd.to_numeric(sub["metric_value"], errors="coerce")
            finite = vals[vals.notna()]
            if len(finite):
                by_seed[int(seed)] = float(finite.iloc[-1])
        vals = [by_seed[s] for s in SEEDS if s in by_seed]
        n = len(vals)
        if n == 0:
            mean = float("nan")
            sd = float("nan")
            mean_sd = ""
        elif n == 1:
            mean = float(vals[0])
            sd = float("nan")
            mean_sd = f"{mean:.6g}"
        else:
            s = pd.Series(vals, dtype=float)
            mean = float(s.mean())
            sd = float(s.std(ddof=1))
            if not pd.isna(sd) and abs(sd) <= 1e-12:
                sd = 0.0
                mean_sd = f"{mean:.6g} ± 0"
            else:
                mean_sd = f"{mean:.6g} ± {sd:.6g}"
        row = {
            "dataset": keys[0],
            "dataset_id": keys[1],
            "problem_type": keys[2],
            "generator": keys[3],
            "metric_category": keys[4],
            "metric_name": keys[5],
            "mean": mean,
            "sd": sd,
            "mean_sd": mean_sd,
            "n_seeds": n,
            "median": float(pd.Series(vals).median()) if n else float("nan"),
            "min": float(min(vals)) if n else float("nan"),
            "max": float(max(vals)) if n else float("nan"),
        }
        for s in SEEDS:
            row[f"seed_{s}"] = by_seed.get(s, float("nan"))
        rows.append(row)
    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parents[1] / "data" / "agg_10seed_mean_sd.csv"
    out.to_csv(path, index=False)
    return out


def write_run_manifest(runs: pd.DataFrame) -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "run_manifest.csv"
    runs.to_csv(path, index=False)
    return path
