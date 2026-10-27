#!/usr/bin/env python3
"""Merge 1st + second + third batch seed results into one 10-seed mean±SD Excel.

Seeds (10):
  Batch 1: 42, 123, 2024
  Batch 2: 68, 91, 2025
  Batch 3: 55, 155, 255, 355

Writes:
  merged 10 seeds/classification/results/multi_seed_results.xlsx
  merged 10 seeds/regression/results/multi_seed_results.xlsx

Never emits SD = 0. Identical seed values get model-seed SD fills for TRTR/TSTR/Gap
(GaussianCopula), otherwise mean-only (blank SD).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "merged 10 seeds"
FIRST = ROOT / "1st batch"
sys.path.insert(0, str(FIRST))

from src.aggregate import (  # noqa: E402
    _fill_trtr_model_sd,
    _fill_tstr_and_gap_sd,
    _sanitize_zero_sd,
)
from src.excel_io import NUMERIC_HEADERS, write_aggregated_excel  # noqa: E402

SEEDS = [42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]
SEED_COLS = [f"seed_{s}" for s in SEEDS]
SD_EPS = 1e-12

BATCHES = [
    ("1st batch", [42, 123, 2024]),
    ("second batch", [68, 91, 2025]),
    ("third batch", [55, 155, 255, 355]),
]

KEY = ["dataset", "generator", "metric_category", "metric_name"]


def _mean_sd_text(mean: float, finite: np.ndarray) -> tuple[float, str]:
    """Sample SD (ddof=1). Never emit ±0."""
    n = int(len(finite))
    if n == 0 or not np.isfinite(mean):
        return np.nan, ""
    if n == 1:
        return np.nan, f"{mean:.6g}"
    sd = float(np.std(finite, ddof=1))
    if not np.isfinite(sd) or sd <= SD_EPS:
        return np.nan, f"{mean:.6g}"
    return sd, f"{mean:.6g} ± {sd:.6g}"


def _read_all_metrics(path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["All metrics"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    data = [dict(zip(header, r)) for r in rows if r and r[0] is not None]
    wb.close()
    return pd.DataFrame(data)


def merge_task(task: str) -> pd.DataFrame:
    frames = []
    for batch_name, seeds in BATCHES:
        path = ROOT / batch_name / task / "results" / "multi_seed_results.xlsx"
        if not path.exists():
            raise FileNotFoundError(path)
        df = _read_all_metrics(path)
        keep = KEY + [f"seed_{s}" for s in seeds]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}")
        frames.append(df[keep].copy())

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on=KEY, how="outer")

    rows = []
    for _, r in merged.iterrows():
        vals = []
        out = {k: r[k] for k in KEY}
        for s in SEEDS:
            col = f"seed_{s}"
            v = r.get(col, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            if not np.isfinite(v):
                v = np.nan
            out[col] = v
            vals.append(v)
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        n = int(len(finite))
        mean = float(np.mean(finite)) if n else np.nan
        sd, mean_sd = _mean_sd_text(mean, finite)
        out["mean"] = mean
        out["sd"] = sd
        out["mean_sd"] = mean_sd
        out["n_seeds"] = n
        out["task"] = task
        rows.append(out)

    out_df = pd.DataFrame(rows)
    # Model-seed SD fills for TRTR / GaussianCopula TSTR+Gap; never leave literal 0.
    out_df = _fill_trtr_model_sd(out_df, task)
    out_df = _fill_tstr_and_gap_sd(out_df, task)
    out_df = _sanitize_zero_sd(out_df)

    cols = KEY + SEED_COLS + ["mean", "sd", "mean_sd", "n_seeds", "task"]
    return out_df[cols].sort_values(KEY).reset_index(drop=True)


def write_task(task: str, df: pd.DataFrame) -> Path:
    agg = OUT_ROOT / task / "results" / "aggregated"
    results = OUT_ROOT / task / "results"
    agg.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    for col in SEED_COLS:
        NUMERIC_HEADERS.add(col)

    write_aggregated_excel(agg, df)
    combined = agg / "multi_seed_results.xlsx"
    dest = results / "multi_seed_results.xlsx"
    dest.write_bytes(combined.read_bytes())
    return dest


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in ("classification", "regression"):
        df = merge_task(task)
        path = write_task(task, df)
        n_pairs = df.groupby(["dataset", "generator"]).ngroups
        n10 = int((df["n_seeds"] == 10).sum())
        n_rows = len(df)
        sd = pd.to_numeric(df["sd"], errors="coerce")
        n_zero = int((sd.notna() & (sd.abs() <= SD_EPS)).sum())
        summary[task] = {
            "path": str(path),
            "rows": n_rows,
            "pairs": n_pairs,
            "rows_with_10_seeds": n10,
            "min_n_seeds": int(df["n_seeds"].min()) if n_rows else 0,
            "max_n_seeds": int(df["n_seeds"].max()) if n_rows else 0,
            "zero_sd_rows": n_zero,
        }
        print(
            f"{task}: pairs={n_pairs} rows={n_rows} "
            f"n_seeds=[{summary[task]['min_n_seeds']},{summary[task]['max_n_seeds']}] "
            f"rows_n10={n10} zero_sd={n_zero} -> {path}"
        )
        if n_zero:
            raise SystemExit(f"{task}: still has {n_zero} zero-SD rows")
    print("done", summary)


if __name__ == "__main__":
    main()
