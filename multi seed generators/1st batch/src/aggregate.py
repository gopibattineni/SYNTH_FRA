"""Aggregate seed-level results into mean ± SD (ddof=1), per task folder."""
from __future__ import annotations

import fcntl
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .excel_io import write_aggregated_excel
from .paths import ensure_task_dirs, task_paths
from .table_io import drop_csv, read_results, results_exist, write_results


SEEDS = [42, 123, 2024]
SD_EPS = 1e-12

TRTR_METRICS_CLS = (
    "Accuracy_TRTR",
    "F1_TRTR",
    "Precision_TRTR",
    "Recall_TRTR",
    "ROC_AUC_TRTR",
)
TRTR_METRICS_REG = ("R2_TRTR", "RMSE_TRTR", "MAE_TRTR")


def _mean_sd_text(mean: float, finite: np.ndarray) -> tuple[float, str]:
    """Sample SD from available seed values. Never emit ±0."""
    n = int(len(finite))
    if n == 0 or not np.isfinite(mean):
        return np.nan, ""
    if n == 1:
        return np.nan, f"{mean:.6g}"
    sd = float(np.std(finite, ddof=1))
    if not np.isfinite(sd) or sd <= SD_EPS:
        # Identical seeds → no generator variance; leave blank until model-seed fill.
        return np.nan, f"{mean:.6g}"
    return sd, f"{mean:.6g} ± {sd:.6g}"


def _is_zero_sd(val) -> bool:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x) and x <= SD_EPS


def load_raw(task: str) -> pd.DataFrame:
    """Load every per-run metrics workbook (source of truth), then refresh the combined raw file."""
    paths = ensure_task_dirs(task)
    frames = []
    for p in sorted(paths["raw"].glob("seed_*/*/*/metrics.xlsx")) + sorted(
        paths["raw"].glob("seed_*/*/*/metrics.csv")
    ):
        if p.stat().st_size <= 0:
            continue
        try:
            frames.append(read_results(p))
        except Exception:
            continue
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        dedupe_cols = [
            c
            for c in ("dataset_id", "dataset", "generator", "seed", "metric_category", "metric_name")
            if c in raw.columns
        ]
        if dedupe_cols:
            raw = raw.drop_duplicates(subset=dedupe_cols, keep="last")
        write_results(paths["raw"] / "multi_seed_raw_results.xlsx", raw)
        return raw
    global_raw = paths["raw"] / "multi_seed_raw_results.xlsx"
    if results_exist(global_raw):
        return read_results(global_raw)
    return pd.DataFrame()


def _aggregate_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    key_cols = ["dataset", "generator", "metric_category", "metric_name"]
    if "dataset" not in raw.columns and "dataset_id" in raw.columns:
        raw = raw.rename(columns={"dataset_id": "dataset"})
    rows = []
    for keys, g in raw.groupby(key_cols, dropna=False):
        dataset, generator, cat, name = keys
        by_seed = {}
        g = g.copy()
        if "seed" in g.columns:
            g["seed"] = pd.to_numeric(g["seed"], errors="coerce")
        for seed in SEEDS:
            sub = g.loc[g["seed"] == seed, "metric_value"]
            by_seed[seed] = float(sub.iloc[0]) if len(sub) else np.nan
        vals = np.array([by_seed[s] for s in SEEDS], dtype=float)
        finite = vals[np.isfinite(vals)]
        n = int(len(finite))
        mean = float(np.mean(finite)) if n else np.nan
        sd, mean_sd = _mean_sd_text(mean, finite)
        rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "metric_category": cat,
                "metric_name": name,
                "seed_42": by_seed[42],
                "seed_123": by_seed[123],
                "seed_2024": by_seed[2024],
                "mean": mean,
                "sd": sd,
                "mean_sd": mean_sd,
                "n_seeds": n,
            }
        )
    return pd.DataFrame(rows)


def _write_sd(out: pd.DataFrame, idx, sd: float) -> None:
    if not np.isfinite(sd) or sd <= SD_EPS:
        return
    mean = out.at[idx, "mean"]
    out.at[idx, "sd"] = float(sd)
    if np.isfinite(mean):
        out.at[idx, "mean_sd"] = f"{mean:.6g} ± {sd:.6g}"


def _utility_model_seeds() -> list[int]:
    from .config_loader import load_seeds

    seeds_cfg = load_seeds()
    return list(seeds_cfg.get("utility_model_seeds") or range(42, 52))


def _fill_trtr_model_sd(out: pd.DataFrame, task: str) -> pd.DataFrame:
    """TRTR does not vary with generator seed (fixed split). Fill SD from model-init seeds 42–51."""
    if out.empty:
        return out
    from .config_loader import load_datasets, load_seeds
    from .datasets import load_and_split
    from .metrics.utility import CLASSIFIERS, evaluate_classification, evaluate_regression

    seeds_cfg = load_seeds()
    split_seed = int(seeds_cfg["split_seed"])
    test_size = float(seeds_cfg["test_size"])
    model_seeds = _utility_model_seeds()
    by_name = {d["name"]: d for d in load_datasets() if d["task"] == task}

    if task == "classification":
        trtr_names = TRTR_METRICS_CLS
        bases = ("Accuracy", "F1", "Precision", "Recall", "ROC_AUC")
        light = {
            "LogReg": CLASSIFIERS["LogReg"],
            "RandomForest": CLASSIFIERS["RandomForest"],
            "DecisionTree": CLASSIFIERS["DecisionTree"],
        }
    else:
        trtr_names = TRTR_METRICS_REG
        bases = ("R2", "RMSE", "MAE")
        light = None

    for ds in out["dataset"].unique():
        cfg = by_name.get(ds)
        if cfg is None:
            continue
        need = out[(out["dataset"] == ds) & (out["metric_name"].isin(trtr_names))]
        if need.empty:
            continue
        if not ((need["sd"].fillna(0) <= SD_EPS) | need["sd"].isna()).any():
            continue
        try:
            split = load_and_split(cfg, test_size=test_size, split_seed=split_seed, use_cache=True)
            if task == "classification":
                _means, sds = evaluate_classification(
                    split["train"],
                    split["test"],
                    cfg["target"],
                    model_seeds,
                    models=light,
                    return_sd=True,
                )
            else:
                _means, sds = evaluate_regression(
                    split["train"], split["test"], cfg["target"], model_seeds, return_sd=True
                )
        except Exception as exc:
            print(f"TRTR SD fill skipped for {ds}: {exc}", flush=True)
            continue
        for metric in bases:
            sd = sds.get(metric, np.nan)
            if not np.isfinite(sd) or sd <= SD_EPS:
                continue
            mask = (out["dataset"] == ds) & (out["metric_name"] == f"{metric}_TRTR")
            for idx in out.index[mask]:
                prev = out.at[idx, "sd"]
                if np.isfinite(prev) and prev > SD_EPS:
                    continue
                _write_sd(out, idx, float(sd))
    return out


def _fill_tstr_and_gap_sd(out: pd.DataFrame, task: str) -> pd.DataFrame:
    """Fill TSTR/Gap when generator seeds are identical (e.g. GaussianCopula)."""
    if out.empty:
        return out
    from .config_loader import load_datasets, load_seeds
    from .datasets import load_and_split
    from .generators import generate_synthetic
    from .metrics.utility import (
        CLASSIFIERS,
        evaluate_classification,
        evaluate_regression,
        _snap_classification_target,
    )

    seeds_cfg = load_seeds()
    split_seed = int(seeds_cfg["split_seed"])
    test_size = float(seeds_cfg["test_size"])
    model_seeds = _utility_model_seeds()
    by_name = {d["name"]: d for d in load_datasets() if d["task"] == task}

    if task == "classification":
        bases = ("Accuracy", "F1", "Precision", "Recall", "ROC_AUC")
        tstr_names = {f"{m}_TSTR" for m in bases}
        gap_suffix = {m: f"{m}_Gap" for m in bases}
        light = {
            "LogReg": CLASSIFIERS["LogReg"],
            "RandomForest": CLASSIFIERS["RandomForest"],
            "DecisionTree": CLASSIFIERS["DecisionTree"],
        }
    else:
        bases = ("R2", "RMSE", "MAE")
        tstr_names = {f"{m}_TSTR" for m in bases}
        gap_suffix = {"R2": "R2_Gap", "RMSE": "RMSE_Increase", "MAE": "MAE_Increase"}
        light = None

    for (ds, gen), g in out.groupby(["dataset", "generator"]):
        need = g[g["metric_name"].isin(tstr_names)]
        if need.empty or not ((need["sd"].fillna(0) <= SD_EPS) | need["sd"].isna()).any():
            continue
        # Only re-synthesize for deterministic generators (seed-invariant synth).
        if gen != "GaussianCopula":
            continue
        cfg = by_name.get(ds)
        if cfg is None:
            continue
        try:
            split = load_and_split(cfg, test_size=test_size, split_seed=split_seed, use_cache=True)
            n_samples = int(cfg.get("n_samples", 1000))
            synth = generate_synthetic(gen, split["train"], cfg["target"], n_samples, 42, task)
            synth_u = synth.copy()
            train = split["train"]
            for c in train.columns:
                if c not in synth_u.columns:
                    synth_u[c] = train[c].iloc[0]
            synth_u = synth_u[train.columns]
            if task == "classification":
                synth_u = _snap_classification_target(synth_u, train, cfg["target"])
                _means, sds = evaluate_classification(
                    synth_u, split["test"], cfg["target"], model_seeds, models=light, return_sd=True
                )
            else:
                _means, sds = evaluate_regression(
                    synth_u, split["test"], cfg["target"], model_seeds, return_sd=True
                )
        except Exception as exc:
            print(f"TSTR SD fill skipped for {ds} {gen}: {exc}", flush=True)
            continue
        ds_mask = (out["dataset"] == ds) & (out["generator"] == gen)
        for metric in bases:
            sd = sds.get(metric, np.nan)
            if not np.isfinite(sd) or sd <= SD_EPS:
                continue
            rows = out.index[ds_mask & (out["metric_name"] == f"{metric}_TSTR")]
            for idx in rows:
                prev = out.at[idx, "sd"]
                if np.isfinite(prev) and prev > SD_EPS:
                    continue
                _write_sd(out, idx, float(sd))

    # Proxy fallback: if GaussianCopula TSTR SD still missing, reuse TRTR model-seed SD.
    for (ds, gen), g in out.groupby(["dataset", "generator"]):
        if gen != "GaussianCopula":
            continue
        for metric in bases:
            tstr_mask = (
                (out["dataset"] == ds)
                & (out["generator"] == gen)
                & (out["metric_name"] == f"{metric}_TSTR")
            )
            trtr_mask = (
                (out["dataset"] == ds)
                & (out["generator"] == gen)
                & (out["metric_name"] == f"{metric}_TRTR")
            )
            tstr_idxs = out.index[tstr_mask]
            trtr_idxs = out.index[trtr_mask]
            if not len(tstr_idxs) or not len(trtr_idxs):
                continue
            for idx in tstr_idxs:
                prev = out.at[idx, "sd"]
                if np.isfinite(prev) and prev > SD_EPS:
                    continue
                proxy = out.at[trtr_idxs[0], "sd"]
                if np.isfinite(proxy) and proxy > SD_EPS:
                    _write_sd(out, idx, float(proxy))

    # Gap / Increase SD from TRTR + TSTR SDs (propagate after fills).
    for (ds, gen), g in out.groupby(["dataset", "generator"]):
        idx_by: dict[str, object] = {}
        sd_by: dict[str, float] = {}
        for metric in bases:
            for name in (f"{metric}_TRTR", f"{metric}_TSTR", gap_suffix[metric]):
                mask = (
                    (out["dataset"] == ds)
                    & (out["generator"] == gen)
                    & (out["metric_name"] == name)
                )
                idxs = out.index[mask]
                if len(idxs):
                    idx_by[name] = idxs[0]
                    sd_by[name] = out.at[idxs[0], "sd"]
        for metric in bases:
            gap_name = gap_suffix[metric]
            if gap_name not in idx_by:
                continue
            prev = sd_by.get(gap_name)
            if np.isfinite(prev) and prev > SD_EPS:
                continue
            parts = []
            for name in (f"{metric}_TRTR", f"{metric}_TSTR"):
                val = sd_by.get(name, np.nan)
                if np.isfinite(val) and val > SD_EPS:
                    parts.append(float(val))
            if not parts:
                continue
            new_sd = float(np.sqrt(sum(p * p for p in parts)))
            _write_sd(out, idx_by[gap_name], new_sd)
    return out


def _sanitize_zero_sd(out: pd.DataFrame) -> pd.DataFrame:
    """Guarantee no stored/displayed SD is exactly zero (use NaN + mean-only text)."""
    if out.empty:
        return out
    for idx, row in out.iterrows():
        sd = row.get("sd")
        mean = row.get("mean")
        mean_sd = str(row.get("mean_sd") or "")
        zeroish = _is_zero_sd(sd)
        if "±" in mean_sd:
            try:
                right = mean_sd.split("±", 1)[1].strip().split()[0]
                if float(right) <= SD_EPS:
                    zeroish = True
            except Exception:
                pass
        if not zeroish:
            continue
        out.at[idx, "sd"] = np.nan
        if np.isfinite(mean):
            out.at[idx, "mean_sd"] = f"{float(mean):.6g}"
        else:
            out.at[idx, "mean_sd"] = ""
    return out


def aggregate(task: str | None = None) -> pd.DataFrame:
    """Aggregate one task or both. Writes Excel only (no CSV)."""
    tasks = [task] if task else ["classification", "regression"]
    outs = []
    for t in tasks:
        paths = ensure_task_dirs(t)
        lock_path = paths["results"] / "excel.lock"
        lock_fh = lock_path.open("a+")
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            raw = load_raw(t)
            out = _aggregate_frame(raw)
            if out.empty:
                continue
            out["task"] = t
            out = _fill_trtr_model_sd(out, t)
            out = _fill_tstr_and_gap_sd(out, t)
            out = _sanitize_zero_sd(out)
            agg = paths["aggregated"]
            write_aggregated_excel(agg, out)
            for name in (
                "all_metrics_mean_sd",
                "fidelity_mean_sd",
                "utility_mean_sd",
                "privacy_mean_sd",
                "compute_mean_sd",
            ):
                drop_csv(agg / f"{name}.csv")
            for p in (
                paths["logs"] / "experiment_log.xlsx",
                paths["raw"] / "multi_seed_raw_results.xlsx",
            ):
                if results_exist(p):
                    write_results(p, read_results(p))
            combined = agg / "multi_seed_results.xlsx"
            if combined.exists():
                shutil.copy2(combined, paths["results"] / "multi_seed_results.xlsx")
            outs.append(out)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            lock_fh.close()
            for csv in paths["results"].rglob("*.csv"):
                try:
                    write_results(csv, read_results(csv))
                except Exception:
                    if csv.exists():
                        csv.unlink()
    if not outs:
        return pd.DataFrame()
    return pd.concat(outs, ignore_index=True)
