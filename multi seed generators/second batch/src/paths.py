"""Path helpers for multi-seed Batch 1 (isolated from the reference pipeline).

Layout:
  multi seed generators/1st batch/
    classification/   ← 9 classification datasets
    regression/       ← 6 regression datasets
    config/, src/     ← shared
"""
from __future__ import annotations

from pathlib import Path

BATCH_ROOT = Path(__file__).resolve().parents[1]
SYNTH_ROOT = BATCH_ROOT.parents[1]  # .../SYNTH
REPO_ROOT = SYNTH_ROOT.parent  # .../SYNTH_BENCHMARK

CONFIG_DIR = BATCH_ROOT / "config"
CLASSIFICATION_ROOT = BATCH_ROOT / "classification"
REGRESSION_ROOT = BATCH_ROOT / "regression"

# Shared defaults (overridden per-task via task_paths)
CACHE_DIR = BATCH_ROOT / "cache"
SPLITS_DIR = CACHE_DIR / "splits"

# Reference assets (read-only reuse)
DATASETS_DIR = SYNTH_ROOT / "Datasets"
CTAB_ROOT = SYNTH_ROOT / "Generators" / "Other GANS" / "CTAB-GAN-Plus"
DIFFUSION_MODULE = SYNTH_ROOT / "Generators" / "Diffusion GANs" / "diffusion_generators.py"
HIVE_DATASETS_JSON = (
    SYNTH_ROOT
    / "Generators"
    / "Experiment with utility data leak"
    / "python_scripts"
    / "hive"
    / "datasets.json"
)


def task_root(task: str) -> Path:
    t = str(task).strip().lower()
    if t.startswith("class"):
        return CLASSIFICATION_ROOT
    if t.startswith("reg"):
        return REGRESSION_ROOT
    raise ValueError(f"Unknown task: {task!r} (expected classification|regression)")


def task_paths(task: str) -> dict[str, Path]:
    """Return result/figure/report/log directories for a task branch."""
    root = task_root(task)
    results = root / "results"
    return {
        "root": root,
        "results": results,
        "raw": results / "raw",
        "aggregated": results / "aggregated",
        "logs": results / "logs",
        "figures": root / "figures",
        "reports": root / "reports",
        "cache": root / "cache",
        "splits": root / "cache" / "splits",
    }


def ensure_task_dirs(task: str) -> dict[str, Path]:
    paths = task_paths(task)
    for key in ("raw", "aggregated", "logs", "figures", "reports", "splits"):
        paths[key].mkdir(parents=True, exist_ok=True)
    # seed subdirs
    for seed in (68, 91, 2025):
        (paths["raw"] / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    return paths


# Back-compat aliases used by older imports (point to classification by default
# for validate-only; experiment.py should use task_paths explicitly).
RESULTS_DIR = CLASSIFICATION_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
AGG_DIR = RESULTS_DIR / "aggregated"
LOG_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = CLASSIFICATION_ROOT / "figures"
REPORTS_DIR = BATCH_ROOT / "reports"
