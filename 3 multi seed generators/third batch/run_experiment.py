#!/usr/bin/env python3
"""CLI entry point for multi-seed Batch 1 experiments.

Folder layout:
  1st batch/classification/   ← 9 classification datasets
  1st batch/regression/       ← 6 regression datasets

Examples
--------
python run_experiment.py --validate-only
python run_experiment.py --all
python run_experiment.py --task classification
python run_experiment.py --task regression
python run_experiment.py --dataset cancer --generator CTGAN --seed 55
python run_experiment.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Keep CUDA worker logs to job status only (SDV/sklearn emit noisy FutureWarnings).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Cap CPU side to 2 threads per worker (6 GPUs stay busy).
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_k, "2")
try:
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
except Exception:
    pass

BATCH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BATCH_ROOT))

from src.aggregate import aggregate  # noqa: E402
from src.config_loader import generator_names, load_datasets, load_seeds  # noqa: E402
from src.experiment import run_experiment, run_one  # noqa: E402
from src.paths import REPORTS_DIR, ensure_task_dirs  # noqa: E402


def validate() -> dict:
    ds = load_datasets()
    gens = generator_names()
    seeds = load_seeds()["generator_seeds"]
    clf = [d["id"] for d in ds if d["task"] == "classification"]
    reg = [d["id"] for d in ds if d["task"] == "regression"]
    ensure_task_dirs("classification")
    ensure_task_dirs("regression")
    report = {
        "n_datasets": len(ds),
        "classification_datasets": clf,
        "n_classification": len(clf),
        "regression_datasets": reg,
        "n_regression": len(reg),
        "n_generators": len(gens),
        "generators": gens,
        "seeds": seeds,
        "expected_runs": len(ds) * len(gens) * len(seeds),
        "expected_classification_runs": len(clf) * len(gens) * len(seeds),
        "expected_regression_runs": len(reg) * len(gens) * len(seeds),
        "split_seed": load_seeds()["split_seed"],
        "folder_layout": {
            "classification": str(BATCH_ROOT / "classification"),
            "regression": str(BATCH_ROOT / "regression"),
        },
        "ok": len(ds) == 15
        and len(clf) == 9
        and len(reg) == 6
        and len(gens) == 8
        and seeds == [55, 155, 255, 355],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "validation_preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(
            "Validation failed: expected 15 datasets (9 clf + 6 reg), 8 generators, seeds [55,155,255,355]"
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed Batch 1 experiment runner")
    p.add_argument("--all", action="store_true", help="Run all 15×8×3 jobs (resumable)")
    p.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=None,
        help="Restrict to classification or regression datasets",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset id; repeat or comma-separate. Used by cuda0.sh–cuda3.sh",
    )
    p.add_argument(
        "--generator",
        action="append",
        default=None,
        help="Generator name; repeat or comma-separate (e.g. CTABGAN,TabDDPM)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Re-run even if complete")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Do not aggregate at the end (for parallel GPU workers)",
    )
    p.add_argument("--smoke", action="store_true", help="cancer × CTGAN × 3 seeds (classification)")
    args = p.parse_args()

    if args.validate_only:
        validate()
        return
    if args.aggregate_only:
        out = aggregate(task=args.task)
        print(
            f"Aggregated {len(out)} rows -> classification|regression/results/ (Excel only)"
        )
        return
    if args.smoke:
        validate()
        status = run_one("cancer", "CTGAN", 55, force=args.force)
        print("Smoke status:", status)
        if status == "FAILED":
            raise SystemExit(1)
        for s in (155, 255, 355):
            print(run_one("cancer", "CTGAN", s, force=args.force))
        agg = aggregate(task="classification")
        print(
            agg[(agg["dataset"] == "Cancer") & (agg["generator"] == "CTGAN")]
            .head(20)
            .to_string(index=False)
        )
        return

    validate()
    datasets = None
    if args.dataset:
        datasets = []
        for item in args.dataset:
            datasets.extend(x.strip() for x in item.split(",") if x.strip())
    generators = None
    if args.generator:
        generators = []
        for item in args.generator:
            generators.extend(x.strip() for x in item.split(",") if x.strip())
    seeds = [args.seed] if args.seed is not None else None
    if not args.all and args.dataset is None and args.task is None:
        print("Specify --all, --task, --smoke, or --dataset/--generator/--seed")
        raise SystemExit(2)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"CUDA_VISIBLE_DEVICES={gpu}  datasets={datasets or 'ALL'}")
    run_experiment(
        datasets=datasets,
        generators=generators,
        seeds=seeds,
        force=args.force,
        task=args.task,
    )
    if not args.skip_aggregate:
        aggregate(task=args.task)


if __name__ == "__main__":
    main()
