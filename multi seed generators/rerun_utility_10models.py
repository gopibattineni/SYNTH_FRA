#!/usr/bin/env python3
"""Re-run utility evaluation with 10 classifiers + 10 regressors.

IMPORTANT: Original multi-seed runs did NOT save synthetic CSVs (only metrics.xlsx).
This script regenerates synth when synth.csv is missing, then persists synth.csv
so later utility-only passes can skip generator training.

Preserves existing fidelity / privacy / compute rows in metrics.xlsx when present;
replaces Utility rows; writes per_model_utility.csv.

Examples
--------
  # Smoke: Cancer + Concrete, one generator seed, one generator
  python rerun_utility_10models.py --smoke

  # Full protocol (all batches / datasets / generators / seeds)
  python rerun_utility_10models.py --all

  # Targeted
  python rerun_utility_10models.py --batch "1st batch" --dataset cancer --seed 42 --generator CTGAN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from shared.utility import (  # noqa: E402
    clear_trtr_cache,
    compute_utility,
    expected_tstr_fit_counts,
    load_utility_model_names,
    per_model_to_long,
    utility_to_rows,
)

BATCHES = {
    "1st batch": [42, 123, 2024],
    "second batch": [68, 91, 2025],
    "third batch": [55, 155, 255, 355],
}


def _import_batch(batch_name: str):
    batch_root = ROOT / batch_name
    # Prefer this batch on sys.path
    sys.path = [p for p in sys.path if Path(p).resolve() != batch_root.resolve()]
    sys.path.insert(0, str(batch_root))
    # Clear cached batch modules so config/paths resolve to this batch
    doomed = [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]
    for k in doomed:
        del sys.modules[k]
    from src.config_loader import dataset_by_id, generator_names, load_datasets, load_seeds
    from src.datasets import load_and_split
    from src.generators import generate_synthetic
    from src.metrics.fidelity import compute_fidelity, fidelity_to_rows
    from src.metrics.privacy import compute_privacy, privacy_to_rows
    from src.paths import ensure_task_dirs
    from src.table_io import read_results, write_results

    return {
        "batch_root": batch_root,
        "dataset_by_id": dataset_by_id,
        "generator_names": generator_names,
        "load_datasets": load_datasets,
        "load_seeds": load_seeds,
        "load_and_split": load_and_split,
        "generate_synthetic": generate_synthetic,
        "compute_fidelity": compute_fidelity,
        "fidelity_to_rows": fidelity_to_rows,
        "compute_privacy": compute_privacy,
        "privacy_to_rows": privacy_to_rows,
        "ensure_task_dirs": ensure_task_dirs,
        "read_results": read_results,
        "write_results": write_results,
    }


def _run_dir(api, seed: int, dataset_id: str, generator: str) -> Path:
    cfg = api["dataset_by_id"](dataset_id)
    paths = api["ensure_task_dirs"](cfg["task"])
    d = paths["raw"] / f"seed_{seed}" / dataset_id / generator
    d.mkdir(parents=True, exist_ok=True)
    return d


def rerun_one(
    api,
    *,
    dataset_id: str,
    generator: str,
    seed: int,
    force_regen_synth: bool = False,
    recompute_fid_priv: bool = False,
) -> str:
    seeds_cfg = api["load_seeds"]()
    split_seed = int(seeds_cfg["split_seed"])
    test_size = float(seeds_cfg["test_size"])
    util_seeds = list(seeds_cfg.get("utility_model_seeds") or range(42, 52))

    cfg = api["dataset_by_id"](dataset_id)
    task = cfg["task"]
    split = api["load_and_split"](cfg, test_size=test_size, split_seed=split_seed, use_cache=True)
    train, test, cfg = split["train"], split["test"], split["cfg"]
    target = cfg["target"]
    n_samples = int(cfg.get("n_samples", 1000))

    run_dir = _run_dir(api, seed, dataset_id, generator)
    synth_path = run_dir / "synth.csv"
    metrics_path = run_dir / "metrics.xlsx"

    t0 = time.perf_counter()
    if synth_path.exists() and not force_regen_synth:
        synth = __import__("pandas").read_csv(synth_path)
        training_time = 0.0
        synth_source = "loaded_synth.csv"
    else:
        t_gen = time.perf_counter()
        synth = api["generate_synthetic"](generator, train, target, n_samples, seed, task)
        training_time = time.perf_counter() - t_gen
        synth.to_csv(synth_path, index=False)
        synth_source = "regenerated"

    util, per_model = compute_utility(
        train,
        test,
        synth,
        target,
        task,
        util_seeds,
        dataset_id=dataset_id,
        return_per_model=True,
    )

    base_meta = {
        "dataset": cfg["name"],
        "dataset_id": cfg["id"],
        "generator": generator,
        "seed": seed,
        "split_seed": split_seed,
        "n_synthetic_samples": n_samples,
        "n_train": len(train),
        "n_test": len(test),
        "task": task,
        "training_time_seconds": training_time,
        "generation_time_seconds": 0.0,
        "evaluation_time_seconds": time.perf_counter() - t0,
        "total_time_seconds": time.perf_counter() - t0,
    }

    per_long = per_model_to_long(per_model, base_meta)
    per_long.to_csv(run_dir / "per_model_utility.csv", index=False)

    # Build / merge metrics workbook
    import pandas as pd

    util_rows = pd.DataFrame(utility_to_rows(util, base_meta))
    other_rows = []
    if metrics_path.exists() and not recompute_fid_priv:
        try:
            old = api["read_results"](metrics_path)
            keep = old[old["metric_category"].isin(["Fidelity", "Privacy", "Compute"])].copy()
            other_rows.append(keep)
        except Exception:
            recompute_fid_priv = True
    if recompute_fid_priv or not metrics_path.exists():
        fid = api["compute_fidelity"](train, synth)
        priv = api["compute_privacy"](train, synth)
        other_rows.append(pd.DataFrame(api["fidelity_to_rows"](fid, base_meta)))
        other_rows.append(pd.DataFrame(api["privacy_to_rows"](priv, base_meta)))
        for k in (
            "training_time_seconds",
            "generation_time_seconds",
            "evaluation_time_seconds",
            "total_time_seconds",
        ):
            other_rows.append(
                pd.DataFrame(
                    [
                        {
                            **base_meta,
                            "metric_category": "Compute",
                            "metric_name": k,
                            "metric_value": base_meta[k],
                        }
                    ]
                )
            )

    parts = other_rows + [util_rows]
    out = pd.concat(parts, ignore_index=True)
    # Drop stale Utility rows if keep had none; util_rows is authoritative
    non_util = out[out["metric_category"] != "Utility"]
    out = pd.concat([non_util, util_rows], ignore_index=True)
    api["write_results"](metrics_path, out)

    meta_path = run_dir / "run_meta.json"
    meta = {
        **base_meta,
        "status": "SUCCESS",
        "utility_models": load_utility_model_names(),
        "synth_source": synth_source,
        "utility_n_models": 10,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            if "split_meta" in prev:
                meta["split_meta"] = prev["split_meta"]
        except Exception:
            pass
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return synth_source


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Cancer+Concrete, seed 42, CTGAN only")
    p.add_argument("--all", action="store_true", help="All batches / datasets / generators / seeds")
    p.add_argument("--batch", type=str, default="")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--generator", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force-regen-synth", action="store_true")
    args = p.parse_args()

    counts = expected_tstr_fit_counts()
    print("Expected TSTR fits (full 10-seed protocol):")
    print("  classification:", counts["formula_classification"], "=>", counts["classification_tstr_fits"])
    print("  regression:    ", counts["formula_regression"], "=>", counts["regression_tstr_fits"])
    print("Model lists:", json.dumps(load_utility_model_names(), indent=2))
    print(
        "NOTE: synth.csv was not saved in the original runs; missing files are regenerated "
        "and then cached as synth.csv."
    )

    jobs: list[tuple[str, str, str, int]] = []
    if args.smoke:
        jobs = [
            ("1st batch", "cancer", "CTGAN", 42),
            ("1st batch", "concrete", "CTGAN", 42),
        ]
    elif args.all:
        for batch, seeds in BATCHES.items():
            api = _import_batch(batch)
            clear_trtr_cache()
            gens = api["generator_names"]()
            for d in api["load_datasets"]():
                for g in gens:
                    for s in seeds:
                        jobs.append((batch, d["id"], g, s))
    else:
        if not args.batch or not args.dataset or args.seed is None:
            raise SystemExit("Provide --smoke, --all, or --batch/--dataset/--seed [--generator]")
        gens = [args.generator] if args.generator else None
        api = _import_batch(args.batch)
        if gens is None:
            gens = api["generator_names"]()
        for g in gens:
            jobs.append((args.batch, args.dataset, g, args.seed))

    # Group by batch for efficient imports
    by_batch: dict[str, list[tuple[str, str, int]]] = {}
    for batch, ds, gen, seed in jobs:
        by_batch.setdefault(batch, []).append((ds, gen, seed))

    ok = fail = 0
    for batch, items in by_batch.items():
        print(f"\n=== {batch} ({len(items)} jobs) ===", flush=True)
        api = _import_batch(batch)
        clear_trtr_cache()
        for ds, gen, seed in items:
            tag = f"{batch} | {ds} | {gen} | seed={seed}"
            try:
                src = rerun_one(
                    api,
                    dataset_id=ds,
                    generator=gen,
                    seed=seed,
                    force_regen_synth=args.force_regen_synth,
                )
                # Verify per-model file
                per_path = _run_dir(api, seed, ds, gen) / "per_model_utility.csv"
                import pandas as pd

                pdf = pd.read_csv(per_path)
                n_models = pdf["model"].nunique()
                paths = sorted(pdf["path"].unique())
                print(f"OK  {tag}  synth={src}  models={n_models}  paths={paths}  rows={len(pdf)}", flush=True)
                if n_models != 10:
                    raise RuntimeError(f"Expected 10 models, got {n_models}")
                ok += 1
            except Exception as e:
                fail += 1
                print(f"FAIL {tag}: {e}\n{traceback.format_exc()}", flush=True)

    print(f"\nDone. ok={ok} fail={fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
