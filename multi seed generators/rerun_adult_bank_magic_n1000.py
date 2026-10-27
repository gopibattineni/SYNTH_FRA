#!/usr/bin/env python3
"""Regenerate Adult / Bank / MAGIC at uniform N=1000 across all 10 seeds.

Resumes from existing metrics.xlsx; keeps GPUs 0,1,2,3,5 busy until
3 datasets × 8 generators × 10 seeds = 240 runs finish.
Tracks in-flight jobs so the same (batch,seed,dataset,generator) is never
launched twice.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PYTHON = REPO / ".venv" / "bin" / "python"

BATCHES = [
    ("1st batch", [42, 123, 2024]),
    ("second batch", [68, 91, 2025]),
    ("third batch", [55, 155, 255, 355]),
]
DATASETS = ["adult", "bank", "magic"]
GENERATORS = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "CTABGAN",
    "WGAN_GP",
    "TabDDPM",
    "ForestDiffusion",
]
GPUS = [0, 1, 2, 3, 5]


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}", flush=True)


def metrics_path(batch: Path, seed: int, dataset: str, generator: str) -> Path:
    return (
        batch
        / "classification"
        / "results"
        / "raw"
        / f"seed_{seed}"
        / dataset
        / generator
        / "metrics.xlsx"
    )


def pending_jobs(
    inflight: set[tuple[str, int, str, str]],
    fail_counts: dict[tuple[str, int, str, str], int],
    max_fails: int = 3,
) -> list[tuple[str, int, str, str]]:
    jobs: list[tuple[str, int, str, str]] = []
    for batch_name, seeds in BATCHES:
        batch = ROOT / batch_name
        for seed in seeds:
            for ds in DATASETS:
                for gen in GENERATORS:
                    key = (batch_name, seed, ds, gen)
                    if key in inflight:
                        continue
                    if fail_counts.get(key, 0) >= max_fails:
                        continue
                    if not metrics_path(batch, seed, ds, gen).is_file():
                        jobs.append(key)
    return jobs


def completed_count() -> int:
    n = 0
    for batch_name, seeds in BATCHES:
        batch = ROOT / batch_name
        for seed in seeds:
            for ds in DATASETS:
                for gen in GENERATORS:
                    if metrics_path(batch, seed, ds, gen).is_file():
                        n += 1
    return n


def gpu_free(gpu: int) -> bool:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
                "-i",
                str(gpu),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() == ""
    except Exception:
        return False


def launch(batch_name: str, seed: int, dataset: str, generator: str, gpu: int) -> subprocess.Popen:
    batch = ROOT / batch_name
    logdir = batch / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"rerun_n1000_gpu{gpu}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTHONWARNINGS": "ignore",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TORCH_NUM_THREADS": "2",
            "CTABGAN_EPOCHS": env.get("CTABGAN_EPOCHS", "150"),
        }
    )
    cmd = [
        str(PYTHON),
        "-u",
        "run_experiment.py",
        "--skip-aggregate",
        "--force",
        "--dataset",
        dataset,
        "--generator",
        generator,
        "--seed",
        str(seed),
    ]
    with open(logfile, "a", encoding="utf-8") as fh:
        fh.write(
            f"\n===== {datetime.now().isoformat()} start "
            f"gpu={gpu} {batch_name} {dataset} {generator} seed={seed} =====\n"
        )
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(batch),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    log(f"launch gpu={gpu} {batch_name} {dataset}/{generator}/seed_{seed} pid={proc.pid}")
    return proc


def main() -> None:
    log("=== Rerun Adult/Bank/MAGIC at N=1000 (resume-safe) ===")
    log(f"already completed={completed_count()}/240")

    # gpu -> (Popen, job_key)
    active: dict[int, tuple[subprocess.Popen, tuple[str, int, str, str]]] = {}
    inflight: set[tuple[str, int, str, str]] = set()
    fail_counts: dict[tuple[str, int, str, str], int] = {}

    while True:
        for gpu, (proc, key) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                batch_name, seed, ds, gen = key
                batch = ROOT / batch_name
                ok = metrics_path(batch, seed, ds, gen).is_file()
                if not ok:
                    fail_counts[key] = fail_counts.get(key, 0) + 1
                    log(
                        f"gpu={gpu} FAILED rc={rc} "
                        f"{batch_name} {ds}/{gen}/seed_{seed} "
                        f"fail#{fail_counts[key]}"
                    )
                else:
                    log(f"gpu={gpu} finished rc={rc} {batch_name} {ds}/{gen}/seed_{seed}")
                del active[gpu]
                inflight.discard(key)

        jobs = pending_jobs(inflight, fail_counts)
        done = completed_count()
        skipped = sum(1 for v in fail_counts.values() if v >= 3)
        log(
            f"done={done}/240 pending={len(jobs)} skipped={skipped} "
            f"active_gpus={sorted(active)}"
        )
        if not jobs and not active:
            log(f"ALL DONE (completed={done} permanent_fails={skipped})")
            break

        for gpu in GPUS:
            if gpu in active:
                continue
            if not jobs:
                break
            if not gpu_free(gpu):
                continue
            key = jobs.pop(0)
            batch_name, seed, ds, gen = key
            # clear prior error so retry is clean
            err = (
                ROOT / batch_name / "classification" / "results" / "raw"
                / f"seed_{seed}" / ds / gen / "error.txt"
            )
            if err.is_file():
                err.unlink(missing_ok=True)
            inflight.add(key)
            active[gpu] = (launch(batch_name, seed, ds, gen, gpu), key)

        time.sleep(15)


if __name__ == "__main__":
    main()
