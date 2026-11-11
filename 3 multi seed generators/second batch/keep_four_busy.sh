#!/usr/bin/env bash
# Keep CUDA 0–5 busy with all Batch 2 jobs. Resume-safe.
set -euo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
LOG="$BATCH/logs/keep_four_busy.log"
mkdir -p "$BATCH/logs"
export GENERATORS="${GENERATORS:-CTGAN CopulaGAN TVAE GaussianCopula CTABGAN WGAN_GP TabDDPM ForestDiffusion}"
export CTABGAN_EPOCHS="${CTABGAN_EPOCHS:-150}"
exec >>"$LOG" 2>&1

echo "============================================================"
echo "$(date -Is) keep_four_busy start generators=$GENERATORS"
echo "============================================================"

PYTHON="${PYTHON:-$BATCH/../../.venv/bin/python}"
export PYTHON

"$PYTHON" -u - <<'PY'
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BATCH = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
# heredoc has no __file__; use cwd (we cd to BATCH)
BATCH = Path.cwd()
GPUS = range(6)
SEEDS = (68, 91, 2025)
GENS = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "CTABGAN",
    "WGAN_GP",
    "TabDDPM",
    "ForestDiffusion",
]
PROTECTED = set()  # fresh batch — run all datasets
QUEUE = [
    "cancer",
    "alzhimers",
    "adult",
    "forest_cover",
    "bank",
    "wine",
    "cdc_diabetes",
    "mushroom",
    "magic",
    "metro",
    "online_shopping",
    "air_quality",
    "concrete",
    "energy_efficiency",
    "real_estate",
]
TASK = {
    "cancer": "classification",
    "alzhimers": "classification",
    "adult": "classification",
    "forest_cover": "classification",
    "bank": "classification",
    "wine": "classification",
    "cdc_diabetes": "classification",
    "mushroom": "classification",
    "magic": "classification",
    "metro": "regression",
    "online_shopping": "regression",
    "air_quality": "regression",
    "concrete": "regression",
    "energy_efficiency": "regression",
    "real_estate": "regression",
}


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {msg}", flush=True)


def gpu_busy(gpu: int) -> bool:
    pidf = BATCH / f"cuda{gpu}.pid"
    if not pidf.exists():
        return False
    try:
        pid = int(pidf.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        try:
            pidf.unlink()
        except OSError:
            pass
        return False


def current_job(gpu: int) -> tuple[str, str, int] | None:
    logp = BATCH / "logs" / f"cuda{gpu}.log"
    if not logp.exists():
        return None
    text = logp.read_text(errors="replace")
    idx = text.rfind(">>> ")
    if idx < 0:
        return None
    line = text[idx:].splitlines()[0]
    parts = [p.strip() for p in line.replace(">>>", "", 1).split("|")]
    if len(parts) < 2:
        return None
    seed = None
    if len(parts) >= 3 and parts[2].startswith("seed="):
        try:
            seed = int(parts[2].split("=", 1)[1])
        except ValueError:
            seed = None
    return parts[0], parts[1], seed


def is_saved(ds: str, gen: str, seed: int) -> bool:
    task = TASK[ds]
    mx = BATCH / task / "results" / "raw" / f"seed_{seed}" / ds / gen / "metrics.xlsx"
    return mx.exists() and mx.stat().st_size > 0


def seed_claimed(ds: str, gen: str, seed: int) -> bool:
    task = TASK[ds]
    pidp = BATCH / task / "results" / "raw" / f"seed_{seed}" / ds / gen / "in_progress.pid"
    if not pidp.exists():
        return False
    try:
        pid = int(pidp.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def pending_jobs() -> list[tuple[str, str]]:
    inflight_seeds: set[tuple[str, str, int]] = set()
    inflight_counts: dict[tuple[str, str], int] = {}
    for g in GPUS:
        if not gpu_busy(g):
            continue
        job = current_job(g)
        if not job:
            continue
        ds_j, gen_j, seed_j = job
        inflight_counts[(ds_j, gen_j)] = inflight_counts.get((ds_j, gen_j), 0) + 1
        if seed_j is not None:
            inflight_seeds.add(job)
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ds in QUEUE:
        if ds in PROTECTED:
            continue
        for gen in GENS:
            leftover = [
                s
                for s in SEEDS
                if not is_saved(ds, gen, s)
                and not seed_claimed(ds, gen, s)
                and (ds, gen, s) not in inflight_seeds
            ]
            if not leftover or (ds, gen) in seen:
                continue
            # Don't stack more workers than remaining seeds for this pair.
            if inflight_counts.get((ds, gen), 0) >= len(leftover):
                continue
            seen.add((ds, gen))
            out.append((ds, gen))
    return out


def launch(gpu: int, ds: str, gen: str) -> None:
    env = os.environ.copy()
    env["GENERATORS"] = gen
    log(f"launch GPU={gpu} dataset={ds} generator={gen}")
    proc = subprocess.Popen(
        ["bash", str(BATCH / "run_cuda_worker.sh"), str(gpu), ds],
        cwd=str(BATCH),
        env=env,
        start_new_session=True,
    )
    # Claim GPU immediately so the next free-slot scan cannot double-launch.
    (BATCH / f"cuda{gpu}.pid").write_text(str(proc.pid), encoding="utf-8")
    time.sleep(0.5)


while True:
    pending = pending_jobs()
    busy = [g for g in GPUS if gpu_busy(g)]
    free = [g for g in GPUS if g not in busy]
    log(f"busy={busy} free={free} pending={pending[:12]}{'...' if len(pending) > 12 else ''}")
    if not pending and not busy:
        py = Path(os.environ.get("PYTHON") or (BATCH.parents[1] / ".venv" / "bin" / "python"))
        log("all Batch-2 jobs done; aggregating")
        subprocess.call([str(py), "-u", "run_experiment.py", "--aggregate-only"], cwd=str(BATCH))
        break
    for gpu in free:
        pending = pending_jobs()
        if not pending:
            break
        ds, gen = pending[0]
        launch(gpu, ds, gen)
    # Fast refill — never leave a free GPU waiting long.
    time.sleep(3)
PY
echo "$(date -Is) keep_four_busy exit"
