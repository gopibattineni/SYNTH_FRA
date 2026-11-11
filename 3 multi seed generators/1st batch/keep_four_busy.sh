#!/usr/bin/env bash
# Keep CUDA 0–5 busy with remaining CTABGAN/TabDDPM jobs. Resume-safe.
set -euo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
LOG="$BATCH/logs/keep_four_busy.log"
mkdir -p "$BATCH/logs"
export GENERATORS="${GENERATORS:-CTABGAN TabDDPM}"
# Faster leftover CTABGAN fits (forest_cover / online_shopping). Completed datasets used 150.
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
SEEDS = (42, 123, 2024)
GENS = [
    "CTABGAN",
]
PROTECTED = {"cancer", "alzhimers", "adult", "bank"}
# Remaining CTABGAN only: online_shopping + forest_cover (3 seeds each).
QUEUE = [
    "online_shopping",
    "forest_cover",
]
TASK = {
    "forest_cover": "classification",
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
    for g in GPUS:
        if not gpu_busy(g):
            continue
        job = current_job(g)
        if job and job[2] is not None:
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
            if leftover and (ds, gen) not in seen:
                seen.add((ds, gen))
                out.append((ds, gen))
    return out


def launch(gpu: int, ds: str, gen: str) -> None:
    env = os.environ.copy()
    env["GENERATORS"] = gen
    log(f"launch GPU={gpu} dataset={ds} generator={gen}")
    subprocess.Popen(
        ["bash", str(BATCH / "run_cuda_worker.sh"), str(gpu), ds],
        cwd=str(BATCH),
        env=env,
        start_new_session=True,
    )


while True:
    pending = pending_jobs()
    busy = [g for g in GPUS if gpu_busy(g)]
    free = [g for g in GPUS if g not in busy]
    log(f"busy={busy} free={free} pending={pending}")
    if not pending and not busy:
        py = Path(os.environ.get("PYTHON") or (BATCH.parents[1] / ".venv" / "bin" / "python"))
        log("all CTABGAN/TabDDPM jobs done; aggregating")
        subprocess.call([str(py), "-u", "run_experiment.py", "--aggregate-only"], cwd=str(BATCH))
        break
    for gpu in free:
        pending = pending_jobs()
        if not pending:
            break
        ds, gen = pending[0]
        launch(gpu, ds, gen)
        time.sleep(2)
    time.sleep(15)
PY
echo "$(date -Is) keep_four_busy exit"
