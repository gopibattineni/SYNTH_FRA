#!/usr/bin/env python3
"""Wait for Adult/Bank/MAGIC N=1000 regen to finish, then re-aggregate + merge Excel."""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PYTHON = REPO / ".venv" / "bin" / "python"
ORCH_LOG = ROOT / "logs" / "rerun_adult_bank_magic_n1000.log"
CRONTAB_BACKUP = Path("/tmp/crontab_backup_before_n1000_regen.txt")

BATCHES = ["1st batch", "second batch", "third batch"]
DATASETS = ["adult", "bank", "magic"]
GENERATORS = [
    "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
    "CTABGAN", "WGAN_GP", "TabDDPM", "ForestDiffusion",
]
SEEDS = {
    "1st batch": [42, 123, 2024],
    "second batch": [68, 91, 2025],
    "third batch": [55, 155, 255, 355],
}


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}", flush=True)


def completed() -> int:
    n = 0
    for batch, seeds in SEEDS.items():
        for seed in seeds:
            for ds in DATASETS:
                for gen in GENERATORS:
                    p = (
                        ROOT / batch / "classification" / "results" / "raw"
                        / f"seed_{seed}" / ds / gen / "metrics.xlsx"
                    )
                    if p.is_file():
                        n += 1
    return n


def orch_done() -> bool:
    if not ORCH_LOG.is_file():
        return False
    text = ORCH_LOG.read_text(encoding="utf-8", errors="ignore")
    return "ALL DONE" in text


def run(cmd: list[str], cwd: Path) -> None:
    log(f"$ {' '.join(cmd)}  (cwd={cwd.name})")
    subprocess.check_call(cmd, cwd=str(cwd))


def main() -> None:
    log("=== Post-regen watcher started ===")
    while True:
        n = completed()
        log(f"progress {n}/240 orch_done={orch_done()}")
        if n >= 240 and orch_done():
            break
        if n >= 240:
            # metrics all present; give orch a moment to exit
            time.sleep(30)
            if completed() >= 240:
                break
        time.sleep(60)

    log("Regeneration complete — aggregating batches")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for batch in BATCHES:
        subprocess.check_call(
            [str(PYTHON), "-u", "run_experiment.py", "--aggregate-only"],
            cwd=str(ROOT / batch),
            env=env,
        )
        log(f"aggregated {batch}")

    log("Merging 10 seeds")
    subprocess.check_call([str(PYTHON), "-u", str(ROOT / "merge_10_seeds.py")], cwd=str(ROOT))

    if CRONTAB_BACKUP.is_file():
        log("Restoring crontab keep_alive entries")
        subprocess.check_call(["crontab", str(CRONTAB_BACKUP)])
        log("crontab restored")
    else:
        log("WARNING: no crontab backup found")

    log("=== Excel update complete ===")


if __name__ == "__main__":
    main()
