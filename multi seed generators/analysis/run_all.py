#!/usr/bin/env python3
"""Build data, validate, and generate the full 10-seed analysis suite."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("validate", [PY, "-c", "from src.validate import run_validation; print(run_validation())"]),
    ("workflow", [PY, "run_workflow_diagrams.py"]),
    ("tables", [PY, "run_tables.py"]),
    ("gap_cd", [PY, "run_gap_cd_analysis.py"]),
    ("rank_tables", [PY, "run_rank_tables.py"]),
    ("heatmaps", [PY, "run_heatmaps.py"]),
    ("seed_stability", [PY, "run_seed_stability.py"]),
    ("tradeoffs", [PY, "run_tradeoffs.py"]),
    ("pack_figures", [PY, "run_pack_manuscript_figures.py"]),
    ("summary", [PY, "write_analysis_summary.py"]),
]


def main():
    for name, cmd in STEPS:
        print("=" * 60)
        print("STEP:", name)
        print("=" * 60)
        r = subprocess.run(cmd, cwd=str(ANALYSIS))
        if r.returncode != 0:
            raise SystemExit(f"Step {name} failed with code {r.returncode}")
    print("ALL STEPS COMPLETE")


if __name__ == "__main__":
    main()
