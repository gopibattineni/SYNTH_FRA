#!/usr/bin/env python3
"""Write reports/analysis_summary.md from validated run counts."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from src.constants import (  # noqa: E402
    ALL_DATASETS,
    BATCHES,
    CLASSIFICATION_DATASETS,
    EXPECTED_TOTAL_RUNS,
    GENERATORS,
    REGRESSION_DATASETS,
    SEEDS,
)
from src.load_results import load_run_index  # noqa: E402
from src.validate import build_batch_summary, build_batch_validation  # noqa: E402


def main():
    runs = load_run_index()
    validation = build_batch_validation(runs)
    summary = build_batch_summary(validation)
    cls = int((runs["problem_type"] == "classification").sum())
    reg = int((runs["problem_type"] == "regression").sum())

    lines = [
        "# Analysis Summary — 10-Seed Evaluation",
        "",
        "## Design",
        "",
        f"- Datasets: **{len(ALL_DATASETS)}**",
        f"- Generators: **{len(GENERATORS)}**",
        f"- Classification datasets: **{len(CLASSIFICATION_DATASETS)}**",
        f"- Regression datasets: **{len(REGRESSION_DATASETS)}**",
        f"- Unique seeds: **{len(SEEDS)}** → `{SEEDS}`",
        f"- Expected configurations: **{len(ALL_DATASETS) * len(GENERATORS)}**",
        f"- Expected generator runs: **{EXPECTED_TOTAL_RUNS}**",
        "",
        "### Batches (execution provenance only)",
        "",
    ]
    for bid, meta in BATCHES.items():
        n = len(meta["seeds"])
        lines.append(
            f"- **{meta['label']}**: seeds = {meta['seeds']}; expected = {n * 120}"
        )
    lines += [
        "",
        "## Observed completion",
        "",
        f"- Completed runs: **{len(runs)}** / {EXPECTED_TOTAL_RUNS}",
        f"- Classification: **{cls}** / {len(SEEDS) * 72}",
        f"- Regression: **{reg}** / {len(SEEDS) * 48}",
        "",
        summary.to_csv(index=False),
        "",
        "## Aggregation rule",
        "",
        "Primary manuscript results use **Mean ± SD across the 10 individual seed observations** "
        "(sample SD, `ddof=1`). Batch means are **not** averaged.",
        "",
        "## Key outputs",
        "",
        "- Figures: `figures/` (workflow, classification, regression, seed_stability, tradeoffs, dataset_level)",
        "- Manuscript packs: `figures/main/`, `figures/supplementary/`",
        "- Tables: `tables/main`, `tables/supplementary`, `tables/batch_validation`",
        "- Data: `data/seed_level_long.csv`, `data/agg_10seed_mean_sd.csv`",
        "- Notebook: `notebooks/01_ten_seed_analysis.ipynb`",
        "- Mapping: `ANALYSIS_MAPPING.md`",
        "- Validation: `reports/data_validation_report.md`",
        "- Captions: `reports/figure_table_captions.md`",
        "",
        "## Statistical analysis (utility gaps)",
        "",
        "- Unit of analysis: dataset-level **10-seed mean** gap",
        "- Sample size: N = 9 (classification) or N = 6 (regression) datasets; K = 8 generators",
        "- Omnibus: Friedman test (α = 0.05)",
        "- Post-hoc: Nemenyi (CD diagrams) + Holm-corrected Wilcoxon",
        "- Effect display: average ranks + critical-difference cliques + rank tables / robustness charts",
        "",
        "## Regeneration",
        "",
        "```bash",
        "cd \"/home/gopi_b/SYNTH_BENCHMARK/multi seed generators/analysis\"",
        "../../.venv/bin/python run_all.py",
        "```",
        "",
    ]
    path = ANALYSIS / "reports" / "analysis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
