"""Data validation for the 10-seed multi-batch experiment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import (
    ALL_DATASETS,
    BATCHES,
    CLASSIFICATION_DATASETS,
    EXPECTED_CLS_PER_SEED,
    EXPECTED_REG_PER_SEED,
    EXPECTED_RUNS_PER_SEED,
    EXPECTED_TOTAL_RUNS,
    GENERATORS,
    REGRESSION_DATASETS,
    SEEDS,
)
from .load_results import load_raw_long, load_run_index


def build_batch_validation(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch_id, meta in BATCHES.items():
        for seed in meta["seeds"]:
            sub = runs[(runs["batch"] == batch_id) & (runs["seed"] == seed)]
            cls_n = int(((sub["problem_type"] == "classification")).sum())
            reg_n = int(((sub["problem_type"] == "regression")).sum())
            total = cls_n + reg_n
            expected = EXPECTED_RUNS_PER_SEED
            status = "OK" if total == expected else "INCOMPLETE"
            rows.append(
                {
                    "batch": batch_id,
                    "batch_label": meta["label"],
                    "seed": seed,
                    "classification_runs": cls_n,
                    "regression_runs": reg_n,
                    "total_runs": total,
                    "expected_runs": expected,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def build_batch_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch_id, meta in BATCHES.items():
        sub = validation[validation["batch"] == batch_id]
        n_seeds = len(meta["seeds"])
        expected = n_seeds * EXPECTED_RUNS_PER_SEED
        completed = int(sub["total_runs"].sum())
        rows.append(
            {
                "Batch": meta["label"],
                "Seeds": ", ".join(str(s) for s in meta["seeds"]),
                "Expected Runs": expected,
                "Completed Runs": completed,
                "Status": "OK" if completed == expected else "INCOMPLETE",
            }
        )
    rows.append(
        {
            "Batch": "Total",
            "Seeds": f"{len(SEEDS)} seeds",
            "Expected Runs": EXPECTED_TOTAL_RUNS,
            "Completed Runs": int(validation["total_runs"].sum()),
            "Status": (
                "OK"
                if int(validation["total_runs"].sum()) == EXPECTED_TOTAL_RUNS
                else "INCOMPLETE"
            ),
        }
    )
    return pd.DataFrame(rows)


def write_validation_report(
    raw: pd.DataFrame,
    runs: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    analysis = Path(__file__).resolve().parents[1]
    reports = analysis / "reports"
    tables = analysis / "tables" / "batch_validation"
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    validation.to_excel(tables / "batch_validation.xlsx", index=False)
    summary.to_excel(tables / "batch_summary.xlsx", index=False)
    validation.to_csv(tables / "batch_validation.csv", index=False)
    summary.to_csv(tables / "batch_summary.csv", index=False)

    # Integrity checks
    unexpected_gens = sorted(set(runs["generator"]) - set(GENERATORS))
    unexpected_ds = sorted(set(runs["dataset_id"]) - set(ALL_DATASETS))
    missing = []
    for seed in SEEDS:
        for ds in ALL_DATASETS:
            task = "classification" if ds in CLASSIFICATION_DATASETS else "regression"
            for gen in GENERATORS:
                hit = runs[
                    (runs["seed"] == seed)
                    & (runs["dataset_id"] == ds)
                    & (runs["generator"] == gen)
                ]
                if hit.empty:
                    missing.append(f"{task}/{ds}/{gen}/seed_{seed}")

    dupes = (
        runs.groupby(["dataset_id", "generator", "seed"])
        .size()
        .reset_index(name="n")
    )
    dupes = dupes[dupes["n"] > 1]

    nan_metrics = int(raw["metric_value"].isna().sum())
    inf_metrics = int((~raw["metric_value"].isna() & ~raw["metric_value"].apply(lambda x: abs(x) != float("inf") if pd.notna(x) else True)).sum()) if False else 0
    # simpler inf check
    import numpy as np

    vals = pd.to_numeric(raw["metric_value"], errors="coerce")
    n_inf = int(np.isinf(vals.fillna(0)).sum())

    cls_runs = int((runs["problem_type"] == "classification").sum())
    reg_runs = int((runs["problem_type"] == "regression").sum())

    lines = [
        "# Data Validation Report — 10-Seed Evaluation",
        "",
        "## Design targets",
        "",
        f"- Datasets: **{len(ALL_DATASETS)}** (classification {len(CLASSIFICATION_DATASETS)}, regression {len(REGRESSION_DATASETS)})",
        f"- Generators: **{len(GENERATORS)}**",
        f"- Unique seeds: **{len(SEEDS)}** → `{SEEDS}`",
        f"- Expected configurations: **{len(ALL_DATASETS) * len(GENERATORS)}**",
        f"- Expected generator runs: **{EXPECTED_TOTAL_RUNS}**",
        f"- Expected classification runs: **{len(SEEDS) * EXPECTED_CLS_PER_SEED}**",
        f"- Expected regression runs: **{len(SEEDS) * EXPECTED_REG_PER_SEED}**",
        "",
        "## Observed completion",
        "",
        f"- Completed generator runs: **{len(runs)}** / {EXPECTED_TOTAL_RUNS}",
        f"- Classification runs: **{cls_runs}** / {len(SEEDS) * EXPECTED_CLS_PER_SEED}",
        f"- Regression runs: **{reg_runs}** / {len(SEEDS) * EXPECTED_REG_PER_SEED}",
        "",
        "## Batch summary",
        "",
        summary.to_csv(index=False),
        "",
        "## Per-seed validation",
        "",
        validation.to_csv(index=False),
        "",
        "## Integrity checks",
        "",
        f"- Missing dataset×generator×seed runs: **{len(missing)}**",
        f"- Duplicate dataset×generator×seed rows: **{len(dupes)}**",
        f"- Unexpected generators: `{unexpected_gens or 'none'}`",
        f"- Unexpected datasets: `{unexpected_ds or 'none'}`",
        f"- NaN metric_value cells: **{nan_metrics}**",
        f"- Infinite metric_value cells: **{n_inf}**",
        "",
        "## Notes",
        "",
        "- Aggregation for manuscript results uses the **10 individual seed observations** (sample SD, `ddof=1`), not the mean of batch means.",
        "- Batches are execution/provenance groups only.",
        "",
    ]
    if missing[:20]:
        lines.append("### Sample missing runs")
        lines.append("")
        for m in missing[:20]:
            lines.append(f"- `{m}`")
        lines.append("")

    path = reports / "data_validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_validation(force_rebuild: bool = False) -> dict:
    raw = load_raw_long(force_rebuild=force_rebuild)
    runs = load_run_index(raw)
    from .load_results import write_run_manifest, aggregate_10_seed

    write_run_manifest(runs)
    aggregate_10_seed(raw)
    validation = build_batch_validation(runs)
    summary = build_batch_summary(validation)
    report = write_validation_report(raw, runs, validation, summary)
    return {
        "n_runs": len(runs),
        "n_metric_rows": len(raw),
        "report": str(report),
        "completed_ok": len(runs) == EXPECTED_TOTAL_RUNS,
    }
