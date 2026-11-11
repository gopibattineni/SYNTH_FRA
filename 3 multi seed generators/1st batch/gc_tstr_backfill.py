from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.aggregate import _fill_tstr_and_gap_sd, _sanitize_zero_sd
from src.excel_io import write_aggregated_excel
from src.paths import ensure_task_dirs

SHEETS = ("Utility", "Privacy", "Fidelity", "Compute")


def load_long(task: str) -> pd.DataFrame:
    path = Path(task) / "results" / "multi_seed_results.xlsx"
    frames = []
    xl = pd.ExcelFile(path)
    for sh in SHEETS:
        if sh in xl.sheet_names:
            frames.append(pd.read_excel(path, sheet_name=sh))
    out = pd.concat(frames, ignore_index=True)
    out["task"] = task
    return out


def main() -> None:
    for task in ("classification", "regression"):
        print(f"=== {task}: load ===", flush=True)
        out = load_long(task)
        gc = out[
            (out.generator == "GaussianCopula")
            & out.metric_name.astype(str).str.endswith("_TSTR")
        ]
        print(
            f"GC TSTR rows={len(gc)} missing_sd="
            f"{(pd.to_numeric(gc.sd, errors='coerce').isna()).sum()}",
            flush=True,
        )
        print(f"=== {task}: fill GC TSTR from model seeds 42-51 ===", flush=True)
        out = _fill_tstr_and_gap_sd(out, task)
        out = _sanitize_zero_sd(out)
        gc2 = out[
            (out.generator == "GaussianCopula")
            & out.metric_name.astype(str).str.endswith("_TSTR")
        ]
        print(
            f"GC TSTR after: finite_sd="
            f"{(pd.to_numeric(gc2.sd, errors='coerce').notna()).sum()}/{len(gc2)}",
            flush=True,
        )
        sample = (
            gc2[["dataset", "metric_name", "mean_sd"]]
            .drop_duplicates(["dataset", "metric_name"])
            .head(15)
        )
        print(sample.to_string(index=False), flush=True)
        paths = ensure_task_dirs(task)
        print(f"=== {task}: write excel ===", flush=True)
        write_aggregated_excel(paths["aggregated"], out)
        combined = paths["aggregated"] / "multi_seed_results.xlsx"
        if combined.exists():
            shutil.copy2(combined, paths["results"] / "multi_seed_results.xlsx")
        print(f"=== {task}: done ===", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
