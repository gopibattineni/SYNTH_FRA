"""Load TRTR/TSTR Excel results from the diffusion data-leak benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "Generators" / "Experiment with utility data leak"
DIFFUSION_ROOT = RESULTS_ROOT / "diffusion_dataleak"
DATASETS_JSON = RESULTS_ROOT / "python_scripts" / "hive" / "datasets.json"

# Eight generators across the benchmark (6 GAN/SDV + 2 diffusion).
GENERATORS_SDV_GAN = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
]
GENERATORS_DIFFUSION = ["TabDDPM", "ForestDiffusion"]
ALL_GENERATORS = GENERATORS_SDV_GAN + GENERATORS_DIFFUSION

# Backwards-compatible aliases used by app.py
GENERATORS_MAIN = GENERATORS_SDV_GAN

CLASSIFICATION_METRICS = {
    "Accuracy_Drop": {"label": "Accuracy drop (TRTR − TSTR)", "better": "lower"},
    "F1_Drop": {"label": "F1 drop (TRTR − TSTR)", "better": "lower"},
    "Precision_Drop": {"label": "Precision drop", "better": "lower"},
    "Recall_Drop": {"label": "Recall drop", "better": "lower"},
}
REGRESSION_METRICS = {
    "R2_Drop": {"label": "R² drop (TRTR − TSTR)", "better": "lower"},
    "RMSE_Increase": {"label": "RMSE increase (TSTR − TRTR)", "better": "lower"},
    "MAE_Increase": {"label": "MAE increase (TSTR − TRTR)", "better": "lower"},
    "MSE_Increase": {"label": "MSE increase (TSTR − TRTR)", "better": "lower"},
}


@dataclass
class DatasetResults:
    dataset_id: str
    name: str
    number: int
    task_type: str
    notebook_dir: str
    excel_path: Optional[Path]
    diffusion_excel_path: Optional[Path] = None
    main_excel_path: Optional[Path] = None
    trtr: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    generators: List[str] = field(default_factory=list)
    missing_generators: List[str] = field(default_factory=list)
    has_notebook: bool = False
    experiment_status: str = "missing"  # complete | partial | pending | missing
    error: Optional[str] = None


def _dataset_number(notebook_dir: str) -> int:
    match = re.match(r"(\d+)", notebook_dir)
    return int(match.group(1)) if match else 0


def _task_type(number: int) -> str:
    return "regression" if number >= 10 else "classification"


def _find_excel(folder: Path, preferred: Optional[str] = None) -> Optional[Path]:
    if not folder.is_dir():
        return None
    if preferred:
        preferred_path = folder / preferred
        if preferred_path.is_file():
            return preferred_path
    matches = sorted(folder.glob("TRTR_TSTR*.xlsx"))
    return matches[0] if matches else None


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Synthetic_Model" in out.columns:
        out = out.rename(columns={"Synthetic_Model": "Generator"})
    if "Generator" in out.columns:
        out["Generator"] = out["Generator"].astype(str)
    return out


def _normalize_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Synthetic_Model" in out.columns:
        out = out.rename(columns={"Synthetic_Model": "Generator"})
    if "Generator" in out.columns:
        out["Generator"] = out["Generator"].astype(str)
    return out


def _load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    return {
        "trtr": sheets.get("TRTR_Results", pd.DataFrame()),
        "summary": _normalize_summary(sheets.get("Summary", pd.DataFrame())),
        "comparisons": _normalize_comparisons(sheets.get("All_Comparisons", pd.DataFrame())),
        "quality": sheets.get("Quality_Metrics", pd.DataFrame()),
    }


def _merge_frames(primary: pd.DataFrame, secondary: pd.DataFrame, key: str) -> pd.DataFrame:
    if primary.empty:
        return secondary.copy()
    if secondary.empty:
        return primary.copy()
    if key not in primary.columns or key not in secondary.columns:
        return pd.concat([primary, secondary], ignore_index=True)

    primary_keys = set(primary[key].astype(str))
    extra = secondary[~secondary[key].astype(str).isin(primary_keys)].copy()
    return pd.concat([primary, extra], ignore_index=True)


def _merge_quality(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return secondary.copy()
    if secondary.empty:
        return primary.copy()

    out = primary.copy()
    gen_col = "Generator" if "Generator" in out.columns else None
    sec_gen_col = "Generator" if "Generator" in secondary.columns else None
    if gen_col and sec_gen_col:
        seen = set(out[gen_col].astype(str))
        extra = secondary[~secondary[sec_gen_col].astype(str).isin(seen)]
        return pd.concat([out, extra], ignore_index=True)
    return pd.concat([out, secondary], ignore_index=True)


def _generators_from_summary(summary: pd.DataFrame) -> List[str]:
    if summary.empty or "Generator" not in summary.columns:
        return []
    found = summary["Generator"].dropna().astype(str).unique().tolist()
    return [g for g in ALL_GENERATORS if g in found]


def _experiment_status(found_generators: List[str], has_notebook: bool, has_any_excel: bool) -> str:
    if not has_notebook and not has_any_excel:
        return "missing"
    if not has_any_excel:
        return "pending"
    if len(found_generators) >= len(ALL_GENERATORS):
        return "complete"
    return "partial"


def load_datasets_metadata() -> List[dict]:
    if DATASETS_JSON.is_file():
        return json.loads(DATASETS_JSON.read_text(encoding="utf-8"))
    return []


def load_all_results() -> Dict[str, DatasetResults]:
    metadata = load_datasets_metadata()
    results: Dict[str, DatasetResults] = {}

    for entry in metadata:
        ds_id = entry["id"]
        notebook_dir = entry["notebook_dir"]
        number = _dataset_number(notebook_dir)
        diffusion_folder = DIFFUSION_ROOT / notebook_dir
        main_folder = RESULTS_ROOT / notebook_dir

        diffusion_excel = _find_excel(diffusion_folder, entry.get("output_file"))
        main_excel = _find_excel(main_folder, entry.get("output_file"))
        excel_path = diffusion_excel or main_excel

        has_notebook = any(diffusion_folder.glob("*.ipynb")) or any(main_folder.glob("*.ipynb"))

        ds = DatasetResults(
            dataset_id=ds_id,
            name=entry.get("name", ds_id),
            number=number,
            task_type=_task_type(number),
            notebook_dir=notebook_dir,
            excel_path=excel_path,
            diffusion_excel_path=diffusion_excel,
            main_excel_path=main_excel,
            has_notebook=has_notebook,
        )

        if diffusion_excel is None and main_excel is None:
            ds.error = "No TRTR/TSTR Excel file in diffusion_dataleak or parent folder"
            ds.experiment_status = _experiment_status([], has_notebook, False)
            ds.missing_generators = ALL_GENERATORS.copy()
            results[ds_id] = ds
            continue

        try:
            diffusion_data = (
                _load_workbook(diffusion_excel) if diffusion_excel is not None else {}
            )
            main_data = _load_workbook(main_excel) if main_excel is not None else {}

            # Prefer diffusion workbook for TRTR baseline; fall back to main.
            ds.trtr = (
                diffusion_data.get("trtr", pd.DataFrame())
                if diffusion_excel is not None
                else main_data.get("trtr", pd.DataFrame())
            )
            if ds.trtr.empty:
                ds.trtr = main_data.get("trtr", pd.DataFrame())

            ds.summary = _merge_frames(
                diffusion_data.get("summary", pd.DataFrame()),
                main_data.get("summary", pd.DataFrame()),
                "Generator",
            )
            ds.comparisons = _merge_frames(
                diffusion_data.get("comparisons", pd.DataFrame()),
                main_data.get("comparisons", pd.DataFrame()),
                "Generator",
            )
            ds.quality = _merge_quality(
                diffusion_data.get("quality", pd.DataFrame()),
                main_data.get("quality", pd.DataFrame()),
            )

            ds.generators = _generators_from_summary(ds.summary)
            ds.missing_generators = [g for g in ALL_GENERATORS if g not in ds.generators]
            ds.experiment_status = _experiment_status(
                ds.generators,
                has_notebook,
                diffusion_excel is not None or main_excel is not None,
            )
        except Exception as exc:
            ds.error = str(exc)
            ds.experiment_status = "missing"
            ds.missing_generators = ALL_GENERATORS.copy()

        results[ds_id] = ds

    return results


def summary_long_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    rows = []
    for ds_id, ds in results.items():
        if ds.summary.empty or "Generator" not in ds.summary.columns:
            continue
        for _, row in ds.summary.iterrows():
            base = {
                "dataset_id": ds_id,
                "dataset": ds.name,
                "dataset_number": ds.number,
                "task_type": ds.task_type,
                "generator": row.get("Generator"),
                "experiment_status": ds.experiment_status,
            }
            for col in ds.summary.columns:
                if col == "Generator":
                    continue
                base[col] = row[col]
            rows.append(base)
    return pd.DataFrame(rows)


def comparisons_long_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    rows = []
    for ds_id, ds in results.items():
        if ds.comparisons.empty:
            continue
        chunk = ds.comparisons.copy()
        chunk["dataset_id"] = ds_id
        chunk["dataset"] = ds.name
        chunk["dataset_number"] = ds.number
        chunk["task_type"] = ds.task_type
        if "Generator" not in chunk.columns and "Synthetic_Model" in chunk.columns:
            chunk = chunk.rename(columns={"Synthetic_Model": "Generator"})
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def coverage_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    """Dataset × generator matrix: 1 when results exist, 0 when missing."""
    rows = []
    for ds in sorted(results.values(), key=lambda d: d.number):
        for generator in ALL_GENERATORS:
            rows.append(
                {
                    "dataset_id": ds.dataset_id,
                    "dataset": ds.name,
                    "dataset_number": ds.number,
                    "task_type": ds.task_type,
                    "generator": generator,
                    "available": int(generator in ds.generators),
                    "experiment_status": ds.experiment_status,
                    "has_diffusion_excel": int(ds.diffusion_excel_path is not None),
                    "has_main_excel": int(ds.main_excel_path is not None),
                }
            )
    return pd.DataFrame(rows)


def experiment_status_frame(results: Dict[str, DatasetResults]) -> pd.DataFrame:
    rows = []
    for ds in sorted(results.values(), key=lambda d: d.number):
        rows.append(
            {
                "dataset": ds.name,
                "dataset_number": ds.number,
                "task_type": ds.task_type,
                "status": ds.experiment_status,
                "generators_done": len(ds.generators),
                "generators_total": len(ALL_GENERATORS),
                "missing_generators": ", ".join(ds.missing_generators),
                "diffusion_excel": (
                    ds.diffusion_excel_path.name if ds.diffusion_excel_path else ""
                ),
                "main_excel": ds.main_excel_path.name if ds.main_excel_path else "",
                "error": ds.error or "",
            }
        )
    return pd.DataFrame(rows)


def primary_metric(task_type: str) -> str:
    return "Accuracy_Drop" if task_type == "classification" else "R2_Drop"


def metrics_for_task(task_type: str) -> Dict[str, dict]:
    return CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
