"""Excel-only result tables (CSV is converted then removed)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def xlsx_path(path: Path) -> Path:
    return Path(path).with_suffix(".xlsx")


def csv_path(path: Path) -> Path:
    return Path(path).with_suffix(".csv")


def read_results(path: Path) -> pd.DataFrame:
    xlsx = xlsx_path(path)
    csv = csv_path(path)
    if xlsx.exists() and xlsx.stat().st_size > 0:
        return pd.read_excel(xlsx)
    if csv.exists() and csv.stat().st_size > 0:
        return pd.read_csv(csv)
    raise FileNotFoundError(xlsx)


def write_results(path: Path, df: pd.DataFrame) -> Path:
    xlsx = xlsx_path(path)
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(xlsx, index=False)
    csv = csv_path(path)
    if csv.exists():
        csv.unlink()
    return xlsx


def results_exist(path: Path) -> bool:
    xlsx = xlsx_path(path)
    csv = csv_path(path)
    return (xlsx.exists() and xlsx.stat().st_size > 0) or (csv.exists() and csv.stat().st_size > 0)


def drop_csv(path: Path) -> None:
    csv = csv_path(path)
    if csv.exists():
        csv.unlink()
