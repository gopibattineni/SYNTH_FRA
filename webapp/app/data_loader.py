from __future__ import annotations

import pandas as pd

from app.models import DatasetSummary
from app.registry import DatasetInfo, get_dataset


def load_training_data(dataset_id: str) -> tuple[DatasetInfo, pd.DataFrame]:
    info = get_dataset(dataset_id)
    path = info.train_csv
    if not path:
        raise FileNotFoundError(f"No training CSV configured for dataset {dataset_id}")
    df = pd.read_csv(path)
    if info.target_col not in df.columns:
        raise ValueError(
            f"Target column {info.target_col!r} not found in {path}. "
            f"Columns: {list(df.columns)}"
        )
    return info, df


def summarize_dataset(dataset_id: str) -> DatasetSummary:
    info, df = load_training_data(dataset_id)
    return DatasetSummary(
        id=info.id,
        name=info.name,
        task=info.task,
        target_col=info.target_col,
        row_count=len(df),
        column_count=len(df.columns),
        description=info.description,
    )


def infer_categorical_columns(df: pd.DataFrame, target_col: str, task: str) -> list[str]:
    cats: list[str] = []
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            cats.append(col)
        elif col == target_col and task == "classification":
            cats.append(col)
    return list(dict.fromkeys(cats))
