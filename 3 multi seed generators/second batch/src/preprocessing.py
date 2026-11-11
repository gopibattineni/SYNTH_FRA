"""Leakage-safe preprocessing: ALWAYS split first, then fit imputation on TRAIN only.

Never:
  full data → mean/median/mode → impute → split

Always:
  full data → split → imputer.fit(train) → transform(train) + transform(test)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TrainOnlyImputer:
    """Median (numeric) / mode (categorical) imputer fitted on train only."""

    def __init__(self) -> None:
        self.fill_: dict[str, Any] = {}
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []

    def fit(self, train: pd.DataFrame) -> "TrainOnlyImputer":
        self.fill_.clear()
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        for col in train.columns:
            s = train[col]
            if pd.api.types.is_numeric_dtype(s):
                self.numeric_cols_.append(col)
                val = s.median() if s.notna().any() else 0.0
                if pd.isna(val):
                    val = 0.0
                self.fill_[col] = float(val)
            else:
                self.categorical_cols_.append(col)
                mode = s.mode(dropna=True)
                self.fill_[col] = mode.iloc[0] if len(mode) else "missing"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, val in self.fill_.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' with NaN. Does NOT compute fill statistics."""
    return df.replace("?", np.nan)


def prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit imputer on TRAIN only; apply the same fitted values to TRAIN and TEST."""
    imputer = TrainOnlyImputer()
    train_i = imputer.fit_transform(train)
    test_i = imputer.transform(test)
    meta: dict[str, Any] = {
        "missing_strategy": "split_first_then_train_only_impute",
        "imputer": "TrainOnlyImputer",
        "imputer_fitted_on": "Real-Train only",
        "imputation_before_or_after_split": "AFTER split",
        "fill_values": {
            k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in imputer.fill_.items()
        },
    }
    if train_i.isna().any().any() or test_i.isna().any().any():
        raise AssertionError("NaNs remain after train-only imputation")
    return train_i, test_i, meta
