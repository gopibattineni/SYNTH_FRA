"""Leakage-safe splitting: row-level, stratified, or entity/participant-level."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def participant_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split so every entity (participant/session) is entirely in train OR test."""
    if group_col not in df.columns:
        raise KeyError(f"group_col {group_col!r} not in columns: {list(df.columns)}")

    # One label per entity (first observation) for stratification when possible
    entity_label = df.groupby(group_col)[target_col].first()
    entities = entity_label.index.to_numpy()
    labels = entity_label.to_numpy()

    stratify = None
    # Stratify only when every class has ≥2 entities
    vals, counts = np.unique(labels, return_counts=True)
    if len(vals) >= 2 and counts.min() >= 2:
        stratify = labels

    ent_train, ent_test = train_test_split(
        entities,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_set = set(ent_train)
    test_set = set(ent_test)
    overlap = train_set & test_set
    if overlap:
        raise AssertionError(f"Entity overlap after split: {len(overlap)} entities")

    train = df[df[group_col].isin(train_set)].copy()
    test = df[df[group_col].isin(test_set)].copy()

    info = {
        "split_type": "entity",
        "group_col": group_col,
        "n_train_entities": len(train_set),
        "n_test_entities": len(test_set),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "entity_overlap": 0,
        "split_seed": seed,
        "test_size": test_size,
    }
    return train, test, info


def row_split(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stratify = None
    if task == "classification":
        y = df[target_col]
        vals, counts = np.unique(y.astype(str), return_counts=True)
        if len(vals) >= 2 and counts.min() >= 2:
            stratify = y
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    info = {
        "split_type": "row",
        "group_col": None,
        "n_train_entities": None,
        "n_test_entities": None,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "entity_overlap": 0,
        "split_seed": seed,
        "test_size": test_size,
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), info


def make_split(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    test_size: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_col: Optional[str] = cfg.get("group_col")
    target = cfg["target"]
    if group_col:
        train, test, info = participant_split(
            df, group_col, target, test_size=test_size, seed=split_seed
        )
        # Drop identifier from frames passed to generators / eval
        train = train.drop(columns=[group_col])
        test = test.drop(columns=[group_col])
        info["dropped_group_col"] = group_col
        return train.reset_index(drop=True), test.reset_index(drop=True), info
    return row_split(df, target, cfg["task"], test_size=test_size, seed=split_seed)
