"""Load Batch 1 YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_datasets() -> list[dict[str, Any]]:
    return list(_load("datasets.yaml")["datasets"])


def load_generators() -> list[dict[str, Any]]:
    return list(_load("generators.yaml")["generators"])


def load_seeds() -> dict[str, Any]:
    return _load("seeds.yaml")


def generator_names() -> list[str]:
    return [g["name"] for g in load_generators()]


def dataset_by_id(dataset_id: str) -> dict[str, Any]:
    for d in load_datasets():
        if d["id"] == dataset_id or d["name"].lower() == dataset_id.lower():
            return d
    raise KeyError(f"Unknown dataset: {dataset_id}")
