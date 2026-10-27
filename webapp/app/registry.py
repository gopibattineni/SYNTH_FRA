from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import BENCHMARK_ROOT, DATASETS_JSON, CTABGAN_DIR, DIFFUSION_MODULE, REPO_ROOT

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    name: str
    task: TaskType
    target_col: str
    train_csv: str
    uci_id: int | None = None
    description: str = ""


@dataclass(frozen=True)
class GeneratorInfo:
    id: str
    name: str
    family: str
    description: str


# Target columns aligned with benchmark train CSV exports.
_TARGET_BY_ID: dict[str, str] = {
    "cancer": "Diagnosis",
    "alzhimers": "Group",
    "adult": "income",
    "forest_cover": "Cover_Type",
    "bank": "y",
    "wine": "quality",
    "cdc_diabetes": "Diabetes_binary",
    "mushroom": "class",
    "magic": "class",
    "metro": "traffic_volume",
    "online_shopping": "price",
    "air_quality": "CO(GT)",
    "concrete": "Concrete compressive strength",
    "energy_efficiency": "Y1",
    "real_estate": "Y house price of unit area",
}

_REGRESSION_IDS = {
    "metro",
    "online_shopping",
    "air_quality",
    "concrete",
    "energy_efficiency",
    "real_estate",
}

_DESCRIPTIONS: dict[str, str] = {
    "cancer": "Breast tumor diagnosis from cell nucleus measurements.",
    "alzhimers": "Dementia group classification from MRI-derived features.",
    "adult": "Income prediction from census demographics.",
    "forest_cover": "Forest cover type from cartographic variables.",
    "bank": "Term deposit subscription from marketing campaign data.",
    "wine": "Wine quality score from physicochemical tests.",
    "cdc_diabetes": "Diabetes indicator from CDC health survey features.",
    "mushroom": "Edible vs poisonous from mushroom morphology.",
    "magic": "Gamma vs hadron from telescope measurements.",
    "metro": "Hourly interstate traffic volume with weather features.",
    "online_shopping": "E-commerce product price from catalog attributes.",
    "air_quality": "Air quality CO concentration from sensor readings.",
    "concrete": "Concrete compressive strength from mix components.",
    "energy_efficiency": "Building heating load from geometry and climate.",
    "real_estate": "House price per unit area from location features.",
}


def _load_datasets_json() -> list[dict]:
    with open(DATASETS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def get_datasets() -> list[DatasetInfo]:
    datasets: list[DatasetInfo] = []
    for entry in _load_datasets_json():
        ds_id = entry["id"]
        task: TaskType = "regression" if ds_id in _REGRESSION_IDS else "classification"
        notebook_dir = entry["notebook_dir"]
        train_csv = str(BENCHMARK_ROOT / notebook_dir / entry["train_csv"])
        datasets.append(
            DatasetInfo(
                id=ds_id,
                name=entry["name"],
                task=task,
                target_col=_TARGET_BY_ID[ds_id],
                train_csv=train_csv,
                uci_id=entry.get("uci_id"),
                description=_DESCRIPTIONS.get(ds_id, ""),
            )
        )
    return datasets


def get_dataset(dataset_id: str) -> DatasetInfo:
    for ds in get_datasets():
        if ds.id == dataset_id:
            return ds
    raise KeyError(f"Unknown dataset: {dataset_id}")


GENERATORS: list[GeneratorInfo] = [
    GeneratorInfo(
        id="ctabgan",
        name="CTAB-GAN+",
        family="GAN",
        description="Conditional tabular GAN with mixed-type column handling.",
    ),
    GeneratorInfo(
        id="wgan_gp",
        name="WGAN-GP",
        family="GAN",
        description="Wasserstein GAN with gradient penalty for tabular synthesis.",
    ),
    GeneratorInfo(
        id="tabddpm",
        name="TabDDPM",
        family="Diffusion",
        description="Diffusion model for tabular data (Yandex Research).",
    ),
    GeneratorInfo(
        id="forestdiffusion",
        name="ForestDiffusion",
        family="Diffusion",
        description="Forest-based flow diffusion for mixed tabular data.",
    ),
    GeneratorInfo(
        id="ctgan",
        name="CTGAN",
        family="SDV",
        description="Conditional tabular GAN from the Synthetic Data Vault.",
    ),
    GeneratorInfo(
        id="copulagan",
        name="CopulaGAN",
        family="SDV",
        description="Copula-based GAN synthesizer from SDV.",
    ),
    GeneratorInfo(
        id="tvae",
        name="TVAE",
        family="SDV",
        description="Tabular variational autoencoder from SDV.",
    ),
    GeneratorInfo(
        id="gaussian_copula",
        name="GaussianCopula",
        family="SDV",
        description="Fast statistical copula model from SDV.",
    ),
]


def get_generators() -> list[GeneratorInfo]:
    return list(GENERATORS)


def get_generator(generator_id: str) -> GeneratorInfo:
    for gen in GENERATORS:
        if gen.id == generator_id:
            return gen
    raise KeyError(f"Unknown generator: {generator_id}")


def check_generator_availability() -> dict[str, dict[str, object]]:
    tab_ddpm_root = REPO_ROOT / "_vendor" / "tab-ddpm"
    availability: dict[str, dict[str, object]] = {}

    for gen in GENERATORS:
        available = True
        reason = None
        if gen.id == "ctabgan" and not CTABGAN_DIR.is_dir():
            available = False
            reason = "CTAB-GAN-Plus will be cloned automatically on first use."
        if gen.id == "tabddpm":
            if not tab_ddpm_root.is_dir():
                available = False
                reason = (
                    "TabDDPM vendor missing. Clone: "
                    "git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"
                )
            elif not DIFFUSION_MODULE.is_file():
                available = False
                reason = "Diffusion module not found in repository."

        availability[gen.id] = {
            "available": available if gen.id != "ctabgan" else True,
            "reason": reason,
            "auto_setup": gen.id == "ctabgan",
        }
    return availability
