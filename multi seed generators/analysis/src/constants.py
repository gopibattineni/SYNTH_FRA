"""Constants for the 10-seed multi-batch analysis."""
from __future__ import annotations

from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
MULTI_SEED_ROOT = ANALYSIS_ROOT.parent
REPO_ROOT = MULTI_SEED_ROOT.parent

BATCHES = {
    "1st_batch": {
        "folder": "1st batch",
        "seeds": [42, 123, 2024],
        "label": "1st batch",
    },
    "2nd_batch": {
        "folder": "second batch",
        "seeds": [68, 91, 2025],
        "label": "2nd batch",
    },
    "3rd_batch": {
        "folder": "third batch",
        "seeds": [55, 155, 255, 355],
        "label": "3rd batch",
    },
}

SEEDS = [42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]
SEED_TO_BATCH = {
    s: bid for bid, meta in BATCHES.items() for s in meta["seeds"]
}

GENERATORS = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "CTABGAN",
    "WGAN_GP",
    "TabDDPM",
    "ForestDiffusion",
]

CLASSIFICATION_DATASETS = [
    "cancer",
    "alzhimers",
    "adult",
    "forest_cover",
    "bank",
    "wine",
    "cdc_diabetes",
    "mushroom",
    "magic",
]

REGRESSION_DATASETS = [
    "metro",
    "online_shopping",
    "air_quality",
    "concrete",
    "energy_efficiency",
    "real_estate",
]

ALL_DATASETS = CLASSIFICATION_DATASETS + REGRESSION_DATASETS

DISPLAY_GENERATOR = {
    "WGAN_GP": "WGAN-GP",
    "ForestDiffusion": "ForestDiffusion",
    "TVAE": "TVAE",
    "CTABGAN": "CTABGAN",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
}

DISPLAY_DATASET = {
    "cancer": "Cancer",
    "alzhimers": "Alzheimer's",
    "adult": "Adult",
    "forest_cover": "Forest Cover",
    "bank": "Bank Marketing",
    "wine": "Wine Quality",
    "cdc_diabetes": "CDC Diabetes",
    "mushroom": "Mushroom",
    "magic": "MAGIC Gamma",
    "metro": "Metro Interstate",
    "online_shopping": "Online Shopping",
    "air_quality": "Air Quality",
    "concrete": "Concrete",
    "energy_efficiency": "Energy Efficiency",
    "real_estate": "Real Estate",
}

# Okabe–Ito palette (Agreed analysis tradeoff package)
GENERATOR_COLORS = {
    "CTGAN": "#0072B2",
    "TVAE": "#E69F00",
    "GaussianCopula": "#009E73",
    "WGAN_GP": "#D55E00",
    "WGAN-GP": "#D55E00",
    "CopulaGAN": "#CC79A7",
    "CTABGAN": "#56B4E9",
    "ForestDiffusion": "#000000",
    "TabDDPM": "#B8860B",
}

NAVY = "#1f3a5f"
INK = "#1c1f24"
ALPHA = 0.05

CLS_GAP_METRICS = ("Accuracy_Gap", "Precision_Gap", "Recall_Gap", "F1_Gap")
REG_GAP_METRICS = ("R2_Gap", "RMSE_Increase", "MAE_Increase")
# Aliases used in Agreed analysis naming
REG_GAP_LABELS = {
    "R2_Gap": "R²",
    "RMSE_Increase": "RMSE",
    "MAE_Increase": "MAE",
}

EXPECTED_RUNS_PER_SEED = len(ALL_DATASETS) * len(GENERATORS)  # 120
EXPECTED_CLS_PER_SEED = len(CLASSIFICATION_DATASETS) * len(GENERATORS)  # 72
EXPECTED_REG_PER_SEED = len(REGRESSION_DATASETS) * len(GENERATORS)  # 48
EXPECTED_TOTAL_RUNS = len(SEEDS) * EXPECTED_RUNS_PER_SEED  # 1200
