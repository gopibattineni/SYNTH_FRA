"""Configuration for publication-quality trade-off visualizations.

Column aliases, styling, generator palettes, and figure defaults live here so
the plotting code stays dataset-agnostic and notebook-independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this package)
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
FIGURES_DIR: Path = PACKAGE_DIR / "figures"
OUTPUT_DIR: Path = PACKAGE_DIR / "output"

# ---------------------------------------------------------------------------
# Canonical column names used internally after normalization
# ---------------------------------------------------------------------------
COL_DATASET: str = "Dataset"
COL_GENERATOR: str = "Generator"
COL_TASK: str = "Task"
COL_QUALITY: str = "QualityScore"
COL_ACCURACY: str = "Accuracy"
COL_R2: str = "R2"
COL_MIA: str = "MIA"

# Optional aliases accepted in input CSVs (case-insensitive matching applied
# after stripping whitespace). First match wins.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    COL_DATASET: ("Dataset", "dataset", "DatasetName", "dataset_name"),
    COL_GENERATOR: ("Generator", "generator", "Model", "model", "Display"),
    COL_TASK: ("Task", "task", "TaskType", "task_type"),
    COL_QUALITY: (
        "QualityScore",
        "Quality_Score",
        "Quality",
        "Fidelity_SDMetrics",
        "Fidelity",
        "SDMetrics_Quality",
        "SDV_Quality",
    ),
    COL_ACCURACY: (
        "Accuracy",
        "Mean_TSTR_Accuracy",
        "TSTR_Accuracy",
        "Utility_Accuracy",
        "Acc",
    ),
    COL_R2: (
        "R2",
        "R²",
        "Mean_TSTR_R2",
        "TSTR_R2",
        "Utility_R2",
        "R2_Score",
    ),
    COL_MIA: (
        "MIA",
        "MIA_AUC",
        "MIA_Success",
        "MIASuccessRate",
        "Privacy_MIA",
        "MembershipInference",
    ),
}

TASK_CLASSIFICATION: str = "classification"
TASK_REGRESSION: str = "regression"

# ---------------------------------------------------------------------------
# Metrics / axis labels
# ---------------------------------------------------------------------------
QUALITY_LABEL: str = "SDV Quality Score (Fidelity)"
ACCURACY_LABEL: str = "Accuracy (Utility)"
R2_LABEL: str = r"$R^{2}$ (Utility)"
MIA_LABEL: str = "MIA (AUC)"

UTILITY_COLUMN: Dict[str, str] = {
    TASK_CLASSIFICATION: COL_ACCURACY,
    TASK_REGRESSION: COL_R2,
}

UTILITY_LABEL: Dict[str, str] = {
    TASK_CLASSIFICATION: ACCURACY_LABEL,
    TASK_REGRESSION: R2_LABEL,
}

# ---------------------------------------------------------------------------
# Styling — journal / IEEE–Elsevier friendly
# ---------------------------------------------------------------------------
DPI: int = 300
FONT_FAMILY: str = "Times New Roman"
FONT_SIZE: float = 11.0
TITLE_SIZE: float = 12.5
LABEL_SIZE: float = 11.5
TICK_SIZE: float = 10.0
LEGEND_SIZE: float = 10.0
MARKER_SIZE: float = 95.0
PARETO_MARKER_SIZE: float = 155.0
LINE_WIDTH: float = 1.4
SHOW_GRID: bool = False
GRID_ALPHA: float = 0.0
FACE_COLOR: str = "white"
SPINE_COLOR: str = "#2f2f2f"
ANNOTATE_GENERATORS: bool = False
IDEAL_REGION_ALPHA: float = 0.08
IDEAL_REGION_COLOR: str = "#2ca02c"
AXIS_PAD_FRAC: float = 0.06
BUBBLE_SIZE_RANGE: Tuple[float, float] = (100.0, 750.0)
BUBBLE_ALPHA: float = 0.65
# Non-Pareto points stay readable (not washed out)
FADED_ALPHA: float = 0.55
MARKER_EDGE_COLOR: str = "#1a1a1a"
MARKER_EDGE_WIDTH: float = 1.15
PARETO_EDGE_WIDTH: float = 2.0

SAVE_FORMATS: Tuple[str, ...] = ("pdf", "svg", "png")

# Colorblind-friendly Okabe–Ito inspired palette (consistent across figures)
GENERATOR_COLORS: Dict[str, str] = {
    "CTGAN": "#0072B2",          # blue
    "TVAE": "#E69F00",           # orange
    "GaussianCopula": "#009E73", # bluish green
    "WGAN_GP": "#D55E00",        # vermillion / red-orange
    "WGAN-GP": "#D55E00",
    "CopulaGAN": "#CC79A7",      # reddish purple
    "CTABGAN": "#56B4E9",        # sky blue
    "ForestDiffusion": "#000000",# black
    "TabDDPM": "#B8860B",        # dark goldenrod (clearer than pale yellow)
}

# Fallback cycle for unknown generators (Okabe–Ito)
FALLBACK_COLORS: Sequence[str] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#000000",
    "#F0E442",
    "#999999",
)

GENERATOR_MARKERS: Dict[str, str] = {
    "CTGAN": "o",
    "TVAE": "s",
    "GaussianCopula": "^",
    "WGAN_GP": "D",
    "WGAN-GP": "D",
    "CopulaGAN": "v",
    "CTABGAN": "P",
    "ForestDiffusion": "*",
    "TabDDPM": "X",
}

FALLBACK_MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "*", "X", "h")

# Preferred generator legend order when present
GENERATOR_ORDER: Sequence[str] = (
    "ForestDiffusion",
    "TVAE",
    "CTABGAN",
    "WGAN_GP",
    "WGAN-GP",
    "GaussianCopula",
    "CopulaGAN",
    "CTGAN",
    "TabDDPM",
)

# Display names for legends / annotations
GENERATOR_DISPLAY: Dict[str, str] = {
    "WGAN_GP": "WGAN-GP",
    "WGAN-GP": "WGAN-GP",
    "ForestDiffusion": "ForestDiffusion",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTABGAN": "CTABGAN",
    "CTGAN": "CTGAN",
    "TVAE": "TVAE",
    "TabDDPM": "TabDDPM",
}

# Subplot grid heuristics
MAX_SUBPLOT_COLS: int = 3

# Figure filenames (stem only; extensions added by saver)
FIG_FILENAMES: Dict[str, str] = {
    "fig1": "Fig1_Fidelity_vs_Utility",
    "fig2": "Fig2_Utility_vs_Privacy",
    "fig3": "Fig3_Fidelity_vs_Privacy",
    "fig4": "Fig4_Bubble_Tradeoff",
    "fig5": "Fig5_Pareto",
}


def figure_dir(task_type: str) -> Path:
    """Return ``figures/<task_type>/`` path."""
    return FIGURES_DIR / task_type


def output_dir(task_type: str) -> Path:
    """Return ``output/<task_type>/`` path."""
    return OUTPUT_DIR / task_type
