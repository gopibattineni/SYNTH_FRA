"""
Add Alzheimer's notebook and renumber Single run_Data_leak_Synth_Quality folders
to match the 11-dataset layout (2=Alzhimers, 11=MAGIC Gamma Telescope, etc.).
"""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[0]
BASE = REPO / "Single run_Data_leak_Synth_Quality"

# Target folder names (aligned with Other GANS / SDV models numbering)
TARGET_FOLDERS = {
    "1. Cancer": "1. Cancer",
    "2. Alzhimers": "2. Alzhimers",  # new
    "3. Adult": "3. Adult",
    "4. Forest cover dataset": "4. Forest cover dataset",
    "5. Bank Markting": "5. Bank Markting",
    "6. Wine dataset": "6. Wine dataset",
    "7. CDC diabetes dataset": "7. CDC diabetes dataset",
    "8. Metro interstate": "8. Metro interstate",
    "9. Mushroom dataset": "9. Mushroom dataset",
    "10. online shopping": "10. online shopping",
    "11. MAGIC Gamma Telescope": "11. MAGIC Gamma Telescope",
}

# Current -> target mapping
RENAME_MAP = {
    "2. MAGIC Gamma Telescope": "11. MAGIC Gamma Telescope",
    "8. CDC diabetes dataset": "7. CDC diabetes dataset",
    "9. Metro interstate": "8. Metro interstate",
    "7. Mushroom dataset": "9. Mushroom dataset",
}

DATA_LOADING_CELL = '''from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
)

from sdv.evaluation.single_table import evaluate_quality

from model.ctabgan import CTABGAN

# ----------------------------------------------------
# Load Dataset (Alzheimer's Disease)
# ----------------------------------------------------
candidate_paths = [
    Path("Alzhimers.xlsx"),
    Path("Alzheimer.xlsx"),
    Path("clean_alzheimer.csv"),
    Path("../Other GANS/clean_alzheimer.csv"),
    Path("../../Other GANS/clean_alzheimer.csv"),
]

data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        "Alzheimer dataset not found. Place Alzhimers.xlsx or clean_alzheimer.csv in this folder."
    )

if data_path.suffix.lower() in {".xlsx", ".xls"}:
    raw_data = pd.read_excel(data_path)
else:
    raw_data = pd.read_csv(data_path)

target_col = "Group"
ad_data = raw_data.drop(columns=["Subject ID", "M/F", "MRI ID", "Hand"], errors="ignore")
ad_data[target_col] = ad_data[target_col].replace({"Demented": 1, "Nondemented": 0, "Converted": 1})
ad_data[target_col] = pd.to_numeric(ad_data[target_col], errors="coerce").fillna(0).astype(int)

for col in ad_data.select_dtypes(include=[np.number]).columns:
    if ad_data[col].isnull().any():
        ad_data[col] = ad_data[col].fillna(ad_data[col].mean())

for col in ad_data.select_dtypes(include=["object"]).columns:
    if ad_data[col].isnull().any():
        modes = ad_data[col].mode()
        ad_data[col] = ad_data[col].fillna(modes[0] if len(modes) else "")

# Use same variable name pattern as other notebooks
alzheimer_data = ad_data.copy()

X = alzheimer_data.drop(columns=[target_col])
y = alzheimer_data[target_col]

# ----------------------------------------------------
# Metadata
# ----------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(alzheimer_data)

# ----------------------------------------------------
# Experiment Settings
# ----------------------------------------------------
N_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------
# Storage Containers
# ----------------------------------------------------
scores = {}

synthetic_datasets = {}

quality_results = []
'''


def _rename_dataset_folders(root: Path) -> None:
    """Renumber dataset folders using temporary names to avoid collisions."""
    if not root.exists():
        return

    planned = dict(RENAME_MAP)
    dataset_dirs = [p for p in root.iterdir() if p.is_dir() and re.match(r"^\d+\.", p.name)]
    involved = {p.name for p in dataset_dirs if p.name in planned or p.name in planned.values()}

    temps: dict[str, Path] = {}
    for name in involved:
        src = root / name
        if not src.exists():
            continue
        tmp = root / f"__tmp__{name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        src.rename(tmp)
        temps[name] = tmp

    for old_name, new_name in planned.items():
        if old_name not in temps:
            continue
        dest = root / new_name
        if dest.exists():
            shutil.rmtree(dest)
        temps[old_name].rename(dest)


def _adapt_cancer_notebook(cancer_nb: dict) -> dict:
    nb = deepcopy(cancer_nb)
    nb["cells"][1]["source"] = [DATA_LOADING_CELL]

    replacements = [
        ("cancer_data", "alzheimer_data"),
        ('label_col = "Diagnosis"', 'label_col = "Group"'),
        ('data_path = "breast_cancer_train.csv"', 'data_path = "alzheimer_train.csv"'),
        ("breast_cancer_train.csv", "alzheimer_train.csv"),
        ('pos_label="M"', "pos_label=1"),
        ("pos_label='M'", "pos_label=1"),
        ("average=\"binary\"", "average=\"binary\""),
        ('categorical_columns=[target_col]', "categorical_columns=[target_col]"),
    ]

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        for old, new in replacements:
            src = src.replace(old, new)
        # WGAN target rounding for binary 0/1
        src = src.replace(
            "synthetic_wgan[target_col] = (\n        synthetic_wgan[target_col]\n        .round()\n        .clip(0, 1)\n        .astype(int)\n    )\n\n    synthetic_wgan[target_col] = encoder.inverse_transform(\n        synthetic_wgan[target_col]\n    )",
            "synthetic_wgan[target_col] = (\n        synthetic_wgan[target_col]\n        .round()\n        .clip(0, 1)\n        .astype(int)\n    )",
        )
        cell["source"] = [src]

    return nb


def create_alzhimers_notebook(dest_dir: Path, template_path: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cancer_nb = json.loads(template_path.read_text(encoding="utf-8"))
    nb = _adapt_cancer_notebook(cancer_nb)
    out = dest_dir / "alzhimers.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    # Copy dataset asset
    src_csv = REPO / "Other GANS" / "clean_alzheimer.csv"
    if src_csv.exists():
        shutil.copy2(src_csv, dest_dir / "clean_alzheimer.csv")


def main() -> None:
    print("Renumbering Single run_Data_leak_Synth_Quality folders...")
    _rename_dataset_folders(BASE)

    print("Creating 2. Alzhimers notebook from cancer template...")
    cancer_template = BASE / "1. Cancer" / "cancer.ipynb"
    alzhimers_dir = BASE / "2. Alzhimers"
    create_alzhimers_notebook(alzhimers_dir, cancer_template)

    print("Renumbering diffusion_dataleak folders...")
    _rename_dataset_folders(BASE / "diffusion_dataleak")

    # Build alzhimers diffusion notebook if build script exists
    diff_build = REPO / "build_diffusion_dataleak_notebooks.py"
    if diff_build.exists():
        import subprocess
        subprocess.run(["python", str(diff_build)], check=True, cwd=str(REPO))

    print("Done. Folders:")
    for p in sorted(BASE.iterdir()):
        if p.is_dir() and re.match(r"^\d+\.", p.name):
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
