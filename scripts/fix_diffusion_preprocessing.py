"""Fix preprocessing before diffusion training in Diffusion GANs notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

# Per-notebook: categorical_columns expression (Python source, not JSON)
CAT_COLS: dict[str, str] = {
    "1. Cancer_diffusion.ipynb": "[target_col]",
    "2. Alzhimers_diffusion.ipynb": "[]",
    "3. Adult census_diffusion.ipynb": "[]",
    "4_Forest_Cover_diffusion.ipynb": "binary_columns",
    "5. Bank marketing_diffusion.ipynb": "[]",
    "6. Winequality_diffusion.ipynb": "[]",
    "7_CDC_diabetes_diffusion.ipynb": "[target_col]",
    "8. Metro Interstate_diffusion.ipynb": "['holiday', 'weather_main']",
    "9. Mushroom_diffusion.ipynb": "[]",
    "10. Online shopping_diffusion.ipynb": (
        "[c for c in ['country', 'page_1_main_category', 'colour', 'location', "
        "'model_photography'] if c in data.columns]"
    ),
    "11_MAGIC_Gamma_Telescope_diffusion.ipynb": "[target_col]",
    "12. Air Quality_diffusion.ipynb": "[]",
    "13. Concrete Compressive Strength_diffusion.ipynb": "[]",
    "14. Energy Efficiency_diffusion.ipynb": "[]",
    "15. Real Estate Valuation_diffusion.ipynb": "[]",
}

# Which dataframe variable to subsample/split/train on
TRAIN_DF: dict[str, str] = {
    "2. Alzhimers_diffusion.ipynb": "ad_data",
}

SUBSAMPLE_SPLIT = """
# -----------------------------
# Benchmark subsample + leak-safe split (generators fit train only)
# -----------------------------
from sklearn.model_selection import train_test_split

_bench_n = min(1000, len({df_var}))
{df_var} = {df_var}.sample(n=_bench_n, random_state=42).reset_index(drop=True)

_strat = {df_var}[target_col] if {df_var}[target_col].nunique() <= 30 else None
train_real, test_real = train_test_split(
    {df_var}, test_size=0.2, random_state=42, stratify=_strat,
)
"""

METRO_LOAD_TRAIN = r'''from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# -----------------------------
# Load + preprocess (Metro Interstate Traffic, UCI 492)
# -----------------------------
metro = fetch_ucirepo(id=492)
X = metro.data.features.copy()
y = metro.data.targets.copy()
X = X.drop(columns=["date_time"], errors="ignore")

if isinstance(y, pd.DataFrame):
    target_col = "traffic_volume" if "traffic_volume" in y.columns else y.columns[0]
    y_series = y[target_col]
else:
    target_col = "traffic_volume"
    y_series = pd.Series(y, name=target_col)

data = X.copy()
for col in ["temp", "rain_1h", "snow_1h", "clouds_all"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data[target_col] = pd.to_numeric(y_series, errors="coerce").fillna(0)
for col in ["holiday", "weather_main", "weather_description"]:
    if col in data.columns:
        data[col] = data[col].fillna("None").astype(str)

# -----------------------------
# Benchmark subsample + leak-safe split
# -----------------------------
from sklearn.model_selection import train_test_split

_bench_n = min(1000, len(data))
data = data.sample(n=_bench_n, random_state=42).reset_index(drop=True)
train_real, test_real = train_test_split(data, test_size=0.2, random_state=42)

# -----------------------------
# Diffusion models (TabDDPM, ForestDiffusion)
# -----------------------------
SYNTHETIC_N = 1000
DIFFUSION_SEED = 42
_categorical_columns = ["holiday", "weather_main"]

print("Training TabDDPM...")
synthetic_tabddpm = train_tabddpm(
    train_real,
    target_col=target_col,
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)
print("Training ForestDiffusion...")
synthetic_forestdiffusion = train_forestdiffusion(
    train_real,
    target_col=target_col,
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)

synthetic_outputs = {
    "TabDDPM": synthetic_tabddpm,
    "ForestDiffusion": synthetic_forestdiffusion,
}
model_order = ["TabDDPM", "ForestDiffusion"]
'''


def _lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _fix_training_cell(src: str, nb_name: str) -> str:
    if "train_tabddpm" not in src or "# Diffusion models" not in src:
        return src

    cat_expr = CAT_COLS.get(nb_name, "[]")
    src = re.sub(
        r"_categorical_columns\s*=\s*data\s*\n",
        f"_categorical_columns = {cat_expr}\n",
        src,
    )
    src = re.sub(
        r"_categorical_columns\s*=\s*\[\s*\n\s*c for c in known_categorical[\s\S]*?\]\s*\n",
        f"_categorical_columns = {cat_expr}\n",
        src,
    )

    df_var = TRAIN_DF.get(nb_name, "data")
    if "train_real, test_real" not in src and df_var in src:
        block = SUBSAMPLE_SPLIT.format(df_var=df_var)
        src = src.replace(
            "# -----------------------------\n# Diffusion models (TabDDPM, ForestDiffusion)\n# -----------------------------",
            block + "# -----------------------------\n# Diffusion models (TabDDPM, ForestDiffusion)\n# -----------------------------",
            1,
        )

    # Generators fit on train split only
    src = re.sub(
        r"(train_tabddpm\(\s*\n\s*)data(,|\s*\n)",
        rf"\1train_real\2",
        src,
    )
    src = re.sub(
        r"(train_forestdiffusion\(\s*\n\s*)data(,|\s*\n)",
        rf"\1train_real\2",
        src,
    )
    if df_var == "ad_data":
        src = re.sub(
            r"(train_tabddpm\(\s*\n\s*)ad_data(,|\s*\n)",
            r"\1train_real\2",
            src,
        )
        src = re.sub(
            r"(train_forestdiffusion\(\s*\n\s*)ad_data(,|\s*\n)",
            r"\1train_real\2",
            src,
        )

    # Explicit target_col where columns[-1] was used
    if nb_name == "3. Adult census_diffusion.ipynb":
        src = src.replace("target_col=data.columns[-1]", "target_col='income'")
        src = src.replace('target_col=data.columns[-1]', "target_col='income'")

    return src


def _should_delete_cell(src: str) -> bool:
    if "train_tabddpm" not in src:
        return False
    if 'target_col="target"' in src or "target_col='target'" in src:
        return True
    if "target_col=\"target\"" in src:
        return True
    # Online shopping: early training before session_id dropped
    if 'target_col="target"' in src.replace(" ", ""):
        return True
    if (
        "Online shopping" in src
        and "file_path = r'C:" in src
        and "train_tabddpm" in src
    ):
        return True
    return False


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    new_cells = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue
        src = "".join(cell.get("source", []))
        if _should_delete_cell(src):
            changed = True
            continue
        new_src = _fix_training_cell(src, path.name)
        if new_src != src:
            cell["source"] = _lines(new_src)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        new_cells.append(cell)

    if path.name == "8. Metro Interstate_diffusion.ipynb":
        has_train = any(
            "synthetic_tabddpm = train_tabddpm" in "".join(c.get("source", []))
            for c in new_cells
        )
        if not has_train:
            metro_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _lines(METRO_LOAD_TRAIN),
            }
            # Insert after import cell (index 2)
            insert_at = 2
            new_cells.insert(insert_at, metro_cell)
            changed = True

    if changed:
        nb["cells"] = new_cells
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def fix_generators() -> None:
    path = FOLDER / "diffusion_generators.py"
    text = path.read_text(encoding="utf-8")
    old = """def infer_column_types(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    cat_cols = list(categorical_columns or [])"""
    new = """def infer_column_types(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    if isinstance(categorical_columns, pd.DataFrame):
        raise TypeError(
            "categorical_columns must be a list of column names, not a DataFrame. "
            "Use e.g. _categorical_columns = [target_col] or []."
        )
    cat_cols = list(categorical_columns or [])"""
    if old in text and new not in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print("Updated diffusion_generators.py infer_column_types guard")


def main() -> None:
    fix_generators()
    fixed = [p.name for p in sorted(FOLDER.glob("*.ipynb")) if fix_notebook(p)]
    print(f"Fixed {len(fixed)} notebooks:")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
