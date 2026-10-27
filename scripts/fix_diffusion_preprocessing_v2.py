"""Second-pass fixes: Metro, Forest Cover, Alzheimer, MAGIC, duplicate subsample."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

FOREST_COVER_LOAD = r'''import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Covertype dataset (UCI id=31)
# -----------------------------
covertype = fetch_ucirepo(id=31)
X = covertype.data.features
y = covertype.data.targets
data = pd.concat([X, y], axis=1)
target_col = y.columns[0]

binary_columns = [
    col for col in X.columns if col.startswith(("Wilderness_Area", "Soil_Type"))
]

# Benchmark subsample + leak-safe split
_bench_n = min(1000, len(data))
data = data.sample(n=_bench_n, random_state=42).reset_index(drop=True)

_strat = data[target_col] if data[target_col].nunique() <= 30 else None
train_real, test_real = train_test_split(
    data, test_size=0.2, random_state=42, stratify=_strat,
)

# Diffusion models (TabDDPM, ForestDiffusion)
SYNTHETIC_N = 1000
DIFFUSION_SEED = 42
_categorical_columns = binary_columns

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

ALZHEIMER_DIFFUSION_TAIL = r'''
# -----------------------------
# Benchmark subsample + leak-safe split (generators fit train only)
# -----------------------------
from sklearn.model_selection import train_test_split

_bench_n = min(1000, len(ad_data))
ad_data = ad_data.sample(n=_bench_n, random_state=42).reset_index(drop=True)

_strat = ad_data[target_col] if ad_data[target_col].nunique() <= 30 else None
train_real, test_real = train_test_split(
    ad_data, test_size=0.2, random_state=42, stratify=_strat,
)

# Diffusion models (TabDDPM, ForestDiffusion)
SYNTHETIC_N = 1000
DIFFUSION_SEED = 42
_categorical_columns = []

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

METRO_LOAD_TRAIN = r'''from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load + preprocess (Metro Interstate Traffic, UCI 492)
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

_bench_n = min(1000, len(data))
data = data.sample(n=_bench_n, random_state=42).reset_index(drop=True)
train_real, test_real = train_test_split(data, test_size=0.2, random_state=42)

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


def _remove_duplicate_subsample(src: str) -> str:
    pattern = (
        r"# Take only 1000 real samples[^\n]*\n"
        r"n_samples = min\(1000, len\([^\)]+\)\)\n"
        r"(?:data|ad_data|magic_data) = \1\.sample\(n=n_samples[^\n]+\n"
        r"(?:    X = data\.drop\(columns=\[target_col\]\)\n\n)?"
    )
    # Simpler: remove block before benchmark subsample
    old = re.compile(
        r"# Take only 1000 real samples[^\n]*\n"
        r"n_samples = min\(1000, len\([^)]+\)\)\n"
        r"(\w+) = \1\.sample\(n=n_samples, random_state=42\)\.reset_index\(drop=True\)\n"
        r"(?:    X = data\.drop\(columns=\[target_col\]\)\n\n)?",
        re.MULTILINE,
    )
    return old.sub("", src)


def fix_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changes: list[str] = []

    if path.name == "4_Forest_Cover_diffusion.ipynb":
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code" and "fetch_ucirepo(id=31)" in "".join(
                cell.get("source", [])
            ):
                cell["source"] = _lines(FOREST_COVER_LOAD)
                cell["outputs"] = []
                cell["execution_count"] = None
                changes.append("rewrote Forest Cover load+train cell")
                break

    if path.name == "2. Alzhimers_diffusion.ipynb":
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "ad_data.to_csv" not in src:
                continue
            head = src.split("# -----------------------------\n# 4. CTABGAN")[0]
            cell["source"] = _lines(head.rstrip() + ALZHEIMER_DIFFUSION_TAIL)
            cell["outputs"] = []
            cell["execution_count"] = None
            changes.append("trimmed Alzheimer GAN leftovers; cat cols []")
            break

    if path.name == "11_MAGIC_Gamma_Telescope_diffusion.ipynb":
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "magic_gamma_telescope = fetch_ucirepo" not in src:
                continue
            src = _remove_duplicate_subsample(src)
            src = src.replace(
                "_bench_n = min(1000, len(data))",
                "_bench_n = min(1000, len(magic_data))",
            )
            src = src.replace(
                "data = data.sample(n=_bench_n",
                "magic_data = magic_data.sample(n=_bench_n",
            )
            src = re.sub(
                r"train_real, test_real = train_test_split\(\s*\n\s*data,",
                "train_real, test_real = train_test_split(\n    magic_data,",
                src,
            )
            src = re.sub(
                r"_strat = data\[target_col\]",
                "_strat = magic_data[target_col]",
                src,
            )
            src = re.sub(
                r"(train_tabddpm\(\s*\n\s*)magic_data(,|\s*\n)",
                r"\1train_real\2",
                src,
            )
            src = re.sub(
                r"(train_forestdiffusion\(\s*\n\s*)magic_data(,|\s*\n)",
                r"\1train_real\2",
                src,
            )
            cell["source"] = _lines(src)
            cell["outputs"] = []
            cell["execution_count"] = None
            changes.append("MAGIC: subsample/split on magic_data; train on train_real")
            break

    if path.name == "8. Metro Interstate_diffusion.ipynb":
        has_train = any(
            "synthetic_tabddpm = train_tabddpm" in "".join(c.get("source", []))
            for c in nb["cells"]
        )
        if not has_train:
            metro_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _lines(METRO_LOAD_TRAIN),
            }
            nb["cells"].insert(2, metro_cell)
            changes.append("inserted Metro load+preprocess+train cell")

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = _remove_duplicate_subsample(src)
        if new_src != src:
            cell["source"] = _lines(new_src)
            cell["outputs"] = []
            cell["execution_count"] = None
            if "removed duplicate subsample" not in changes:
                changes.append("removed duplicate subsample")

    if changes:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changes


def main() -> None:
    for path in sorted(FOLDER.glob("*.ipynb")):
        changes = fix_notebook(path)
        if changes:
            print(f"{path.name}:")
            for c in changes:
                print(f"  - {c}")


if __name__ == "__main__":
    main()
