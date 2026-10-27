"""
Build Diffusion GANs notebooks from Other GANS templates.

Deep-copies every cell from the GAN notebooks and replaces only:
  - CTAB-GAN-Plus setup cell
  - GAN training cell(s) -> diffusion training block

All evaluation / fidelity / utility / privacy cells are preserved.
"""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[0]
SRC = REPO / "Other GANS"
DST = REPO / "Diffusion GANs"

MODEL_ORDER = '["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]'

SETUP_CELL = '''# Diffusion model dependencies
# TabDDPM: yandex-research/tab-ddpm (_vendor/tab-ddpm)
# CoDi: ChaejeongLee/CoDi (_vendor/CoDi)
# GOGGLE: tennisonliu/goggle (_vendor/goggle) — graph-based generative model (benchmarked alongside diffusion methods)
# ForestDiffusion: pip install ForestDiffusion
%pip install -q ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(".").resolve()
REPO_ROOT = NOTEBOOK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "goggle" / "src"))
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "CoDi"))
sys.path.insert(0, str(NOTEBOOK_DIR))

from diffusion_generators import (
    MODEL_ORDER,
    train_tabddpm,
    train_codi,
    train_goggle,
    train_forestdiffusion,
)
'''

GENERATOR_TEMPLATE = '''
# -----------------------------
# Diffusion models (TabDDPM, CoDi, GOGGLE, ForestDiffusion)
# -----------------------------
SYNTHETIC_N = 1000
DIFFUSION_SEED = 42
_categorical_columns = {categorical_columns}

print("Training TabDDPM...")
synthetic_tabddpm = train_tabddpm(
    {data_var},
    target_col={target_col},
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)
print("Training CoDi...")
synthetic_codi = train_codi(
    {data_var},
    target_col={target_col},
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)
print("Training GOGGLE...")
synthetic_goggle = train_goggle(
    {data_var},
    target_col={target_col},
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)
print("Training ForestDiffusion...")
synthetic_forestdiffusion = train_forestdiffusion(
    {data_var},
    target_col={target_col},
    categorical_columns=_categorical_columns,
    n_samples=SYNTHETIC_N,
    seed=DIFFUSION_SEED,
)

synthetic_outputs = {{
    "TabDDPM": synthetic_tabddpm,
    "CoDi": synthetic_codi,
    "GOGGLE": synthetic_goggle,
    "ForestDiffusion": synthetic_forestdiffusion,
}}
model_order = {model_order}
'''

REPLACEMENTS = [
    ("CTAB-GAN-Plus", "TabDDPM/CoDi/GOGGLE/ForestDiffusion"),
    ("CTAB GAN Plus", "TabDDPM"),
    ("CTAB-GAN", "TabDDPM"),
    ('synthetic_outputs["CTABGAN"]', 'synthetic_outputs["TabDDPM"]'),
    ("synthetic_ctabgan", "synthetic_outputs['TabDDPM']"),
    ("synthetic_wgan_gp", "synthetic_outputs['CoDi']"),
    ('"CTABGAN"', '"TabDDPM"'),
    ('"WGAN_GP"', '"CoDi"'),
    ("CTABGAN", "TabDDPM"),
    ("WGAN-GP", "CoDi"),
    ("WGAN_GP", "CoDi"),
    ('model_order = ["TabDDPM", "CoDi"]', f"model_order = {MODEL_ORDER}"),
    ('model_order = ["CTABGAN", "WGAN_GP"]', f"model_order = {MODEL_ORDER}"),
    ('model_order = ["WGAN_GP", "CTABGAN"]', f"model_order = {MODEL_ORDER}"),
    ('model_list = ["CTABGAN", "WGAN_GP"]', f"model_list = {MODEL_ORDER}"),
    ('model_list = ["TabDDPM", "CoDi"]', f"model_list = {MODEL_ORDER}"),
    ('for name in ["WGAN_GP", "CTABGAN"]', f"for name in {MODEL_ORDER}"),
    ('for name in ["CTABGAN", "WGAN_GP"]', f"for name in {MODEL_ORDER}"),
    ("Hungarian_Mahalanobis_CTABGAN_WGAN_GP", "Hungarian_Mahalanobis_Diffusion_Models"),
    ("Hungarian_Matchings_CTABGAN_WGAN_GP", "Hungarian_Matchings_Diffusion_Models"),
    ("Hungarian_Mahalanobis_WGAN_CTABGAN", "Hungarian_Mahalanobis_Diffusion_Models"),
    ("CTABGAN_WGAN_GP", "TabDDPM_CoDi_GOGGLE_ForestDiffusion"),
    (
        'synthetic_data = {\n    "CoDi": synthetic_outputs[\'CoDi\'],\n    "TabDDPM": synthetic_outputs[\'TabDDPM\']\n}',
        "synthetic_data = synthetic_outputs",
    ),
    (
        'synthetic_data = {\n    "WGAN_GP": synthetic_wgan_gp,\n    "CTABGAN": synthetic_outputs["CTABGAN"]\n}',
        "synthetic_data = synthetic_outputs",
    ),
    (
        'synthetic_data = {\n    "WGAN_GP": synthetic_outputs[\'CoDi\'],\n    "CTABGAN": synthetic_outputs["TabDDPM"]\n}',
        "synthetic_data = synthetic_outputs",
    ),
]

SUBPLOT_FIXES = [
    (r"plt\.subplots\(1,\s*2,", "plt.subplots(1, len(model_order),"),
    (r"plt\.subplots\(2,\s*1,", "plt.subplots(1, len(model_order),"),
    (r"plt\.subplots\(1,\s*2\)", "plt.subplots(1, len(model_order))"),
]


def _detect_training_context(src: str) -> dict:
    """Infer data variable, target column, and categorical columns from training code."""
    data_var = "data"
    if re.search(r"\bmagic_data\b", src):
        data_var = "magic_data"
    elif re.search(r"\bbank_data\b", src):
        data_var = "bank_data"
    elif re.search(r"\bwine_data\b", src):
        data_var = "wine_data"

    target_match = re.search(r'target_col\s*=\s*["\']([^"\']+)["\']', src)
    if not target_match:
        target_match = re.search(r'target_col\s*=\s*y\.columns\[0\]', src)
        target_col = "data.columns[-1]" if target_match else '"target"'
    else:
        target_col = repr(target_match.group(1))

    cat_match = re.search(r"categorical_columns\s*=\s*(\[[^\]]*\])", src)
    if cat_match:
        categorical_columns = cat_match.group(1)
    elif re.search(r"categorical_columns\s*=\s*([A-Za-z_][\w]*)", src):
        categorical_columns = re.search(
            r"categorical_columns\s*=\s*([A-Za-z_][\w]*)", src
        ).group(1)
    elif target_match and isinstance(target_col, str) and target_col.startswith('"'):
        categorical_columns = f"[{target_col}]"
    else:
        categorical_columns = "[]"

    return {
        "data_var": data_var,
        "target_col": target_col,
        "categorical_columns": categorical_columns,
        "model_order": MODEL_ORDER,
    }


def _strip_gan_training(src: str) -> str:
    """Remove CTABGAN / WGAN-GP training blocks; keep data loading and preprocessing."""
    out = src

    # Drop CTAB-GAN-Plus clone / import lines
    out = re.sub(r"!git clone.*CTAB-GAN-Plus.*\n", "", out)
    out = re.sub(r"sys\.path\.append\(['\"]\./CTAB-GAN-Plus['\"]\).*\n", "", out)
    out = re.sub(r"from model\.ctabgan import CTABGAN.*\n", "", out)

    # CTABGAN training block
    out = re.sub(
        r"# -+\s*\n#\s*CTABGAN.*?(?=# -+\s*\n#\s*WGAN|$)",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(r"ctabgan\s*=\s*CTABGAN\(.*?\)\s*\n", "", out, flags=re.DOTALL)
    out = re.sub(r"ctabgan\.fit\(\).*\n", "", out)
    out = re.sub(
        r"synthetic_(?:ctabgan|data)\s*=\s*ctabgan\..*?(?=\n#|\n\n|\Z)",
        "",
        out,
        flags=re.DOTALL,
    )

    # WGAN-GP training block
    out = re.sub(
        r"# -+\s*\n#\s*WGAN-GP.*?(?=synthetic_outputs\s*=|\Z)",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = re.sub(
        r"class Generator\(.*?(?=synthetic_wgan|\nsynthetic_outputs|\Z)",
        "",
        out,
        flags=re.DOTALL,
    )
    out = re.sub(
        r"synthetic_wgan_gp\s*=.*?(?=\n\n|\nsynthetic_outputs|\Z)",
        "",
        out,
        flags=re.DOTALL,
    )

    # Old synthetic_outputs / model_order from GAN training
    out = re.sub(
        r"synthetic_outputs\s*=\s*\{[^}]*(?:CTABGAN|WGAN_GP)[^}]*\}\s*\n?",
        "",
        out,
        flags=re.DOTALL,
    )
    out = re.sub(r'model_order\s*=\s*\["CTABGAN",\s*"WGAN_GP"\]\s*\n?', "", out)
    out = re.sub(r'model_order\s*=\s*\["WGAN_GP",\s*"CTABGAN"\]\s*\n?', "", out)

    # Torch imports only needed for WGAN
    if "class Generator" not in out and "gradient_penalty" not in out:
        out = re.sub(r"import torch\.nn as nn\n", "", out)
        out = re.sub(r"import torch\.optim as optim\n", "", out)
        if "torch." not in out:
            out = re.sub(r"import torch\n", "", out)

    return out.strip()


def _apply_replacements(src: str) -> str:
    out = src
    for old, new in REPLACEMENTS:
        out = out.replace(old, new)
    for pattern, repl in SUBPLOT_FIXES:
        out = re.sub(pattern, repl, out)
    return out


def _is_pure_setup_cell(src: str) -> bool:
    """Git-clone / CTAB-GAN-Plus import cell only (no dataset loading)."""
    if "CTAB-GAN-Plus" not in src and "!git clone" not in src:
        return False
    if "fetch_ucirepo" in src or "read_csv" in src:
        return False
    if re.search(r"ctabgan\.fit|class Generator", src, re.I):
        return False
    return True


def _has_data_loading(src: str) -> bool:
    return bool(
        re.search(
            r"fetch_ucirepo|read_csv|Load dataset|Load Covertype|Load CDC",
            src,
            re.I,
        )
    )


def _has_gan_training(src: str) -> bool:
    lowered = src.lower()
    return bool(
        re.search(
            r"ctabgan\.fit|ctabgan\s*=\s*ctabgan|ctabgan\s*=\s*tabddpm|class Generator|CTABGAN\(",
            src,
            re.I,
        )
        or (
            "synthetic_outputs" in lowered
            and ("ctabgan" in lowered or "wgan_gp" in lowered or "tabddpm" in lowered)
            and len(src) < 800
        )
    )


def _is_combined_training_cell(src: str) -> bool:
    return _has_data_loading(src) and _has_gan_training(src)


def _is_data_prelude_cell(src: str) -> bool:
    return _has_data_loading(src) and not _has_gan_training(src)


def _is_gan_fragment_cell(src: str) -> bool:
    if _has_data_loading(src):
        return False
    return _has_gan_training(src)


def _flush_training_buffer(buffer: list[str]) -> str:
    raw_merged = "\n\n".join(part for part in buffer if part.strip())
    ctx = _detect_training_context(raw_merged)
    merged = "\n\n".join(_strip_gan_training(part) for part in buffer if part.strip())
    merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
    gen = GENERATOR_TEMPLATE.format(**ctx)
    return merged + "\n" + gen


def _clear_cell_outputs(cell: dict) -> dict:
    cell = deepcopy(cell)
    cell["outputs"] = []
    cell["execution_count"] = None
    return cell


def _training_sequence_complete(last_src: str) -> bool:
    lowered = last_src.lower()
    if "synthetic_outputs" in lowered and (
        "ctabgan" in lowered or "wgan_gp" in lowered or "tabddpm" in lowered
    ):
        return True
    return "synthetic_wgan_gp" in lowered or "synthetic_wgan" in lowered


def _collect_training_sequence(cells: list[dict], start: int) -> tuple[list[str], int]:
    """Collect consecutive data-prelude and/or GAN-training cells."""
    buffer = ["".join(cells[start].get("source", []))]
    idx = start + 1
    while idx < len(cells):
        nxt = cells[idx]
        if nxt["cell_type"] != "code":
            break
        nxt_src = "".join(nxt.get("source", []))
        if _is_combined_training_cell(nxt_src):
            buffer.append(nxt_src)
            idx += 1
            break
        if _is_gan_fragment_cell(nxt_src) or (
            _has_gan_training(nxt_src) and not _has_data_loading(nxt_src)
        ):
            buffer.append(nxt_src)
            idx += 1
            if _training_sequence_complete(nxt_src):
                break
            continue
        if _is_data_prelude_cell(nxt_src):
            buffer.append(nxt_src)
            idx += 1
            continue
        break
    return buffer, idx


def transform_notebook(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    new_cells: list[dict] = []
    setup_done = False
    cells = nb["cells"]
    idx = 0

    if cells:
        title = "".join(cells[0].get("source", []))
        title = title.replace(
            "CTAB-GAN & WGAN-GP",
            "Diffusion Models (TabDDPM, CoDi, GOGGLE, ForestDiffusion)",
        )
        title = title.replace(
            "another two synthetic data",
            "four diffusion-based synthetic data",
        )
        title = title.replace("CTAB GAN Plus", "TabDDPM")
        title = title.replace("WGAN-GP", "CoDi")
        cells[0]["source"] = [title]

    while idx < len(cells):
        cell = cells[idx]
        src = "".join(cell.get("source", []))

        if _is_pure_setup_cell(src):
            if not setup_done:
                setup_cell = _clear_cell_outputs(cell)
                setup_cell["source"] = [SETUP_CELL]
                new_cells.append(setup_cell)
                setup_done = True
            idx += 1
            continue

        is_training_start = (
            _is_combined_training_cell(src)
            or _is_data_prelude_cell(src)
            or (_has_gan_training(src) and not _is_pure_setup_cell(src))
        )
        if is_training_start:
            buffer, idx = _collect_training_sequence(cells, idx)
            out_cell = _clear_cell_outputs(cell)
            out_cell["source"] = [_flush_training_buffer(buffer)]
            new_cells.append(out_cell)
            continue

        if cell["cell_type"] == "code":
            out_cell = _clear_cell_outputs(cell)
            out_cell["source"] = [_apply_replacements(src)]
            new_cells.append(out_cell)
        else:
            out_cell = deepcopy(cell)
            if out_cell["cell_type"] == "markdown":
                out_cell["source"] = [_apply_replacements(src)]
            new_cells.append(out_cell)
        idx += 1

    nb["cells"] = new_cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def verify_notebook_pair(gan_path: Path, diff_path: Path) -> dict:
    gan_nb = json.loads(gan_path.read_text(encoding="utf-8"))
    diff_nb = json.loads(diff_path.read_text(encoding="utf-8"))
    gan_src = "".join("".join(c.get("source", [])) for c in gan_nb["cells"])
    diff_src = "".join("".join(c.get("source", [])) for c in diff_nb["cells"])
    metrics = [
        "bivariate_quality",
        "evaluate_models",
        "privacy_metrics_mia",
        "hungarian_match",
        "compute_mmd",
        "c2st",
    ]
    return {
        "gan_cells": len(gan_nb["cells"]),
        "diff_cells": len(diff_nb["cells"]),
        "missing_metrics": [m for m in metrics if m in gan_src and m not in diff_src],
    }


def main() -> None:
    gen_src = DST / "diffusion_generators.py"
    if not gen_src.exists():
        alt = REPO / "diffusion_generators.py"
        if alt.exists():
            shutil.copy2(alt, gen_src)

    DST.mkdir(parents=True, exist_ok=True)

    for item in SRC.iterdir():
        if item.name.endswith(".ipynb"):
            continue
        dest = DST / item.name
        if item.is_dir():
            if not dest.exists():
                shutil.copytree(item, dest)
        elif not dest.exists():
            shutil.copy2(item, dest)

    print("Building diffusion notebooks (full cell clone)...")
    for nb in sorted(SRC.glob("*.ipynb")):
        out_name = nb.name.replace("_other_GAN", "_diffusion").replace(
            "_other GAN", "_diffusion"
        )
        if out_name == nb.name:
            out_name = nb.stem + "_diffusion.ipynb"
        out_path = DST / out_name
        transform_notebook(nb, out_path)
        report = verify_notebook_pair(nb, out_path)
        status = "OK" if not report["missing_metrics"] else "MISSING"
        print(
            f"  {out_name}: GAN={report['gan_cells']} DIFF={report['diff_cells']} "
            f"[{status}]"
        )
        if report["missing_metrics"]:
            print(f"    missing: {report['missing_metrics']}")


if __name__ == "__main__":
    main()
