"""
Build diffusion_dataleak notebooks from Single run_Data_leak_Synth_Quality templates.

Replaces CTAB-GAN+ and WGAN-GP with TabDDPM and CoDi while preserving:
  - train/test split before synthesis (no leakage)
  - SDV baselines (CTGAN, CopulaGAN, TVAE, GaussianCopula)
  - SDV evaluate_quality fidelity + TRTR/TSTR utility pipeline
"""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[0]
SRC = REPO / "Single run_Data_leak_Synth_Quality"
DST = SRC / "diffusion_dataleak"

SETUP_CELL = '''# Diffusion model dependencies (TabDDPM + CoDi)
# TabDDPM: yandex-research/tab-ddpm (_vendor/tab-ddpm)
# CoDi: ChaejeongLee/CoDi (_vendor/CoDi)
%pip install -q ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(".").resolve()
DIFFUSION_PKG = NOTEBOOK_DIR.parent
_repo = NOTEBOOK_DIR
while not (_repo / "_vendor" / "tab-ddpm").is_dir() and _repo.parent != _repo:
    _repo = _repo.parent
REPO_ROOT = _repo
if not (REPO_ROOT / "_vendor" / "tab-ddpm").is_dir():
    raise FileNotFoundError(
        "Missing _vendor/tab-ddpm. Clone per README: "
        "git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"
    )
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "goggle" / "src"))
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "CoDi"))
sys.path.insert(0, str(DIFFUSION_PKG))

from diffusion_generators import train_tabddpm, train_codi
'''

REPLACEMENTS = [
    ("CTAB-GAN-Plus", "TabDDPM/CoDi"),
    ("CTAB-GAN", "TabDDPM"),
    ('"CTABGAN"', '"TabDDPM"'),
    ("'CTABGAN'", "'TabDDPM'"),
    ('"WGAN_GP"', '"CoDi"'),
    ("'WGAN_GP'", "'CoDi'"),
    ("CTABGAN", "TabDDPM"),
    ("WGAN_GP", "CoDi"),
    ("WGAN-GP", "CoDi"),
    ("from model.ctabgan import TabDDPM", ""),
    ("from model.ctabgan import CTABGAN", ""),
    ("ctabgan", "tabddpm_removed"),
    ("tabddpm_removed", "tabddpm_removed"),
    ("tabddpm_unused", ""),
]

METRO_GENERATORS_LINE = (
    "'CTGAN', 'CopulaGAN', 'TVAE', 'GaussianCopula', 'TabDDPM', 'CoDi'"
)


def _detect_n_samples(src: str) -> str:
    for name in ("N_SYNTH_SAMPLES", "N_SAMPLES", "SYNTHETIC_N"):
        if re.search(rf"\b{name}\b", src):
            return name
    return "N_SAMPLES"


def _detect_categorical_columns(src: str) -> str:
    for pat in (
        r"CTABGAN_CATEGORICAL\s*=\s*(\[[^\]]*\])",
        r"categorical_columns\s*=\s*(\[[^\]]*\])",
    ):
        m = re.search(pat, src)
        if m:
            return m.group(1)
    return "[target_col]"


def _quality_block_simple(model_key: str, synth_var: str) -> str:
    return f"""    quality = evaluate_quality(
        real_data=train_real,
        synthetic_data={synth_var},
        metadata=train_metadata,
    )

    scores["{model_key}"] = quality.get_score()

    print("{model_key}:", round(scores["{model_key}"], 4))"""


def _quality_block_metro(model_key: str, synth_var: str) -> str:
    return f"""        if RUN_QUALITY_EVAL:
            quality = evaluate_quality(
                real_data=train_real,
                synthetic_data={synth_var},
                metadata=train_metadata,
            )
            scores['{model_key}'] = quality.get_score()
            print('{model_key}:', round(scores['{model_key}'], 4))
        else:
            print('{model_key}: trained (quality eval skipped)')"""


def _tabddpm_block(src: str, metro_style: bool) -> str:
    n_var = _detect_n_samples(src)
    cat_cols = _detect_categorical_columns(src)
    quality = _quality_block_metro("TabDDPM", "synthetic_tabddpm") if metro_style else _quality_block_simple(
        "TabDDPM", "synthetic_tabddpm"
    )
    indent = "    " if metro_style else ""
    try_line = f"{indent}try:\n" if not metro_style else ""
    except_line = (
        f"\n{indent}except Exception as e:\n{indent}    print(\"TabDDPM Failed:\", e)"
        if not metro_style
        else ""
    )
    if metro_style:
        return f"""if 'TabDDPM' in GENERATORS_TO_EVAL:
    import traceback
    try:
        print('Training TabDDPM...')
        synthetic_tabddpm = train_tabddpm(
            train_real,
            target_col=target_col,
            categorical_columns={cat_cols},
            n_samples={n_var},
            seed=seed,
        )
        synthetic_datasets['TabDDPM'] = synthetic_tabddpm.copy()
{quality}
    except Exception as e:
        print('TabDDPM Failed:', e)
        traceback.print_exc()
else:
    print('TabDDPM: skipped (not in GENERATORS_TO_EVAL)')
"""
    return f"""# ---------------------------------------------------
# TabDDPM
# ---------------------------------------------------

{try_line}
    print("Training TabDDPM...")
    synthetic_tabddpm = train_tabddpm(
        train_real,
        target_col=target_col,
        categorical_columns={cat_cols},
        n_samples={n_var},
        seed=seed,
    )

    synthetic_datasets["TabDDPM"] = synthetic_tabddpm.copy()

{quality}
{except_line}
"""


def _codi_block(src: str, metro_style: bool) -> str:
    n_var = _detect_n_samples(src)
    cat_cols = _detect_categorical_columns(src)
    quality = _quality_block_metro("CoDi", "synthetic_codi") if metro_style else _quality_block_simple(
        "CoDi", "synthetic_codi"
    )
    if metro_style:
        return f"""# CoDi
if 'CoDi' in GENERATORS_TO_EVAL:
    import traceback
    try:
        print('Training CoDi...')
        synthetic_codi = train_codi(
            train_real,
            target_col=target_col,
            categorical_columns={cat_cols},
            n_samples={n_var},
            seed=seed,
        )
        synthetic_datasets['CoDi'] = synthetic_codi.copy()
        print('CoDi: synthesis complete')
{quality}
    except Exception as e:
        print('CoDi Failed (training/sampling):')
        traceback.print_exc()
    if 'CoDi' in synthetic_datasets and RUN_QUALITY_EVAL:
        pass
else:
    print('CoDi: skipped (not in GENERATORS_TO_EVAL)')
"""
    return f"""# CoDi

try:
    import traceback

    print("Training CoDi...")
    synthetic_codi = train_codi(
        train_real,
        target_col=target_col,
        categorical_columns={cat_cols},
        n_samples={n_var},
        seed=seed,
    )

    synthetic_datasets["CoDi"] = synthetic_codi.copy()

{quality}

    del synthetic_codi

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception as e:

    print("CoDi Failed:")
    traceback.print_exc()
"""


def _preamble_before_ctabgan(src: str) -> str:
    for marker in (
        r"# -+\s*\n#\s*CTABGAN",
        r"if 'CTABGAN' in GENERATORS_TO_EVAL",
        r"# CTABGAN",
    ):
        m = re.search(marker, src, re.IGNORECASE)
        if m:
            return src[: m.start()].rstrip() + "\n\n"
    m = re.search(
        r"(.*?train_metadata\.detect_from_dataframe\(train_real\)\s*\n)",
        src,
        re.DOTALL,
    )
    if m:
        return m.group(1)
    return src


def _replace_ctabgan_cell(src: str, metro_style: bool) -> str:
    preamble = _preamble_before_ctabgan(src)
  # drop CTAB-only helpers from metro cells
    preamble = re.sub(
        r"def prepare_ctabgan_train_df.*?return out\n\n",
        "",
        preamble,
        flags=re.DOTALL,
    )
    preamble = re.sub(
        r"def encode_ctabgan_synthetic.*?return align_to_train_schema.*?\n\n",
        "",
        preamble,
        flags=re.DOTALL,
    )
    preamble = re.sub(r"CTABGAN_[A-Z_]+\s*=\s*[^\n]+\n", "", preamble)
    merged = preamble.rstrip() + "\n\n" + _tabddpm_block(src, metro_style)
    return _apply_replacements(merged)


def _is_setup_cell(src: str) -> bool:
    return "CTAB-GAN-Plus" in src and "git clone" in src


def _is_wgan_cell(src: str) -> bool:
    return "class Generator" in src and "gradient_penalty" in src


def _has_ctabgan_training(src: str) -> bool:
    return bool(
        re.search(r"ctabgan\.fit|if 'CTABGAN' in GENERATORS_TO_EVAL|CTABGAN\(", src, re.I)
    )


def _apply_replacements(src: str) -> str:
    out = src
    for old, new in REPLACEMENTS:
        out = out.replace(old, new)
    out = re.sub(
        r"'CTGAN', 'CopulaGAN', 'TVAE', 'GaussianCopula', 'CoDi', 'TabDDPM'",
        METRO_GENERATORS_LINE,
        out,
    )
    out = re.sub(
        r"\['CTGAN', 'CopulaGAN', 'TVAE', 'GaussianCopula', 'CoDi', 'TabDDPM'\]",
        f"[{METRO_GENERATORS_LINE}]",
        out,
    )
    out = re.sub(
        r"model_order = \[m for m in \['TabDDPM', 'CoDi'\]",
        "model_order = [m for m in ['TabDDPM', 'CoDi']",
        out,
    )
    return out


def _clear_cell(cell: dict) -> dict:
    cell = deepcopy(cell)
    cell["outputs"] = []
    cell["execution_count"] = None
    return cell


def transform_notebook(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    new_cells: list[dict] = []
    setup_done = False
    metro_style = "GENERATORS_TO_EVAL" in "".join(
        "".join(c.get("source", [])) for c in nb["cells"]
    )

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if _is_setup_cell(src):
            if not setup_done:
                c = _clear_cell(cell)
                c["source"] = [SETUP_CELL]
                new_cells.append(c)
                setup_done = True
            continue

        if _is_wgan_cell(src):
            c = _clear_cell(cell)
            c["source"] = [_codi_block(src, metro_style)]
            new_cells.append(c)
            continue

        if _has_ctabgan_training(src):
            c = _clear_cell(cell)
            c["source"] = [_replace_ctabgan_cell(src, metro_style)]
            new_cells.append(c)
            continue

        c = _clear_cell(cell) if cell["cell_type"] == "code" else deepcopy(cell)
        if cell["cell_type"] == "code":
            c["source"] = [_apply_replacements(src)]
        else:
            c["source"] = [_apply_replacements(src)] if c.get("source") else []
        new_cells.append(c)

    nb["cells"] = new_cells
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def copy_assets() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    gen_src = REPO / "Diffusion GANs" / "diffusion_generators.py"
    if gen_src.exists():
        shutil.copy2(gen_src, DST / "diffusion_generators.py")

    skip = {"diffusion_dataleak", "python_scripts"}
    for item in SRC.iterdir():
        if item.name in skip or not item.is_dir():
            continue
        dest_dir = DST / item.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in item.iterdir():
            if f.suffix == ".ipynb":
                continue
            if not f.is_file():
                continue
            dest = dest_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)


def main() -> None:
    copy_assets()
    skip = {"diffusion_dataleak", "python_scripts"}
    notebooks = []
    for item in sorted(SRC.iterdir()):
        if item.name in skip or not item.is_dir():
            continue
        for nb_path in sorted(item.glob("*.ipynb")):
            notebooks.append(nb_path)
    print(f"Building {len(notebooks)} diffusion_dataleak notebooks...")
    for nb_path in notebooks:
        rel = nb_path.relative_to(SRC)
        out_path = DST / rel
        transform_notebook(nb_path, out_path)
        print(f"  {rel}")
    print("Done.")


if __name__ == "__main__":
    main()
