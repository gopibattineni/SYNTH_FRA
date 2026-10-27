"""Keep only TabDDPM and ForestDiffusion in diffusion_dataleak notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Single run_Data_leak_Synth_Quality" / "diffusion_dataleak"
GENERATORS = FOLDER / "diffusion_generators.py"

REPLACEMENTS = [
    ("# Diffusion model dependencies (TabDDPM + CoDi)", "# Diffusion model dependencies (TabDDPM + ForestDiffusion)"),
    ("# CoDi: ChaejeongLee/CoDi (_vendor/CoDi)\n", "# ForestDiffusion: pip install ForestDiffusion\n"),
    ('sys.path.insert(0, str(REPO_ROOT / "_vendor" / "goggle" / "src"))\n', ""),
    ('sys.path.insert(0, str(REPO_ROOT / "_vendor" / "CoDi"))\n', ""),
    ('sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))\n', 'sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))\nsys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm" / "scripts"))\n'),
    ("from diffusion_generators import train_tabddpm, train_codi", "from diffusion_generators import train_tabddpm, train_forestdiffusion"),
    ("train_codi", "train_forestdiffusion"),
    ("synthetic_codi", "synthetic_forestdiffusion"),
    ("'CoDi'", "'ForestDiffusion'"),
    ('"CoDi"', '"ForestDiffusion"'),
    ("# CoDi\n", "# ForestDiffusion\n"),
    ("# ---------------------------------------------------\n# CoDi\n", "# ---------------------------------------------------\n# ForestDiffusion\n"),
    ("Training CoDi...", "Training ForestDiffusion..."),
    ("CoDi:", "ForestDiffusion:"),
    ("CoDi Failed", "ForestDiffusion Failed"),
    ("'TabDDPM', 'CoDi'", "'TabDDPM', 'ForestDiffusion'"),
    ("['TabDDPM', 'CoDi']", "['TabDDPM', 'ForestDiffusion']"),
    (
        "%pip install -q ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX",
        """# libzero/rtdl pin torch<2; use --no-deps on torch 2.x (TabDDPM still works)
%pip install -q ForestDiffusion xgboost category-encoders imbalanced-learn absl-py tensorboardX icecream dython optuna skorch pyarrow tomli tomli-w
%pip install -q "pynvml>=11,<12"
%pip install -q "libzero==0.0.8" "rtdl==0.0.13" --no-deps""",
    ),
]

# Fix accidental double-replacement if script re-run
UNDO = [
    ("train_forestdiffusionusion", "train_forestdiffusion"),
    ("synthetic_forestdiffusionusion", "synthetic_forestdiffusion"),
    ("ForestDiffusionForestDiffusion", "ForestDiffusion"),
]


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in UNDO:
        text = text.replace(old, new)
    return text


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        new_src = apply_replacements(src)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def fix_generators() -> bool:
    text = GENERATORS.read_text(encoding="utf-8")
    new_text = text.replace(
        'MODEL_ORDER = ["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]',
        'MODEL_ORDER = ["TabDDPM", "ForestDiffusion"]',
    )
    new_text = new_text.replace(
        """    return {
        "TabDDPM": train_tabddpm(df, target_col, categorical_columns, n_samples, seed),
        "CoDi": train_codi(df, target_col, categorical_columns, n_samples, seed),
        "GOGGLE": train_goggle(df, target_col, categorical_columns, n_samples, seed),
        "ForestDiffusion": train_forestdiffusion(
            df, target_col, categorical_columns, n_samples, seed
        ),
    }""",
        """    return {
        "TabDDPM": train_tabddpm(df, target_col, categorical_columns, n_samples, seed),
        "ForestDiffusion": train_forestdiffusion(
            df, target_col, categorical_columns, n_samples, seed
        ),
    }""",
    )
    if new_text != text:
        GENERATORS.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    if fix_generators():
        print("Updated diffusion_dataleak/diffusion_generators.py")
    fixed = []
    for path in sorted(FOLDER.rglob("*.ipynb")):
        if fix_notebook(path):
            fixed.append(str(path.relative_to(FOLDER)))
    print(f"Updated {len(fixed)} notebooks:")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
