"""Add tab-ddpm/scripts to sys.path and TabDDPM pip deps in Diffusion notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

OLD_PATH = 'sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))'
NEW_PATH = (
    'sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm"))\n'
    'sys.path.insert(0, str(REPO_ROOT / "_vendor" / "tab-ddpm" / "scripts"))'
)

OLD_PIP = "%pip install -q ForestDiffusion xgboost category-encoders imbalanced-learn absl-py tensorboardX"
NEW_PIP = "%pip install -q ForestDiffusion xgboost category-encoders imbalanced-learn absl-py tensorboardX icecream dython optuna skorch pyarrow tomli tomli-w"


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        if OLD_PATH in src and 'tab-ddpm" / "scripts"' not in src:
            src = src.replace(OLD_PATH, NEW_PATH)
        if OLD_PIP in src:
            src = src.replace(OLD_PIP, NEW_PIP)
        if src != orig:
            lines = src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = [p.name for p in sorted(FOLDER.glob("*.ipynb")) if fix_notebook(p)]
    print(f"Fixed {len(fixed)} notebooks")


if __name__ == "__main__":
    main()
