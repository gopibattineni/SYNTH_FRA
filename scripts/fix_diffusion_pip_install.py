"""Fix Diffusion GANs pip install cells (libzero/rtdl ResolutionImpossible on torch 2.x)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

OLD = '%pip install -q ForestDiffusion xgboost category-encoders libzero rtdl imbalanced-learn absl-py tensorboardX'

NEW = """# libzero/rtdl pin torch<2; use --no-deps on torch 2.x (TabDDPM still works)
%pip install -q ForestDiffusion xgboost category-encoders imbalanced-learn absl-py tensorboardX icecream dython optuna skorch pyarrow tomli tomli-w
%pip install -q "pynvml>=11,<12"
%pip install -q "libzero==0.0.8" "rtdl==0.0.13" --no-deps"""

VENDOR_CHECK = """
# Verify vendored TabDDPM / CoDi / GOGGLE checkouts (see README Setup)
_vendor_missing = [
    p.name for p in [
        REPO_ROOT / "_vendor" / "tab-ddpm",
        REPO_ROOT / "_vendor" / "CoDi",
        REPO_ROOT / "_vendor" / "goggle",
    ]
    if not p.exists()
]
if _vendor_missing:
    raise FileNotFoundError(
        "Missing _vendor checkouts: "
        + ", ".join(_vendor_missing)
        + ". Clone per README: git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"
        + " && git clone https://github.com/ChaejeongLee/CoDi _vendor/CoDi"
        + " && git clone https://github.com/tennisonliu/goggle _vendor/goggle"
    )
"""


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if OLD not in src:
            continue
        src = src.replace(OLD, NEW)
        if "_vendor_missing" not in src and "from diffusion_generators import" in src:
            src = src.replace(
                "from diffusion_generators import",
                VENDOR_CHECK.strip() + "\n\nfrom diffusion_generators import",
            )
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cell["source"] = lines
        cell["outputs"] = []
        cell["execution_count"] = None
        changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = []
    for p in sorted(FOLDER.glob("*.ipynb")):
        if fix_notebook(p):
            fixed.append(p.name)
    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)


if __name__ == "__main__":
    main()
