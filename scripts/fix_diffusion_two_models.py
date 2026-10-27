"""Keep only TabDDPM and ForestDiffusion in Diffusion GANs notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"
GENERATORS = ROOT / "Diffusion GANs" / "diffusion_generators.py"

MODEL_ORDER_OLD = '["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]'
MODEL_ORDER_NEW = '["TabDDPM", "ForestDiffusion"]'

REPLACEMENTS = [
    (MODEL_ORDER_OLD, MODEL_ORDER_NEW),
    ('["CoDi", "TabDDPM"]', MODEL_ORDER_NEW),
    ('["TabDDPM", "CoDi"]', MODEL_ORDER_NEW),
    ('model_list = ["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]', f"model_list = {MODEL_ORDER_NEW}"),
    ('generators = ["TabDDPM", "CoDi"]', f'generators = {MODEL_ORDER_NEW}'),
    ("TabDDPM, CoDi, GOGGLE, ForestDiffusion", "TabDDPM, ForestDiffusion"),
    ("TabDDPM / CoDi / GOGGLE", "TabDDPM"),
    ("# CoDi: ChaejeongLee/CoDi (_vendor/CoDi)\n", ""),
    (
        "# GOGGLE: tennisonliu/goggle (_vendor/goggle) — graph-based generative model (benchmarked alongside diffusion methods)\n",
        "",
    ),
    ('sys.path.insert(0, str(REPO_ROOT / "_vendor" / "goggle" / "src"))\n', ""),
    ('sys.path.insert(0, str(REPO_ROOT / "_vendor" / "CoDi"))\n', ""),
    ("    train_codi,\n", ""),
    ("    train_goggle,\n", ""),
    ('    "CoDi": synthetic_codi,\n', ""),
    ('    "GOGGLE": synthetic_goggle,\n', ""),
    (
        "# Verify vendored TabDDPM / CoDi / GOGGLE checkouts (see README Setup)",
        "# Verify vendored TabDDPM checkout (see README Setup)",
    ),
    ('        REPO_ROOT / "_vendor" / "CoDi",\n', ""),
    ('        REPO_ROOT / "_vendor" / "goggle",\n', ""),
    (
        '        + ". Clone per README: git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"\n'
        '        + " && git clone https://github.com/ChaejeongLee/CoDi _vendor/CoDi"\n'
        '        + " && git clone https://github.com/tennisonliu/goggle _vendor/goggle"\n',
        '        + ". Clone per README: git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"\n',
    ),
    (
        'print("Mahalanobis MIA - Real vs Synthetic (CoDi & TabDDPM)")',
        'print("Mahalanobis MIA - Real vs Synthetic (TabDDPM & ForestDiffusion)")',
    ),
    ("TabDDPM_CoDi", "TabDDPM_ForestDiffusion"),
    (
        '    "CoDi": synthetic_outputs["CoDi"][target_col].value_counts().sort_index()\n',
        '    "ForestDiffusion": synthetic_outputs["ForestDiffusion"][target_col].value_counts().sort_index()\n',
    ),
    ('    "CoDi"\n', '    "ForestDiffusion"\n'),
    (
        '    "CoDi": synthetic_outputs[\'CoDi\'],\n',
        '    "ForestDiffusion": synthetic_outputs["ForestDiffusion"],\n',
    ),
    (
        "# Replace synthetic_outputs['CoDi'] and synthetic_outputs['TabDDPM'] with the variables you really have",
        "# Use TabDDPM and ForestDiffusion synthetic outputs",
    ),
    (
        '    "CoDi": synthetic_outputs[\'CoDi\'],\n    "TabDDPM": synthetic_outputs[\'TabDDPM\'],',
        '    "TabDDPM": synthetic_outputs["TabDDPM"],\n    "ForestDiffusion": synthetic_outputs["ForestDiffusion"],',
    ),
    (
        '    "CoDi": synthetic_outputs["CoDi"],\n    "TabDDPM": synthetic_outputs["TabDDPM"]',
        '    "TabDDPM": synthetic_outputs["TabDDPM"],\n    "ForestDiffusion": synthetic_outputs["ForestDiffusion"]',
    ),
    ("2. **CoDi**", "2. **ForestDiffusion**"),
    ("Wassertian GAN with gradiant penalty", "Forest-based diffusion for tabular data"),
    ("Wasserstein GAN with gradient penalty", "Forest-based diffusion for tabular data"),
    ("four diffusion-based synthetic data generation models", "two diffusion-based synthetic data generation models"),
    ("two synthetic data generation models", "two diffusion-based synthetic data generation models"),
]

TRAIN_CODI_RE = re.compile(
    r'print\("Training CoDi\.\.\."\)\s*\n'
    r"synthetic_codi = train_codi\(\s*\n.*?\n\)\s*\n",
    re.DOTALL,
)
TRAIN_GOGGLE_RE = re.compile(
    r'print\("Training GOGGLE\.\.\."\)\s*\n'
    r"synthetic_goggle = train_goggle\(\s*\n.*?\n\)\s*\n",
    re.DOTALL,
)


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    text = TRAIN_CODI_RE.sub("", text)
    text = TRAIN_GOGGLE_RE.sub("", text)
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
    if new_text != text:
        GENERATORS.write_text(new_text, encoding="utf-8")
        return True
    return False


def main() -> None:
    if fix_generators():
        print("Updated diffusion_generators.py")
    fixed = []
    for path in sorted(set(FOLDER.glob("*.ipynb"))):
        if fix_notebook(path):
            fixed.append(path.name)
    print(f"Updated {len(fixed)} notebooks:")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
