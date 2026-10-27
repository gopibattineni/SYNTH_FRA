"""Add label dtype coercion to classification evaluate_models() in Diffusion GANs notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

COERCE_BLOCK = """
    def _coerce_class_labels(y):
        y = pd.Series(y)
        if y.dtype == object or str(y.dtype) == "string":
            y = pd.to_numeric(y, errors="coerce")
        return y.round().astype(int)

    y_train = _coerce_class_labels(y_train)
    y_test = _coerce_class_labels(y_test)
"""

ANCHORS = [
    """    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=seed, stratify=y_test
    )

    scaler = StandardScaler().fit(X_train)""",
    """    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=seed, stratify=y_test
    )

    # Scale using TRAIN statistics only
    scaler = StandardScaler().fit(X_train)""",
    """    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=seed, stratify=y_test
    )

    scaler = StandardScaler()""",
]


def fix_cell(src: str) -> str:
    if "def evaluate_models" not in src or "accuracy_score" not in src:
        return src
    if "_coerce_class_labels" in src:
        return src
    for anchor in ANCHORS:
        if anchor in src:
            replaced = anchor.replace("\n    scaler", COERCE_BLOCK + "\n    scaler", 1)
            replaced = replaced.replace(
                "\n    # Scale using TRAIN statistics only\n    scaler",
                COERCE_BLOCK + "\n    # Scale using TRAIN statistics only\n    scaler",
                1,
            )
            return src.replace(anchor, replaced, 1)
    return src


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "import pandas" not in src and "def evaluate_models" in src and "accuracy_score" in src:
            src = "import pandas as pd\n" + src
        new_src = fix_cell(src)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    fixed = [p.name for p in sorted(FOLDER.glob("*.ipynb")) if fix_notebook(p)]
    print(f"Fixed {len(fixed)} notebooks:")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
