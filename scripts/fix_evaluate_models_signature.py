"""Fix broken evaluate_models signatures from label= -> label_col= migration."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"


def fix_cell(src: str) -> str:
    if "def evaluate_models" not in src:
        return src

    # Broken default: label_col=label_col at function definition time
    src = re.sub(
        r"def evaluate_models\(([^)]*?)label_col=label_col",
        r"def evaluate_models(\1label_col",
        src,
    )

    # Regression helpers still named `label` but called with label_col=
    if re.search(r"def evaluate_models\([^)]*\blabel\b", src):
        src = re.sub(
            r"def evaluate_models\(train_df, test_df, label, models",
            "def evaluate_models(train_df, test_df, label_col, models",
            src,
        )
        src = re.sub(r"\[label\]", "[label_col]", src)
        src = re.sub(r"train_df\[label\]", "train_df[label_col]", src)
        src = re.sub(r"test_df\[label\]", "test_df[label_col]", src)
        src = re.sub(r"y_train\[label\]", "y_train[label_col]", src)
        src = re.sub(r"y_test\[label\]", "y_test[label_col]", src)

    return src


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
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
