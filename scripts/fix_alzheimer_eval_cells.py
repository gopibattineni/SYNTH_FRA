"""Remove Cancer B/M label mapping leftovers from Alzheimer diffusion notebook."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Diffusion GANs"
    / "2. Alzhimers_diffusion.ipynb"
)

OLD_BLOCK = '''data_eval = data.copy()
if data_eval[label_col].dtype == object:
    data_eval[label_col] = data_eval[label_col].map({"B": 0, "M": 1})

synthetic_eval = {}
for name in model_order:
    df = synthetic_outputs[name].copy()
    if label_col in df.columns and df[label_col].dtype == object:
        df[label_col] = df[label_col].map({"B": 0, "M": 1})
    synthetic_eval[name] = df'''

NEW_BLOCK = '''data_eval = ad_data.copy()
synthetic_eval = {name: synthetic_outputs[name].copy() for name in model_order}'''

OLD_BLOCK2 = '''data_eval = data.copy()
if data_eval[label_col].dtype == object:
    data_eval[label_col] = data_eval[label_col].map({"B": 0, "M": 1})'''

NEW_BLOCK2 = "data_eval = ad_data.copy()"


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = src
        new_src = new_src.replace(OLD_BLOCK, NEW_BLOCK)
        new_src = new_src.replace(OLD_BLOCK2, NEW_BLOCK2)
        new_src = new_src.replace(
            'label_col = "Group" if "Group" in data.columns else "Group"',
            'label_col = "Group"',
        )
        # Per-generator loop with B/M map on synth
        new_src = new_src.replace(
            '''    if label_col in synth_eval.columns and synth_eval[label_col].dtype == object:
        synth_eval[label_col] = synth_eval[label_col].map({"B": 0, "M": 1})

    tstr_results''',
            "    tstr_results",
        )
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed += 1
    if changed:
        NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {changed} cells in {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
