"""Keep only TabDDPM + ForestDiffusion in diffusion_dataleak notebooks."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OLD_MODEL_ORDER = """model_order = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "ForestDiffusion",
    "TabDDPM"
]"""

NEW_MODEL_ORDER = """model_order = ["TabDDPM", "ForestDiffusion"]"""

SDV_CELL_MARKER = "# SDV MODELS"
TRTR_DUPLICATE_MARKER = "--- Starting TRTR Evaluation (Train Real, Test Real) ---"

SDV_IMPORT_RE = re.compile(
    r"from sdv\.single_table import \(\s*"
    r"CTGANSynthesizer,\s*"
    r"CopulaGANSynthesizer,\s*"
    r"TVAESynthesizer,\s*"
    r"GaussianCopulaSynthesizer,?\s*"
    r"\)\s*\n",
    re.MULTILINE,
)

LOOP_GUARD = (
    "for synth_name in model_order:\n"
    "\n"
    "    print(f\"{synth_name} - TSTR\")\n"
    "\n"
    "    synthetic_train_df = synthetic_datasets[synth_name]\n"
)

LOOP_GUARD_REPLACEMENT = (
    "for synth_name in model_order:\n"
    "\n"
    "    if synth_name not in synthetic_datasets:\n"
    "        print(f\"Skipping {synth_name} — not trained\")\n"
    "        continue\n"
    "\n"
    "    print(f\"{synth_name} - TSTR\")\n"
    "\n"
    "    synthetic_train_df = synthetic_datasets[synth_name]\n"
)


def strip_notebook(notebook_path: Path) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    new_cells = []

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        src = "".join(cell.get("source", []))

        if src.strip().startswith(SDV_CELL_MARKER):
            continue

        if TRTR_DUPLICATE_MARKER in src:
            continue

        src = SDV_IMPORT_RE.sub("", src)

        if OLD_MODEL_ORDER in src:
            src = src.replace(OLD_MODEL_ORDER, NEW_MODEL_ORDER)

        if LOOP_GUARD in src:
            src = src.replace(LOOP_GUARD, LOOP_GUARD_REPLACEMENT)

        cell["source"] = src.splitlines(keepends=True) or [""]
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)

    nb["cells"] = new_cells
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {notebook_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "Single run_Data_leak_Synth_Quality"
        / "diffusion_dataleak"
        / "1. Cancer"
        / "cancer.ipynb",
    )
    args = parser.parse_args()
    strip_notebook(args.notebook.resolve())


if __name__ == "__main__":
    main()
