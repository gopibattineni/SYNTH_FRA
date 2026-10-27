"""Keep only TabDDPM + ForestDiffusion in diffusion_dataleak Cancer notebook."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "Single run_Data_leak_Synth_Quality"
    / "diffusion_dataleak"
    / "1. Cancer"
    / "cancer.ipynb"
)

SDV_IMPORT_BLOCK = """from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer
)

"""

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


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    new_cells = []

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        src = "".join(cell.get("source", []))

        # Drop entire SDV training cell
        if src.strip().startswith(SDV_CELL_MARKER):
            continue

        # Drop duplicate standalone TRTR cell (main cell does TRTR + TSTR)
        if "--- Starting TRTR Evaluation (Train Real, Test Real) ---" in src:
            continue

        # Remove SDV synthesizer imports; keep metadata + evaluate_quality for diffusion
        if SDV_IMPORT_BLOCK in src:
            src = src.replace(SDV_IMPORT_BLOCK, "")

        if OLD_MODEL_ORDER in src:
            src = src.replace(OLD_MODEL_ORDER, NEW_MODEL_ORDER)

        # Only evaluate generators present in synthetic_datasets
        loop_guard = (
            "for synth_name in model_order:\n"
            "\n"
            "    print(f\"{synth_name} - TSTR\")\n"
            "\n"
            "    synthetic_train_df = synthetic_datasets[synth_name]\n"
        )
        if loop_guard in src:
            src = src.replace(
                loop_guard,
                "for synth_name in model_order:\n"
                "\n"
                "    if synth_name not in synthetic_datasets:\n"
                "        print(f\"Skipping {synth_name} — not trained\")\n"
                "        continue\n"
                "\n"
                "    print(f\"{synth_name} - TSTR\")\n"
                "\n"
                "    synthetic_train_df = synthetic_datasets[synth_name]\n",
            )

        cell["source"] = src.splitlines(keepends=True) or [""]
        cell["outputs"] = []
        cell["execution_count"] = None
        new_cells.append(cell)

    nb["cells"] = new_cells
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
