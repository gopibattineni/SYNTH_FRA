"""Audit diffusion preprocessing across all 15 Diffusion GANs notebooks."""
from __future__ import annotations

import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1] / "Diffusion GANs"

ISSUES = []


def audit(name: str, src: str) -> None:
    if "synthetic_tabddpm = train_tabddpm" not in src:
        ISSUES.append(f"{name}: missing diffusion training cell")
        return
    if "_categorical_columns = data" in src:
        ISSUES.append(f"{name}: _categorical_columns = data (bug)")
    if "known_categorical" in src and "train_tabddpm" in src:
        ISSUES.append(f"{name}: known_categorical still referenced in training cell")
    if 'target_col="target"' in src or "target_col='target'" in src:
        ISSUES.append(f"{name}: wrong target_col='target'")
    if "train_tabddpm(" in src and "train_real" not in src.split("train_tabddpm(")[1].split(")")[0]:
        # crude check - train first arg should be train_real
        if "train_tabddpm(\n    data," in src or "train_tabddpm(\n    magic_data," in src:
            ISSUES.append(f"{name}: train_tabddpm not using train_real")
    if "train_real, test_real" not in src:
        ISSUES.append(f"{name}: missing train_test_split -> train_real")
    if "_bench_n = min(1000" not in src and "sample(n=_bench_n" not in src:
        if "metric_subsample" not in src:  # MAGIC has separate metric subsample
            ISSUES.append(f"{name}: missing N=1000 subsample before split")


def main() -> None:
    for path in sorted(FOLDER.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        combined = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                combined += "".join(cell.get("source", [])) + "\n"
        audit(path.name, combined)

    if ISSUES:
        print("ISSUES:")
        for i in ISSUES:
            print(f"  - {i}")
    else:
        print("All 15 notebooks pass preprocessing audit.")
    print(f"\nChecked {len(list(FOLDER.glob('*.ipynb')))} notebooks.")


if __name__ == "__main__":
    main()
