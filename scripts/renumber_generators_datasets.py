#!/usr/bin/env python3
"""Renumber Generators datasets: classification 1-9, regression 10-15."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATORS = REPO / "Generators"
TMP = "_renum_tmp_metro"

# current numbered prefix -> target numbered prefix (folder or notebook prefix)
PREFIX_MAP = {
    "8": "10",
    "9": "8",
    "10": "11",
    "11": "9",
}

NOTEBOOK_RENAMES = {
    GENERATORS / "Diffusion GANs": [
        ("8. Metro Interstate_diffusion.ipynb", "10. Metro Interstate_diffusion.ipynb"),
        ("9. Mushroom_diffusion.ipynb", "8. Mushroom_diffusion.ipynb"),
        ("10. Online shopping_diffusion.ipynb", "11. Online shopping_diffusion.ipynb"),
        ("11_MAGIC_Gamma_Telescope_diffusion.ipynb", "9. MAGIC Gamma Telescope_diffusion.ipynb"),
    ],
    GENERATORS / "Other GANS": [
        ("8. Metro Interstate_other GAN.ipynb", "10. Metro Interstate_other GAN.ipynb"),
        ("9. Mushroom_other GAN.ipynb", "8. Mushroom_other GAN.ipynb"),
        ("10. Online shopping_other GAN.ipynb", "11. Online shopping_other GAN.ipynb"),
        ("11_MAGIC_Gamma_Telescope_other_GAN.ipynb", "9. MAGIC Gamma Telescope_other GAN.ipynb"),
    ],
    GENERATORS / "SDV models": [
        ("8. Metro Interstate Traffic Volume.ipynb", "10. Metro Interstate Traffic Volume.ipynb"),
        ("9. Mushroom.ipynb", "8. Mushroom.ipynb"),
        ("10. Online Shopping.ipynb", "11. Online Shopping.ipynb"),
        ("11. MAGIC Gamma Telescope.ipynb", "9. MAGIC Gamma Telescope.ipynb"),
    ],
}

FOLDER_PARENTS = [
    GENERATORS / "Experiment with utility data leak",
    GENERATORS / "Experiment with utility data leak" / "diffusion_dataleak",
]

FOLDER_RENAMES = [
    ("8. Metro interstate", "10. Metro interstate"),
    ("9. Mushroom dataset", "8. Mushroom dataset"),
    ("10. online shopping", "11. online shopping"),
    ("11. MAGIC Gamma Telescope", "9. MAGIC Gamma Telescope"),
]


def _rename_path(src: Path, dst: Path) -> None:
    if not src.exists():
        if dst.exists():
            print(f"skip (already done): {dst.relative_to(REPO)}")
            return
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    src.rename(dst)
    print(f"renamed: {src.relative_to(REPO)} -> {dst.name}")


def renumber_notebooks() -> None:
    for parent, pairs in NOTEBOOK_RENAMES.items():
        staged: list[tuple[Path, Path]] = []
        for old_name, new_name in pairs:
            src = parent / old_name
            if not src.exists() and (parent / new_name).exists():
                continue
            staged.append((src, parent / TMP / old_name, parent / new_name))

        tmp_dir = parent / TMP
        tmp_dir.mkdir(exist_ok=True)

        for src, tmp, _final in staged:
            if src.exists():
                _rename_path(src, tmp)

        for src, tmp, final in staged:
            if tmp.exists():
                _rename_path(tmp, final)

        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()


def renumber_folders(parent: Path) -> None:
    tmp_parent = parent / TMP
    tmp_parent.mkdir(exist_ok=True)

    staged: list[tuple[Path, Path, Path]] = []
    for old_name, new_name in FOLDER_RENAMES:
        src = parent / old_name
        if not src.exists() and (parent / new_name).exists():
            continue
        staged.append((src, tmp_parent / old_name, parent / new_name))

    for src, tmp, _final in staged:
        if src.exists():
            _rename_path(src, tmp)

    for _src, tmp, final in staged:
        if tmp.exists():
            _rename_path(tmp, final)

    if tmp_parent.exists() and not any(tmp_parent.iterdir()):
        tmp_parent.rmdir()


def update_datasets_json() -> None:
    path = (
        GENERATORS
        / "Experiment with utility data leak"
        / "python_scripts"
        / "hive"
        / "datasets.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    updates = {
        "metro": "10. Metro interstate",
        "mushroom": "8. Mushroom dataset",
        "online_shopping": "11. online shopping",
        "magic": "9. MAGIC Gamma Telescope",
    }
    for entry in data:
        ds_id = entry.get("id")
        if ds_id in updates:
            entry["notebook_dir"] = updates[ds_id]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"updated: {path.relative_to(REPO)}")


def update_implementation_log() -> None:
    path = GENERATORS / "Diffusion GANs" / "IMPLEMENTATION_LOG.md"
    text = path.read_text(encoding="utf-8")
    replacements = [
        ("| 8 | `8. Metro Interstate_diffusion.ipynb` |", "| 8 | `8. Mushroom_diffusion.ipynb` |"),
        ("| 9 | `9. Mushroom_diffusion.ipynb` |", "| 9 | `9. MAGIC Gamma Telescope_diffusion.ipynb` |"),
        ("| 8 | `8. Mushroom_diffusion.ipynb` |", "| 8 | `8. Mushroom_diffusion.ipynb` |"),  # noop guard
    ]
    # Apply in careful order using direct line replacements
    text = text.replace(
        "| 8 | `8. Metro Interstate_diffusion.ipynb` |",
        "| 8 | `8. Mushroom_diffusion.ipynb` |",
    )
    text = text.replace(
        "| 9 | `9. Mushroom_diffusion.ipynb` |",
        "| 9 | `9. MAGIC Gamma Telescope_diffusion.ipynb` |",
    )
    text = text.replace(
        "| 10 | `10. Online shopping_diffusion.ipynb` |",
        "| 10 | `10. Metro Interstate_diffusion.ipynb` |",
    )
    text = text.replace(
        "| 11 | `11_MAGIC_Gamma_Telescope_diffusion.ipynb` |",
        "| 11 | `11. Online shopping_diffusion.ipynb` |",
    )
    path.write_text(text, encoding="utf-8")
    print(f"updated: {path.relative_to(REPO)}")


def main() -> None:
    for parent in FOLDER_PARENTS:
        renumber_folders(parent)
    renumber_notebooks()
    update_datasets_json()
    update_implementation_log()
    print("Done.")


if __name__ == "__main__":
    main()
