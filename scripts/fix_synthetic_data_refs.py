"""Replace undefined synthetic_wgan_gp / synthetic_ctabgan dicts with synthetic_outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [ROOT / "Other GANS", ROOT / "Diffusion GANs", ROOT / "SDV models"]


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def patch_source(src: str) -> str:
    out = src

    # Broken manual dicts -> use the canonical dict from training cells
    out = re.sub(
        r"# Map your synthetic outputs here\s*\n"
        r"synthetic_data = \{\s*\n"
        r'[\s"]*WGAN_GP["\']?\s*:\s*synthetic_wgan_gp,?\s*\n'
        r'[\s"]*CTABGAN["\']?\s*:\s*synthetic_ctabgan,?\s*\n'
        r"\s*\}",
        "synthetic_data = synthetic_outputs",
        out,
    )
    out = re.sub(
        r"# Use your actual synthetic DataFrames here\s*\n"
        r"# Replace synthetic_wgan_gp and synthetic_ctabgan with the variables you really have\s*\n"
        r"synthetic_data = \{\s*\n"
        r'[\s"]*WGAN_GP["\']?\s*:\s*synthetic_wgan_gp,?\s*\n'
        r'[\s"]*CTABGAN["\']?\s*:\s*synthetic_ctabgan,?\s*\n'
        r"\s*\}",
        "synthetic_data = synthetic_outputs",
        out,
    )
    out = re.sub(
        r"# Replace synthetic_outputs\['CoDi'\].*?\n"
        r"synthetic_data = \{\s*\n"
        r'[\s"]*CoDi["\']?\s*:\s*synthetic_outputs\[\'CoDi\'\],?\s*\n'
        r'[\s"]*TabDDPM["\']?\s*:\s*synthetic_outputs\[\'TabDDPM\'\],?\s*\n'
        r"\s*\}",
        "synthetic_data = synthetic_outputs",
        out,
        flags=re.DOTALL,
    )

    # Any remaining undefined alias
    out = out.replace("synthetic_wgan_gp", "synthetic_wgan")

    # Prefer synthetic_outputs when building cosine/hungarian tables
    if "synthetic_data = synthetic_outputs" not in out and "synthetic_wgan" in out:
        out = re.sub(
            r"synthetic_data = \{\s*\n\s*\"WGAN_GP\": synthetic_wgan,?\s*\n\s*\"CTABGAN\": synthetic_ctabgan,?\s*\n\s*\}",
            "synthetic_data = synthetic_outputs",
            out,
        )

    return out


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "synthetic_wgan_gp" not in src and "synthetic_ctabgan" not in src:
            continue
        # skip training cell that legitimately defines synthetic_ctabgan
        if "synthetic_outputs = {" in src and "synthetic_ctabgan" in src:
            patched = src.replace("synthetic_wgan_gp", "synthetic_wgan")
            if patched != src:
                set_src(cell, patched)
                changed = True
            continue
        patched = patch_source(src)
        if patched != src:
            set_src(cell, patched)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = []
    for folder in FOLDERS:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in p.as_posix():
                continue
            raw = p.read_text(encoding="utf-8")
            if "synthetic_wgan_gp" not in raw:
                continue
            if fix_notebook(p):
                fixed.append(p.relative_to(ROOT).as_posix())
    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)


if __name__ == "__main__":
    main()
