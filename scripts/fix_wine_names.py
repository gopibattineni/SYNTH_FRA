"""Replace leftover Wine Quality template names in notebook titles/exports."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    ROOT / "Other GANS",
    ROOT / "Diffusion GANs",
    ROOT / "SDV models",
    ROOT / "Single run_Data_leak_Synth_Quality",
]

# path fragment -> display name for plots/titles
DISPLAY = [
    ("1. Cancer", "Cancer"),
    ("2. Alzhimers", "Alzheimer's"),
    ("3. Adult", "Adult Census"),
    ("4. Forest", "Forest Cover"),
    ("4_Forest", "Forest Cover"),
    ("5. Bank", "Bank Marketing"),
    ("6. Wine", "Wine Quality"),
    ("Winequality", "Wine Quality"),
    ("Wine dataset", "Wine Quality"),
    ("7. CDC", "CDC Diabetes"),
    ("7_CDC", "CDC Diabetes"),
    ("8. Metro", "Metro Traffic"),
    ("9. Mushroom", "Mushroom"),
    ("10. Online", "Online Shopping"),
    ("10. online", "Online Shopping"),
    ("11. MAGIC", "MAGIC Gamma"),
    ("11_MAGIC", "MAGIC Gamma"),
    ("12. Air Quality", "Air Quality"),
    ("13. Concrete", "Concrete Strength"),
    ("14. Energy", "Energy Efficiency"),
    ("15. Real Estate", "Real Estate"),
]

# path fragment -> short name for files/exports
SHORT = [
    ("1. Cancer", "Cancer"),
    ("2. Alzhimers", "Alzheimers"),
    ("3. Adult", "Adult"),
    ("4. Forest", "ForestCover"),
    ("4_Forest", "ForestCover"),
    ("5. Bank", "BankMarketing"),
    ("6. Wine", "WineQuality"),
    ("Winequality", "WineQuality"),
    ("Wine dataset", "WineQuality"),
    ("7. CDC", "CDCDiabetes"),
    ("7_CDC", "CDCDiabetes"),
    ("8. Metro", "Metro"),
    ("9. Mushroom", "Mushroom"),
    ("10. Online", "OnlineShopping"),
    ("10. online", "OnlineShopping"),
    ("11. MAGIC", "MAGIC"),
    ("11_MAGIC", "MAGIC"),
    ("12. Air Quality", "AirQuality"),
    ("13. Concrete", "Concrete"),
    ("14. Energy", "EnergyEfficiency"),
    ("15. Real Estate", "RealEstate"),
]

WINE_FRAGS = ("6. Wine", "Winequality", "Wine dataset", "6. Winequality")


def match_maps(path: Path):
    s = path.as_posix().lower()
    display = short = None
    is_wine = any(f.lower().replace(" ", "") in s.replace(" ", "") for f in WINE_FRAGS)
    for frag, name in DISPLAY:
        if frag.lower().replace(" ", "") in s.replace(" ", "") or frag.lower() in s:
            display = name
            break
    for frag, name in SHORT:
        if frag.lower().replace(" ", "") in s.replace(" ", "") or frag.lower() in s:
            short = name
            break
    return display, short, is_wine


def patch_text(text: str, display: str, short: str, is_wine: bool) -> str:
    if is_wine:
        # fix typo only on wine notebooks
        return (
            text.replace("Wine qaulity", "Wine Quality")
            .replace("wine qaulity", "Wine Quality")
        )

    out = text
    replacements = [
        ("Wine qaulity", display),
        ("wine qaulity", display),
        ("Wine quality", display),
        ("wine quality", display),
        ("Wine Quality", display),
        ("WineQuality", short),
        ("Wine_Summary", f"{short}_Summary"),
        ("Hungarian_Mahalanobis_Four_Models_WineQuality.xlsx", f"Hungarian_Mahalanobis_Four_Models_{short}.xlsx"),
        ("Hungarian_Matchings_All_Models_Heart.xlsx", f"Hungarian_Matchings_All_Models_{short}.xlsx"),
        ("mahalanobis_winequality_by_model.png", f"mahalanobis_{short.lower()}_by_model.png"),
        ("# The target column for the Wine Quality dataset is 'quality'", f"# Target column for {display}"),
        ("# Load Wine Quality dataset", f"# Load {display} dataset"),
    ]
    for old, new in replacements:
        if old != new:
            out = out.replace(old, new)
    return out


def clear_wine_outputs(cell) -> None:
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        data = out.get("data", {})
        png = data.get("image/png", "")
        combined = text + str(png)
        if re.search(r"Wine\s*qau?lity|WineQuality", combined, re.I):
            cell["outputs"] = []
            cell["execution_count"] = None
            return


def fix_notebook(path: Path) -> bool:
    display, short, is_wine = match_maps(path)
    if not display:
        return False
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        new_src = patch_text(src, display, short, is_wine)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        elif not is_wine:
            clear_wine_outputs(cell)
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
            if fix_notebook(p):
                fixed.append(p.relative_to(ROOT).as_posix())
    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)


if __name__ == "__main__":
    main()
