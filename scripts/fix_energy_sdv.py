"""Apply regression SDV fixes to Energy Efficiency notebook."""
import runpy
import sys
from pathlib import Path

path = Path("scripts/fix_concrete_sdv.py")
src = path.read_text(encoding="utf-8")
src = src.replace('TARGET = "Concrete compressive strength"', 'TARGET = "Y1"')
src = src.replace('DATASET_NAME = "Concrete"', 'DATASET_NAME = "EnergyEfficiency"')
src = src.replace('EXCEL_SUFFIX = "Concrete"', 'EXCEL_SUFFIX = "EnergyEfficiency"')
src = src.replace(
    'NOTEBOOK = Path("SDV models/13. Concrete Compressive Strength.ipynb")',
    'NOTEBOOK = Path("SDV models/14. Energy Efficiency.ipynb")',
)
src = src.replace("# CONCRETE EXPORT", "# ENERGY EFFICIENCY EXPORT")
tmp = Path("scripts/_fix_energy_tmp.py")
tmp.write_text(src, encoding="utf-8")
runpy.run_path(str(tmp), run_name="__main__")
tmp.unlink(missing_ok=True)

# kernel guard + markdown
import json

GUARD = '''# REGRESSION TRTR/TSTR — run cell 35 (regressors) and 36 (evaluate_models) first after kernel restart
from sklearn.base import RegressorMixin
if "models" not in globals():
    raise NameError("Run cell 35 to define `models` (regressors).")
if not all(isinstance(m, RegressorMixin) for m in models.values()):
    raise TypeError("Stale classifier `models` in kernel. Restart kernel, then run cells 35 and 36.")
'''

nb_path = Path("SDV models/14. Energy Efficiency.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") == "markdown":
        s = "".join(cell.get("source", []))
        if "10 classifiers" in s:
            cell["source"] = [s.replace("10 classifiers", "10 regressors")]
    if cell.get("cell_type") != "code":
        continue
    s = "".join(cell.get("source", []))
    if 'target_col = "diagnosis"' in s:
        s = s.replace('target_col = "diagnosis"', 'target_col = "Y1"')
        cell["source"] = [line + "\n" for line in s.splitlines()]
    if s.startswith("import pandas as pd\n\nlabel_col =") and "evaluate_models" in s and "RegressorMixin" not in s:
        cell["source"] = [line + "\n" for line in (GUARD + s).splitlines()]
        cell["outputs"] = []
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Energy Efficiency notebook patched")
