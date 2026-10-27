import json
from pathlib import Path

NOTEBOOK = Path("SDV models/13. Concrete Compressive Strength.ipynb")
TARGET = "Concrete compressive strength"

GUARD = f'''# REGRESSION TRTR/TSTR — run cell 35 (regressors) and 36 (evaluate_models) first after kernel restart
from sklearn.base import RegressorMixin

if "models" not in globals():
    raise NameError("Run cell 35 to define `models` (regressors).")
if not all(isinstance(m, RegressorMixin) for m in models.values()):
    raise TypeError(
        "Stale classifier `models` in kernel. Kernel -> Restart, then run cells 35 and 36 before this cell."
    )
'''

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "markdown":
        src = "".join(cell.get("source", []))
        if "10 classifiers" in src:
            cell["source"] = [src.replace("10 classifiers", "10 regressors")]

    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))

    if src.startswith("import pandas as pd\n\nlabel_col =") and "evaluate_models" in src:
        if "RegressorMixin" not in src:
            src = GUARD + src
            cell["source"] = [line + "\n" for line in src.splitlines()]
        cell["outputs"] = []
        cell["execution_count"] = None

    if "def evaluate_models" in src and "mean_absolute_error" in src:
        cell["outputs"] = []
        cell["execution_count"] = None

    if src.strip().startswith("from sklearn.linear_model import LinearRegression"):
        cell["outputs"] = []
        cell["execution_count"] = None

    # remove stored tracebacks referencing quality/classifiers
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if "label_col = \"quality\"" in text or "Unknown label type: continuous" in text:
            cell["outputs"] = []
            cell["execution_count"] = None
            break

nb_path = NOTEBOOK
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Updated guards and cleared stale errors")
