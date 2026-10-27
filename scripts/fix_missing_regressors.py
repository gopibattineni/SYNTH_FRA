"""Restore missing regressors cell and fix TRTR plot cells in regression notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGRESSORS = """from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'SVR-RBF': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'GradientBoost': GradientBoostingRegressor(random_state=42),
    'MLP': MLPRegressor(max_iter=2000, random_state=42)
}
"""

REG_FRAGS = [
    "6. Wine", "Winequality", "8. Metro", "10. Online", "10. online",
    "12. Air Quality", "13. Concrete", "14. Energy", "15. Real Estate",
]

FOLDERS = [ROOT / "Other GANS", ROOT / "Diffusion GANs"]


def is_reg_notebook(path: Path) -> bool:
    s = path.name.lower()
    return any(f.replace(" ", "").lower() in s.replace(" ", "") for f in REG_FRAGS)


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def has_models_dict(nb) -> bool:
    return any(
        "models = {" in "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def has_trtr(nb) -> bool:
    return any(
        "trtr_results = evaluate_models" in "".join(c.get("source", []))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if not has_trtr(nb):
        return False
    changed = False

    if not has_models_dict(nb):
        inserted = False
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "def evaluate_models" in src and "mean_absolute_error" in src:
                nb["cells"].insert(i, {"cell_type": "code", "metadata": {}, "source": [], "outputs": [], "execution_count": None})
                set_src(nb["cells"][i], REGRESSORS)
                inserted = True
                changed = True
                break
            if not src.strip():
                set_src(cell, REGRESSORS)
                inserted = True
                changed = True
                break
        if not inserted:
            for i, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") == "markdown" and "regressor" in "".join(cell.get("source", [])).lower():
                    nb["cells"].insert(i + 1, {"cell_type": "code", "metadata": {}, "source": [], "outputs": [], "execution_count": None})
                    set_src(nb["cells"][i + 1], REGRESSORS)
                    changed = True
                    break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        if "def plot_trtr_vs_tstr" in src:
            src = src.replace('metric="AUC"', 'metric="RMSE"')
            src = src.replace('metric="Accuracy"', 'metric="RMSE"')
            src = src.replace("AUC_Drop", "RMSE_Diff")
            src = src.replace("Accuracy_Drop", "RMSE_Diff")
            src = src.replace('["Accuracy", "F1", "AUC"]', '["RMSE", "MAE", "R2"]')
        if "label_col = " in src and "plot_trtr_vs_tstr" in src:
            pass  # label_col handled by main script
        if src != orig:
            set_src(cell, src)
            changed = True

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = []
    still_missing = []
    for folder in FOLDERS:
        for p in sorted(folder.glob("*.ipynb")):
            if not is_reg_notebook(p):
                continue
            if fix_notebook(p):
                fixed.append(p.relative_to(ROOT).as_posix())
            nb = json.loads(p.read_text(encoding="utf-8"))
            if has_trtr(nb) and not has_models_dict(nb):
                still_missing.append(p.relative_to(ROOT).as_posix())

    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)
    if still_missing:
        print("Still missing models:", still_missing)
    else:
        print("All regression TRTR notebooks have models dict.")


if __name__ == "__main__":
    main()
