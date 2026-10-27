"""Batch-fix notebooks across Other GANS, Diffusion GANs, Single run, SDV folders."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    ROOT / "Other GANS",
    ROOT / "Diffusion GANs",
    ROOT / "Single run_Data_leak_Synth_Quality",
    ROOT / "SDV models",
]

# (path_fragment, target, task, dataset_name, excel_suffix, sample_cap)
CONFIGS = [
    ("1. Cancer", "diagnosis", "cls", "Cancer", "Cancer", 1000),
    ("2. Alzhimers", "Group", "cls", "Alzheimers", "Alzheimers", 1000),
    ("3. Adult", "income", "cls", "Adult", "Adult", 10000),
    ("4. Forest", "Cover_Type", "cls", "ForestCover", "ForestCover", 1000),
    ("4_Forest", "Cover_Type", "cls", "ForestCover", "ForestCover", 1000),
    ("5. Bank", "y", "cls", "BankMarketing", "BankMarketing", 10000),
    ("6. Wine", "quality", "reg", "WineQuality", "WineQuality", 1000),
    ("Winequality", "quality", "reg", "WineQuality", "WineQuality", 1000),
    ("7. CDC", "Diabetes_binary", "cls", "CDCDiabetes", "CDCDiabetes", 1000),
    ("7_CDC", "Diabetes_binary", "cls", "CDCDiabetes", "CDCDiabetes", 1000),
    ("8. Metro", "traffic_volume", "reg", "Metro", "Metro", 1000),
    ("9. Mushroom", "class", "cls", "Mushroom", "Mushroom", 1000),
    ("10. Online", "price", "reg", "OnlineShopping", "OnlineShopping", 1000),
    ("10. online", "price", "reg", "OnlineShopping", "OnlineShopping", 1000),
    ("11. MAGIC", "class", "cls", "MAGIC", "MAGIC", 1000),
    ("11_MAGIC", "class", "cls", "MAGIC", "MAGIC", 1000),
    ("12. Air Quality", "CO(GT)", "reg", "AirQuality", "AirQuality", 1000),
    ("13. Concrete", "Concrete compressive strength", "reg", "Concrete", "Concrete", 1000),
    ("14. Energy", "Y1", "reg", "EnergyEfficiency", "EnergyEfficiency", 768),
    ("15. Real Estate", "Y house price of unit area", "reg", "RealEstate", "RealEstate", 414),
]


def match_config(path: Path):
    s = path.as_posix().lower()
    for frag, target, task, dname, excel, cap in CONFIGS:
        key = frag.replace(" ", "").lower()
        if key in s.replace(" ", "") or frag.lower() in s:
            return target, task, dname, excel, cap
    return None


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


REG_EVAL = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def evaluate_models(train_df, test_df, label, models, test_size=0.2, seed=42):
    X_train = train_df.drop(columns=[label]).copy()
    y_train = train_df[label].copy()
    X_test = test_df.drop(columns=[label]).copy()
    y_test = test_df[label].copy()
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=seed)
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[num_cols].astype(np.float64)
    X_test = X_test[num_cols].astype(np.float64)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({"Model": name, "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)), "R2": r2_score(y_test, preds)})
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
"""

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


def patch_sample(src: str, cap: int) -> str:
    return re.sub(
        r"data = data\.sample\(n=(\d+), random_state=([^)]+)\)\.reset_index\(drop=True\)",
        lambda m: f"n_samples = min({m.group(1)}, len(data))\ndata = data.sample(n=n_samples, random_state={m.group(2)}).reset_index(drop=True)",
        src,
    )


def patch_targets(src: str, target: str, dname: str, excel: str, task: str) -> str:
    t = target
    if t != "quality":
        src = src.replace('possible_targets = ["quality"]', f'possible_targets = ["{t}"]')
        src = src.replace('label_col = "quality"', f'label_col = "{t}"')
        src = src.replace('if c not in ["quality"]', f'if c != "{t}"')
        src = src.replace('if c != "quality"', f'if c != "{t}"')
        src = src.replace('drop_from_metrics.append("quality")', "drop_from_metrics.append(target_col)")
        src = src.replace('if "quality" in num_cols:', "if target_col in num_cols:")
        src = src.replace('target="quality"', f'target="{t}"')
    if t != "y":
        src = re.sub(r'target_col = "y"\s*$', f'target_col = "{t}"', src, flags=re.MULTILINE)
    src = src.replace('target_col = "diagnosis"', f'target_col = "{t}"')
    src = src.replace('target_col = "Diabetes_binary"', f'target_col = "{t}"')
    src = src.replace("Hungarian_Mahalanobis_Four_Models_WineQuality.xlsx", f"Hungarian_Mahalanobis_Four_Models_{excel}.xlsx")
    src = src.replace("Hungarian_Matchings_All_Models_Heart.xlsx", f"Hungarian_Matchings_All_Models_{excel}.xlsx")
    src = src.replace("mahalanobis_winequality_by_model.png", f"mahalanobis_{excel.lower()}_by_model.png")
    if dname != "WineQuality":
        src = src.replace('"WineQuality"', f'"{dname}"')
        src = src.replace("WineQuality", dname)
        src = src.replace('sheet_name="Wine_Summary"', f'sheet_name="{dname}_Summary"')
        src = src.replace('sheet_name = f"Wine_{model_name}"', f'sheet_name = f"{dname}_{{model_name}}"')
    src = src.replace("Wine quality:", f"{dname}:")
    src = src.replace("Corrected target column to 'quality'", f"Target column: {t}")
    src = src.replace("bivariate_quality_wine", "bivariate_quality_regression" if task == "reg" else "bivariate_quality_wine")
    src = src.replace("label_col=label_col", f'label="{t}"')
    src = src.replace('if c not in ["ID", "Diabetes_binary"]', f'if c != "{t}"')
    src = src.replace(
        'num_cols = [c for c in num_cols if c not in ["ID", "Diabetes_binary"]]',
        f'num_cols = [c for c in num_cols if c != "{t}"]',
    )
    src = src.replace(
        'feature_cols = [c for c in feature_cols if c not in ["ID", "Diabetes_binary"]]',
        f'feature_cols = [c for c in feature_cols if c != "{t}"]',
    )
    src = re.sub(
        r"(\n[ \t]+)synth_df = synth_df\[real_df\.columns\]",
        r"\1_common = [c for c in real_df.columns if c in synth_df.columns]\n\1synth_df = synth_df[_common]",
        src,
    )
    src = re.sub(
        r"synth_train = synthetic_datasets\[(\w+)\]\[data\.columns\]\.copy\(\)",
        r"_common = [c for c in data.columns if c in synthetic_datasets[\1].columns]\n    synth_train = synthetic_datasets[\1][_common].copy()",
        src,
    )
    src = re.sub(
        r"synthetic_datasets\[gen\]\[data\.columns\]",
        r"synthetic_datasets[gen][[c for c in data.columns if c in synthetic_datasets[gen].columns]]",
        src,
    )
    src = src.replace(
        'if "bivariate_quality_regression" in globals():\n    quality_fn = bivariate_quality_regression\nelse:\n    raise NameError("Define `bivariate_quality_wine` before running this cell.")',
        "quality_fn = bivariate_quality_regression",
    )
    if task == "reg":
        src = src.replace('["Accuracy", "F1", "AUC"]', '["RMSE", "MAE", "R2"]')
        src = src.replace("per classifier", "per regressor")
        # remove wine-quality integer clip on continuous targets
        if t != "quality":
            src = re.sub(
                r"synthetic_data\[target_col\] = \(\s*synthetic_data\[target_col\]\s*\.round\(\)\s*\.clip\(0, 10\)\s*\.astype\(int\)\s*\)",
                "pass  # keep continuous regression target as-is",
                src,
            )
            src = re.sub(
                r"synthetic_\w+\[target_col\] = \(\s*[^)]+\.round\(\)\s*\.clip\(0, 10\)\s*\.astype\(int\)\s*\)",
                "pass  # keep continuous regression target as-is",
                src,
            )
    return src


def mia_regression_cell(target: str) -> str:
    return f'''import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve, roc_auc_score

def model_loss_per_sample_regression(model, X, y):
    return (y - model.predict(X)) ** 2

def privacy_metrics_mia_regression(model, X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    loss_train = model_loss_per_sample_regression(model, X_train_s, y_train)
    loss_test = model_loss_per_sample_regression(model, X_test_s, y_test)
    scores = -np.concatenate([loss_train, loss_test])
    labels = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return {{"AUC": float(roc_auc_score(labels, scores)), "Advantage": float(np.max(tpr - fpr))}}

target_col = "{target}"
feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != target_col]
real_df = data.copy()
synth_key = next(iter(synthetic_data)) if "synthetic_data" in globals() else next(iter(synthetic_outputs))
synth_src = synthetic_data if "synthetic_data" in globals() else synthetic_outputs
_common = [c for c in real_df.columns if c in synth_src[synth_key].columns]
synth_df = synth_src[synth_key][_common].copy()
model = Ridge(alpha=1.0, random_state=42)
print(privacy_metrics_mia_regression(
    model,
    real_df[feature_cols].to_numpy(dtype=np.float64),
    real_df[target_col].to_numpy(dtype=np.float64),
    synth_df[feature_cols].to_numpy(dtype=np.float64),
    synth_df[target_col].to_numpy(dtype=np.float64),
))
'''


def fix_notebook(path: Path) -> bool:
    cfg = match_config(path)
    if not cfg:
        return False
    target, task, dname, excel, cap = cfg
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            s = "".join(cell.get("source", []))
            if "10 classifiers" in s and task == "reg":
                set_src(cell, s.replace("10 classifiers", "10 regressors"))
                changed = True
            continue
        if cell.get("cell_type") != "code":
            continue

        src = "".join(cell.get("source", []))
        orig = src
        src = patch_sample(src, cap)
        src = patch_targets(src, target, dname, excel, task)

        if task == "reg":
            if "def evaluate_models" in src and "roc_auc_score" in src:
                src = REG_EVAL
            elif src.strip().startswith("from sklearn.linear_model import LogisticRegression") and "models = {" in src:
                src = REGRESSORS
            elif "def privacy_metrics_mia(" in src and "LogisticRegression" in src:
                src = mia_regression_cell(target)
            elif "RandomForestClassifier" in src and "accuracy_score" in src and "target_col" in src and task == "reg":
                src = src.replace("RandomForestClassifier", "RandomForestRegressor")
                src = src.replace("accuracy_score", "r2_score")
                src = src.replace(
                    "from sklearn.metrics import accuracy_score",
                    "from sklearn.metrics import r2_score",
                )

        if src != orig:
            set_src(cell, src)
            changed = True
            cell["outputs"] = []
            cell["execution_count"] = None

        # drop stale outputs referencing wrong targets
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if target != "quality" and ('label_col = "quality"' in text or "KeyError: \"['quality']" in text):
                cell["outputs"] = []
                break

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def verify_sample(path: Path) -> list[str]:
    cfg = match_config(path)
    if not cfg:
        return []
    target, task, dname, excel, cap = cfg
    issues = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            sources.append("".join(cell.get("source", [])))
    raw = "\n".join(sources)
    if target != "quality" and 'possible_targets = ["quality"]' in raw:
        issues.append("possible_targets quality")
    if target != "quality" and 'label_col = "quality"' in raw:
        issues.append("label_col quality")
    if target != "y" and re.search(r'target_col = "y"', raw):
        issues.append('target_col = "y"')
    if dname != "WineQuality" and "WineQuality" in raw:
        issues.append("WineQuality name")
    if task == "reg" and "LogisticRegression(max_iter" in raw and "def privacy_metrics_mia(" in raw:
        issues.append("MIA classifier on regression")
    if re.search(r"data = data\.sample\(n=1000", raw) and cap < 1000:
        issues.append("sample > dataset size")
    return issues


def main() -> None:
    fixed: list[str] = []
    remaining: dict[str, list[str]] = {}

    for folder in FOLDERS:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in p.as_posix():
                continue
            if fix_notebook(p):
                fixed.append(p.relative_to(ROOT).as_posix())
            issues = verify_sample(p)
            if issues:
                remaining[p.relative_to(ROOT).as_posix()] = issues

    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)
    if remaining:
        print(f"\nRemaining issues in {len(remaining)} notebooks:")
        for path, iss in sorted(remaining.items())[:40]:
            print(f"  {path}: {', '.join(iss)}")
        if len(remaining) > 40:
            print(f"  ... and {len(remaining) - 40} more")
        sys.exit(1)
    print("All notebooks passed verification.")


if __name__ == "__main__":
    main()
