"""Fix evaluate_models: coerce labels before split and drop NaN rows."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"

# Replacement body inserted after models dict / imports — only the function.
EVALUATE_MODELS_FN = '''
def evaluate_models(train_df, test_df, label_col, models=models, test_size=0.2, seed=42):
    def _coerce_class_labels(y):
        y = pd.Series(y).copy()
        text_map = {
            "demented": 1, "nondemented": 0, "converted": 1,
            "g": 1, "h": 0, "b": 0, "m": 1,
            "0": 0, "1": 1, "0.0": 0, "1.0": 1,
        }
        num = pd.to_numeric(y, errors="coerce")
        if y.dtype == object or str(y.dtype) == "string":
            mapped = y.astype(str).str.strip().str.lower().map(text_map)
            out = num.round()
            out = out.fillna(mapped)
            return out
        return num.round()

    def _prepare_xy(df):
        X = df.drop(columns=[label_col])
        y = _coerce_class_labels(df[label_col])
        mask = y.notna()
        return X.loc[mask], y.loc[mask].astype(int)

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    strat_train = y_train if y_train.nunique() > 1 else None
    strat_test = y_test if y_test.nunique() > 1 else None

    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=test_size, random_state=seed, stratify=strat_train
    )
    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=seed, stratify=strat_test
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_s)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test_s)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")

        rows.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})

    return pd.DataFrame(rows).sort_values(by="AUC", ascending=False)
'''.strip() + "\n"

PATTERN = re.compile(
    r"def evaluate_models\(train_df, test_df, label_col[^\n]*\):.*?"
    r"return pd\.DataFrame\(rows\)\.sort_values\(by=\"AUC\", ascending=False\)\n",
    re.DOTALL,
)


def fix_cell(src: str) -> str:
    if "def evaluate_models" not in src or "accuracy_score" not in src:
        return src
    if "_prepare_xy" in src:
        return src
    if "import pandas as pd" not in src:
        src = "import pandas as pd\n" + src
    return PATTERN.sub(EVALUATE_MODELS_FN, src, count=1)


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = fix_cell(src)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    fixed = [p.name for p in sorted(FOLDER.glob("*.ipynb")) if fix_notebook(p)]
    print(f"Fixed {len(fixed)} notebooks:")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
