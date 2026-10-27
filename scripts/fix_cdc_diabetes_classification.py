"""Fix CDC diabetes diffusion notebook: classification target + numeric coercion."""
from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "Diffusion GANs" / "7_CDC_diabetes_diffusion.ipynb"

NUMERIC_COERCE_HELPER = """
def _coerce_numeric_features(df, cols):
    out = df[cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
"""

EVALUATE_MODELS_BLOCK = """def evaluate_models(train_df, test_df, label_col, models=models, test_size=0.2, seed=42):
    def _coerce_class_labels(y):
        y = pd.to_numeric(pd.Series(y), errors="coerce").round()
        return y

    def _prepare_xy(df):
        df = df.copy()
        feature_cols = [c for c in df.columns if c != label_col]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        medians = df[feature_cols].median(numeric_only=True)
        for col in feature_cols:
            df[col] = df[col].fillna(medians[col])
        y = _coerce_class_labels(df[label_col])
        mask = y.notna()
        X = df.loc[mask, feature_cols].astype(np.float64)
        y = y.loc[mask].astype(int)
        return X, y

    X_train, y_train = _prepare_xy(train_df)
    X_test, y_test = _prepare_xy(test_df)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    strat_train = y_train if y_train.nunique() > 1 and y_train.value_counts().min() >= 2 else None
    strat_test = y_test if y_test.nunique() > 1 and y_test.value_counts().min() >= 2 else None

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
            y_prob = clf.predict_proba(X_test_s)
            if y_test.nunique() == 2:
                pos = 1 if 1 in clf.classes_ else clf.classes_[-1]
                pos_idx = list(clf.classes_).index(pos)
                auc = roc_auc_score(y_test, y_prob[:, pos_idx])
            else:
                auc = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="weighted", labels=clf.classes_
                )
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test_s)
            if scores.ndim == 1:
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = float("nan")
        else:
            auc = float("nan")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        rows.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})

    return pd.DataFrame(rows).sort_values(by="AUC", ascending=False).reset_index(drop=True)
"""


def patch_src(src: str) -> str:
    src = src.replace('target_col = "quality"', 'target_col = "Diabetes_binary"')
    src = src.replace("target_col = 'quality'", "target_col = 'Diabetes_binary'")
    src = src.replace('label_col = "quality"', 'label_col = "Diabetes_binary"')
    src = src.replace("# Target column: Diabetes_binary", "# Target column: Diabetes_binary")
    src = src.replace("target column for the CDC Diabetes dataset is 'quality'", "target column is 'Diabetes_binary'")
    src = src.replace("# Correcting label_col to 'quality'", "# label_col for CDC diabetes classification")
    src = src.replace("# Ensure 'quality' column is integer type for classification", "# Ensure Diabetes_binary is integer")
    src = src.replace("# Ensure 'quality' column in synthetic data is integer type for classification", "# Coerce Diabetes_binary in synthetic data")
    src = src.replace('only if \'quality\' was categorical', "only if target was categorical")
    src = src.replace("# For numeric 'quality' column, we will use it directly as target values", "# Diabetes_binary is numeric 0/1")
    src = src.replace("# For numeric 'quality' column, use it directly as target values for synthetic data", "# Use Diabetes_binary directly for synthetic data")

    if 'for cand in ["quality", "Class"' in src:
        src = src.replace(
            'for cand in ["quality", "Class", "class", "target", "Target"]:',
            'for cand in ["Diabetes_binary", "quality", "Class", "class", "target", "Target"]:',
        )

    if "real_means = real_df[num_cols].mean()" in src and "_coerce_numeric_features" not in src:
        src = src.replace(
            "model_order = [\"TabDDPM\", \"ForestDiffusion\"]\n",
            "model_order = [\"TabDDPM\", \"ForestDiffusion\"]\n" + NUMERIC_COERCE_HELPER,
            1,
        )
        src = src.replace(
            "    real_means = real_df[num_cols].mean()\n    synth_means = synth_df[num_cols].mean()\n",
            "    real_num = _coerce_numeric_features(real_df, num_cols)\n    synth_num = _coerce_numeric_features(synth_df, num_cols)\n    real_means = real_num.mean()\n    synth_means = synth_num.mean()\n",
        )
        src = src.replace(
            "    real_stds = real_df[num_cols].std(ddof=1)\n    synth_stds = synth_df[num_cols].std(ddof=1)\n",
            "    real_stds = real_num.std(ddof=1)\n    synth_stds = synth_num.std(ddof=1)\n",
        )
        src = src.replace(
            "    real_means, synth_means = real_df[num_cols].mean(), synth_df[num_cols].mean()\n",
            "    real_num = _coerce_numeric_features(real_df, num_cols)\n    synth_num = _coerce_numeric_features(synth_df, num_cols)\n    real_means, synth_means = real_num.mean(), synth_num.mean()\n",
        )

    if "def evaluate_models(train_df, test_df, label_col, models=models" in src:
        src = re.sub(
            r"def evaluate_models\(train_df, test_df, label_col, models=models.*?\n    return pd\.DataFrame\(rows\)\.sort_values\(by=\"AUC\", ascending=False\)\n",
            EVALUATE_MODELS_BLOCK + "\n",
            src,
            count=1,
            flags=re.DOTALL,
        )

    if "fetch_ucirepo(id=891)" in src and "Diabetes_binary is a binary classification target" not in src:
        anchor = 'target_col = "Diabetes_binary"\n'
        insert = (
            anchor
            + "\n"
            + "# Diabetes_binary is a binary classification target (0/1)\n"
            + "data[target_col] = pd.to_numeric(data[target_col], errors=\"coerce\")\n"
            + "data = data.dropna(subset=[target_col])\n"
            + "data[target_col] = data[target_col].round().astype(int)\n"
        )
        if anchor in src and "data[target_col] = pd.to_numeric" not in src:
            src = src.replace(anchor, insert, 1)

    if "synthetic_eval = {}" in src and "all_possible_labels = sorted" in src:
        src = re.sub(
            r"synthetic_eval = \{\}\nfor name, df in synthetic_outputs\.items\(\):.*?synthetic_eval\[name\] = tmp\n",
            """synthetic_eval = {}
for name, df in synthetic_outputs.items():
    tmp = df.copy()
    if label_col in tmp.columns:
        tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce").round()
        tmp = tmp.dropna(subset=[label_col])
        tmp[label_col] = tmp[label_col].astype(int)
    synthetic_eval[name] = tmp

data_eval[label_col] = pd.to_numeric(data_eval[label_col], errors="coerce").round()
data_eval = data_eval.dropna(subset=[label_col])
data_eval[label_col] = data_eval[label_col].astype(int)
""",
            src,
            count=1,
            flags=re.DOTALL,
        )

    if "all_possible_labels = sorted" in src:
        src = re.sub(
            r"# --- New: Collect all possible labels for the target column ---\nall_possible_labels = sorted\(\n    np\.unique\(\n        np\.concatenate\(\n            \[data_eval\[label_col\]\.unique\(\)\] \+\n            \[df\[label_col\]\.unique\(\) for df in synthetic_eval\.values\(\) if label_col in df\.columns\]\n        \)\n    \)\.tolist\(\)\n\)\n# --- End New ---\n",
            "all_possible_labels = sorted(\n    pd.to_numeric(\n        pd.concat(\n            [data_eval[label_col]]\n            + [df[label_col] for df in synthetic_eval.values() if label_col in df.columns],\n            ignore_index=True,\n        ),\n        errors=\"coerce\",\n    ).dropna().astype(int).unique().tolist()\n)\n",
            src,
            count=1,
        )

    src = src.replace(",\n    all_possible_labels=all_possible_labels", "")
    src = src.replace(",\n        all_possible_labels=all_possible_labels", "")

    if "data_eval[label_col] = data_eval[label_col].astype(int)" in src:
        src = src.replace(
            "data_eval[label_col] = data_eval[label_col].astype(int)",
            "data_eval[label_col] = pd.to_numeric(data_eval[label_col], errors=\"coerce\").round().astype(int)",
        )

    if "tmp[label_col] = tmp[label_col].round().astype(int)" in src:
        src = src.replace(
            "tmp[label_col] = tmp[label_col].round().astype(int)",
            "tmp[label_col] = pd.to_numeric(tmp[label_col], errors=\"coerce\").round()\n        tmp = tmp.dropna(subset=[label_col])\n        tmp[label_col] = tmp[label_col].astype(int)",
        )

    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = patch_src(src)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed += 1

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {changed} cells in {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
