"""Convert Diffusion GANs/6. Winequality_diffusion.ipynb from regression to classification."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "Diffusion GANs" / "6. Winequality_diffusion.ipynb"

MODELS_CELL = """from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

models = {
    'LogReg': LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42),
    'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'GradientBoost': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=2000, random_state=42),
}
"""

EVALUATE_CELL = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_models(train_df, test_df, label_col, models, test_size=0.2, seed=42):
    def _coerce_quality_labels(y):
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
        y = _coerce_quality_labels(df[label_col])
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
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test_s)
            if scores.ndim == 1:
                y_prob = np.column_stack([-scores, scores])
            else:
                y_prob = scores
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        if y_prob is not None and y_test.nunique() >= 2:
            if y_test.nunique() == 2:
                pos = 1 if 1 in clf.classes_ else clf.classes_[-1]
                pos_idx = list(clf.classes_).index(pos)
                auc = roc_auc_score(y_test, y_prob[:, pos_idx])
            else:
                auc = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="weighted", labels=clf.classes_
                )
        else:
            auc = float("nan")

        rows.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})

    return pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
"""

TRTR_CELL = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric="AUC"):
    if metric not in trtr_results.columns or metric not in tstr_results.columns:
        return
    df = trtr_results[["Model", metric]].merge(
        tstr_results[["Model", metric]], on="Model", suffixes=("_TRTR", "_TSTR")
    )
    df["Drop"] = df[f"{metric}_TRTR"] - df[f"{metric}_TSTR"]
    df = df.sort_values("Drop", ascending=False)
    x = np.arange(len(df))
    w = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, df[f"{metric}_TRTR"], w, label="TRTR")
    plt.bar(x + w / 2, df[f"{metric}_TSTR"], w, label="TSTR")
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{synth_name}: TRTR vs TSTR ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.show()

label_col = "quality"
model_order = ["TabDDPM", "ForestDiffusion"]

trtr_results = evaluate_models(train_df=data, test_df=data, label_col=label_col, models=models)
print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)

all_comp = []
for synth_name in model_order:
    synth_df = synthetic_outputs[synth_name].copy()
    _common = [c for c in data.columns if c in synth_df.columns]
    synth_train = synth_df[_common]
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label_col=label_col, models=models)
    print(f"{synth_name} - TSTR (Train Synthetic, Test Real)")
    display(tstr_results)
    comparison = trtr_results.merge(tstr_results, on="Model", suffixes=("_TRTR", "_TSTR"))
    comparison["AUC_Drop"] = comparison["AUC_TRTR"] - comparison["AUC_TSTR"]
    comparison["F1_Drop"] = comparison["F1_TRTR"] - comparison["F1_TSTR"]
    comparison["Accuracy_Drop"] = comparison["Accuracy_TRTR"] - comparison["Accuracy_TSTR"]
    comparison["Synthetic_Model"] = synth_name
    print(f"{synth_name} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)
    all_comp.append(comparison)
    for metric in ["AUC", "F1", "Accuracy"]:
        plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric)

combined_comparison = pd.concat(all_comp, ignore_index=True)
summary = (
    combined_comparison.groupby("Synthetic_Model", as_index=False)["AUC_Drop"]
    .mean()
    .sort_values("AUC_Drop")
)
print("Average AUC drop by synthetic generator (lower is better):")
display(summary)
"""

GRID_CELL = """import numpy as np
import matplotlib.pyplot as plt

def _metric_delta(metric, trtr_val, tstr_val):
    return trtr_val - tstr_val

def plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name, ncols=5):
    metrics = [m for m in ["AUC", "F1", "Accuracy"] if m in trtr_results.columns and m in tstr_results.columns]
    if not metrics:
        print(f"No common metrics found for {generator_name}.")
        return

    x = np.arange(len(metrics))
    trtr = trtr_results.set_index("Model")[metrics]
    tstr = tstr_results.set_index("Model")[metrics]
    models_local = [m for m in trtr.index if m in tstr.index]

    n = len(models_local)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, max(9, 3 * nrows)), dpi=150, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, model_name in enumerate(models_local):
        ax = axes[i]
        a = trtr.loc[model_name].values.astype(float)
        b = tstr.loc[model_name].values.astype(float)
        deltas = [_metric_delta(m, ai, bi) for m, ai, bi in zip(metrics, a, b)]

        ax.plot(x, a, marker="o", label="TRTR")
        ax.plot(x, b, marker="s", label="TSTR")
        ax.fill_between(x, np.minimum(a, b), np.maximum(a, b), alpha=0.2)

        lo = float(np.nanmin(np.concatenate([a, b])))
        hi = float(np.nanmax(np.concatenate([a, b])))
        pad = max((hi - lo) * 0.15, 0.05) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)

        for xi, ai, bi, di in zip(x, a, b, deltas):
            ax.text(xi, (ai + bi) / 2, f"{di:+.3f}", ha="center", va="center", fontsize=8)

        ax.set_title(model_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        if i % ncols == 0:
            ax.set_ylabel("Score")
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{generator_name} - TRTR vs TSTR per classifier", fontsize=14)
    plt.show()

label_col = "quality"
model_order = ["TabDDPM", "ForestDiffusion"]
trtr_results = evaluate_models(train_df=data, test_df=data, label_col=label_col, models=models)
for synth_name in model_order:
    synth_df = synthetic_outputs[synth_name].copy()
    _common = [c for c in data.columns if c in synth_df.columns]
    tstr_results = evaluate_models(
        train_df=synth_df[_common],
        test_df=data,
        label_col=label_col,
        models=models,
    )
    plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name=synth_name, ncols=5)
"""

MIA_CELL = """import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

def model_loss_per_sample(model, X, y, eps=1e-15):
    proba = model.predict_proba(X)
    n = len(y)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    y_idx = np.array([class_to_idx[c] for c in y])
    p_true = proba[np.arange(n), y_idx] + eps
    return -np.log(p_true)

def privacy_metrics_mia(model, X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    loss_train = model_loss_per_sample(model, X_train_s, y_train)
    loss_test = model_loss_per_sample(model, X_test_s, y_test)
    scores = -np.concatenate([loss_train, loss_test])
    labels = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return {"AUC": float(roc_auc_score(labels, scores)), "Advantage": float(np.max(tpr - fpr))}

def _coerce_quality_series(s):
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")

target_col = "quality"
feature_cols = [c for c in data.columns if c != target_col]
real_df = data.copy()
real_df[target_col] = _coerce_quality_series(real_df[target_col])
real_df = real_df.dropna(subset=[target_col])
real_df[target_col] = real_df[target_col].astype(int)

synth_key = next(iter(synthetic_data)) if "synthetic_data" in globals() else next(iter(synthetic_outputs))
synth_src = synthetic_data if "synthetic_data" in globals() else synthetic_outputs
_common = [c for c in real_df.columns if c in synth_src[synth_key].columns]
synth_df = synth_src[synth_key][_common].copy()
synth_df[target_col] = _coerce_quality_series(synth_df[target_col])
synth_df = synth_df.dropna(subset=[target_col])
synth_df[target_col] = synth_df[target_col].astype(int)

model = LogisticRegression(max_iter=5000, random_state=42)
print(privacy_metrics_mia(
    model,
    real_df[feature_cols].to_numpy(dtype=np.float64),
    real_df[target_col].to_numpy(dtype=int),
    synth_df[feature_cols].to_numpy(dtype=np.float64),
    synth_df[target_col].to_numpy(dtype=int),
))
"""


def set_cell_source(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None


def patch_data_loading(src: str) -> str:
    if "_categorical_columns = [target_col]" in src:
        return src
    anchor = 'data = data.replace("?", np.nan)\n'
    insert = (
        anchor
        + "\n"
        + "# Wine quality is a multi-class classification target (scores 3-9)\n"
        + "data[target_col] = pd.to_numeric(data[target_col], errors=\"coerce\")\n"
        + "data = data.dropna(subset=[target_col])\n"
        + "data[target_col] = data[target_col].round().astype(int)\n"
    )
    if anchor not in src:
        raise ValueError("Could not find data-loading anchor")
    src = src.replace(anchor, insert, 1)
    return src.replace("_categorical_columns = []", "_categorical_columns = [target_col]")


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if i == 3 and "fetch_ucirepo(id=186)" in src:
            set_cell_source(cell, patch_data_loading(src))
        elif i == 33 and "# 4 Utility" in src:
            cell["source"] = ["# 4 Utility with 10 classifiers\n"]
        elif i == 34 and "LinearRegression" in src:
            set_cell_source(cell, MODELS_CELL)
        elif i == 35 and "def evaluate_models" in src:
            set_cell_source(cell, EVALUATE_CELL)
        elif i == 36 and "plot_trtr_vs_tstr" in src:
            set_cell_source(cell, TRTR_CELL)
        elif i == 37 and "plot_grid_line_metrics_with_gap" in src:
            set_cell_source(cell, GRID_CELL)
        elif i == 59 and "privacy_metrics_mia_regression" in src:
            set_cell_source(cell, MIA_CELL)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NOTEBOOK.name} for classification")


if __name__ == "__main__":
    main()
