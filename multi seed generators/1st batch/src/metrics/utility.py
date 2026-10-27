"""Utility TRTR/TSTR metrics (existing Acc/F1/Prec/Recall or R2/RMSE)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


CLASSIFIERS = {
    "LogReg": LogisticRegression(max_iter=5000, solver="lbfgs"),
    "SVM-RBF": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100),
    "GradientBoost": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
}

REGRESSORS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
}


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    # encode object features
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    return X, y


def _encode_y(y_train, y_test):
    if y_train.dtype == object or str(y_train.dtype) == "category" or y_train.dtype == bool:
        le = LabelEncoder()
        # fit on union
        le.fit(pd.concat([y_train.astype(str), y_test.astype(str)], ignore_index=True))
        return le.transform(y_train.astype(str)), le.transform(y_test.astype(str)), le
    return y_train.to_numpy(), y_test.to_numpy(), None


def evaluate_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    classifier_seeds: list[int],
    models: dict | None = None,
    return_sd: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """
    Fixed held-out test (no re-split). Classifier seeds only affect model init.
    Returns mean metrics across models × classifier seeds.
    """
    models = models or CLASSIFIERS
    X_tr, y_tr = _xy(train_df, target)
    X_te, y_te = _xy(test_df, target)
    # align columns
    cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr_e, y_te_e, _ = _encode_y(y_tr, y_te)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    buckets = {"Accuracy": [], "F1": [], "Precision": [], "Recall": [], "ROC_AUC": []}
    n_classes = len(np.unique(y_tr_e))
    # Generator collapse / tiny N can yield a single-class synthetic train set.
    # Record NaN utility rather than failing the whole run.
    if n_classes < 2:
        nan_m = {k: float("nan") for k in buckets}
        if return_sd:
            return nan_m, {k: float("nan") for k in buckets}
        return nan_m
    for name, model in models.items():
        for seed in classifier_seeds:
            clf = clone(model)
            if hasattr(clf, "set_params"):
                try:
                    clf.set_params(random_state=seed)
                except ValueError:
                    pass
            clf.fit(X_tr_s, y_tr_e)
            pred = clf.predict(X_te_s)
            buckets["Accuracy"].append(accuracy_score(y_te_e, pred))
            buckets["F1"].append(f1_score(y_te_e, pred, average="weighted", zero_division=0))
            buckets["Precision"].append(
                precision_score(y_te_e, pred, average="weighted", zero_division=0)
            )
            buckets["Recall"].append(recall_score(y_te_e, pred, average="weighted", zero_division=0))
            try:
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_te_s)
                    if n_classes == 2:
                        buckets["ROC_AUC"].append(roc_auc_score(y_te_e, proba[:, 1]))
                    else:
                        buckets["ROC_AUC"].append(
                            roc_auc_score(y_te_e, proba, multi_class="ovr", average="weighted")
                        )
            except Exception:
                pass
    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in buckets.items()}
    if not return_sd:
        return means
    sds: dict[str, float] = {}
    for k, v in buckets.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        sds[k] = float(np.std(a, ddof=1)) if len(a) >= 2 else float("nan")
    return means, sds


def evaluate_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    classifier_seeds: list[int],
    return_sd: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    """
    Fixed held-out test (no re-split). Model seeds only affect model init.
    Returns mean metrics across models × utility seeds.
    """
    X_tr, y_tr = _xy(train_df, target)
    X_te, y_te = _xy(test_df, target)
    cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr = pd.to_numeric(y_tr, errors="coerce").fillna(0).to_numpy()
    y_te = pd.to_numeric(y_te, errors="coerce").fillna(0).to_numpy()
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    buckets = {"R2": [], "RMSE": [], "MAE": []}
    for name, model in REGRESSORS.items():
        for seed in classifier_seeds:
            reg = clone(model)
            if hasattr(reg, "set_params"):
                try:
                    reg.set_params(random_state=seed)
                except ValueError:
                    pass
            reg.fit(X_tr_s, y_tr)
            pred = reg.predict(X_te_s)
            buckets["R2"].append(r2_score(y_te, pred))
            buckets["RMSE"].append(float(np.sqrt(mean_squared_error(y_te, pred))))
            buckets["MAE"].append(mean_absolute_error(y_te, pred))
    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in buckets.items()}
    if not return_sd:
        return means
    sds: dict[str, float] = {}
    for k, v in buckets.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        sds[k] = float(np.std(a, ddof=1)) if len(a) >= 2 else float("nan")
    return means, sds


def _snap_classification_target(synth: pd.DataFrame, train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Map synthetic target values onto the discrete label set from Real-Train."""
    out = synth.copy()
    if target not in out.columns or target not in train.columns:
        return out
    classes = pd.unique(train[target])
    if len(classes) == 0:
        return out
    # numeric / continuous synth labels → nearest class
    if pd.api.types.is_numeric_dtype(train[target]) or pd.api.types.is_numeric_dtype(out[target]):
        class_vals = np.array(sorted(pd.to_numeric(classes, errors="coerce")))
        class_vals = class_vals[np.isfinite(class_vals)]
        if len(class_vals) == 0:
            return out
        y = pd.to_numeric(out[target], errors="coerce")
        y = y.fillna(class_vals[0]).to_numpy(dtype=float)
        idx = np.abs(y.reshape(-1, 1) - class_vals.reshape(1, -1)).argmin(axis=1)
        out[target] = class_vals[idx]
        # preserve int dtype when train labels are ints
        if pd.api.types.is_integer_dtype(train[target]):
            out[target] = out[target].astype(int)
        return out
    # categorical: strip / map unknown → mode
    mode = train[target].astype(str).mode()
    fill = mode.iloc[0] if len(mode) else str(classes[0])
    allowed = set(map(str, classes))
    out[target] = out[target].astype(str).map(lambda v: v if v in allowed else fill)
    return out


def compute_utility(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str,
    task: str,
    classifier_seeds: list[int],
) -> dict[str, float]:
    """TRTR, TSTR, and gaps (existing naming)."""
    out: dict[str, float] = {}
    if task == "classification":
        light = {
            "LogReg": CLASSIFIERS["LogReg"],
            "RandomForest": CLASSIFIERS["RandomForest"],
            "DecisionTree": CLASSIFIERS["DecisionTree"],
        }
        trtr = evaluate_classification(train_real, test_real, target, classifier_seeds, models=light)
        synth_u = synth.copy()
        for c in train_real.columns:
            if c not in synth_u.columns:
                synth_u[c] = train_real[c].iloc[0]
        synth_u = synth_u[train_real.columns]
        synth_u = _snap_classification_target(synth_u, train_real, target)
        tstr = evaluate_classification(synth_u, test_real, target, classifier_seeds, models=light)
        for m in ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC"]:
            out[f"{m}_TRTR"] = trtr[m]
            out[f"{m}_TSTR"] = tstr[m]
            out[f"{m}_Gap"] = trtr[m] - tstr[m]
    else:
        trtr = evaluate_regression(train_real, test_real, target, classifier_seeds)
        synth_u = synth.copy()
        for c in train_real.columns:
            if c not in synth_u.columns:
                synth_u[c] = train_real[c].iloc[0]
        synth_u = synth_u[train_real.columns]
        tstr = evaluate_regression(synth_u, test_real, target, classifier_seeds)
        for m in ["R2", "RMSE", "MAE"]:
            out[f"{m}_TRTR"] = trtr[m]
            out[f"{m}_TSTR"] = tstr[m]
            if m == "R2":
                out[f"{m}_Gap"] = trtr[m] - tstr[m]
            else:
                out[f"{m}_Increase"] = tstr[m] - trtr[m]
    return out


def utility_to_rows(metrics: dict[str, float], meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**meta, "metric_category": "Utility", "metric_name": k, "metric_value": v}
        for k, v in metrics.items()
    ]
