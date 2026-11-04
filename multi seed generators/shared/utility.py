"""Shared utility TRTR/TSTR evaluation (10 classifiers + 10 regressors).

Single source of truth for 1st / second / third batch. Model names come from
``shared/config/utility_models.yaml``. Features are StandardScaler-fitted on the
**training** frame of each path (Real-Train for TRTR, synthetic for TSTR).

y is NOT scaled for regression (matches prior protocol / notebooks). SVR and MLP
are therefore sensitive to target magnitude; X standardisation is still applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SHARED_DIR = Path(__file__).resolve().parent
UTILITY_MODELS_YAML = SHARED_DIR / "config" / "utility_models.yaml"

# n_jobs=-1 on tree ensembles; outer loop is models × seeds (no nested joblib).
CLASSIFIERS: dict[str, Any] = {
    "LogReg": LogisticRegression(max_iter=5000, solver="lbfgs"),  # default L2, C=1.0
    "SVM-RBF": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoost": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=5000),
}

REGRESSORS: dict[str, Any] = {
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
    "GradientBoost": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR_RBF": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
}

CLS_METRICS = ("Accuracy", "F1", "Precision", "Recall", "ROC_AUC")
REG_METRICS = ("R2", "RMSE", "MAE")

# TRTR depends only on Real-Train / Real-Test (fixed split) — cache per dataset_id.
_TRTR_CACHE: dict[str, tuple[dict[str, float], pd.DataFrame]] = {}


def clear_trtr_cache() -> None:
    _TRTR_CACHE.clear()


def load_utility_model_names() -> dict[str, list[str]]:
    with UTILITY_MODELS_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "classification": list(cfg["classification"]),
        "regression": list(cfg["regression"]),
    }


def get_classifiers(names: list[str] | None = None) -> dict[str, Any]:
    names = names or load_utility_model_names()["classification"]
    missing = [n for n in names if n not in CLASSIFIERS]
    if missing:
        raise KeyError(f"Unknown classifiers in utility_models.yaml: {missing}")
    return {n: CLASSIFIERS[n] for n in names}


def get_regressors(names: list[str] | None = None) -> dict[str, Any]:
    names = names or load_utility_model_names()["regression"]
    missing = [n for n in names if n not in REGRESSORS]
    if missing:
        raise KeyError(f"Unknown regressors in utility_models.yaml: {missing}")
    return {n: REGRESSORS[n] for n in names}


def expected_tstr_fit_counts() -> dict[str, int]:
    """Printable expected TSTR fit counts for the full 10-seed protocol."""
    n_cls_ds, n_reg_ds = 9, 6
    n_gen, n_seeds, n_models, n_util = 8, 10, 10, 10
    return {
        "classification_tstr_fits": n_cls_ds * n_gen * n_seeds * n_models * n_util,
        "regression_tstr_fits": n_reg_ds * n_gen * n_seeds * n_models * n_util,
        "formula_classification": "9 × 8 × 10 × 10 × 10 = 72,000",
        "formula_regression": "6 × 8 × 10 × 10 × 10 = 48,000",
    }


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
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
        le.fit(pd.concat([y_train.astype(str), y_test.astype(str)], ignore_index=True))
        return le.transform(y_train.astype(str)), le.transform(y_test.astype(str)), le
    return y_train.to_numpy(), y_test.to_numpy(), None


def _set_seed(est: Any, seed: int) -> Any:
    if hasattr(est, "set_params"):
        try:
            est.set_params(random_state=seed)
        except ValueError:
            pass
    return est


def evaluate_classification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    classifier_seeds: list[int],
    models: dict | None = None,
    return_sd: bool = False,
    return_per_model: bool = False,
    path_label: str = "TRTR",
) -> Any:
    """Mean metrics across models × utility seeds; optional per-model long frame."""
    models = models or get_classifiers()
    X_tr, y_tr = _xy(train_df, target)
    X_te, y_te = _xy(test_df, target)
    cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr_e, y_te_e, _ = _encode_y(y_tr, y_te)

    # Scaler fitted on this path's training data (Real-Train or synthetic).
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    buckets = {k: [] for k in CLS_METRICS}
    per_rows: list[dict[str, Any]] = []
    n_classes = len(np.unique(y_tr_e))
    if n_classes < 2:
        nan_m = {k: float("nan") for k in buckets}
        empty = pd.DataFrame(columns=["path", "model", "utility_seed", "metric", "value"])
        if return_per_model and return_sd:
            return nan_m, {k: float("nan") for k in buckets}, empty
        if return_per_model:
            return nan_m, empty
        if return_sd:
            return nan_m, {k: float("nan") for k in buckets}
        return nan_m

    for name, model in models.items():
        for seed in classifier_seeds:
            clf = _set_seed(clone(model), seed)
            clf.fit(X_tr_s, y_tr_e)
            pred = clf.predict(X_te_s)
            vals = {
                "Accuracy": float(accuracy_score(y_te_e, pred)),
                "F1": float(f1_score(y_te_e, pred, average="weighted", zero_division=0)),
                "Precision": float(
                    precision_score(y_te_e, pred, average="weighted", zero_division=0)
                ),
                "Recall": float(recall_score(y_te_e, pred, average="weighted", zero_division=0)),
            }
            try:
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_te_s)
                    if n_classes == 2:
                        vals["ROC_AUC"] = float(roc_auc_score(y_te_e, proba[:, 1]))
                    else:
                        vals["ROC_AUC"] = float(
                            roc_auc_score(y_te_e, proba, multi_class="ovr", average="weighted")
                        )
                else:
                    vals["ROC_AUC"] = float("nan")
            except Exception:
                vals["ROC_AUC"] = float("nan")

            for k, v in vals.items():
                buckets[k].append(v)
                per_rows.append(
                    {
                        "path": path_label,
                        "model": name,
                        "utility_seed": int(seed),
                        "metric": k,
                        "value": v,
                    }
                )

    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in buckets.items()}
    per_df = pd.DataFrame(per_rows)
    if return_sd:
        sds = {}
        for k, v in buckets.items():
            a = np.asarray(v, dtype=float)
            a = a[np.isfinite(a)]
            sds[k] = float(np.std(a, ddof=1)) if len(a) >= 2 else float("nan")
        if return_per_model:
            return means, sds, per_df
        return means, sds
    if return_per_model:
        return means, per_df
    return means


def evaluate_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    classifier_seeds: list[int],
    models: dict | None = None,
    return_sd: bool = False,
    return_per_model: bool = False,
    path_label: str = "TRTR",
) -> Any:
    """Mean metrics across models × utility seeds; optional per-model long frame.

    Note: target y is left in original units (not StandardScaler'd). Only features X
    are scaled, identically for TRTR and TSTR (scaler fit on each path's train frame).
    """
    models = models or get_regressors()
    X_tr, y_tr = _xy(train_df, target)
    X_te, y_te = _xy(test_df, target)
    cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr = pd.to_numeric(y_tr, errors="coerce").fillna(0).to_numpy()
    y_te = pd.to_numeric(y_te, errors="coerce").fillna(0).to_numpy()
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    buckets = {k: [] for k in REG_METRICS}
    per_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        for seed in classifier_seeds:
            reg = _set_seed(clone(model), seed)
            reg.fit(X_tr_s, y_tr)
            pred = reg.predict(X_te_s)
            vals = {
                "R2": float(r2_score(y_te, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                "MAE": float(mean_absolute_error(y_te, pred)),
            }
            for k, v in vals.items():
                buckets[k].append(v)
                per_rows.append(
                    {
                        "path": path_label,
                        "model": name,
                        "utility_seed": int(seed),
                        "metric": k,
                        "value": v,
                    }
                )

    means = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in buckets.items()}
    per_df = pd.DataFrame(per_rows)
    if return_sd:
        sds = {}
        for k, v in buckets.items():
            a = np.asarray(v, dtype=float)
            a = a[np.isfinite(a)]
            sds[k] = float(np.std(a, ddof=1)) if len(a) >= 2 else float("nan")
        if return_per_model:
            return means, sds, per_df
        return means, sds
    if return_per_model:
        return means, per_df
    return means


def _snap_classification_target(synth: pd.DataFrame, train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Map synthetic target values onto the discrete label set from Real-Train."""
    out = synth.copy()
    if target not in out.columns or target not in train.columns:
        return out
    classes = pd.unique(train[target])
    if len(classes) == 0:
        return out
    if pd.api.types.is_numeric_dtype(train[target]) or pd.api.types.is_numeric_dtype(out[target]):
        class_vals = np.array(sorted(pd.to_numeric(classes, errors="coerce")))
        class_vals = class_vals[np.isfinite(class_vals)]
        if len(class_vals) == 0:
            return out
        y = pd.to_numeric(out[target], errors="coerce")
        y = y.fillna(class_vals[0]).to_numpy(dtype=float)
        idx = np.abs(y.reshape(-1, 1) - class_vals.reshape(1, -1)).argmin(axis=1)
        out[target] = class_vals[idx]
        if pd.api.types.is_integer_dtype(train[target]):
            out[target] = out[target].astype(int)
        return out
    mode = train[target].astype(str).mode()
    fill = mode.iloc[0] if len(mode) else str(classes[0])
    allowed = set(map(str, classes))
    out[target] = out[target].astype(str).map(lambda v: v if v in allowed else fill)
    return out


def _align_synth(synth: pd.DataFrame, train_real: pd.DataFrame) -> pd.DataFrame:
    synth_u = synth.copy()
    for c in train_real.columns:
        if c not in synth_u.columns:
            synth_u[c] = train_real[c].iloc[0]
    return synth_u[train_real.columns]


def compute_utility(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str,
    task: str,
    classifier_seeds: list[int],
    *,
    dataset_id: Optional[str] = None,
    return_per_model: bool = True,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    """TRTR, TSTR, and gaps. Optionally return per-model long DataFrame."""
    out: dict[str, float] = {}
    per_parts: list[pd.DataFrame] = []

    if task == "classification":
        models = get_classifiers()
        cache_key = f"cls::{dataset_id}" if dataset_id else None
        if cache_key and cache_key in _TRTR_CACHE:
            trtr, trtr_per = _TRTR_CACHE[cache_key]
        else:
            trtr, trtr_per = evaluate_classification(
                train_real,
                test_real,
                target,
                classifier_seeds,
                models=models,
                return_per_model=True,
                path_label="TRTR",
            )
            if cache_key:
                _TRTR_CACHE[cache_key] = (trtr, trtr_per)

        synth_u = _align_synth(synth, train_real)
        synth_u = _snap_classification_target(synth_u, train_real, target)
        tstr, tstr_per = evaluate_classification(
            synth_u,
            test_real,
            target,
            classifier_seeds,
            models=models,
            return_per_model=True,
            path_label="TSTR",
        )
        for m in CLS_METRICS:
            out[f"{m}_TRTR"] = trtr[m]
            out[f"{m}_TSTR"] = tstr[m]
            out[f"{m}_Gap"] = trtr[m] - tstr[m]
        per_parts = [trtr_per, tstr_per]
    else:
        models = get_regressors()
        cache_key = f"reg::{dataset_id}" if dataset_id else None
        if cache_key and cache_key in _TRTR_CACHE:
            trtr, trtr_per = _TRTR_CACHE[cache_key]
        else:
            trtr, trtr_per = evaluate_regression(
                train_real,
                test_real,
                target,
                classifier_seeds,
                models=models,
                return_per_model=True,
                path_label="TRTR",
            )
            if cache_key:
                _TRTR_CACHE[cache_key] = (trtr, trtr_per)

        synth_u = _align_synth(synth, train_real)
        tstr, tstr_per = evaluate_regression(
            synth_u,
            test_real,
            target,
            classifier_seeds,
            models=models,
            return_per_model=True,
            path_label="TSTR",
        )
        for m in REG_METRICS:
            out[f"{m}_TRTR"] = trtr[m]
            out[f"{m}_TSTR"] = tstr[m]
            if m == "R2":
                out[f"{m}_Gap"] = trtr[m] - tstr[m]
            else:
                out[f"{m}_Increase"] = tstr[m] - trtr[m]
        per_parts = [trtr_per, tstr_per]

    if not return_per_model:
        return out
    per_df = pd.concat(per_parts, ignore_index=True) if per_parts else pd.DataFrame()
    return out, per_df


def utility_to_rows(metrics: dict[str, float], meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**meta, "metric_category": "Utility", "metric_name": k, "metric_value": v}
        for k, v in metrics.items()
    ]


def per_model_to_long(
    per_df: pd.DataFrame,
    meta: dict[str, Any],
) -> pd.DataFrame:
    """Attach run metadata: dataset, generator, generator_seed, path, model, utility_seed, metric, value."""
    if per_df is None or per_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "dataset_id",
                "generator",
                "generator_seed",
                "path",
                "model",
                "utility_seed",
                "metric",
                "value",
            ]
        )
    out = per_df.copy()
    out.insert(0, "dataset", meta.get("dataset"))
    out.insert(1, "dataset_id", meta.get("dataset_id"))
    out.insert(2, "generator", meta.get("generator"))
    out.insert(3, "generator_seed", meta.get("seed"))
    return out
