#!/usr/bin/env python3
"""Recompute Adult/Alzheimer TRTR–TSTR utility gaps for Accuracy, Precision, Recall, F1.

Uses the same participant-wise / complete-case prep as the patched generator notebooks.
Prefers cached synthetic CSVs under each generator folder's synthetic_cache/;
otherwise retrains SDV models (Alzheimer always feasible; Adult uses FAST 1000-row subsample).

Other-GAN / Diffusion caches are used when present; those generators are not retrained here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Agreed analysis" / "classification" / "adult_alzh_four_metrics_kernel"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LogReg": LogisticRegression(max_iter=5000, solver="liblinear", random_state=42),
    "SVM-RBF": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42),
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "MLP": MLPClassifier(max_iter=2000, random_state=42),
}


def participant_split(df, subject_col="Subject ID", label_col="Group", test_size=0.2, seed=42):
    subj_label = df.groupby(subject_col)[label_col].first()
    subjects = subj_label.index.to_numpy()
    subj_y = subj_label.to_numpy()
    subj_train, subj_test = train_test_split(
        subjects, test_size=test_size, random_state=seed, stratify=subj_y
    )
    train_df = (
        df[df[subject_col].isin(subj_train)].drop(columns=[subject_col]).reset_index(drop=True)
    )
    test_df = (
        df[df[subject_col].isin(subj_test)].drop(columns=[subject_col]).reset_index(drop=True)
    )
    return train_df, test_df


def load_alzheimer():
    path = ROOT / "Datasets" / "Alzhimers.xlsx"
    ad = pd.read_excel(path)
    ad = ad.drop(columns=["M/F", "MRI ID", "Hand"], errors="ignore")
    ad["Group"] = ad["Group"].replace({"Demented": 1, "Nondemented": 0, "Converted": 1})
    ad = ad.dropna().reset_index(drop=True)
    train_real, test_real = participant_split(ad)
    return train_real, test_real, train_real.copy()


def load_adult():
    # Match SDV Adult notebook complete-case prep as closely as practical
    candidates = [
        ROOT / "Datasets" / "adult_census.csv",
        ROOT / "Generators" / "SDV models" / "adult_data.csv",
        ROOT / "Generators" / "Other GANS" / "adult_data.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Adult dataset not found")
    df = pd.read_csv(path)
    # normalize income col
    if "income" not in df.columns:
        for c in df.columns:
            if "income" in c.lower():
                df = df.rename(columns={c: "income"})
                break
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Adult complete-case: dropped {before - len(df)} NA rows → {len(df)}")
    train_real, test_real = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["income"]
    )
    return train_real.reset_index(drop=True), test_real.reset_index(drop=True), df


def _encode_features(X_train, X_test):
    common = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].copy()
    X_test = X_test[common].copy()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
        X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    train_keep = ~X_train.isna().any(axis=1)
    test_keep = ~X_test.isna().any(axis=1)
    X_train = X_train.loc[train_keep].astype(np.float64)
    X_test = X_test.loc[test_keep].astype(np.float64)
    return X_train, X_test, train_keep, test_keep


def _clean_binary_labels(y):
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    y = y.astype(str).str.strip().str.replace(".", "", regex=False)
    mapping = {"<=50K": 0, ">50K": 1, "0": 0, "1": 1}
    return y.map(mapping).astype(int)


def evaluate_models(train_df, test_df, label_col, models=MODELS, test_size=0.2, seed=42, resplit=True):
    X_train = train_df.drop(columns=[label_col])
    y_train = _clean_binary_labels(train_df[label_col])
    X_test = test_df.drop(columns=[label_col])
    y_test = _clean_binary_labels(test_df[label_col])

    if resplit:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, test_size=test_size, random_state=seed, stratify=y_train
        )
        _, X_test, _, y_test = train_test_split(
            X_test, y_test, test_size=test_size, random_state=seed, stratify=y_test
        )

    X_train, X_test, train_keep, test_keep = _encode_features(X_train, X_test)
    y_train = y_train.loc[train_keep]
    y_test = y_test.loc[test_keep]
    if len(X_train) == 0 or len(X_test) == 0 or y_train.nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1": np.nan,
                    "AUC": np.nan,
                }
                for name in models
            ]
        )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []
    for name, base in models.items():
        clf = clone(base)
        try:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_s)[:, 1]
            else:
                y_prob = None
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(
                        y_test, y_pred, average="binary", zero_division=0
                    ),
                    "Recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
                    "F1": f1_score(y_test, y_pred, average="binary", zero_division=0),
                    "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
                }
            )
        except Exception as exc:
            print(f"  {name}: {exc}")
            rows.append(
                {
                    "Model": name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1": np.nan,
                    "AUC": np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare_generators(train_real, test_real, synthetic, label_col, resplit):
    trtr = evaluate_models(
        train_real, test_real, label_col=label_col, resplit=resplit
    )
    rows = []
    for name, synth_df in synthetic.items():
        sdf = synth_df.copy()
        # align label name
        if label_col not in sdf.columns:
            for c in sdf.columns:
                if c.lower() == label_col.lower():
                    sdf = sdf.rename(columns={c: label_col})
                    break
        tstr = evaluate_models(sdf, test_real, label_col=label_col, resplit=resplit)
        cmp = trtr.merge(tstr, on="Model", suffixes=("_TRTR", "_TSTR"))
        for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
            a, b = f"{metric}_TRTR", f"{metric}_TSTR"
            if a in cmp.columns and b in cmp.columns:
                cmp[f"{metric}_Drop"] = cmp[a] - cmp[b]
        cmp["Synthetic_Model"] = name
        rows.append(cmp)
        print(
            f"  {name}: AccDrop={cmp['Accuracy_Drop'].mean():.4f} "
            f"P={cmp['Precision_Drop'].mean():.4f} "
            f"R={cmp['Recall_Drop'].mean():.4f} "
            f"F1={cmp['F1_Drop'].mean():.4f}"
        )
    return pd.concat(rows, ignore_index=True)


def load_synth_cache(cache_dirs, names):
    found = {}
    for d in cache_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for name in names:
            for cand in [d / f"{name}.csv", d / f"{name.replace('-', '_')}.csv"]:
                if cand.exists() and name not in found:
                    found[name] = pd.read_csv(cand)
    return found


def train_sdv(train_df, names, n_samples, epochs=40):
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        CTGANSynthesizer,
        CopulaGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(train_df)
    builders = {
        "CTGAN": lambda: CTGANSynthesizer(meta, epochs=epochs, verbose=False),
        "CopulaGAN": lambda: CopulaGANSynthesizer(meta, epochs=epochs, verbose=False),
        "TVAE": lambda: TVAESynthesizer(meta, epochs=epochs, verbose=False),
        "GaussianCopula": lambda: GaussianCopulaSynthesizer(meta),
    }
    out = {}
    for name in names:
        print(f"Training SDV {name} …")
        model = builders[name]()
        model.fit(train_df)
        out[name] = model.sample(num_rows=n_samples)
    return out


def summarize(combined: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Accuracy_Drop", "Precision_Drop", "Recall_Drop", "F1_Drop", "AUC_Drop"] if c in combined.columns]
    return (
        combined.groupby("Synthetic_Model", as_index=False)[cols]
        .mean()
        .sort_values("Accuracy_Drop")
    )


def run_alzheimer():
    print("=== Alzheimer ===")
    train_real, test_real, fit_df = load_alzheimer()
    print(f"train={len(train_real)} test={len(test_real)}")
    caches = [
        ROOT / "Generators/SDV models/synthetic_cache",
        ROOT / "Generators/Other GANS/synthetic_cache",
        ROOT / "Generators/Diffusion GANs/synthetic_cache",
    ]
    synth = load_synth_cache(
        caches,
        ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula", "CTABGAN", "WGAN_GP", "TabDDPM", "ForestDiffusion"],
    )
    need_sdv = [n for n in ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"] if n not in synth]
    if need_sdv:
        synth.update(train_sdv(fit_df, need_sdv, n_samples=len(fit_df), epochs=100))
        cache = ROOT / "Generators/SDV models/synthetic_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for k, v in synth.items():
            if k in need_sdv:
                v.to_csv(cache / f"{k}.csv", index=False)

    missing = [n for n in ["CTABGAN", "WGAN_GP", "TabDDPM", "ForestDiffusion"] if n not in synth]
    if missing:
        print(f"WARNING: no cache for {missing}; skip those generators (re-run Other/Diffusion notebooks).")

    combined = compare_generators(
        train_real, test_real, synth, label_col="Group", resplit=False
    )
    summary = summarize(combined)
    xlsx = OUT / "TRTR_TSTR_four_metrics_Alzheimer.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        combined.to_excel(w, sheet_name="All_Comparisons", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {xlsx}")
    return summary


def run_adult():
    print("=== Adult ===")
    train_real, test_real, full = load_adult()
    caches = [
        ROOT / "Generators/SDV models/synthetic_cache",
        ROOT / "Generators/Other GANS/synthetic_cache",
        ROOT / "Generators/Diffusion GANs/synthetic_cache",
    ]
    synth = load_synth_cache(
        caches,
        ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula", "CTABGAN", "WGAN_GP", "TabDDPM", "ForestDiffusion"],
    )
    need_sdv = [n for n in ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"] if n not in synth]
    if need_sdv:
        fit_df = full.sample(n=min(1000, len(full)), random_state=42).reset_index(drop=True)
        synth.update(train_sdv(fit_df, need_sdv, n_samples=1000, epochs=40))
        cache = ROOT / "Generators/SDV models/synthetic_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for k, v in synth.items():
            if k in need_sdv:
                v.to_csv(cache / f"{k}.csv", index=False)

    missing = [n for n in ["CTABGAN", "WGAN_GP", "TabDDPM", "ForestDiffusion"] if n not in synth]
    if missing:
        print(f"WARNING: no cache for {missing}; skip those generators (re-run Other/Diffusion notebooks).")

    # Adult notebook uses resplit=True inside evaluate_models
    combined = compare_generators(
        train_real, test_real, synth, label_col="income", resplit=True
    )
    summary = summarize(combined)
    xlsx = OUT / "TRTR_TSTR_four_metrics_Adult.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        combined.to_excel(w, sheet_name="All_Comparisons", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {xlsx}")
    return summary


def main():
    alzh = run_alzheimer()
    adult = run_adult()
    # Combined wide table for chat
    rows = []
    for ds, summ in [("Alzheimer", alzh), ("Adult", adult)]:
        for _, r in summ.iterrows():
            rows.append({"Dataset": ds, **r.to_dict()})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUT / "adult_alzh_four_metric_summary.csv", index=False)
    print(f"\nCombined summary → {OUT / 'adult_alzh_four_metric_summary.csv'}")


if __name__ == "__main__":
    main()
