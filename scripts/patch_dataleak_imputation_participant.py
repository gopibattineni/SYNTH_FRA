#!/usr/bin/env python3
"""Update utility-data-leak Adult/Alzheimer notebooks:

- Alzheimer: complete-case (no imputation) + Subject ID participant split
- Adult: complete-case (no mean/mode fill)
- Evaluation: TRTR/TSTR use train_real / test_real (no train-on-full leak)
- Alzheimer evaluate_models: fixed participant frames; seed only varies classifiers
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAK = ROOT / "Generators" / "Experiment with utility data leak"
DIFF = LEAK / "diffusion_dataleak"

ALZH_LOAD_OLD = '''target_col = "Group"
ad_data = raw_data.drop(columns=["Subject ID", "M/F", "MRI ID", "Hand"], errors="ignore")
ad_data[target_col] = ad_data[target_col].replace({"Demented": 1, "Nondemented": 0, "Converted": 1})
ad_data[target_col] = pd.to_numeric(ad_data[target_col], errors="coerce").fillna(0).astype(int)

for col in ad_data.select_dtypes(include=[np.number]).columns:
    if ad_data[col].isnull().any():
        ad_data[col] = ad_data[col].fillna(ad_data[col].mean())

for col in ad_data.select_dtypes(include=["object"]).columns:
    if ad_data[col].isnull().any():
        modes = ad_data[col].mode()
        ad_data[col] = ad_data[col].fillna(modes[0] if len(modes) else "")

# Use same variable name pattern as other notebooks
alzheimer_data = ad_data.copy()

X = alzheimer_data.drop(columns=[target_col])
y = alzheimer_data[target_col]'''

ALZH_LOAD_NEW = '''target_col = "Group"
subject_col = "Subject ID"

# Keep Subject ID for participant-level splits; drop other IDs / unused cols
ad_data = raw_data.drop(columns=["M/F", "MRI ID", "Hand"], errors="ignore")
if subject_col not in ad_data.columns:
    raise KeyError(f"{subject_col} is required for participant-level train/test splits")

ad_data[target_col] = ad_data[target_col].replace(
    {"Demented": 1, "Nondemented": 0, "Converted": 1}
)
ad_data[target_col] = pd.to_numeric(ad_data[target_col], errors="coerce")

# Complete-case only — no mean / median / mode imputation
_before = len(ad_data)
_n_missing_rows = int(ad_data.isna().any(axis=1).sum())
ad_data = ad_data.dropna().reset_index(drop=True)
ad_data[target_col] = ad_data[target_col].astype(int)
print(
    f"Dropped {_before - len(ad_data)} rows with missing feature values "
    f"(complete-case; no imputation; rows_with_na={_n_missing_rows})"
)
assert not ad_data.isna().any().any(), "Unexpected NaNs remain after dropna"

def _participant_split(df, subject_col=subject_col, label_col=target_col, test_size=0.2, seed=42):
    """Split by Subject ID so all sessions of a participant stay in one fold."""
    subj_label = df.groupby(subject_col)[label_col].first()
    subjects = subj_label.index.to_numpy()
    subj_y = subj_label.to_numpy()
    subj_train, subj_test = train_test_split(
        subjects, test_size=test_size, random_state=seed, stratify=subj_y
    )
    assert set(subj_train).isdisjoint(set(subj_test)), "Subject overlap between train and test"
    train_df = (
        df[df[subject_col].isin(subj_train)]
        .drop(columns=[subject_col])
        .reset_index(drop=True)
    )
    test_df = (
        df[df[subject_col].isin(subj_test)]
        .drop(columns=[subject_col])
        .reset_index(drop=True)
    )
    print(
        f"Participant split: subjects={len(subjects)} "
        f"(train={len(subj_train)}, test={len(subj_test)}); "
        f"sessions train={len(train_df)}, test={len(test_df)}"
    )
    print("Train Group:", train_df[label_col].value_counts().to_dict())
    print("Test Group:", test_df[label_col].value_counts().to_dict())
    return train_df, test_df

# Fixed participant split used for generation + TRTR/TSTR
train_real, test_real = _participant_split(ad_data, test_size=TEST_SIZE if "TEST_SIZE" in dir() else 0.2, seed=SEED if "SEED" in dir() else 42)

# Generators / fidelity train on train subjects only (leak-safe)
alzheimer_data = train_real.copy()

X = pd.concat([train_real, test_real], ignore_index=True).drop(columns=[target_col])
y = pd.concat([train_real, test_real], ignore_index=True)[target_col]'''

# TEST_SIZE/SEED are defined AFTER load in current notebooks — fix order.
# Better rewrite load block so participant split happens AFTER SEED/TEST_SIZE are set,
# OR hardcode 0.2/42 in split and re-split in single-run cell.

ALZH_LOAD_NEW = '''target_col = "Group"
subject_col = "Subject ID"

# Keep Subject ID for participant-level splits; drop other IDs / unused cols
ad_data = raw_data.drop(columns=["M/F", "MRI ID", "Hand"], errors="ignore")
if subject_col not in ad_data.columns:
    raise KeyError(f"{subject_col} is required for participant-level train/test splits")

ad_data[target_col] = ad_data[target_col].replace(
    {"Demented": 1, "Nondemented": 0, "Converted": 1}
)
ad_data[target_col] = pd.to_numeric(ad_data[target_col], errors="coerce")

# Complete-case only — no mean / median / mode imputation
_before = len(ad_data)
_n_missing_rows = int(ad_data.isna().any(axis=1).sum())
ad_data = ad_data.dropna().reset_index(drop=True)
ad_data[target_col] = ad_data[target_col].astype(int)
print(
    f"Dropped {_before - len(ad_data)} rows with missing feature values "
    f"(complete-case; no imputation; rows_with_na={_n_missing_rows})"
)
assert not ad_data.isna().any().any(), "Unexpected NaNs remain after dropna"

def _participant_split(df, subject_col=subject_col, label_col=target_col, test_size=0.2, seed=42):
    """Split by Subject ID so all sessions of a participant stay in one fold."""
    subj_label = df.groupby(subject_col)[label_col].first()
    subjects = subj_label.index.to_numpy()
    subj_y = subj_label.to_numpy()
    subj_train, subj_test = train_test_split(
        subjects, test_size=test_size, random_state=seed, stratify=subj_y
    )
    assert set(subj_train).isdisjoint(set(subj_test)), "Subject overlap between train and test"
    train_df = (
        df[df[subject_col].isin(subj_train)]
        .drop(columns=[subject_col])
        .reset_index(drop=True)
    )
    test_df = (
        df[df[subject_col].isin(subj_test)]
        .drop(columns=[subject_col])
        .reset_index(drop=True)
    )
    print(
        f"Participant split: subjects={len(subjects)} "
        f"(train={len(subj_train)}, test={len(subj_test)}); "
        f"sessions train={len(train_df)}, test={len(test_df)}"
    )
    print("Train Group:", train_df[label_col].value_counts().to_dict())
    print("Test Group:", test_df[label_col].value_counts().to_dict())
    return train_df, test_df

# Placeholder until SEED/TEST_SIZE are set below; single-run cell re-applies split
alzheimer_data = ad_data.drop(columns=[subject_col]).copy()
X = alzheimer_data.drop(columns=[target_col])
y = alzheimer_data[target_col]
train_real = test_real = None'''

ALZH_SPLIT_OLD = '''train_real, test_real = train_test_split(
    alzheimer_data,
    test_size=TEST_SIZE,
    stratify=alzheimer_data[target_col],
    random_state=seed
)'''

ALZH_SPLIT_NEW = '''# Participant-level split (no subject leakage across train/test)
train_real, test_real = _participant_split(
    ad_data, subject_col=subject_col, label_col=target_col, test_size=TEST_SIZE, seed=seed
)
alzheimer_data = train_real.copy()  # generators fit on train subjects only
X = pd.concat([train_real, test_real], ignore_index=True).drop(columns=[target_col])
y = pd.concat([train_real, test_real], ignore_index=True)[target_col]'''

ALZH_CANDIDATES_OLD = '''candidate_paths = [
    Path("Alzhimers.xlsx"),
    Path("Alzheimer.xlsx"),
    Path("clean_alzheimer.csv"),
    Path("../Other GANS/clean_alzheimer.csv"),
    Path("../../Other GANS/clean_alzheimer.csv"),
]'''

ALZH_CANDIDATES_NEW = '''_here = Path.cwd().resolve()
candidate_paths = [
    Path("Alzhimers.xlsx"),
    Path("Alzheimer.xlsx"),
    Path("clean_alzheimer.csv"),
    Path("../Other GANS/clean_alzheimer.csv"),
    Path("../../Other GANS/clean_alzheimer.csv"),
    Path("../../../Datasets/Alzhimers.xlsx"),
    Path("../../../../Datasets/Alzhimers.xlsx"),
    _here.parents[3] / "Datasets" / "Alzhimers.xlsx" if len(_here.parents) >= 4 else Path("Datasets/Alzhimers.xlsx"),
]
# Also walk up for Datasets/Alzhimers.xlsx
for _root in [_here, *_here.parents]:
    candidate_paths.append(_root / "Datasets" / "Alzhimers.xlsx")
    candidate_paths.append(_root / "Datasets" / "Alzheimer.xlsx")
    candidate_paths.append(_root / "clean_alzheimer.csv")'''

EVAL_ALZH_OLD = '''def evaluate_models(
    train_df,
    test_df,
    label_col,
    models,
    test_size=0.2,
    seeds=[42,43,44,45,46,47,48,49,50,51]
):

    results = []
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[label_col] = pd.to_numeric(train_df[label_col], errors="coerce").astype(int)
    test_df[label_col] = pd.to_numeric(test_df[label_col], errors="coerce").astype(int)

    for name, model in models.items():

        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for seed in seeds:

            X_train = train_df.drop(columns=[label_col])
            y_train = train_df[label_col]

            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                test_size=test_size,
                random_state=seed,
                stratify=y_train
            )

            X_test = test_df.drop(columns=[label_col])
            y_test = test_df[label_col]

            _, X_test, _, y_test = train_test_split(
                X_test,
                y_test,
                test_size=test_size,
                random_state=seed,
                stratify=y_test
            )

            scaler = StandardScaler().fit(X_train)

            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = clone(model)

            if hasattr(clf, "random_state"):
                clf.set_params(random_state=seed)

            clf.fit(X_train_s, y_train)

            y_pred = clf.predict(X_test_s)'''

EVAL_ALZH_NEW = '''def evaluate_models(
    train_df,
    test_df,
    label_col=None,
    models=None,
    test_size=0.2,
    seeds=[42,43,44,45,46,47,48,49,50,51],
    label=None,
    resplit=False,
):
    """Evaluate classifiers with Acc/F1/Precision/Recall.

    For Alzheimer participant splits, keep resplit=False so sessions stay in the
    fixed train_real / test_real folds (seed only varies classifier RNG).
    """
    if label is not None and label_col is None:
        label_col = label
    if label_col is None:
        raise ValueError("label_col is required")
    if models is None:
        raise ValueError("models is required")

    results = []
    train_df = train_df.copy()
    test_df = test_df.copy()
    # Drop Subject ID if still present
    for _df in (train_df, test_df):
        if "Subject ID" in _df.columns:
            _df.drop(columns=["Subject ID"], inplace=True)
    train_df[label_col] = pd.to_numeric(train_df[label_col], errors="coerce").astype(int)
    test_df[label_col] = pd.to_numeric(test_df[label_col], errors="coerce").astype(int)

    for name, model in models.items():

        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for seed in seeds:

            X_train = train_df.drop(columns=[label_col])
            y_train = train_df[label_col]
            X_test = test_df.drop(columns=[label_col])
            y_test = test_df[label_col]

            if resplit:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    test_size=test_size,
                    random_state=seed,
                    stratify=y_train
                )
                _, X_test, _, y_test = train_test_split(
                    X_test,
                    y_test,
                    test_size=test_size,
                    random_state=seed,
                    stratify=y_test
                )

            # Complete-case residual drop (no imputation)
            train_keep = ~X_train.isna().any(axis=1)
            test_keep = ~X_test.isna().any(axis=1)
            X_train, y_train = X_train.loc[train_keep], y_train.loc[train_keep]
            X_test, y_test = X_test.loc[test_keep], y_test.loc[test_keep]

            scaler = StandardScaler().fit(X_train)

            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = clone(model)

            if hasattr(clf, "random_state"):
                clf.set_params(random_state=seed)

            clf.fit(X_train_s, y_train)

            y_pred = clf.predict(X_test_s)'''

ADULT_IMPUTE_OLD = '''# Fill missing values
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])'''

ADULT_IMPUTE_NEW = '''# Complete-case only — no mean / mode imputation
_before = len(data)
_n_missing_rows = int(data.isna().any(axis=1).sum())
data = data.dropna().reset_index(drop=True)
print(
    f"Dropped {_before - len(data)} rows with missing feature values "
    f"(complete-case; no imputation; rows_with_na={_n_missing_rows})"
)
assert not data.isna().any().any(), "Unexpected NaNs remain after dropna"

numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns'''


def _set_source(cell: dict, src: str) -> None:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def _patch_text(src: str, replacements: list[tuple[str, str]]) -> tuple[str, int]:
    n = 0
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            n += 1
    return src, n


def patch_alzh(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    stats = {"file": str(path), "cells": 0, "hits": []}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        reps = [
            (ALZH_CANDIDATES_OLD, ALZH_CANDIDATES_NEW),
            (ALZH_LOAD_OLD, ALZH_LOAD_NEW),
            (ALZH_SPLIT_OLD, ALZH_SPLIT_NEW),
            (EVAL_ALZH_OLD, EVAL_ALZH_NEW),
            # Fix comparison to use participant folds + label_col
            (
                '''trtr_results = evaluate_models(
    train_df=alzheimer_data,
    test_df=alzheimer_data,
    label="Group",
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
                '''trtr_results = evaluate_models(
    train_df=train_real,
    test_df=test_real,
    label_col="Group",
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds,
    resplit=False,
)''',
            ),
            (
                '''trtr_results = evaluate_models(
    train_df=alzheimer_data,
    test_df=alzheimer_data,
    label_col="Group",
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
                '''trtr_results = evaluate_models(
    train_df=train_real,
    test_df=test_real,
    label_col="Group",
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds,
    resplit=False,
)''',
            ),
            (
                '''    tstr_results = evaluate_models(
        train_df=synthetic_train_df,
        test_df=alzheimer_data,
        label="Group",
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds
    )''',
                '''    tstr_results = evaluate_models(
        train_df=synthetic_train_df,
        test_df=test_real,
        label_col="Group",
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds,
        resplit=False,
    )''',
            ),
            (
                '''    tstr_results = evaluate_models(
        train_df=synthetic_train_df,
        test_df=alzheimer_data,
        label_col="Group",
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds
    )''',
                '''    tstr_results = evaluate_models(
        train_df=synthetic_train_df,
        test_df=test_real,
        label_col="Group",
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds,
        resplit=False,
    )''',
            ),
        ]
        src, n = _patch_text(src, reps)

        # TRTR diagnostic cell: use fixed participant frames; seed varies clf only
        if "Starting TRTR Evaluation" in src and "train_test_split(\n            X," in src:
            src2 = src.replace(
                '''        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed
        )

        clf = clone(model)

        if hasattr(clf, "random_state"):
            clf.set_params(random_state=seed)

        clf.fit(X_train_real, y_train_real)

        y_pred = clf.predict(X_test_real)''',
                '''        # Fixed participant fold; seed only changes classifier RNG
        X_train_real = train_real.drop(columns=[target_col])
        y_train_real = train_real[target_col]
        X_test_real = test_real.drop(columns=[target_col])
        y_test_real = test_real[target_col]

        clf = clone(model)

        if hasattr(clf, "random_state"):
            clf.set_params(random_state=seed)

        clf.fit(X_train_real, y_train_real)

        y_pred = clf.predict(X_test_real)'''
            )
            if src2 != src:
                src = src2
                n += 1

        if src != orig:
            _set_source(cell, src)
            stats["cells"] += 1
            stats["hits"].append(n)
    if stats["cells"]:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def patch_adult(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    stats = {"file": str(path), "cells": 0, "hits": []}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        reps = [
            (ADULT_IMPUTE_OLD, ADULT_IMPUTE_NEW),
            # Prefer sampling AFTER complete-case on full data when pattern is sample-then-impute
            (
                '''data = data.replace("?", np.nan)
n_samples = min(1000, len(data))
data = data.sample(n=n_samples, random_state=42).reset_index(drop=True)

# Complete-case only — no mean / mode imputation''',
                '''data = data.replace("?", np.nan)

# Complete-case only — no mean / mode imputation''',
            ),
            (
                '''assert not data.isna().any().any(), "Unexpected NaNs remain after dropna"

numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns

# Encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Prepare features, target, and metadata
X = data.drop(columns=[target_col])
y = data[target_col]

processed_data = pd.concat([X, y], axis=1)''',
                '''assert not data.isna().any().any(), "Unexpected NaNs remain after dropna"

# Sample after complete-case so NA handling is not confounded by imputation
n_samples = min(1000, len(data))
data = data.sample(n=n_samples, random_state=42).reset_index(drop=True)
print(f"Adult subsample after complete-case: {len(data)} rows")

numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns

# Encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Prepare features, target, and metadata
X = data.drop(columns=[target_col])
y = data[target_col]

processed_data = pd.concat([X, y], axis=1)''',
            ),
            # Second Adult load cell (income clean then sample then impute)
            (
                '''# Take 1000 real samples
n_samples = min(1000, len(data))
data = data.sample(n=n_samples, random_state=42).reset_index(drop=True)

# Complete-case only — no mean / mode imputation''',
                '''# Complete-case only — no mean / mode imputation''',
            ),
            (
                '''trtr_results = evaluate_models(
    train_df=real_data,
    test_df=real_data,
    label_col=target_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
                '''trtr_results = evaluate_models(
    train_df=train_real,
    test_df=test_real,
    label_col=target_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
            ),
            (
                '''        test_df=real_data,
        label_col=target_col,
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds
    )''',
                '''        test_df=test_real,
        label_col=target_col,
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds
    )''',
            ),
        ]
        src, n = _patch_text(src, reps)

        # Main Adult comparison cell variants using processed_data / label=
        for old, new in [
            (
                '''trtr_results = evaluate_models(
    train_df=processed_data,
    test_df=processed_data,
    label="income",
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
                '''trtr_results = evaluate_models(
    train_df=train_real,
    test_df=test_real,
    label_col=target_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
            ),
            (
                '''trtr_results = evaluate_models(
    train_df=processed_data,
    test_df=processed_data,
    label_col=target_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
                '''trtr_results = evaluate_models(
    train_df=train_real,
    test_df=test_real,
    label_col=target_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)''',
            ),
            (
                '''        test_df=processed_data,
        label="income",''',
                '''        test_df=test_real,
        label_col=target_col,''',
            ),
            (
                '''        test_df=processed_data,
        label_col=target_col,''',
                '''        test_df=test_real,
        label_col=target_col,''',
            ),
        ]:
            if old in src:
                src = src.replace(old, new)
                n += 1

        if src != orig:
            _set_source(cell, src)
            stats["cells"] += 1
            stats["hits"].append(n)
    if stats["cells"]:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    targets = [
        (LEAK / "2. Alzhimers" / "alzhimers.ipynb", patch_alzh),
        (DIFF / "2. Alzhimers" / "alzhimers.ipynb", patch_alzh),
        (LEAK / "3. Adult" / "adult.ipynb", patch_adult),
        (DIFF / "3. Adult" / "adult.ipynb", patch_adult),
    ]
    for path, fn in targets:
        if not path.exists():
            print("MISSING", path)
            continue
        stats = fn(path)
        print(f"{path.relative_to(ROOT)}: cells_updated={stats['cells']} hits={stats['hits']}")


if __name__ == "__main__":
    main()
