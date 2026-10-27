"""Privacy metrics aligned with existing names (Mahalanobis, NNDR, MIA_AUC)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _encode_numeric(df: pd.DataFrame) -> np.ndarray:
    work = df.copy()
    for c in work.columns:
        if work[c].dtype == object or str(work[c].dtype) == "category":
            work[c] = LabelEncoder().fit_transform(work[c].astype(str))
        else:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    return work.fillna(0).to_numpy(dtype=float)


def compute_privacy(train_real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    cols = [c for c in train_real.columns if c in synth.columns]
    R = _encode_numeric(train_real[cols])
    S = _encode_numeric(synth[cols])
    # subsample for cost
    rng = np.random.default_rng(0)
    n_r = min(500, len(R))
    n_s = min(500, len(S))
    R = R[rng.choice(len(R), n_r, replace=False)]
    S = S[rng.choice(len(S), n_s, replace=False)]

    scaler = StandardScaler().fit(R)
    Rs = scaler.transform(R)
    Ss = scaler.transform(S)

    out: dict[str, float] = {}

    # Mahalanobis mean distance of synth → real distribution
    try:
        cov = np.cov(Rs, rowvar=False)
        # regularize
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        inv = np.linalg.pinv(cov)
        mu = Rs.mean(axis=0)
        diff = Ss - mu
        d = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
        out["Mahalanobis_Distance"] = float(np.mean(d))
        out["Mean_Distance"] = float(np.mean(d))
        out["Median_Distance"] = float(np.median(d))
    except Exception:
        out["Mahalanobis_Distance"] = float("nan")
        out["Mean_Distance"] = float("nan")
        out["Median_Distance"] = float("nan")

    # NNDR: ratio of distance to closest real vs 2nd closest
    try:
        nn = NearestNeighbors(n_neighbors=2).fit(Rs)
        dists, _ = nn.kneighbors(Ss)
        d1 = dists[:, 0]
        d2 = np.maximum(dists[:, 1], 1e-12)
        out["NNDR"] = float(np.mean(d1 / d2))
    except Exception:
        out["NNDR"] = float("nan")

    # Simple MIA AUC: shadow classifier members(train) vs non-members(synth holdout)
    try:
        n = min(len(Rs), len(Ss), 300)
        mem = Rs[:n]
        non = Ss[:n]
        X = np.vstack([mem, non])
        y = np.array([1] * n + [0] * n)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        out["MIA_AUC"] = float(roc_auc_score(yte, proba))
    except Exception:
        out["MIA_AUC"] = float("nan")

    return out


def privacy_to_rows(metrics: dict[str, float], meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**meta, "metric_category": "Privacy", "metric_name": k, "metric_value": v}
        for k, v in metrics.items()
    ]
