"""Fidelity metrics aligned with existing SYNTH nomenclature (no new metrics)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import LabelEncoder


def _align_frames(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in real.columns if c in synth.columns]
    return real[cols].copy(), synth[cols].copy()


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def compute_fidelity(train_real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    """Return dict of fidelity metric_name -> value (existing metric names)."""
    real, syn = _align_frames(train_real, synth)
    out: dict[str, float] = {}

    # SDV Quality Score (when available)
    try:
        from sdv.evaluation import evaluate_quality
        from sdv.metadata import Metadata

        md = Metadata.detect_from_dataframe(data=real, table_name="table", verbose=False)
        quality = evaluate_quality(
            real_data=real, synthetic_data=syn, metadata=md, verbose=False
        )
        out["Quality_Score"] = float(quality.get_score())
    except Exception:
        out["Quality_Score"] = float("nan")

    num_cols = _numeric_cols(real)
    ks_vals = []
    wass_vals = []
    for c in num_cols:
        a = pd.to_numeric(real[c], errors="coerce").dropna()
        b = pd.to_numeric(syn[c], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        # KS complement = 1 - KS statistic (higher better; used in SDV Column Shapes)
        stat = ks_2samp(a, b).statistic
        ks_vals.append(1.0 - float(stat))
        try:
            wass_vals.append(float(wasserstein_distance(a, b)))
        except Exception:
            pass
    out["KS_Complement"] = float(np.mean(ks_vals)) if ks_vals else float("nan")
    out["Wasserstein_Distance"] = float(np.mean(wass_vals)) if wass_vals else float("nan")

    # Simple multivariate MMD proxy on standardized numeric features (RBF not required —
    # use mean absolute difference of column-wise standardized means as lightweight proxy
    # ONLY if we also compute a proper MMD; prefer sklearn pairwise if small)
    try:
        from sklearn.metrics.pairwise import rbf_kernel

        X = real[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        Y = syn[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        # subsample for cost
        rng = np.random.default_rng(0)
        n = min(200, len(X), len(Y))
        Xi = X[rng.choice(len(X), n, replace=False)]
        Yi = Y[rng.choice(len(Y), n, replace=False)]
        # standardize
        mu, sd = Xi.mean(0), Xi.std(0) + 1e-8
        Xi = (Xi - mu) / sd
        Yi = (Yi - mu) / sd
        Kxx = rbf_kernel(Xi, Xi)
        Kyy = rbf_kernel(Yi, Yi)
        Kxy = rbf_kernel(Xi, Yi)
        mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        out["MMD"] = float(max(mmd2, 0.0))
    except Exception:
        out["MMD"] = float("nan")

    return out


def fidelity_to_rows(metrics: dict[str, float], meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, val in metrics.items():
        rows.append(
            {
                **meta,
                "metric_category": "Fidelity",
                "metric_name": name,
                "metric_value": val,
            }
        )
    return rows
