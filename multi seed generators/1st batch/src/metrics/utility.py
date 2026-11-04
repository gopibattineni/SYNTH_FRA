"""Batch shim — delegates to shared multi-seed utility (10 classifiers + 10 regressors)."""
from __future__ import annotations

import sys
from pathlib import Path

_MULTI_ROOT = Path(__file__).resolve().parents[3]
if str(_MULTI_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTI_ROOT))

from shared.utility import (  # noqa: E402,F401
    CLASSIFIERS,
    REGRESSORS,
    clear_trtr_cache,
    compute_utility,
    evaluate_classification,
    evaluate_regression,
    expected_tstr_fit_counts,
    get_classifiers,
    get_regressors,
    load_utility_model_names,
    per_model_to_long,
    utility_to_rows,
    _snap_classification_target,
)
