"""Pareto-frontier helpers for multi-objective generator ranking.

Maximize fidelity (QualityScore) and utility (Accuracy / R²). Points that are
weakly dominated by another generator are non-Pareto.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def is_dominated(
    point: Sequence[float],
    others: Iterable[Sequence[float]],
    *,
    maximize: Sequence[bool] | None = None,
) -> bool:
    """Return True if ``point`` is dominated by any vector in ``others``.

    Parameters
    ----------
    point :
        Objective vector.
    others :
        Candidate dominating vectors (typically all points including ``point``;
        self-comparison is skipped).
    maximize :
        Per-objective flag. ``True`` means larger is better. Defaults to all
        maximize.
    """
    p = np.asarray(point, dtype=float)
    if maximize is None:
        maximize = [True] * len(p)
    max_flags = np.asarray(maximize, dtype=bool)
    if p.shape[0] != max_flags.shape[0]:
        raise ValueError("point and maximize must have the same length")

    for other in others:
        o = np.asarray(other, dtype=float)
        if o.shape != p.shape:
            raise ValueError("all objective vectors must share the same shape")
        if np.allclose(o, p, equal_nan=False):
            continue

        # Flip minimized objectives so "larger is better" uniformly.
        signed_p = np.where(max_flags, p, -p)
        signed_o = np.where(max_flags, o, -o)
        if np.all(signed_o >= signed_p - 1e-12) and np.any(signed_o > signed_p + 1e-12):
            return True
    return False


def pareto_mask(
    objectives: np.ndarray,
    *,
    maximize: Sequence[bool] | None = None,
) -> np.ndarray:
    """Boolean mask of non-dominated rows in ``objectives`` (shape n × k)."""
    arr = np.asarray(objectives, dtype=float)
    if arr.ndim != 2:
        raise ValueError("objectives must be a 2-D array")
    n = arr.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if np.any(~np.isfinite(arr[i])):
            mask[i] = False
            continue
        mask[i] = not is_dominated(arr[i], arr, maximize=maximize)
    return mask


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """Return Pareto-optimal rows, sorted along ``x_col`` for plotting.

    Both objectives are maximized by default (higher fidelity and higher
    utility are better).
    """
    work = df.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return work

    objs = work[[x_col, y_col]].to_numpy(dtype=float)
    mask = pareto_mask(objs, maximize=[maximize_x, maximize_y])
    frontier = work.loc[mask].copy()
    # Sort for a clean dashed frontier polyline
    frontier = frontier.sort_values(by=[x_col, y_col], ascending=[True, True])
    return frontier.reset_index(drop=True)


def annotate_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
    flag_col: str = "Pareto",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a boolean ``flag_col`` column."""
    out = df.copy()
    out[flag_col] = False
    valid = out.dropna(subset=[x_col, y_col])
    if valid.empty:
        return out
    objs = valid[[x_col, y_col]].to_numpy(dtype=float)
    mask = pareto_mask(objs, maximize=[maximize_x, maximize_y])
    out.loc[valid.index[mask], flag_col] = True
    return out
