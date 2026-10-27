"""Shared helpers for trade-off figure generation."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as cfg

# Reuse project Times New Roman helper when available
_AGREED = Path(__file__).resolve().parents[1]
if str(_AGREED) not in sys.path:
    sys.path.insert(0, str(_AGREED))
try:
    from latex_fonts import apply_font_to_figure, configure_times_font  # type: ignore
except ImportError:  # pragma: no cover
    apply_font_to_figure = None  # type: ignore
    configure_times_font = None  # type: ignore


def apply_publication_style() -> None:
    """Configure matplotlib/seaborn: Times New Roman, no grid lines."""
    # Seaborn first (it resets rcParams), then lock Times + no-grid.
    sns.set_theme(style="ticks", context="paper")

    if configure_times_font is not None:
        configure_times_font()
    else:
        available = {f.name for f in mpl.font_manager.fontManager.ttflist}
        chosen = (
            "Times New Roman"
            if "Times New Roman" in available
            else cfg.FONT_FAMILY
        )
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [chosen, "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            }
        )

    mpl.rcParams.update(
        {
            "figure.facecolor": cfg.FACE_COLOR,
            "axes.facecolor": cfg.FACE_COLOR,
            "savefig.facecolor": cfg.FACE_COLOR,
            "font.size": cfg.FONT_SIZE,
            "axes.titlesize": cfg.TITLE_SIZE,
            "axes.labelsize": cfg.LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": cfg.TICK_SIZE,
            "ytick.labelsize": cfg.TICK_SIZE,
            "legend.fontsize": cfg.LEGEND_SIZE,
            "axes.linewidth": 1.1,
            "axes.edgecolor": cfg.SPINE_COLOR,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.alpha": 0.0,
            "grid.linewidth": 0.0,
            "xtick.bottom": True,
            "ytick.left": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )


def finalize_figure_fonts(fig: plt.Figure) -> None:
    """Force Times New Roman on every text artist."""
    if apply_font_to_figure is not None:
        apply_font_to_figure(fig)


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical names → actual CSV column names present in ``df``."""
    lookup = {_normalize_key(c): c for c in df.columns}
    resolved: Dict[str, str] = {}
    for canonical, aliases in cfg.COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_key(alias)
            if key in lookup:
                resolved[canonical] = lookup[key]
                break
    return resolved


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and coerce numeric metrics."""
    mapping = resolve_columns(df)
    rename = {src: dst for dst, src in mapping.items()}
    out = df.rename(columns=rename).copy()

    for col in (cfg.COL_QUALITY, cfg.COL_ACCURACY, cfg.COL_R2, cfg.COL_MIA):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if cfg.COL_GENERATOR in out.columns:
        out[cfg.COL_GENERATOR] = out[cfg.COL_GENERATOR].astype(str).str.strip()
        # Unify WGAN naming
        out[cfg.COL_GENERATOR] = out[cfg.COL_GENERATOR].replace({"WGAN-GP": "WGAN_GP"})

    if cfg.COL_DATASET in out.columns:
        out[cfg.COL_DATASET] = out[cfg.COL_DATASET].astype(str).str.strip()

    if cfg.COL_TASK in out.columns:
        out[cfg.COL_TASK] = (
            out[cfg.COL_TASK]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "clf": cfg.TASK_CLASSIFICATION,
                    "class": cfg.TASK_CLASSIFICATION,
                    "reg": cfg.TASK_REGRESSION,
                }
            )
        )
    return out


def infer_task_type(df: pd.DataFrame, task_type: Optional[str] = None) -> str:
    """Infer ``classification`` vs ``regression`` from columns / Task values."""
    if task_type is not None:
        t = task_type.strip().lower()
        if t not in (cfg.TASK_CLASSIFICATION, cfg.TASK_REGRESSION):
            raise ValueError(f"Unknown task_type: {task_type!r}")
        return t

    if cfg.COL_TASK in df.columns:
        values = set(df[cfg.COL_TASK].dropna().astype(str).str.lower().unique())
        values -= {""}
        if values == {cfg.TASK_CLASSIFICATION}:
            return cfg.TASK_CLASSIFICATION
        if values == {cfg.TASK_REGRESSION}:
            return cfg.TASK_REGRESSION
        if values == {cfg.TASK_CLASSIFICATION, cfg.TASK_REGRESSION}:
            raise ValueError(
                "Input mixes classification and regression rows; "
                "pass task_type= or split the CSV first."
            )

    has_acc = cfg.COL_ACCURACY in df.columns and df[cfg.COL_ACCURACY].notna().any()
    has_r2 = cfg.COL_R2 in df.columns and df[cfg.COL_R2].notna().any()
    if has_acc and not has_r2:
        return cfg.TASK_CLASSIFICATION
    if has_r2 and not has_acc:
        return cfg.TASK_REGRESSION
    if has_acc and has_r2:
        # Prefer the denser utility column
        n_acc = int(df[cfg.COL_ACCURACY].notna().sum())
        n_r2 = int(df[cfg.COL_R2].notna().sum())
        return cfg.TASK_CLASSIFICATION if n_acc >= n_r2 else cfg.TASK_REGRESSION
    raise ValueError(
        "Cannot infer task type: need Accuracy (classification) or R2 (regression)."
    )


def load_tradeoff_csv(
    path: str | Path,
    *,
    task_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load, normalize, and validate a trade-off CSV.

    Returns
    -------
    df, task
        Normalized frame restricted to the inferred/requested task, and the
        task string.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    df = normalize_dataframe(raw)
    task = infer_task_type(df, task_type=task_type)

    required = [cfg.COL_DATASET, cfg.COL_GENERATOR, cfg.COL_QUALITY, cfg.COL_MIA]
    util_col = cfg.UTILITY_COLUMN[task]
    required.append(util_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    if cfg.COL_TASK in df.columns:
        df = df[df[cfg.COL_TASK].isna() | (df[cfg.COL_TASK] == task)].copy()
    else:
        df[cfg.COL_TASK] = task

    df = df.dropna(subset=[cfg.COL_DATASET, cfg.COL_GENERATOR, cfg.COL_QUALITY, util_col, cfg.COL_MIA])
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows left for task={task!r} in {path}")
    return df, task


def display_name(generator: str) -> str:
    """Pretty generator name for legends / annotations."""
    return cfg.GENERATOR_DISPLAY.get(generator, generator)


def generator_color_map(generators: Iterable[str]) -> Dict[str, str]:
    """Stable colorblind-friendly colors for the given generators."""
    gens = list(dict.fromkeys(generators))
    colors: Dict[str, str] = {}
    fallback_i = 0
    for g in gens:
        if g in cfg.GENERATOR_COLORS:
            colors[g] = cfg.GENERATOR_COLORS[g]
        else:
            colors[g] = cfg.FALLBACK_COLORS[fallback_i % len(cfg.FALLBACK_COLORS)]
            fallback_i += 1
    return colors


def generator_marker_map(generators: Iterable[str]) -> Dict[str, str]:
    """Stable marker styles for the given generators."""
    gens = list(dict.fromkeys(generators))
    markers: Dict[str, str] = {}
    fallback_i = 0
    for g in gens:
        if g in cfg.GENERATOR_MARKERS:
            markers[g] = cfg.GENERATOR_MARKERS[g]
        else:
            markers[g] = cfg.FALLBACK_MARKERS[fallback_i % len(cfg.FALLBACK_MARKERS)]
            fallback_i += 1
    return markers


def ordered_generators(generators: Iterable[str]) -> List[str]:
    """Return generators in preferred legend order, unknowns appended."""
    present = list(dict.fromkeys(generators))
    ordered = [g for g in cfg.GENERATOR_ORDER if g in present]
    ordered.extend([g for g in present if g not in ordered])
    return ordered


def axis_limits(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    pad_frac: float = cfg.AXIS_PAD_FRAC,
    hard_min: Optional[float] = None,
    hard_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute padded axis limits from finite values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(lo, hi):
        delta = max(abs(lo) * 0.05, 0.05)
        lo, hi = lo - delta, hi + delta
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    if hard_min is not None:
        lo = max(lo, hard_min)
    if hard_max is not None:
        hi = min(hi, hard_max)
    return lo, hi


def shared_xy_limits(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Global axis limits shared across all dataset subplots."""
    return axis_limits(df[x_col]), axis_limits(df[y_col])


def bubble_sizes(
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    size_range: Tuple[float, float] = cfg.BUBBLE_SIZE_RANGE,
) -> np.ndarray:
    """Map metric values to scatter ``s`` sizes (points²)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, size_range[0], dtype=float)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if math.isclose(lo, hi):
        return np.full_like(arr, 0.5 * (size_range[0] + size_range[1]), dtype=float)
    norm = (arr - lo) / (hi - lo)
    return size_range[0] + norm * (size_range[1] - size_range[0])


def subplot_grid(n: int, *, max_cols: int = cfg.MAX_SUBPLOT_COLS) -> Tuple[int, int]:
    """Return (nrows, ncols) for ``n`` subplots."""
    if n <= 0:
        raise ValueError("n must be positive")
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def dataset_list(df: pd.DataFrame) -> List[str]:
    """Unique datasets in first-seen order."""
    return list(dict.fromkeys(df[cfg.COL_DATASET].tolist()))


def make_generator_legend_handles(
    generators: Sequence[str],
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    *,
    size: float = cfg.MARKER_SIZE,
    alpha: float = 1.0,
) -> List[mpl.lines.Line2D]:
    """Scatter-style legend handles for generators (large, edged markers)."""
    handles: List[mpl.lines.Line2D] = []
    for g in generators:
        # Stars read smaller; enlarge slightly in the legend
        ms = math.sqrt(size) * (1.05 if markers[g] == "*" else 0.95)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers[g],
                color="none",
                markerfacecolor=colors[g],
                markeredgecolor=cfg.MARKER_EDGE_COLOR,
                markeredgewidth=1.2,
                markersize=ms,
                linestyle="None",
                alpha=alpha,
                label=display_name(g),
            )
        )
    return handles


def shade_ideal_region(
    ax: plt.Axes,
    *,
    corner: str = "bottom-right",
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    frac: float = 0.28,
) -> None:
    """Lightly shade the ideal corner of a 2-D trade-off plot."""
    x0, x1 = x_lim
    y0, y1 = y_lim
    dx = (x1 - x0) * frac
    dy = (y1 - y0) * frac
    if corner == "bottom-right":
        xs = [x1 - dx, x1, x1, x1 - dx]
        ys = [y0, y0, y0 + dy, y0 + dy]
    elif corner == "top-right":
        xs = [x1 - dx, x1, x1, x1 - dx]
        ys = [y1 - dy, y1 - dy, y1, y1]
    else:
        raise ValueError(f"Unsupported ideal corner: {corner}")
    ax.fill(xs, ys, color=cfg.IDEAL_REGION_COLOR, alpha=cfg.IDEAL_REGION_ALPHA, zorder=0)
    ax.text(
        np.mean(xs),
        np.mean(ys),
        "Ideal",
        ha="center",
        va="center",
        fontsize=cfg.TICK_SIZE - 1,
        color=cfg.IDEAL_REGION_COLOR,
        alpha=0.85,
        fontstyle="italic",
        zorder=1,
    )


def annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    enabled: bool = cfg.ANNOTATE_GENERATORS,
) -> None:
    """Optionally label each generator point."""
    if not enabled:
        return
    for _, row in df.iterrows():
        ax.annotate(
            display_name(str(row[cfg.COL_GENERATOR])),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7.5,
            alpha=0.9,
        )


def save_figure(fig: plt.Figure, stem: Path, *, formats: Sequence[str] = cfg.SAVE_FORMATS) -> List[Path]:
    """Save ``fig`` as pdf/svg/png next to ``stem`` (no extension)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight", facecolor=cfg.FACE_COLOR)
        written.append(out)
    return written


def style_axes(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    """Apply shared axis cosmetics (no interior grid lines)."""
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    # Remove any seaborn leftover grid artists
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(cfg.SPINE_COLOR)
        spine.set_linewidth(1.1)
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
