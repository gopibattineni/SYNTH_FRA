"""Publication-quality trade-off visualizations for synthetic-data evaluation.

Generate five figures for classification (Accuracy) or regression (R²) tasks
from a single per-generator metrics CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config as cfg
from .pareto import annotate_pareto, compute_pareto_frontier
from .utils import (
    annotate_points,
    apply_publication_style,
    bubble_sizes,
    dataset_list,
    display_name,
    finalize_figure_fonts,
    generator_color_map,
    generator_marker_map,
    load_tradeoff_csv,
    make_generator_legend_handles,
    ordered_generators,
    save_figure,
    shared_xy_limits,
    style_axes,
    subplot_grid,
)


def _prepare_figure(
    n_datasets: int,
    *,
    figsize_scale: Tuple[float, float] = (4.2, 3.6),
) -> Tuple[plt.Figure, np.ndarray, int, int]:
    nrows, ncols = subplot_grid(n_datasets)
    fig_w = figsize_scale[0] * ncols
    fig_h = figsize_scale[1] * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )
    return fig, axes, nrows, ncols


def _hide_extra_axes(axes: np.ndarray, n_used: int) -> None:
    flat = axes.ravel()
    for ax in flat[n_used:]:
        ax.set_visible(False)


def _add_shared_legend(
    fig: plt.Figure,
    handles: Sequence,
    *,
    title: str = "Generator",
    ncol: Optional[int] = None,
) -> None:
    if not handles:
        return
    ncol = ncol or min(4, len(handles))
    fig.legend(
        handles=handles,
        title=title,
        loc="outside lower center",
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=cfg.LEGEND_SIZE,
        title_fontsize=cfg.LEGEND_SIZE,
    )


def _scatter_generators(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    colors: Dict[str, str],
    markers: Dict[str, str],
    *,
    alpha: float = 0.98,
    size: float = cfg.MARKER_SIZE,
    edge: str = cfg.MARKER_EDGE_COLOR,
    lw: float = cfg.MARKER_EDGE_WIDTH,
    faded: Optional[pd.Series] = None,
) -> None:
    for _, row in sub.iterrows():
        g = str(row[cfg.COL_GENERATOR])
        a = alpha
        # Stars render smaller; bump area for visibility
        s = size * (1.35 if markers[g] == "*" else 1.0)
        ec = edge
        elw = lw
        if faded is not None and bool(faded.loc[row.name]):
            a = cfg.FADED_ALPHA
            ec = "#555555"
            elw = max(0.9, lw * 0.85)
        ax.scatter(
            row[x_col],
            row[y_col],
            s=s,
            marker=markers[g],
            c=colors[g],
            alpha=a,
            edgecolors=ec,
            linewidths=elw,
            zorder=3,
        )


def _finish_and_save(fig: plt.Figure, stem: Path) -> List[Path]:
    finalize_figure_fonts(fig)
    paths = save_figure(fig, stem)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Figure 1 — Fidelity vs Utility
# ---------------------------------------------------------------------------
def plot_fig1_fidelity_vs_utility(
    df: pd.DataFrame,
    task_type: str,
    out_dir: Path,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> List[Path]:
    """Fidelity (QualityScore) vs utility (Accuracy / R²), one panel per dataset."""
    util_col = cfg.UTILITY_COLUMN[task_type]
    util_label = cfg.UTILITY_LABEL[task_type]
    datasets = dataset_list(df)
    gens = ordered_generators(df[cfg.COL_GENERATOR])
    colors = generator_color_map(gens)
    markers = generator_marker_map(gens)
    xlim, ylim = shared_xy_limits(df, cfg.COL_QUALITY, util_col)

    fig, axes, _, _ = _prepare_figure(len(datasets))
    for i, name in enumerate(datasets):
        ax = axes.ravel()[i]
        sub = df[df[cfg.COL_DATASET] == name]
        _scatter_generators(ax, sub, cfg.COL_QUALITY, util_col, colors, markers)
        annotate_points(ax, sub, cfg.COL_QUALITY, util_col, enabled=annotate)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axes(ax, title=name, xlabel=cfg.QUALITY_LABEL, ylabel=util_label)

    _hide_extra_axes(axes, len(datasets))
    _add_shared_legend(fig, make_generator_legend_handles(gens, colors, markers))
    stem = out_dir / cfg.FIG_FILENAMES["fig1"]
    return _finish_and_save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 2 — Utility vs Privacy
# ---------------------------------------------------------------------------
def plot_fig2_utility_vs_privacy(
    df: pd.DataFrame,
    task_type: str,
    out_dir: Path,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> List[Path]:
    """Utility vs MIA (AUC); ideal region is bottom-right (high utility, low risk)."""
    util_col = cfg.UTILITY_COLUMN[task_type]
    util_label = cfg.UTILITY_LABEL[task_type]
    datasets = dataset_list(df)
    gens = ordered_generators(df[cfg.COL_GENERATOR])
    colors = generator_color_map(gens)
    markers = generator_marker_map(gens)
    xlim, ylim = shared_xy_limits(df, util_col, cfg.COL_MIA)

    fig, axes, _, _ = _prepare_figure(len(datasets))
    for i, name in enumerate(datasets):
        ax = axes.ravel()[i]
        sub = df[df[cfg.COL_DATASET] == name]
        _scatter_generators(ax, sub, util_col, cfg.COL_MIA, colors, markers)
        annotate_points(ax, sub, util_col, cfg.COL_MIA, enabled=annotate)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axes(ax, title=name, xlabel=util_label, ylabel=cfg.MIA_LABEL)

    _hide_extra_axes(axes, len(datasets))
    _add_shared_legend(fig, make_generator_legend_handles(gens, colors, markers))
    stem = out_dir / cfg.FIG_FILENAMES["fig2"]
    return _finish_and_save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 3 — Fidelity vs Privacy
# ---------------------------------------------------------------------------
def plot_fig3_fidelity_vs_privacy(
    df: pd.DataFrame,
    task_type: str,
    out_dir: Path,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> List[Path]:
    """Fidelity vs MIA (AUC); ideal region is bottom-right (high fidelity, low risk)."""
    datasets = dataset_list(df)
    gens = ordered_generators(df[cfg.COL_GENERATOR])
    colors = generator_color_map(gens)
    markers = generator_marker_map(gens)
    xlim, ylim = shared_xy_limits(df, cfg.COL_QUALITY, cfg.COL_MIA)

    fig, axes, _, _ = _prepare_figure(len(datasets))
    for i, name in enumerate(datasets):
        ax = axes.ravel()[i]
        sub = df[df[cfg.COL_DATASET] == name]
        _scatter_generators(ax, sub, cfg.COL_QUALITY, cfg.COL_MIA, colors, markers)
        annotate_points(ax, sub, cfg.COL_QUALITY, cfg.COL_MIA, enabled=annotate)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axes(ax, title=name, xlabel=cfg.QUALITY_LABEL, ylabel=cfg.MIA_LABEL)

    _hide_extra_axes(axes, len(datasets))
    _add_shared_legend(fig, make_generator_legend_handles(gens, colors, markers))
    stem = out_dir / cfg.FIG_FILENAMES["fig3"]
    return _finish_and_save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 4 — Bubble trade-off
# ---------------------------------------------------------------------------
def plot_fig4_bubble_tradeoff(
    df: pd.DataFrame,
    task_type: str,
    out_dir: Path,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> List[Path]:
    """Three-way bubble plot: fidelity × utility, size = MIA (AUC)."""
    util_col = cfg.UTILITY_COLUMN[task_type]
    util_label = cfg.UTILITY_LABEL[task_type]
    datasets = dataset_list(df)
    gens = ordered_generators(df[cfg.COL_GENERATOR])
    colors = generator_color_map(gens)
    markers = generator_marker_map(gens)
    xlim, ylim = shared_xy_limits(df, cfg.COL_QUALITY, util_col)
    # Global bubble scaling so sizes are comparable across panels
    global_sizes = bubble_sizes(df[cfg.COL_MIA])

    fig, axes, _, _ = _prepare_figure(len(datasets), figsize_scale=(4.4, 3.8))
    for i, name in enumerate(datasets):
        ax = axes.ravel()[i]
        sub = df[df[cfg.COL_DATASET] == name].copy()
        sizes = global_sizes[sub.index.to_numpy()]
        for (idx, row), s in zip(sub.iterrows(), sizes):
            g = str(row[cfg.COL_GENERATOR])
            ax.scatter(
                row[cfg.COL_QUALITY],
                row[util_col],
                s=s * (1.25 if markers[g] == "*" else 1.0),
                marker=markers[g],
                c=colors[g],
                alpha=cfg.BUBBLE_ALPHA,
                edgecolors=cfg.MARKER_EDGE_COLOR,
                linewidths=cfg.MARKER_EDGE_WIDTH,
                zorder=3,
            )
        annotate_points(ax, sub, cfg.COL_QUALITY, util_col, enabled=annotate)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axes(ax, title=name, xlabel=cfg.QUALITY_LABEL, ylabel=util_label)

    _hide_extra_axes(axes, len(datasets))

    # Generator legend
    gen_handles = make_generator_legend_handles(gens, colors, markers)
    # Bubble-size legend (MIA)
    mia_vals = df[cfg.COL_MIA].to_numpy(dtype=float)
    mia_lo, mia_hi = float(np.nanmin(mia_vals)), float(np.nanmax(mia_vals))
    mia_mid = 0.5 * (mia_lo + mia_hi)
    size_examples = bubble_sizes(np.array([mia_lo, mia_mid, mia_hi]))
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#555555",
            markersize=max(4.0, math_sqrt_size(s)),
            linestyle="None",
            label=f"MIA (AUC) = {v:.2f}",
        )
        for v, s in zip((mia_lo, mia_mid, mia_hi), size_examples)
    ]

    legend1 = fig.legend(
        handles=gen_handles,
        title="Generator",
        loc="outside lower left",
        ncol=min(4, len(gen_handles)),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=cfg.LEGEND_SIZE,
        title_fontsize=cfg.LEGEND_SIZE,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=size_handles,
        title="Bubble size (MIA AUC)",
        loc="outside lower right",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=cfg.LEGEND_SIZE,
        title_fontsize=cfg.LEGEND_SIZE,
    )

    stem = out_dir / cfg.FIG_FILENAMES["fig4"]
    return _finish_and_save(fig, stem)


def math_sqrt_size(s: float) -> float:
    """Convert scatter area ``s`` to legend markersize (points)."""
    return float(np.sqrt(max(s, 1.0)) * 0.55)


# ---------------------------------------------------------------------------
# Figure 5 — Pareto frontier
# ---------------------------------------------------------------------------
def plot_fig5_pareto(
    df: pd.DataFrame,
    task_type: str,
    out_dir: Path,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> List[Path]:
    """Pareto frontier over fidelity × utility (both maximized)."""
    util_col = cfg.UTILITY_COLUMN[task_type]
    util_label = cfg.UTILITY_LABEL[task_type]
    datasets = dataset_list(df)
    gens = ordered_generators(df[cfg.COL_GENERATOR])
    colors = generator_color_map(gens)
    markers = generator_marker_map(gens)
    xlim, ylim = shared_xy_limits(df, cfg.COL_QUALITY, util_col)

    fig, axes, _, _ = _prepare_figure(len(datasets))
    for i, name in enumerate(datasets):
        ax = axes.ravel()[i]
        sub = df[df[cfg.COL_DATASET] == name].copy()
        tagged = annotate_pareto(sub, cfg.COL_QUALITY, util_col)
        faded = ~tagged["Pareto"]
        # Non-Pareto faded
        _scatter_generators(
            ax,
            tagged,
            cfg.COL_QUALITY,
            util_col,
            colors,
            markers,
            size=cfg.MARKER_SIZE,
            faded=faded,
        )
        # Re-draw Pareto points highlighted
        frontier = tagged[tagged["Pareto"]]
        for _, row in frontier.iterrows():
            g = str(row[cfg.COL_GENERATOR])
            psize = cfg.PARETO_MARKER_SIZE * (1.35 if markers[g] == "*" else 1.0)
            ax.scatter(
                row[cfg.COL_QUALITY],
                row[util_col],
                s=psize,
                marker=markers[g],
                c=colors[g],
                alpha=1.0,
                edgecolors="black",
                linewidths=cfg.PARETO_EDGE_WIDTH,
                zorder=5,
            )
        # Dashed frontier polyline (sorted by fidelity)
        frontier_line = compute_pareto_frontier(sub, cfg.COL_QUALITY, util_col)
        if len(frontier_line) >= 2:
            ax.plot(
                frontier_line[cfg.COL_QUALITY],
                frontier_line[util_col],
                linestyle="--",
                color="#222222",
                linewidth=cfg.LINE_WIDTH,
                zorder=4,
                label="Pareto frontier",
            )
        annotate_points(ax, frontier, cfg.COL_QUALITY, util_col, enabled=annotate)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_axes(ax, title=name, xlabel=cfg.QUALITY_LABEL, ylabel=util_label)

    _hide_extra_axes(axes, len(datasets))
    handles = make_generator_legend_handles(gens, colors, markers)
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="#222222",
            linewidth=cfg.LINE_WIDTH,
            label="Pareto frontier",
        )
    )
    _add_shared_legend(fig, handles, title="Legend", ncol=min(5, len(handles)))
    stem = out_dir / cfg.FIG_FILENAMES["fig5"]
    return _finish_and_save(fig, stem)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_tradeoff_figures(
    input_csv: str | Path,
    output_dir: str | Path | None = None,
    task_type: Optional[str] = None,
    *,
    annotate: bool = cfg.ANNOTATE_GENERATORS,
) -> Dict[str, List[Path]]:
    """Generate all five trade-off figures for one task.

    Parameters
    ----------
    input_csv :
        Path to a CSV with one row per generator (and typically multiple
        datasets). Column names are resolved via ``config.COLUMN_ALIASES``.
    output_dir :
        Root directory for figures. Defaults to ``tradeoff/figures``.
        Figures are written under ``<output_dir>/<task>/``.
    task_type :
        ``"classification"``, ``"regression"``, or ``None`` to auto-detect.
    annotate :
        If True, label generator names next to markers.

    Returns
    -------
    dict
        Mapping of figure key (``fig1``…``fig5``) to saved file paths.
    """
    apply_publication_style()
    df, task = load_tradeoff_csv(input_csv, task_type=task_type)

    root = Path(output_dir) if output_dir is not None else cfg.FIGURES_DIR
    fig_dir = root / task
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Also mirror a normalized copy under output/<task>/
    data_out = cfg.output_dir(task)
    data_out.mkdir(parents=True, exist_ok=True)
    normalized_path = data_out / "normalized_metrics.csv"
    df.to_csv(normalized_path, index=False)

    results: Dict[str, List[Path]] = {
        "fig1": plot_fig1_fidelity_vs_utility(df, task, fig_dir, annotate=annotate),
        "fig2": plot_fig2_utility_vs_privacy(df, task, fig_dir, annotate=annotate),
        "fig3": plot_fig3_fidelity_vs_privacy(df, task, fig_dir, annotate=annotate),
        "fig4": plot_fig4_bubble_tradeoff(df, task, fig_dir, annotate=annotate),
        "fig5": plot_fig5_pareto(df, task, fig_dir, annotate=annotate),
    }
    return results


def build_combined_csv_from_individual(
    individual_root: str | Path,
    out_csv: str | Path,
    *,
    task_type: Optional[str] = None,
) -> Path:
    """Assemble a combined API CSV from per-dataset ``generator_metrics_raw.csv``.

    This bridges the existing ``trade_off/Individual dataset/*/`` layout to the
    standalone tradeoff module input schema.
    """
    individual_root = Path(individual_root)
    rows: List[dict] = []
    for folder in sorted(individual_root.iterdir()):
        if not folder.is_dir():
            continue
        raw_path = folder / "generator_metrics_raw.csv"
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        # Detect task from columns
        is_clf = "Mean_TSTR_Accuracy" in raw.columns or "Accuracy" in raw.columns
        is_reg = "Mean_TSTR_R2" in raw.columns or "R2" in raw.columns
        if task_type == cfg.TASK_CLASSIFICATION and not is_clf:
            continue
        if task_type == cfg.TASK_REGRESSION and not is_reg:
            continue
        if task_type is None:
            local_task = cfg.TASK_CLASSIFICATION if is_clf and not is_reg else (
                cfg.TASK_REGRESSION if is_reg and not is_clf else None
            )
            if local_task is None:
                continue
        else:
            local_task = task_type

        for _, r in raw.iterrows():
            gen = r.get("Generator", r.get("Display", ""))
            quality = r.get("Fidelity_SDMetrics", r.get("QualityScore", np.nan))
            mia = r.get("MIA", np.nan)
            row = {
                cfg.COL_DATASET: folder.name,
                cfg.COL_GENERATOR: gen,
                cfg.COL_TASK: local_task,
                cfg.COL_QUALITY: quality,
                cfg.COL_MIA: mia,
            }
            if local_task == cfg.TASK_CLASSIFICATION:
                row[cfg.COL_ACCURACY] = r.get("Mean_TSTR_Accuracy", r.get("Accuracy", np.nan))
            else:
                row[cfg.COL_R2] = r.get("Mean_TSTR_R2", r.get("R2", np.nan))
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No generator_metrics_raw.csv found under {individual_root}")

    out = pd.DataFrame(rows)
    # Friendly dataset titles
    title_map = {
        "Alzheimer's": "Alzheimer's",
        "CDCDiabetes": "CDC Diabetes",
        "ForestCover": "Forest Cover",
        "BankMarketing": "Bank Marketing",
        "WineQuality": "Wine Quality",
        "MAGICGamma": "MAGIC Gamma",
        "MetroInterstate": "Metro Interstate",
        "OnlineShopping": "Online Shopping",
        "AirQuality": "Air Quality",
        "EnergyEfficiency": "Energy Efficiency",
        "RealEstate": "Real Estate",
    }
    out[cfg.COL_DATASET] = out[cfg.COL_DATASET].replace(title_map)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality synthetic-data trade-off figures."
    )
    parser.add_argument("input_csv", type=Path, help="Per-generator metrics CSV")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Figure root (default: tradeoff/figures)",
    )
    parser.add_argument(
        "-t",
        "--task-type",
        choices=[cfg.TASK_CLASSIFICATION, cfg.TASK_REGRESSION],
        default=None,
        help="Force task type (default: auto-detect)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate generator names on markers",
    )
    args = parser.parse_args(argv)
    results = generate_tradeoff_figures(
        args.input_csv,
        output_dir=args.output_dir,
        task_type=args.task_type,
        annotate=args.annotate,
    )
    for key, paths in results.items():
        print(f"{key}:")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
