"""Trade-off visualization package for synthetic-data evaluation.

Typical usage::

    from tradeoff.tradeoff_plots import generate_tradeoff_figures

    generate_tradeoff_figures("metrics.csv", output_dir="tradeoff/figures")
"""

from .pareto import annotate_pareto, compute_pareto_frontier, pareto_mask
from .tradeoff_plots import (
    build_combined_csv_from_individual,
    generate_tradeoff_figures,
    plot_fig1_fidelity_vs_utility,
    plot_fig2_utility_vs_privacy,
    plot_fig3_fidelity_vs_privacy,
    plot_fig4_bubble_tradeoff,
    plot_fig5_pareto,
)
from .utils import (
    axis_limits,
    bubble_sizes,
    generator_color_map,
    generator_marker_map,
    load_tradeoff_csv,
    save_figure,
)

__all__ = [
    "annotate_pareto",
    "axis_limits",
    "bubble_sizes",
    "build_combined_csv_from_individual",
    "compute_pareto_frontier",
    "generate_tradeoff_figures",
    "generator_color_map",
    "generator_marker_map",
    "load_tradeoff_csv",
    "pareto_mask",
    "plot_fig1_fidelity_vs_utility",
    "plot_fig2_utility_vs_privacy",
    "plot_fig3_fidelity_vs_privacy",
    "plot_fig4_bubble_tradeoff",
    "plot_fig5_pareto",
    "save_figure",
]
