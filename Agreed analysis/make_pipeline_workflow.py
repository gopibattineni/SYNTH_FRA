"""Publication-quality diagram of the TSTR experimental pipeline.

Renders the full evaluation protocol (real data -> split -> synthetic
generation -> downstream classifier -> TSTR evaluation), annotated with
what changes across the 10 random seeds and what stays fixed (the
real train/test split, to guarantee no information leakage).

Outputs:
    Conor/experimental_pipeline_workflow.png  (300 dpi, publication-ready)
    Conor/experimental_pipeline_workflow.svg  (vector, editable)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from latex_fonts import apply_font_to_figure, configure_times_font
configure_times_font()

# ---------------------------------------------------------------- palette --
NAVY = "#1f3a5f"
BLUE = "#2f6fb5"
BLUE_LIGHT = "#eaf1fb"
GREEN = "#2f7d5b"
GREEN_LIGHT = "#e9f5ef"
AMBER = "#b5762f"
AMBER_LIGHT = "#fbf2e6"
RED = "#b3402f"
RED_LIGHT = "#fbeceb"
GRAY = "#5a5f66"
GRAY_LIGHT = "#f4f5f6"
INK = "#1c1f24"
WHITE = "#ffffff"


def _box(ax, xy, w, h, text, *, fc, ec, tc=INK, fontsize=11.5, weight="bold",
         boxstyle="round,pad=0.06,rounding_size=0.08", lw=1.5, zorder=3):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    ))
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fontsize,
        fontweight=weight, color=tc, zorder=zorder + 1, linespacing=1.35,
    )


def _stack_box(ax, xy, w, h, text, *, fc, ec, n=3, offset=0.075, **kw):
    """'Deck of cards' stack (offset up-right) to convey multiplicity."""
    x, y = xy
    for i in range(n - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + i * offset, y - h / 2 + i * offset), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.08", linewidth=1.1,
            edgecolor=ec, facecolor=WHITE, zorder=2.5,
        ))
    _box(ax, xy, w, h, text, fc=fc, ec=ec, zorder=3.5, **kw)


def _arrow(ax, p0, p1, *, color=GRAY, lw=2.0, style="-|>", connstyle="arc3,rad=0.0",
           mutation_scale=16, linestyle="solid", zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, color=color, linewidth=lw,
        connectionstyle=connstyle, mutation_scale=mutation_scale,
        linestyle=linestyle, zorder=zorder, shrinkA=1, shrinkB=1,
    ))


def build_figure() -> plt.Figure:
    font_name = configure_times_font()
    fig_w, fig_h = 14.6, 9.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.patch.set_visible(False)
    fig.patch.set_facecolor(WHITE)

    # ---- main pipeline geometry --------------------------------------------
    cx = 5.15
    bw, bh = 4.2, 0.86
    gap = 1.18

    y6 = 2.05
    y5, y4, y3, y2, y1 = (y6 + i * gap for i in (1, 2, 3, 4, 5))
    ys = [y1, y2, y3, y4, y5, y6]

    # Title block
    ax.text(
        fig_w / 2, 9.55,
        "Experimental Pipeline \u2014 Train-Synthetic-Test-Real (TSTR) Protocol",
        ha="center", va="center", fontsize=16.5, fontweight="bold", color=NAVY,
    )
    ax.text(
        fig_w / 2, 9.13,
        "The real train/test split is fixed across all runs; only generator, sampling, and classifier\n"
        "randomness vary across the 10 seeds \u2014 by design, this guarantees zero information leakage.",
        ha="center", va="center", fontsize=10, color=GRAY, linespacing=1.4,
    )

    # ---- main pipeline boxes ------------------------------------------------
    _box(ax, (cx, y1), bw, bh, "Real Dataset", fc=BLUE_LIGHT, ec=BLUE, tc=NAVY, fontsize=12.5)

    _box(
        ax, (cx, y2), bw, bh,
        "Train / Test Split\n(fixed \u2014 no leakage)",
        fc=AMBER_LIGHT, ec=AMBER, tc="#5c3d15", fontsize=11.5,
    )

    _stack_box(
        ax, (cx, y3), bw - 0.6, bh,
        "Train Synthetic Generator\n(training set only)",
        fc=BLUE_LIGHT, ec=BLUE, tc=NAVY, fontsize=11, n=3,
    )

    _box(
        ax, (cx, y4), bw, bh, "Generate Synthetic Dataset",
        fc=BLUE_LIGHT, ec=BLUE, tc=NAVY, fontsize=12,
    )

    _stack_box(
        ax, (cx, y5), bw - 0.6, bh,
        "Train Downstream Classifier\non Synthetic Data",
        fc=GREEN_LIGHT, ec=GREEN, tc="#1c4a34", fontsize=11, n=3,
    )

    _box(
        ax, (cx, y6), bw, 1.02,
        "Evaluate on Real Test Set\n(TSTR)",
        fc=GREEN_LIGHT, ec=GREEN, tc="#1c4a34", fontsize=12.5,
    )

    # vertical connectors between consecutive boxes
    for a, b in zip(ys[:-1], ys[1:]):
        _arrow(ax, (cx, a - bh / 2 - 0.03), (cx, b + bh / 2 + 0.03), lw=2.2)

    box_right = cx + bw / 2

    # ---- bracket connecting the 3 seed-dependent stages to a single callout --
    spine_x = box_right + 0.55
    for ty in (y3, y4, y5):
        _arrow(ax, (box_right + 0.08, ty), (spine_x, ty), color=GRAY, lw=1.5,
               style="-", zorder=1.8)
    _arrow(ax, (spine_x, y3), (spine_x, y5), color=GRAY, lw=1.5, style="-", zorder=1.8)

    call_w, call_h = 3.55, 3.0
    call_x = spine_x + 0.55 + call_w / 2
    call_y = y4
    _arrow(ax, (spine_x, y4), (call_x - call_w / 2 - 0.03, y4), color=GRAY, lw=1.6,
           style="-|>", mutation_scale=15, zorder=1.8)

    _box(
        ax, (call_x, call_y), call_w, call_h, "",
        fc=GRAY_LIGHT, ec=GRAY, boxstyle="round,pad=0.08,rounding_size=0.10", zorder=3,
    )
    ax.text(
        call_x, call_y + call_h / 2 - 0.34, "VARIES across the 10 seeds",
        ha="center", va="center", fontsize=10.6, fontweight="bold", color=INK,
    )
    items = [
        "Generator initialization\n(weights / sampling RNG)",
        "Synthetic sample generation\n(draw from trained generator)",
        "Classifier initialization\n(model weights / fit RNG)",
    ]
    item_ys = [call_y + call_h / 2 - 0.95, call_y, call_y - call_h / 2 + 0.95]
    for text, iy in zip(items, item_ys):
        ax.text(
            call_x, iy, f"\u2022  {text}",
            ha="center", va="center", fontsize=9.3, color="#2b2f35", linespacing=1.4,
        )

    # ---- real test-set bypass path (held out, never touches synthetic path) --
    bypass_x = call_x + call_w / 2 + 0.7
    _arrow(ax, (box_right + 0.05, y2), (bypass_x, y2), color=AMBER, lw=1.9, style="-", zorder=1.5)
    _arrow(ax, (bypass_x, y2), (bypass_x, y6), color=AMBER, lw=1.9, style="-", zorder=1.5)
    _arrow(ax, (bypass_x, y6), (box_right + 0.05, y6), color=AMBER, lw=1.9, style="-|>",
           mutation_scale=15, zorder=1.5)
    ax.text(
        bypass_x + 0.2, (y2 + y6) / 2, "held-out real test set\n(never touched by the generator)",
        rotation=90, ha="left", va="center", fontsize=8.6, color="#7a5219",
        style="italic", linespacing=1.3,
    )

    # ---- "fixed / no leakage" callout ---------------------------------------
    fix_w, fix_h = 2.55, 1.35
    box_left = cx - bw / 2
    fix_x = box_left - 0.55 - fix_w / 2
    fix_y = y2
    _box(
        ax, (fix_x, fix_y), fix_w, fix_h,
        "FIXED across all seeds\n\nReal train/test split\n(identical for every run)",
        fc=RED_LIGHT, ec=RED, tc="#7a2113", fontsize=9.5,
        boxstyle="round,pad=0.08,rounding_size=0.10",
    )
    _arrow(
        ax, (fix_x + fix_w / 2, fix_y), (box_left - 0.05, fix_y),
        color=RED, lw=1.7, style="-|>", mutation_scale=14, zorder=2,
    )

    # ---- bottom summary banner ----------------------------------------------
    banner_y = 0.58
    banner_w, banner_h = fig_w - 1.0, 0.86
    _box(
        ax, (fig_w / 2, banner_y), banner_w, banner_h, "",
        fc=NAVY, ec=NAVY, boxstyle="round,pad=0.08,rounding_size=0.10", zorder=3,
    )
    ax.text(
        fig_w / 2, banner_y + 0.20,
        "10 random seeds  \u00d7  multiple synthetic generators  \u00d7  multiple downstream classifiers",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=WHITE,
    )
    ax.text(
        fig_w / 2, banner_y - 0.22,
        "Real train/test split held fixed throughout  \u2192  zero train\u2013test information leakage",
        ha="center", va="center", fontsize=10, color="#cfe0f5",
    )

    apply_font_to_figure(fig, font_name)
    return fig


def main() -> None:
    fig = build_figure()
    png_path = OUT_DIR / "experimental_pipeline_workflow.png"
    svg_path = OUT_DIR / "experimental_pipeline_workflow.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
