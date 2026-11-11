#!/usr/bin/env python3
"""Workflow + experimental-design diagrams (Agreed analysis visual language)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
from src.latex_fonts import apply_font_to_figure, configure_times_font  # noqa: E402

OUT = ANALYSIS / "figures" / "workflow"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1f3a5f"
BLUE = "#2f6fb5"
BLUE_LIGHT = "#eaf1fb"
GREEN = "#2f7d5b"
GREEN_LIGHT = "#e9f5ef"
AMBER = "#b5762f"
AMBER_LIGHT = "#fbf2e6"
GRAY = "#5a5f66"
GRAY_LIGHT = "#f4f5f6"
INK = "#1c1f24"
WHITE = "#ffffff"


def _box(ax, xy, w, h, text, *, fc, ec, tc=INK, fontsize=10.5, weight="bold", lw=1.5, z=3):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=z,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=tc, zorder=z + 1, linespacing=1.3)


def _arrow(ax, p0, p1, *, color=GRAY, lw=2.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            color=color,
            linewidth=lw,
            mutation_scale=14,
            zorder=2,
        )
    )


def workflow_figure():
    font = configure_times_font()
    fig, ax = plt.subplots(figsize=(12.5, 14.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    y = 15.2
    _box(ax, (5, y), 6.2, 0.7, "15 DATASETS  (9 classification + 6 regression)", fc=BLUE_LIGHT, ec=BLUE)
    y -= 1.1
    _arrow(ax, (5, y + 0.75), (5, y + 0.35))
    _box(ax, (5, y), 7.2, 0.85, "LEAKAGE-SAFE TRAIN/TEST SPLIT\n(split seed = 42; fixed across all generator seeds)", fc=AMBER_LIGHT, ec=AMBER)
    y -= 1.15
    _arrow(ax, (5, y + 0.8), (5, y + 0.4))
    _box(ax, (5, y), 7.4, 0.9, "FIT IMPUTER ON REAL-TRAIN ONLY\nApply train-fitted imputer to REAL-TEST\n(OASIS: participant-level split; no ID overlap)", fc=AMBER_LIGHT, ec=AMBER)
    y -= 1.2
    _arrow(ax, (5, y + 0.85), (5, y + 0.4))
    _box(ax, (3.2, y), 3.4, 0.7, "REAL-TRAIN", fc=GREEN_LIGHT, ec=GREEN)
    _box(ax, (6.8, y), 3.4, 0.7, "REAL-TEST (held out)", fc=GRAY_LIGHT, ec=GRAY)
    y -= 1.15
    _arrow(ax, (3.2, y + 0.8), (3.2, y + 0.4))
    _box(ax, (5, y), 6.5, 0.7, "8 GENERATORS", fc=BLUE_LIGHT, ec=BLUE)
    y -= 1.35
    _arrow(ax, (5, y + 1.0), (5, y + 0.55))
    _box(
        ax,
        (5, y),
        8.2,
        1.5,
        "10 INDEPENDENT GENERATOR-TRAINING SEEDS\n"
        "Batch 1 (execution): 42, 123, 2024\n"
        "Batch 2 (execution): 68, 91, 2025\n"
        "Batch 3 (execution): 55, 155, 255, 355",
        fc="#fff7e8",
        ec=AMBER,
        fontsize=10,
    )
    y -= 1.55
    _arrow(ax, (5, y + 0.85), (5, y + 0.4))
    _box(ax, (5, y), 6.5, 0.7, "1,200 GENERATOR RUNS  (15 × 8 × 10)", fc=BLUE_LIGHT, ec=BLUE)
    y -= 1.15
    _arrow(ax, (5, y + 0.8), (5, y + 0.35))
    _box(ax, (2.2, y), 2.4, 0.7, "FIDELITY", fc=GREEN_LIGHT, ec=GREEN, fontsize=10)
    _box(ax, (5.0, y), 2.4, 0.7, "UTILITY", fc=GREEN_LIGHT, ec=GREEN, fontsize=10)
    _box(ax, (7.8, y), 2.4, 0.7, "PRIVACY", fc=GREEN_LIGHT, ec=GREEN, fontsize=10)
    y -= 1.15
    _arrow(ax, (5, y + 0.8), (5, y + 0.35))
    _box(ax, (5, y), 7.0, 0.85, "10-SEED AGGREGATION\n(individual seed observations → Mean ± SD, ddof=1)", fc=BLUE_LIGHT, ec=NAVY)
    y -= 1.15
    _arrow(ax, (5, y + 0.8), (5, y + 0.35))
    _box(ax, (5, y), 6.8, 0.7, "DATASET × GENERATOR ANALYSIS  (Mean ± SD)", fc=GREEN_LIGHT, ec=GREEN)

    ax.set_title(
        "Experimental Workflow — 10-Seed Synthetic Data Evaluation",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        pad=12,
    )
    fig.text(
        0.5,
        0.015,
        "Batches are computational execution groups only; primary results pool all 10 seed-level observations.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    apply_font_to_figure(fig, font)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"experimental_workflow.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote experimental_workflow")


def design_figure():
    font = configure_times_font()
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    _box(ax, (5, 11.2), 5.5, 0.65, "15 DATASETS", fc=BLUE_LIGHT, ec=BLUE)
    _arrow(ax, (5, 10.85), (5, 10.45))
    _box(ax, (5, 10.1), 5.5, 0.65, "8 GENERATORS (per dataset)", fc=BLUE_LIGHT, ec=BLUE)
    _arrow(ax, (5, 9.75), (5, 9.2))

    _box(ax, (2.0, 7.6), 3.0, 2.4, "Batch 1\n(execution)\n\n42\n123\n2024\n\n360 runs", fc=AMBER_LIGHT, ec=AMBER, fontsize=10)
    _box(ax, (5.0, 7.6), 3.0, 2.4, "Batch 2\n(execution)\n\n68\n91\n2025\n\n360 runs", fc=AMBER_LIGHT, ec=AMBER, fontsize=10)
    _box(ax, (8.0, 7.6), 3.0, 2.4, "Batch 3\n(execution)\n\n55  155\n255  355\n\n480 runs", fc=AMBER_LIGHT, ec=AMBER, fontsize=10)

    for x in (2.0, 5.0, 8.0):
        _arrow(ax, (x, 6.35), (x, 5.55))
    _box(ax, (5, 5.15), 8.2, 0.7, "10 SEED OBSERVATIONS per dataset × generator", fc=GREEN_LIGHT, ec=GREEN)
    _arrow(ax, (5, 4.75), (5, 4.25))
    _box(ax, (5, 3.9), 6.5, 0.7, "MEAN ± SD  (sample SD, ddof = 1)", fc=BLUE_LIGHT, ec=NAVY)
    _arrow(ax, (5, 3.5), (5, 3.0))
    _box(ax, (5, 2.55), 7.5, 0.85, "Total: 15 × 8 × 10 = 1,200 generator runs\n(720 classification + 480 regression)", fc=GRAY_LIGHT, ec=GRAY, fontsize=10)

    ax.set_title(
        "Experimental Design — Datasets × Generators × Seeds",
        fontsize=14.5,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )
    fig.text(
        0.5,
        0.02,
        "Do not treat batches as separate scientific experiments; they preserve provenance only.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    apply_font_to_figure(fig, font)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"experimental_design.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote experimental_design")


def main():
    workflow_figure()
    design_figure()


if __name__ == "__main__":
    main()
