"""Generate SYNTH benchmark pipeline flowchart PNG (matplotlib, no Graphviz required)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "SYNTH_benchmark_pipeline_15datasets_8generators.png"


def box(ax, xy, w, h, text, fc="#F7F9FC", ec="#4A5568", fontsize=9, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )
    return (x + w / 2, y, x + w / 2, y + h)  # bottom center, top center


def arrow(ax, p1, p2, color="#4A5568", style="-|>", lw=1.2, label=None, label_offset=(0, 0)):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8, color=color, fontweight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.axis("off")

    ax.text(
        7,
        21.3,
        "SYNTH Benchmark Pipeline — 15 Datasets · 8 Generators · 20% Holdout",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    # Dataset cluster
    box(
        ax,
        (1.0, 19.2),
        12.0,
        1.5,
        "15 Datasets (#1–#15)\n"
        "9 Classification: Cancer, Alzheimer's, Adult, Forest Cover, Bank, CDC Diabetes, Mushroom, MAGIC Gamma (+ Wine as ordinal)\n"
        "6 Regression: Wine, Metro, Online Shopping, Air Quality, Concrete, Energy Efficiency, Real Estate",
        fc="#EBF8FF",
        ec="#3182CE",
        fontsize=8.5,
        bold=True,
    )

    load = box(ax, (4.5, 17.8), 5.0, 0.9, "Load & preprocess\n(drop date / time / session / ID features)", fc="#EDF2F7")
    sample = box(ax, (4.5, 16.5), 5.0, 0.9, "Random subsample N = 1000 (seed = 42)", fc="#EDF2F7")
    split = box(ax, (4.2, 15.0), 5.6, 1.0, "Train / test split\nTEST_SIZE = 0.2  →  80% train  |  20% test", fc="#FFF5F5", ec="#C53030", bold=True)

    train = box(ax, (1.0, 12.8), 5.2, 1.2, "train_real (80%)\n800 rows\nTRAIN GENERATORS HERE ONLY", fc="#C6F6D5", ec="#276749", bold=True)
    test = box(ax, (7.8, 12.8), 5.2, 1.2, "test_real (20%)\n200 rows\nHELD OUT — UNSEEN BY GENERATORS", fc="#FED7D7", ec="#C53030", bold=True)

    gens = box(
        ax,
        (1.0, 10.8),
        12.0,
        1.3,
        "8 Generators (each trained only on train_real)\n"
        "CTAB-GAN+  |  WGAN-GP  |  TabDDPM  |  CoDi  |  CTGAN  |  CopulaGAN  |  TVAE  |  GaussianCopula",
        fc="#FAF5FF",
        ec="#6B46C1",
        fontsize=8.5,
        bold=True,
    )

    synth = box(ax, (4.0, 9.2), 6.0, 0.9, "Generate 1,000 synthetic rows per generator", fc="#E9D8FD", ec="#6B46C1")
    fidelity = box(ax, (4.0, 7.7), 6.0, 1.0, "Fidelity: SDV evaluate_quality\n(synthetic vs train_real)", fc="#BEE3F8", ec="#2B6CB0", bold=True)

    trtr = box(ax, (1.0, 5.5), 5.5, 1.1, "TRTR baseline\nTrain on REAL  |  Test on REAL", fc="#FEEBC8", ec="#C05621", bold=True)
    tstr = box(ax, (7.5, 5.5), 5.5, 1.1, "TSTR\nTrain on SYNTHETIC  |  Test on HELD-OUT REAL", fc="#FBD38D", ec="#C05621", bold=True)

    drops = box(
        ax,
        (3.0, 3.5),
        8.0,
        1.1,
        "TRTR vs TSTR drops\n(Accuracy / F1 / Precision / Recall  or  RMSE / MAE / R²)",
        fc="#E6FFFA",
        ec="#285E61",
        bold=True,
    )
    export = box(ax, (4.0, 1.7), 6.0, 0.9, "Export: TRTR_TSTR_results_*.xlsx", fc="#EDF2F7", bold=True)

    # Arrows main flow
    arrow(ax, (7, 19.2), (7, 18.7))
    arrow(ax, (7, 17.8), (7, 17.4))
    arrow(ax, (7, 16.5), (7, 16.0))
    arrow(ax, (7, 15.0), (3.6, 14.0))
    arrow(ax, (7, 15.0), (10.4, 14.0))
    arrow(ax, (3.6, 12.8), (7, 12.1))
    arrow(ax, (7, 10.8), (7, 10.1))
    arrow(ax, (7, 9.2), (7, 8.7))
    arrow(ax, (3.6, 12.8), (7, 8.2), color="#2B6CB0")
    arrow(ax, (3.6, 12.8), (3.25, 6.6))
    arrow(ax, (10.4, 12.8), (3.25, 6.6))
    arrow(ax, (7, 9.2), (10.25, 6.6))
    arrow(ax, (10.4, 12.8), (10.25, 6.6), color="#C53030", lw=2.0, label="20% holdout\n→ TSTR test set", label_offset=(1.3, 0.2))
    arrow(ax, (3.25, 5.5), (5.5, 4.6))
    arrow(ax, (10.25, 5.5), (8.5, 4.6))
    arrow(ax, (7, 3.5), (7, 2.6))

    ax.text(
        7,
        0.6,
        "Note: Each Single run notebook uses 6 of 8 generators at a time\n"
        "(GAN+SDV or Diffusion+SDV). Full benchmark spans Single run, diffusion_dataleak, Other GANS, Diffusion GANs, SDV models.",
        ha="center",
        va="center",
        fontsize=8,
        color="#718096",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
