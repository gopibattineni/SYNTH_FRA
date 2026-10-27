"""Generate a publication-quality SYNTH benchmark workflow diagram.

Style target: clean IEEE / Nature / Elsevier journal figure — white background,
landscape layout, rounded rectangles with subtle drop shadows, thin consistent
borders, sans-serif typography, generous whitespace, and color-coded stages.

All connectors are routed as orthogonal (elbow) polylines through the blank
space between columns/rows so no line ever crosses a box.

Outputs (vector + high-res raster) are written to both:
  - figures/SYNTH_workflow.{svg,pdf,png}
  - Single run_Data_leak_Synth_Quality/forge_paper_workflow.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent
LEGACY_OUT = ROOT / "Single run_Data_leak_Synth_Quality" / "forge_paper_workflow.png"

# ---------------------------------------------------------------- journal styling (Times / mathptmx — matches LaTeX body + math)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

C_DATA = "#2E6DB4"
C_SYNTH = "#7B4FBF"
C_QUAL = "#0E9488"
C_EVAL = "#D97F1E"
C_OUT = "#3B9142"
C_INK = "#1F2733"
C_SUB = "#6B7684"

BG = {
    "data": "#EAF1FB",
    "synth": "#F1ECFB",
    "qual": "#E4F7F4",
    "eval": "#FCEFDD",
    "out": "#E9F6EA",
    "train": "#DFF5E4",
    "test": "#FBE1E1",
}
EC_TRAIN = "#2A8C42"
EC_TEST = "#C23B3B"


def shadow(ax, x, y, w, h, dx=-0.045, dy=-0.06, alpha=0.10):
    ax.add_patch(
        FancyBboxPatch(
            (x + dx, y + dy), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=0, facecolor="#000000", alpha=alpha, zorder=1,
        )
    )


def box(ax, x, y, w, h, text, fc, ec, fs=8.2, bold=False, lw=1.15, title=None, title_fs=8.6):
    shadow(ax, x, y, w, h)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
        )
    )
    if title:
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
                 fontsize=title_fs, fontweight="bold", color=C_INK, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.10, text, ha="center", va="center",
                 fontsize=fs, color=C_INK, multialignment="center", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                 fontweight="bold" if bold else "normal", color=C_INK,
                 multialignment="center", zorder=4)
    return {"cx": x + w / 2, "cy": y + h / 2, "top": y + h, "bot": y,
            "left": x, "right": x + w, "w": w, "h": h}


def seg_arrow(ax, p1, p2, color, lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
            color=color, shrinkA=0, shrinkB=1.5, zorder=2, capstyle="round",
        )
    )


def routed(ax, points, color, lw=1.5):
    if len(points) > 2:
        xs = [p[0] for p in points[:-1]]
        ys = [p[1] for p in points[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round",
                 solid_joinstyle="round", zorder=2, alpha=0.95)
    seg_arrow(ax, points[-2], points[-1], color, lw)


def stage_header(ax, x, w, y, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.36, boxstyle="round,pad=0.01,rounding_size=0.05",
                                  linewidth=0, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + 0.18, label, ha="center", va="center",
             fontsize=10, fontweight="bold", color="white", zorder=4)


def main() -> None:
    fig, ax = plt.subplots(figsize=(19.5, 11.8))
    ax.set_xlim(0, 19.5)
    ax.set_ylim(0, 11.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(9.75, 11.42, "SYNTH Benchmark: Synthetic Data Evaluation Workflow",
             ha="center", va="center", fontsize=18, fontweight="bold", color=C_INK)
    ax.text(9.75, 10.98,
             "15 Datasets  \u2022  8 Generators  \u2022  20% Hold-Out Test  \u2022  Leakage-Aware Training",
             ha="center", va="center", fontsize=11, color=C_SUB)

    col_w = 3.35
    gap = 0.55
    x0 = 0.35
    col_x = [x0 + i * (col_w + gap) for i in range(5)]
    g = [(col_x[i] + col_w + col_x[i + 1]) / 2 for i in range(4)]

    hdr_y = 10.28
    stage_header(ax, col_x[0], col_w, hdr_y, "STAGE 1 \u00b7 DATA", C_DATA)
    stage_header(ax, col_x[1], col_w, hdr_y, "STAGE 2 \u00b7 SYNTHESIS", C_SYNTH)
    stage_header(ax, col_x[2], col_w, hdr_y, "STAGE 3 \u00b7 QUALITY", C_QUAL)
    stage_header(ax, col_x[3], col_w, hdr_y, "STAGE 4 \u00b7 EVALUATION", C_EVAL)
    stage_header(ax, col_x[4], col_w, hdr_y, "STAGE 5 \u00b7 ANALYSIS", C_OUT)

    pad = 0.14
    bw = col_w - 2 * pad
    bx = lambda i: col_x[i] + pad

    # =========================================================== STAGE 1 — DATA
    b1 = box(ax, bx(0), 9.28, bw, 0.72,
             "15 UCI / Curated Datasets\n9 Classification \u00b7 6 Regression",
             BG["data"], C_DATA, fs=8.1, bold=True)
    b2 = box(ax, bx(0), 8.40, bw, 0.62,
             "Load \u0026 Preprocess\n(remove date, session, IDs; clean missing values)",
             BG["data"], C_DATA, fs=7.6)
    b3 = box(ax, bx(0), 7.62, bw, 0.53,
             "Subsample  $N = 1000$  (Seed = 42)", BG["data"], C_DATA, fs=8.0)
    b4 = box(ax, bx(0), 6.84, bw, 0.53,
             "Stratified 80 / 20 Split", BG["data"], C_DATA, fs=8.4, bold=True)

    half = (bw - 0.16) / 2
    train = box(ax, bx(0), 5.42, half, 1.02,
                "Real Train (80%)\n800 rows\nGenerators train ONLY here",
                BG["train"], EC_TRAIN, fs=7.3, bold=True)
    test = box(ax, bx(0) + half + 0.16, 5.42, half, 1.02,
               "Real Test (20%)\n200 rows \u2014 Held-Out\nNever seen by generators",
               BG["test"], EC_TEST, fs=7.3, bold=True)
    ax.text(bx(0) + bw / 2, 5.20, "Never used during synthesis",
             fontsize=7.4, color=EC_TEST, ha="center", style="italic")

    seg_arrow(ax, (b1["cx"], b1["bot"]), (b1["cx"], b2["top"]), C_DATA)
    seg_arrow(ax, (b2["cx"], b2["bot"]), (b2["cx"], b3["top"]), C_DATA)
    seg_arrow(ax, (b3["cx"], b3["bot"]), (b3["cx"], b4["top"]), C_DATA)
    seg_arrow(ax, (b4["cx"], b4["bot"]), (train["cx"], train["top"]), C_DATA)
    seg_arrow(ax, (b4["cx"], b4["bot"]), (test["cx"], test["top"]), C_DATA)

    # =========================================================== STAGE 2 — SYNTHESIS
    b7 = box(ax, bx(1), 9.38, bw, 0.62,
             "8 Generators\ntrained ONLY on Real Train", BG["synth"], C_SYNTH, fs=8.3, bold=True)
    gens = ["CTAB-GAN+", "TabDDPM", "CTGAN", "TVAE", "WGAN-GP", "CoDi", "CopulaGAN", "Gaussian Copula"]
    gtop, gbot = 9.18, 7.28
    rows, cols = 4, 2
    gw = (bw - 0.16) / cols
    gh = (gtop - gbot - 3 * 0.10) / rows
    for i, name in enumerate(gens):
        r, c = divmod(i, cols)
        gx = bx(1) + c * (gw + 0.16)
        gy = gtop - (r + 1) * gh - r * 0.10
        box(ax, gx, gy, gw, gh, name, "white", C_SYNTH, fs=7.4)
    b8 = box(ax, bx(1), 6.40, bw, 0.68,
             "Synthetic Data\n$N = 1000$ rows per generator", BG["synth"], C_SYNTH, fs=8.2, bold=True)
    seg_arrow(ax, (b8["cx"], gbot), (b8["cx"], b8["top"]), C_SYNTH)

    # =========================================================== STAGE 3 — QUALITY
    b9 = box(ax, bx(2), 8.05, bw, 0.82,
             "Synthetic vs Real Train", BG["qual"], C_QUAL, fs=8.0, bold=False,
             title="SDV evaluate_quality()", title_fs=8.6)
    b10 = box(ax, bx(2), 6.72, bw, 0.90,
              "\u2022 Identity Similarity\n\u2022 Distribution Similarity\n\u2022 Feature Similarity",
              BG["qual"], C_QUAL, fs=7.6, title="Quality Score", title_fs=8.4)
    seg_arrow(ax, (b9["cx"], b9["bot"]), (b10["cx"], b10["top"]), C_QUAL)

    seg_arrow(ax, (b8["right"], b8["cy"]), (b9["left"], b9["cy"] - 0.15), C_SYNTH, lw=1.7)

    # =========================================================== STAGE 4 — EVALUATION
    r1y, r1h = 9.15, 0.62
    r2y, r2h = 7.75, 0.62
    r3y, r3h = 6.72, 0.52
    r4y, r4h = 5.35, 1.02
    hhalf = (bw - 0.16) / 2

    row1L = box(ax, bx(3), r1y, hhalf, r1h,
                "Path A\nTrain classifiers on Real Train", BG["eval"], C_EVAL, fs=7.1, bold=True)
    row1R = box(ax, bx(3) + hhalf + 0.16, r1y, hhalf, r1h,
                "Path B\nTrain classifiers on Synthetic Data", BG["eval"], C_EVAL, fs=7.1, bold=True)
    row2L = box(ax, bx(3), r2y, hhalf, r2h,
                "Test on Real Test\n\u2192 TRTR Metrics", BG["eval"], C_EVAL, fs=7.3)
    row2R = box(ax, bx(3) + hhalf + 0.16, r2y, hhalf, r2h,
                "Test on Real Test\n\u2192 TSTR Metrics", BG["eval"], C_EVAL, fs=7.3)
    row3 = box(ax, bx(3), r3y, bw, r3h,
               "10 Classifiers $\\times$ 10 Seeds  (mean $\\pm$ std)", BG["eval"], C_EVAL, fs=7.8, bold=True)
    row4 = box(ax, bx(3), r4y, bw, r4h,
               "Classification: Accuracy, Precision,\nRecall, $F_1$, ROC-AUC\nRegression: RMSE, MAE, $R^2$",
               BG["eval"], C_EVAL, fs=7.5)

    seg_arrow(ax, (row1L["cx"], row1L["bot"]), (row2L["cx"], row2L["top"]), C_EVAL)
    seg_arrow(ax, (row1R["cx"], row1R["bot"]), (row2R["cx"], row2R["top"]), C_EVAL)
    seg_arrow(ax, (row2L["cx"], row2L["bot"]), (row3["cx"] - 0.65, row3["top"]), C_EVAL)
    seg_arrow(ax, (row2R["cx"], row2R["bot"]), (row3["cx"] + 0.65, row3["top"]), C_EVAL)
    seg_arrow(ax, (row3["cx"], row3["bot"]), (row4["cx"], row4["top"]), C_EVAL)

    # =========================================================== STAGE 5 — ANALYSIS
    b17 = box(ax, bx(4), 8.55, bw, 0.75,
              "$\\Delta = \\mathrm{TRTR} - \\mathrm{TSTR}$", BG["out"], C_OUT, fs=8.6,
              title="Performance Drop", title_fs=8.8)
    b18 = box(ax, bx(4), 7.35, bw, 0.65,
              "Generator Ranking\nLower drop = Better synthesis", BG["out"], C_OUT, fs=8.0)
    b19 = box(ax, bx(4), 6.15, bw, 0.78,
              "Export Results\nTRTR_TSTR_results.xlsx\nPer Dataset \u00d7 Generator",
              BG["out"], C_OUT, fs=8.0, bold=True)
    seg_arrow(ax, (b17["cx"], b17["bot"]), (b18["cx"], b18["top"]), C_OUT)
    seg_arrow(ax, (b18["cx"], b18["bot"]), (b19["cx"], b19["top"]), C_OUT)
    seg_arrow(ax, (row4["right"], row4["cy"]), (b17["left"], b17["cy"] - 0.1), C_EVAL, lw=1.7)

    # =========================================================== LONG-DISTANCE BUS ROUTES
    bus_train_y = 3.55
    bus_test_y = 3.28
    bus_synth_y = 3.00
    band_top = 10.05

    # Clear horizontal band above the Real Train / Real Test row (below the split
    # box) spans the FULL column-1 width, so it is safe to cross over either half.
    band_y0 = (test["top"] + b4["bot"]) / 2
    exit1 = train["cx"] - 0.16   # -> Stage-2 generators (b7)
    exit2 = train["cx"] + 0.16   # -> Stage-4 Path A bus

    routed(
        ax,
        [
            (exit1, train["top"]),
            (exit1, band_y0),
            (g[0] - 0.06, band_y0),
            (g[0] - 0.06, b7["cy"] + 0.10),
            (b7["left"], b7["cy"] + 0.10),
        ],
        C_DATA, lw=1.7,
    )

    train_pts = [
        (exit2, train["top"]),
        (exit2, band_y0),
        (g[0] + 0.06, band_y0),
        (g[0] + 0.06, bus_train_y),
        (g[2], bus_train_y),
        (g[2], band_top),
        (row1L["cx"], band_top),
        (row1L["cx"], row1L["top"]),
    ]
    routed(ax, train_pts, C_DATA, lw=1.6)

    train_q_pts = [
        (g[0] + 0.06, bus_train_y),
        (g[1], bus_train_y),
        (g[1], b9["cy"] - 0.18),
        (b9["left"], b9["cy"] - 0.18),
    ]
    routed(ax, train_q_pts, C_DATA, lw=1.35)

    row1_bot = r1y
    row2_top = r2y + r2h
    band_mid = (row1_bot + row2_top) / 2
    test_pts_common = [
        (test["right"], test["cy"] - 0.24),
        (g[0], test["cy"] - 0.24),
        (g[0], bus_test_y),
        (g[2] + 0.10, bus_test_y),
        (g[2] + 0.10, band_mid),
    ]
    routed(ax, test_pts_common, EC_TEST, lw=1.6)
    seg_arrow(ax, (g[2] + 0.10, band_mid), (row2L["cx"], row2L["top"]), EC_TEST, lw=1.6)
    routed(ax, [(g[2] + 0.10, band_mid), (row2R["cx"], band_mid), (row2R["cx"], row2R["top"])], EC_TEST, lw=1.6)

    synth_pts = [
        (b8["right"], b8["cy"] - 0.20),
        (g[1], b8["cy"] - 0.20),
        (g[1], bus_synth_y),
        (g[2] - 0.10, bus_synth_y),
        (g[2] - 0.10, band_top - 0.26),
        (row1R["cx"], band_top - 0.26),
        (row1R["cx"], row1R["top"]),
    ]
    routed(ax, synth_pts, C_SYNTH, lw=1.6)

    ax.text(g[2] + 0.7, bus_test_y - 0.30, "20% hold-out \u2192 TSTR test set",
             fontsize=7.8, color=EC_TEST, fontweight="bold", ha="center")

    # ---- legend ----
    legend_y = 0.62
    items = [("Data", C_DATA), ("Synthesis", C_SYNTH), ("Quality", C_QUAL),
             ("Evaluation", C_EVAL), ("Analysis", C_OUT)]
    total_w = sum(0.34 + 0.14 + len(n) * 0.088 + 0.5 for n, _ in items)
    lx = (19.5 - total_w) / 2 + 1.0
    for name, col in items:
        ax.add_patch(FancyBboxPatch((lx, legend_y), 0.34, 0.24, boxstyle="round,pad=0.005,rounding_size=0.05",
                                      linewidth=0, facecolor=col, zorder=3))
        ax.text(lx + 0.48, legend_y + 0.12, name, va="center", fontsize=9.2, color=C_INK)
        lx += 0.34 + 0.14 + len(name) * 0.088 + 0.5

    ax.text(
        9.75, 0.15,
        "Each Single-run notebook trains 6 of 8 generators (GAN + SDV, or Diffusion + SDV).  "
        "Full benchmark spans Single run, diffusion_dataleak, Other GANS, Diffusion GANs, and SDV models.",
        ha="center", va="center", fontsize=7.8, color=C_SUB, style="italic",
    )

    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    svg_out = FIG_DIR / "SYNTH_workflow.svg"
    pdf_out = FIG_DIR / "SYNTH_workflow.pdf"
    png_out = FIG_DIR / "SYNTH_workflow.png"

    fig.savefig(svg_out, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_out, bbox_inches="tight", facecolor="white")
    fig.savefig(png_out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(LEGACY_OUT, dpi=300, bbox_inches="tight", facecolor="white")

    benchmark_out = ROOT / "SYNTH_benchmark.png"
    fig.savefig(benchmark_out, dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved: {svg_out}")
    print(f"Saved: {pdf_out}")
    print(f"Saved: {png_out}")
    print(f"Saved: {LEGACY_OUT}")
    print(f"Saved: {benchmark_out}")


if __name__ == "__main__":
    main()
