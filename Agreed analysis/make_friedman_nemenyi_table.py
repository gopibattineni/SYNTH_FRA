"""Publication-quality Friedman + Nemenyi statistical summary table.

Uses OverallScore ranks across the 9 classification datasets (no leakage).
Exports a PNG with:
  - Friedman χ², degrees of freedom, p-value
  - Pairwise Nemenyi adjusted p-values
  - Significant pairs (α = 0.05) highlighted
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import friedmanchisquare

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classification"
ROOT = SCRIPT_DIR.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES = ROOT / "Results" / "Processed_Data" / "overall_scores.csv"
ALPHA = 0.05

CLASSIFICATION_DATASETS = [
    "1. Cancer",
    "2. Alzhimers",
    "3. Adult",
    "4. Forest cover dataset",
    "5. Bank Markting",
    "6. Wine dataset",
    "7. CDC diabetes dataset",
    "8. Mushroom dataset",
    "9. MAGIC Gamma Telescope",
]

DISPLAY = {
    "ForestDiffusion": "ForestDiffusion",
    "TVAE": "TVAE",
    "CTABGAN": "CTABGAN",
    "WGAN_GP": "WGAN-GP",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTGAN": "CTGAN",
    "TabDDPM": "TabDDPM",
}

NAVY = "#1f3a5f"
INK = "#1c1f24"
GRID = "#c5d0dc"
HEADER_BG = "#1f3a5f"
ROW_A = "#ffffff"
ROW_B = "#eef3f9"
SIG_BG = "#fde8e4"       # light coral for significant cells
SIG_FG = "#9b2c1f"
NS_FG = "#3a3f45"
SUMMARY_BG = "#eaf1fb"


def load_matrices():
    df = pd.read_csv(SCORES)
    sub = df[
        df["Dataset"].isin(CLASSIFICATION_DATASETS) & (df["LeakageLevel"] == 0.0)
    ].copy()
    piv_score = sub.pivot_table(index="Dataset", columns="Generator", values="OverallScore")
    piv_rank = sub.pivot_table(index="Dataset", columns="Generator", values="OverallRank")
    return piv_score, piv_rank


def compute_stats(piv_score: pd.DataFrame, piv_rank: pd.DataFrame):
    arrays = [piv_score[c].to_numpy() for c in piv_score.columns]
    chi2, p = friedmanchisquare(*arrays)
    k = piv_score.shape[1]
    n = piv_score.shape[0]
    df_f = k - 1
    avg_rank = piv_rank.mean().sort_values()
    order = avg_rank.index.tolist()
    # Nemenyi on score matrix (higher = better); reorder by avg rank
    nemenyi = sp.posthoc_nemenyi_friedman(piv_score[order])
    nemenyi = nemenyi.loc[order, order]
    return {
        "chi2": float(chi2),
        "df": int(df_f),
        "p": float(p),
        "n_datasets": int(n),
        "k": int(k),
        "avg_rank": avg_rank,
        "order": order,
        "nemenyi": nemenyi,
    }


def _fmt_p(p: float) -> str:
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def render(stats: dict) -> plt.Figure:
    from latex_fonts import apply_font_to_figure, configure_times_font
    font_name = configure_times_font()

    order = stats["order"]
    labels = [DISPLAY.get(g, g) for g in order]
    k = len(order)
    nemenyi = stats["nemenyi"]
    avg = stats["avg_rank"]

    # Layout geometry (axes units)
    label_w = 2.35
    cell_w = 1.05
    cell_h = 0.48
    header_h = 0.72
    summary_h = 1.15
    title_h = 0.85
    note_h = 0.55
    rank_h = 0.48

    table_w = label_w + k * cell_w
    fig_w = table_w + 0.6
    fig_h = title_h + summary_h + 0.25 + header_h + rank_h + k * cell_h + note_h + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, table_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- Title ----
    ax.text(
        table_w / 2, fig_h - 0.32,
        "Friedman Test & Nemenyi Post-hoc Comparisons",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY,
    )
    ax.text(
        table_w / 2, fig_h - 0.62,
        "Overall composite score ranks across 9 classification datasets (no information leakage)",
        ha="center", va="center", fontsize=9, color="#5a5f66", style="italic",
    )

    # ---- Friedman summary strip ----
    y_sum = fig_h - title_h - summary_h
    ax.add_patch(FancyBboxPatch(
        (0.05, y_sum), table_w - 0.1, summary_h - 0.08,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.2, edgecolor=NAVY, facecolor=SUMMARY_BG, zorder=1,
    ))
    p_str = _fmt_p(stats["p"])
    sig_tag = "significant" if stats["p"] < ALPHA else "not significant"
    ax.text(
        table_w / 2, y_sum + summary_h - 0.32,
        "Friedman omnibus test",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY, zorder=2,
    )
    ax.text(
        table_w / 2, y_sum + summary_h / 2 - 0.08,
        rf"$\chi^2$ = {stats['chi2']:.3f}"
        f"      |      df = {stats['df']}"
        f"      |      p = {p_str}  ({sig_tag} at α = {ALPHA})"
        f"      |      N = {stats['n_datasets']} datasets,  k = {stats['k']} generators",
        ha="center", va="center", fontsize=10, color=INK, zorder=2,
    )

    # ---- Matrix origin ----
    y_top = y_sum - 0.2  # top of header row
    x0 = 0.0

    # Corner cell
    ax.add_patch(Rectangle(
        (x0, y_top - header_h), label_w, header_h,
        facecolor=HEADER_BG, edgecolor=GRID, linewidth=0.6, zorder=1,
    ))
    ax.text(
        x0 + label_w / 2, y_top - header_h / 2, "Generator",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=2,
    )

    # Column headers (abbreviated for width)
    short = {
        "ForestDiffusion": "FD",
        "TVAE": "TVAE",
        "CTABGAN": "CTAB",
        "WGAN-GP": "WGAN",
        "GaussianCopula": "GC",
        "CopulaGAN": "CGAN",
        "CTGAN": "CTGAN",
        "TabDDPM": "DDPM",
    }
    for j, lab in enumerate(labels):
        x = x0 + label_w + j * cell_w
        ax.add_patch(Rectangle(
            (x, y_top - header_h), cell_w, header_h,
            facecolor=HEADER_BG, edgecolor=GRID, linewidth=0.6, zorder=1,
        ))
        ax.text(
            x + cell_w / 2, y_top - header_h / 2, short.get(lab, lab),
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=2, rotation=0,
        )

    # Average-rank row
    y_rank = y_top - header_h - rank_h
    ax.add_patch(Rectangle(
        (x0, y_rank), label_w, rank_h,
        facecolor="#dce6f2", edgecolor=GRID, linewidth=0.6, zorder=1,
    ))
    ax.text(
        x0 + label_w / 2, y_rank + rank_h / 2, "Avg. rank",
        ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY, zorder=2,
    )
    for j, g in enumerate(order):
        x = x0 + label_w + j * cell_w
        ax.add_patch(Rectangle(
            (x, y_rank), cell_w, rank_h,
            facecolor="#dce6f2", edgecolor=GRID, linewidth=0.6, zorder=1,
        ))
        ax.text(
            x + cell_w / 2, y_rank + rank_h / 2, f"{avg[g]:.2f}",
            ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY, zorder=2,
        )

    # Pairwise Nemenyi cells (row i vs column j); diagonal = "—"
    for i, gi in enumerate(order):
        y = y_rank - (i + 1) * cell_h
        # row label
        bg = ROW_A if i % 2 == 0 else ROW_B
        ax.add_patch(Rectangle(
            (x0, y), label_w, cell_h,
            facecolor=bg, edgecolor=GRID, linewidth=0.6, zorder=1,
        ))
        ax.text(
            x0 + 0.12, y + cell_h / 2, labels[i],
            ha="left", va="center", fontsize=9, color=INK, zorder=2,
        )

        for j, gj in enumerate(order):
            x = x0 + label_w + j * cell_w
            if i == j:
                ax.add_patch(Rectangle(
                    (x, y), cell_w, cell_h,
                    facecolor="#f0f2f5", edgecolor=GRID, linewidth=0.6, zorder=1,
                ))
                ax.text(
                    x + cell_w / 2, y + cell_h / 2, "—",
                    ha="center", va="center", fontsize=10, color="#8a9098", zorder=2,
                )
                continue

            pval = float(nemenyi.loc[gi, gj])
            sig = pval < ALPHA
            cell_bg = SIG_BG if sig else (ROW_A if i % 2 == 0 else ROW_B)
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=cell_bg, edgecolor=GRID, linewidth=0.6, zorder=1,
            ))
            txt = _fmt_p(pval)
            if sig:
                txt = f"{txt}*"
            ax.text(
                x + cell_w / 2, y + cell_h / 2, txt,
                ha="center", va="center",
                fontsize=8.2 if not sig else 8.4,
                fontweight="bold" if sig else "normal",
                color=SIG_FG if sig else NS_FG, zorder=2,
            )

    # Outer border
    y_bottom = y_rank - k * cell_h
    ax.add_patch(Rectangle(
        (x0, y_bottom), table_w, header_h + rank_h + k * cell_h,
        facecolor="none", edgecolor=NAVY, linewidth=1.3, zorder=4,
    ))

    # Legend / notes
    ax.text(
        0.05, note_h - 0.05,
        f"*  Significant pairwise difference (Nemenyi adjusted p < {ALPHA}).  "
        "Cells show adjusted p-values.  Generators ordered by average rank (best → worst).  "
        "Abbreviations: FD = ForestDiffusion, CTAB = CTABGAN, WGAN = WGAN-GP, "
        "GC = GaussianCopula, CGAN = CopulaGAN, DDPM = TabDDPM.",
        ha="left", va="top", fontsize=7.8, color="#5a5f66", wrap=True,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    apply_font_to_figure(fig, font_name)
    return fig


def save_csv(stats: dict) -> Path:
    # Long-form pairwise table
    rows = []
    order = stats["order"]
    nemenyi = stats["nemenyi"]
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            p = float(nemenyi.loc[a, b])
            rows.append({
                "Generator_A": DISPLAY.get(a, a),
                "Generator_B": DISPLAY.get(b, b),
                "AvgRank_A": round(stats["avg_rank"][a], 4),
                "AvgRank_B": round(stats["avg_rank"][b], 4),
                "Nemenyi_p": p,
                "Significant_alpha_0.05": p < ALPHA,
            })
    pair_df = pd.DataFrame(rows).sort_values("Nemenyi_p")
    pair_path = OUT_DIR / "friedman_nemenyi_pairwise.csv"
    pair_df.to_csv(pair_path, index=False)

    summary = pd.DataFrame([{
        "Test": "Friedman",
        "Statistic_chi2": stats["chi2"],
        "DegreesOfFreedom": stats["df"],
        "p_value": stats["p"],
        "N_Datasets": stats["n_datasets"],
        "N_Generators": stats["k"],
        "Alpha": ALPHA,
    }])
    sum_path = OUT_DIR / "friedman_nemenyi_summary.csv"
    summary.to_csv(sum_path, index=False)

    # Full p-matrix
    mat = stats["nemenyi"].copy()
    mat.index = [DISPLAY.get(g, g) for g in mat.index]
    mat.columns = [DISPLAY.get(g, g) for g in mat.columns]
    mat_path = OUT_DIR / "friedman_nemenyi_pmatrix.csv"
    mat.to_csv(mat_path)
    return pair_path


def main() -> None:
    piv_score, piv_rank = load_matrices()
    stats = compute_stats(piv_score, piv_rank)
    save_csv(stats)

    fig = render(stats)
    png_path = OUT_DIR / "friedman_nemenyi_summary_table.png"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print(f"Friedman χ² = {stats['chi2']:.4f}, df = {stats['df']}, p = {stats['p']:.6g}")
    print(f"Significant pairs (α={ALPHA}):")
    nemenyi = stats["nemenyi"]
    order = stats["order"]
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            p = float(nemenyi.loc[a, b])
            if p < ALPHA:
                print(f"  {DISPLAY.get(a,a)} vs {DISPLAY.get(b,b)}: p = {_fmt_p(p)}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
