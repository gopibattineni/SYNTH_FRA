#!/usr/bin/env python3
"""Generate independent Critical-Difference diagrams for classification utility gaps.

Reads merged 10-seed classification Excel and writes three figures:
  Accuracy_Gap, Precision_Gap, Recall_Gap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "classification" / "results" / "multi_seed_results.xlsx"
OUT = Path(__file__).resolve().parent

METRICS = [
    ("Accuracy_Gap", "Accuracy"),
    ("Precision_Gap", "Precision"),
    ("Recall_Gap", "Recall"),
]

DISPLAY_NAME = {
    "WGAN_GP": "WGAN-GP",
    "ForestDiffusion": "ForestDiffusion",
    "GaussianCopula": "GaussianCopula",
    "CopulaGAN": "CopulaGAN",
    "CTABGAN": "CTABGAN",
    "CTGAN": "CTGAN",
    "TVAE": "TVAE",
    "TabDDPM": "TabDDPM",
}

# Nemenyi critical values (Demšar 2006), alpha=0.05, two-tailed studentized range / sqrt(2)
NEMENYI_Q_05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def load_gap_matrix(metric: str) -> pd.DataFrame:
    """Return datasets × generators matrix of mean utility-gap values (lower is better)."""
    wb = openpyxl.load_workbook(EXCEL, read_only=True, data_only=True)
    ws = wb["Utility"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    di, gi, mi, mean_i = (
        header.index("dataset"),
        header.index("generator"),
        header.index("metric_name"),
        header.index("mean"),
    )
    records = []
    for r in rows:
        if r[mi] != metric:
            continue
        records.append({"dataset": r[di], "generator": r[gi], "value": float(r[mean_i])})
    wb.close()
    df = pd.DataFrame(records)
    wide = df.pivot(index="dataset", columns="generator", values="value")
    # stable generator order matching experiment config
    gens = [
        "CTGAN",
        "CopulaGAN",
        "TVAE",
        "GaussianCopula",
        "CTABGAN",
        "WGAN_GP",
        "TabDDPM",
        "ForestDiffusion",
    ]
    wide = wide.reindex(columns=gens)
    if wide.isna().any().any():
        missing = wide.isna().stack()
        missing = missing[missing].index.tolist()
        raise ValueError(f"{metric}: missing cells {missing[:10]}")
    return wide


def average_ranks(wide: pd.DataFrame) -> pd.Series:
    """Rank generators within each dataset (lower gap = rank 1), then average."""
    ranks = wide.rank(axis=1, method="average", ascending=True)
    return ranks.mean(axis=0)


def friedman_test(wide: pd.DataFrame) -> tuple[float, int, float]:
    """Friedman chi-square across generators (columns), datasets (rows)."""
    samples = [wide[c].to_numpy(dtype=float) for c in wide.columns]
    stat, p = stats.friedmanchisquare(*samples)
    df = len(wide.columns) - 1
    return float(stat), int(df), float(p)


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    if alpha != 0.05 or k not in NEMENYI_Q_05:
        raise ValueError(f"Unsupported Nemenyi params k={k} alpha={alpha}")
    q = NEMENYI_Q_05[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def cliques_not_significant(avg_ranks: pd.Series, cd: float) -> list[tuple[float, float]]:
    """Return [low_rank, high_rank] intervals for maximal non-significant cliques."""
    ordered = avg_ranks.sort_values()
    names = list(ordered.index)
    ranks = ordered.to_numpy(dtype=float)
    k = len(names)
    # adjacency: |rank_i - rank_j| < CD  => not significantly different
    adj = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(k):
            adj[i, j] = abs(ranks[i] - ranks[j]) < cd

    # maximal contiguous cliques on the sorted rank axis (standard CD-diagram bars)
    bars: list[tuple[float, float]] = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and adj[i : j + 2, i : j + 2].all():
            j += 1
        if j > i:
            bars.append((float(ranks[i]), float(ranks[j])))
        i += 1

    # drop bars fully contained in a wider bar
    kept: list[tuple[float, float]] = []
    for lo, hi in bars:
        if any(lo >= a and hi <= b and (lo, hi) != (a, b) for a, b in bars):
            continue
        kept.append((lo, hi))
    return kept


def format_p(p: float) -> str:
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.001:
        return f"p = {p:.1e}"
    return f"p = {p:.4f}"


def plot_cd(
    avg_ranks: pd.Series,
    cd: float,
    title_metric: str,
    friedman_stat: float,
    friedman_df: int,
    friedman_p: float,
    out_path: Path,
) -> None:
    ordered = avg_ranks.sort_values()
    labels = [DISPLAY_NAME.get(g, g) for g in ordered.index]
    ranks = ordered.to_numpy(dtype=float)
    k = len(ranks)
    low, high = float(ranks.min()) - 0.35, float(ranks.max()) + 0.35

    fig_h = 3.6
    fig_w = 8.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(low, high)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # rank axis
    axis_y = 0.82
    ax.plot([low, high], [axis_y, axis_y], color="#222222", lw=1.4, solid_capstyle="round")
    ticks = np.arange(np.floor(low), np.ceil(high) + 0.1, 1.0)
    for t in ticks:
        if low <= t <= high:
            ax.plot([t, t], [axis_y - 0.015, axis_y + 0.015], color="#222222", lw=1.2)
            ax.text(t, axis_y + 0.05, f"{t:g}", ha="center", va="bottom", fontsize=9, color="#222")

    # CD scale bar (left)
    cd_y = 0.93
    cd_x0 = low + 0.05
    ax.plot([cd_x0, cd_x0 + cd], [cd_y, cd_y], color="#222", lw=2.0)
    ax.plot([cd_x0, cd_x0], [cd_y - 0.02, cd_y + 0.02], color="#222", lw=2.0)
    ax.plot([cd_x0 + cd, cd_x0 + cd], [cd_y - 0.02, cd_y + 0.02], color="#222", lw=2.0)
    ax.text(cd_x0 + cd / 2, cd_y + 0.035, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    # points on axis
    cmap = plt.get_cmap("tab10")
    for i, (name, rank) in enumerate(zip(labels, ranks)):
        ax.scatter([rank], [axis_y], s=55, zorder=5, color=cmap(i % 10), edgecolors="#111", linewidths=0.6)

    # alternating left/right labels with stems
    mid = (k + 1) // 2
    left = list(range(0, mid))
    right = list(range(mid, k))
    # left side: best ranks, labels go down-left
    left_ys = np.linspace(0.62, 0.12, len(left)) if left else []
    right_ys = np.linspace(0.62, 0.12, len(right)) if right else []

    for idx, y in zip(left, left_ys):
        rank = ranks[idx]
        name = labels[idx]
        ax.plot([rank, rank], [axis_y, y + 0.03], color="#444", lw=0.9)
        ax.plot([rank, low + 0.08], [y + 0.03, y + 0.03], color="#444", lw=0.9)
        ax.text(
            low + 0.05,
            y + 0.03,
            f"{name} ({rank:.2g})",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="medium",
        )

    for idx, y in zip(right, right_ys):
        rank = ranks[idx]
        name = labels[idx]
        ax.plot([rank, rank], [axis_y, y + 0.03], color="#444", lw=0.9)
        ax.plot([rank, high - 0.08], [y + 0.03, y + 0.03], color="#444", lw=0.9)
        ax.text(
            high - 0.05,
            y + 0.03,
            f"{name} ({rank:.2g})",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="medium",
        )

    # significance bars
    bars = cliques_not_significant(ordered, cd)
    bar_ys = np.linspace(axis_y - 0.08, axis_y - 0.08 - 0.055 * max(0, len(bars) - 1), max(1, len(bars)))
    for (lo, hi), y in zip(bars, bar_ys):
        ax.plot([lo, hi], [y, y], color="#b00020", lw=3.2, solid_capstyle="round", zorder=4)

    subtitle = (
        f"Friedman test: $\\chi^2$ = {friedman_stat:.2f}, df = {friedman_df}, {format_p(friedman_p)}  |  "
        f"Nemenyi post-hoc ($\\alpha$ = 0.05). Generators joined by a red bar are not significantly different."
    )
    ax.set_title(
        f"Critical-Difference Diagram — {title_metric} Utility Gap Ranking\n",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=8.5, color="#333")

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not EXCEL.exists():
        raise SystemExit(f"Missing Excel: {EXCEL}")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"excel={EXCEL}")
    for metric, label in METRICS:
        wide = load_gap_matrix(metric)
        avg = average_ranks(wide)
        chi2, df, p = friedman_test(wide)
        cd = nemenyi_cd(k=wide.shape[1], n=wide.shape[0], alpha=0.05)
        out = OUT / f"cd_{label.lower()}_utility_gap.png"
        plot_cd(avg, cd, label, chi2, df, p, out)
        print(
            f"{label}: N={wide.shape[0]} k={wide.shape[1]} "
            f"chi2={chi2:.2f} df={df} p={p:.4g} CD={cd:.3f}"
        )
        print("  avg ranks:", {DISPLAY_NAME.get(g, g): round(float(v), 3) for g, v in avg.sort_values().items()})
        print(f"  wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
