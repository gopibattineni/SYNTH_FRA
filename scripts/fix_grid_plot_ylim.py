"""Fix plot_grid_line_metrics_with_gap y-axis stuck at 0-1 for regression metrics."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGRESSION_TARGETS = {
    "12. Air Quality": "CO(GT)",
    "13. Concrete": "Concrete compressive strength",
    "14. Energy": "Y1",
    "15. Real Estate": "Y house price of unit area",
    "6. Wine": "quality",
    "Wine quality": "quality",
    "8. Metro": "traffic_volume",
    "10. Online": "price",
}


def match_target(path: Path) -> str | None:
    name = path.name
    for frag, target in REGRESSION_TARGETS.items():
        if frag.lower().replace(" ", "") in name.lower().replace(" ", ""):
            return target
    return None


def grid_plot_cell(target: str) -> str:
    return f'''import numpy as np
import matplotlib.pyplot as plt

def _metric_delta(metric, trtr_val, tstr_val):
    if metric in ("RMSE", "MAE"):
        return tstr_val - trtr_val
    if metric == "R2":
        return trtr_val - tstr_val
    return trtr_val - tstr_val

def plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name, ncols=5):
    metrics = [m for m in ["RMSE", "MAE", "R2"] if m in trtr_results.columns and m in tstr_results.columns]
    if not metrics:
        metrics = [m for m in ["Accuracy", "F1", "AUC"] if m in trtr_results.columns and m in tstr_results.columns]
    if not metrics:
        print(f"No common metrics found for {{generator_name}}.")
        return

    x = np.arange(len(metrics))
    trtr = trtr_results.set_index("Model")[metrics]
    tstr = tstr_results.set_index("Model")[metrics]
    models_local = [m for m in trtr.index if m in tstr.index]

    n = len(models_local)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, max(9, 3 * nrows)), dpi=150, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, model_name in enumerate(models_local):
        ax = axes[i]
        a = trtr.loc[model_name].values.astype(float)
        b = tstr.loc[model_name].values.astype(float)
        deltas = [_metric_delta(m, ai, bi) for m, ai, bi in zip(metrics, a, b)]

        ax.plot(x, a, marker="o", label="TRTR")
        ax.plot(x, b, marker="s", label="TSTR")
        ax.fill_between(x, np.minimum(a, b), np.maximum(a, b), alpha=0.2)

        lo = float(np.nanmin(np.concatenate([a, b])))
        hi = float(np.nanmax(np.concatenate([a, b])))
        pad = max((hi - lo) * 0.15, 0.05) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)

        for xi, ai, bi, di in zip(x, a, b, deltas):
            ax.text(xi, (ai + bi) / 2, f"{{di:+.3f}}", ha="center", va="center", fontsize=8)

        ax.set_title(model_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        if i % ncols == 0:
            ax.set_ylabel("Score")
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{{generator_name}} - TRTR vs TSTR per regressor", fontsize=14)
    plt.show()

label_col = "{target}"
generators = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
for gen in generators:
    _common = [c for c in data.columns if c in synthetic_datasets[gen].columns]
    tstr_results = evaluate_models(
        train_df=synthetic_datasets[gen][_common],
        test_df=data,
        label=label_col,
        models=models,
    )
    plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name=gen, ncols=5)
'''


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def patch_ylim_only(src: str) -> str:
    if "def plot_grid_line_metrics_with_gap" not in src or "set_ylim(0, 1.05)" not in src:
        return src
    block = """        lo = float(np.nanmin(np.concatenate([a, b])))
        hi = float(np.nanmax(np.concatenate([a, b])))
        pad = max((hi - lo) * 0.15, 0.05) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)"""
    return src.replace("        ax.set_ylim(0, 1.05)", block)


def fix_notebook(path: Path, target: str | None) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def plot_grid_line_metrics_with_gap" not in src:
            continue
        if target and "RMSE" in src and "for gen in generators" in src:
            new_src = grid_plot_cell(target)
        else:
            new_src = patch_ylim_only(src)
        if new_src != src:
            set_src(cell, new_src)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    folders = [ROOT / "SDV models", ROOT / "Other GANS", ROOT / "Diffusion GANs"]
    fixed = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in p.as_posix():
                continue
            target = match_target(p)
            raw = p.read_text(encoding="utf-8")
            if "plot_grid_line_metrics_with_gap" not in raw:
                continue
            if "set_ylim(0, 1.05)" not in raw and not (target and folder.name == "SDV models"):
                continue
            if fix_notebook(p, target):
                fixed.append(p.relative_to(ROOT).as_posix())
    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)


if __name__ == "__main__":
    main()
