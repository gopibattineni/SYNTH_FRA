"""Fix broken TRTR plot cells (SyntaxError + AUC leftovers) in GAN notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REG_TARGETS = [
    ("12. Air Quality", "CO(GT)"),
    ("13. Concrete", "Concrete compressive strength"),
    ("14. Energy", "Y1"),
    ("15. Real Estate", "Y house price of unit area"),
    ("6. Wine", "quality"),
    ("Winequality", "quality"),
    ("8. Metro", "traffic_volume"),
    ("10. Online", "price"),
    ("10. online", "price"),
]


def match_target(path: Path) -> str | None:
    s = path.name.lower()
    for frag, target in REG_TARGETS:
        if frag.lower().replace(" ", "") in s.replace(" ", ""):
            return target
    return None


def gan_kind(path: Path) -> str | None:
    if "Other GANS" in path.as_posix():
        return "other"
    if "Diffusion GANs" in path.as_posix():
        return "diffusion"
    return None


def plot_cell(target: str, kind: str) -> str:
    if kind == "other":
        models_list = '["CTABGAN", "WGAN_GP"]'
        synth_loop = """    synth_df = synthetic_outputs[synth_name].copy()
    _common = [c for c in data.columns if c in synth_df.columns]
    synth_train = synth_df[_common]"""
    else:
        models_list = '["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]'
        synth_loop = """    synth_df = synthetic_outputs[synth_name].copy()
    _common = [c for c in data.columns if c in synth_df.columns]
    synth_train = synth_df[_common]"""

    return f'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric="RMSE"):
    if metric not in trtr_results.columns or metric not in tstr_results.columns:
        return
    df = trtr_results[["Model", metric]].merge(
        tstr_results[["Model", metric]], on="Model", suffixes=("_TRTR", "_TSTR")
    )
    df["Diff"] = (
        df[f"{{metric}}_TSTR"] - df[f"{{metric}}_TRTR"]
        if metric in ["MAE", "RMSE"]
        else df[f"{{metric}}_TRTR"] - df[f"{{metric}}_TSTR"]
    )
    df = df.sort_values("Diff", ascending=False)
    x = np.arange(len(df))
    w = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, df[f"{{metric}}_TRTR"], w, label="TRTR")
    plt.bar(x + w / 2, df[f"{{metric}}_TSTR"], w, label="TSTR")
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{{synth_name}}: TRTR vs TSTR ({{metric}})")
    plt.legend()
    plt.tight_layout()
    plt.show()

label_col = "{target}"
model_order = {models_list}

trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)

all_comp = []
for synth_name in model_order:
{synth_loop}
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label=label_col, models=models)
    print(f"{{synth_name}} - TSTR (Train Synthetic, Test Real)")
    display(tstr_results)
    comparison = trtr_results.merge(tstr_results, on="Model", suffixes=("_TRTR", "_TSTR"))
    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["MAE_Diff"] = comparison["MAE_TSTR"] - comparison["MAE_TRTR"]
    comparison["R2_Drop"] = comparison["R2_TRTR"] - comparison["R2_TSTR"]
    comparison["Synthetic_Model"] = synth_name
    print(f"{{synth_name}} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)
    all_comp.append(comparison)
    for metric in ["RMSE", "MAE", "R2"]:
        plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric)

combined_comparison = pd.concat(all_comp, ignore_index=True)
summary = (
    combined_comparison.groupby("Synthetic_Model", as_index=False)["RMSE_Diff"]
    .mean()
    .sort_values("RMSE_Diff")
)
print("Average RMSE increase by synthetic generator (lower is better):")
display(summary)
'''


def grid_cell(target: str, kind: str) -> str:
    if kind == "other":
        models_list = '["CTABGAN", "WGAN_GP"]'
    else:
        models_list = '["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]'

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
model_order = {models_list}
trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
for synth_name in model_order:
    synth_df = synthetic_outputs[synth_name].copy()
    _common = [c for c in data.columns if c in synth_df.columns]
    tstr_results = evaluate_models(
        train_df=synth_df[_common],
        test_df=data,
        label=label_col,
        models=models,
    )
    plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name=synth_name, ncols=5)
'''


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def needs_plot_fix(src: str) -> bool:
    return any(
        x in src
        for x in [
            "LabelEncoder removed for regression",
            "AUC_TRTR",
            'for metric in ["AUC", "F1", "Accuracy"]',
            "all_possible_labels",
            "synthetic_datasets[gen]",
        ]
    )


def fix_notebook(path: Path) -> bool:
    target = match_target(path)
    kind = gan_kind(path)
    if not target or not kind:
        return False

    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def plot_grid_line_metrics_with_gap" in src and needs_plot_fix(src):
            set_src(cell, grid_cell(target, kind))
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        elif "def plot_trtr_vs_tstr" in src and needs_plot_fix(src):
            set_src(cell, plot_cell(target, kind))
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True

    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = []
    for folder in [ROOT / "Other GANS", ROOT / "Diffusion GANs"]:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.ipynb")):
            if fix_notebook(p):
                fixed.append(p.relative_to(ROOT).as_posix())
    print(f"Fixed {len(fixed)} notebooks")
    for f in fixed:
        print(" ", f)


if __name__ == "__main__":
    main()
