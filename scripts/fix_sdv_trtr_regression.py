"""Fix TRTR/TSTR cells that still reference AUC/Accuracy/F1 in regression SDV notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    (ROOT / "SDV models" / "12. Air Quality.ipynb", "CO(GT)"),
    (ROOT / "SDV models" / "13. Concrete Compressive Strength.ipynb", "Concrete compressive strength"),
    (ROOT / "SDV models" / "14. Energy Efficiency.ipynb", "Y1"),
    (ROOT / "SDV models" / "15. Real Estate Valuation.ipynb", "Y house price of unit area"),
    (ROOT / "SDV models" / "6. Wine quality.ipynb", "quality"),
    (ROOT / "SDV models" / "8. Metro Interstate Traffic Volume.ipynb", "traffic_volume"),
    (ROOT / "SDV models" / "10. Online Shopping.ipynb", "price"),
]


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def trtr_cell(target: str) -> str:
    return f'''import pandas as pd

label_col = "{target}"
model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]

trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)

all_comparisons = []
for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label=label_col, models=models)
    print(f"{{synth_name}} - TSTR (Train Synthetic, Test Real)")
    display(tstr_results)
    comparison = trtr_results.merge(tstr_results, on="Model", suffixes=("_TRTR", "_TSTR"))
    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["MAE_Diff"] = comparison["MAE_TSTR"] - comparison["MAE_TRTR"]
    comparison["R2_Drop"] = comparison["R2_TRTR"] - comparison["R2_TSTR"]
    comparison["Synthetic_Model"] = synth_name
    comparison = comparison.sort_values("RMSE_Diff", ascending=False)
    print(f"{{synth_name}} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)
    all_comparisons.append(comparison)

combined_comparison = pd.concat(all_comparisons, ignore_index=True)
summary = combined_comparison.groupby("Synthetic_Model", as_index=False)["RMSE_Diff"].mean().sort_values("RMSE_Diff")
print("Average RMSE increase by synthetic generator (lower is better):")
display(summary)
'''


def plot_cell(target: str) -> str:
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
    plt.bar(x - w/2, df[f"{{metric}}_TRTR"], w, label="TRTR")
    plt.bar(x + w/2, df[f"{{metric}}_TSTR"], w, label="TSTR")
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{{synth_name}}: TRTR vs TSTR ({{metric}})")
    plt.legend()
    plt.tight_layout()
    plt.show()

label_col = "{target}"
model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)
for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label=label_col, models=models)
    print(f"{{synth_name}} - TSTR")
    display(tstr_results)
    comparison = trtr_results.merge(tstr_results, on="Model", suffixes=("_TRTR", "_TSTR"))
    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["Synthetic_Model"] = synth_name
    print(f"{{synth_name}} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)
    for metric in ["RMSE", "MAE", "R2"]:
        plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric)
'''


def grid_plot_cell() -> str:
    return '''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_grid_line_metrics_with_gap(trtr_results, tstr_results, generator_name, ncols=5):
    metrics = ["RMSE", "MAE", "R2"]
    x = np.arange(len(metrics))
    models = trtr_results["Model"].tolist()
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        trtr_row = trtr_results.loc[trtr_results["Model"] == model, metrics]
        tstr_row = tstr_results.loc[tstr_results["Model"] == model, metrics]
        if trtr_row.empty or tstr_row.empty:
            ax.set_visible(False)
            continue
        trtr_vals = trtr_row.iloc[0].to_numpy(dtype=float)
        tstr_vals = tstr_row.iloc[0].to_numpy(dtype=float)
        ax.plot(x, trtr_vals, marker="o", label="TRTR")
        ax.plot(x, tstr_vals, marker="s", label="TSTR")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(model)
        ax.legend(fontsize=8)
    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(f"{generator_name} - TRTR vs TSTR per regressor")
    plt.tight_layout()
    plt.show()

model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label=label_col, models=models)
    plot_grid_line_metrics_with_gap(trtr_results, tstr_results, synth_name)
'''


def fix_notebook(path: Path, target: str) -> bool:
    if not path.exists():
        return False
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if (
            src.startswith("import pandas as pd\n\nlabel_col =")
            and "AUC_Drop" in src
            and "plot_trtr_vs_tstr" not in src
        ):
            set_src(cell, trtr_cell(target))
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        elif "def plot_trtr_vs_tstr" in src and ('metric="AUC"' in src or "AUC_Drop" in src):
            set_src(cell, plot_cell(target))
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        elif "def plot_grid_line_metrics_with_gap" in src and "Accuracy" in src:
            set_src(cell, grid_plot_cell())
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
        elif "AUC_TRTR" in src or "AUC_Drop" in src:
            patched = src
            patched = re.sub(
                r'comparison\["AUC_Drop"\][^\n]+',
                'comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]',
                patched,
            )
            patched = patched.replace("AUC_Drop", "RMSE_Diff")
            patched = patched.replace("AUC_TRTR", "RMSE_TRTR")
            patched = patched.replace("AUC_TSTR", "RMSE_TSTR")
            patched = patched.replace("Average AUC drop", "Average RMSE increase")
            patched = patched.replace('metric="AUC"', 'metric="RMSE"')
            if patched != src:
                set_src(cell, patched)
                cell["outputs"] = []
                cell["execution_count"] = None
                changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main():
    fixed = []
    for path, target in NOTEBOOKS:
        if fix_notebook(path, target):
            fixed.append(path.name)
    print("Fixed:", fixed or "none")


if __name__ == "__main__":
    main()
