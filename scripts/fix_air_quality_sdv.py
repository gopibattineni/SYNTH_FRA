"""Fix Wine-template leftovers in SDV models/12. Air Quality.ipynb."""
import json
from pathlib import Path

NOTEBOOK = Path("SDV models/12. Air Quality.ipynb")
TARGET = "CO(GT)"

METRO_EVALUATE = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def evaluate_models(train_df, test_df, label, models, test_size=0.2, seed=42):
    X_train = train_df.drop(columns=[label]).copy()
    y_train = train_df[label].copy()

    X_test = test_df.drop(columns=[label]).copy()
    y_test = test_df[label].copy()

    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=test_size, random_state=seed
    )
    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=test_size, random_state=seed
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns

    X_train = X_train[num_cols].astype(np.float64)
    X_test = X_test[num_cols].astype(np.float64)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds)
        })

    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
"""

METRO_TRTR = """import pandas as pd

label_col = "CO(GT)"
model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]

trtr_results = evaluate_models(
    train_df=data,
    test_df=data,
    label=label_col,
    models=models
)

print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)

all_comparisons = []

for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()

    tstr_results = evaluate_models(
        train_df=synth_train,
        test_df=data,
        label=label_col,
        models=models
    )

    print(f"{synth_name} - TSTR (Train Synthetic, Test Real)")
    display(tstr_results)

    comparison = trtr_results.merge(
        tstr_results, on="Model", suffixes=("_TRTR", "_TSTR")
    )

    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["MAE_Diff"] = comparison["MAE_TSTR"] - comparison["MAE_TRTR"]
    comparison["R2_Drop"] = comparison["R2_TRTR"] - comparison["R2_TSTR"]

    comparison["Synthetic_Model"] = synth_name
    comparison = comparison.sort_values("RMSE_Diff", ascending=False)

    print(f"{synth_name} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)

    all_comparisons.append(comparison)

combined_comparison = pd.concat(all_comparisons, ignore_index=True)

summary = (combined_comparison
          .groupby("Synthetic_Model", as_index=False)["RMSE_Diff"]
          .mean()
          .sort_values("RMSE_Diff"))

print("Average RMSE increase by synthetic generator (lower is better):")
display(summary)
"""

METRO_PLOT = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric="RMSE"):
    if metric not in trtr_results.columns or metric not in tstr_results.columns:
        return

    df = trtr_results[["Model", metric]].merge(
        tstr_results[["Model", metric]],
        on="Model",
        suffixes=("_TRTR", "_TSTR")
    )

    if metric in ["MAE", "RMSE"]:
        df["Diff"] = df[f"{metric}_TSTR"] - df[f"{metric}_TRTR"]
    else:
        df["Diff"] = df[f"{metric}_TRTR"] - df[f"{metric}_TSTR"]

    df = df.sort_values("Diff", ascending=False)

    x = np.arange(len(df))
    w = 0.38

    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, df[f"{metric}_TRTR"], w, label="TRTR")
    plt.bar(x + w/2, df[f"{metric}_TSTR"], w, label="TSTR")
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{synth_name}: TRTR vs TSTR ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_metric_diff(comparison_df, synth_name, metric):
    if metric not in comparison_df.columns:
        return

    df = comparison_df.sort_values(metric, ascending=False)
    x = np.arange(len(df))

    plt.figure(figsize=(12, 5))
    plt.bar(x, df[metric])
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{synth_name}: {metric}")
    plt.tight_layout()
    plt.show()

label_col = "CO(GT)"
model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]

trtr_results = evaluate_models(
    train_df=data,
    test_df=data,
    label=label_col,
    models=models
)

all_comp = []

print("TRTR (Train Real, Test Real)")
display(trtr_results)
print("=" * 70)

for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()

    tstr_results = evaluate_models(
        train_df=synth_train,
        test_df=data,
        label=label_col,
        models=models
    )

    print(f"{synth_name} - TSTR (Train Synthetic, Test Real)")
    display(tstr_results)

    comparison = trtr_results.merge(
        tstr_results, on="Model", suffixes=("_TRTR", "_TSTR")
    )

    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["MAE_Diff"] = comparison["MAE_TSTR"] - comparison["MAE_TRTR"]
    comparison["R2_Drop"] = comparison["R2_TRTR"] - comparison["R2_TSTR"]
    comparison["Synthetic_Model"] = synth_name
    comparison = comparison.sort_values("RMSE_Diff", ascending=False)

    print(f"{synth_name} - TRTR vs TSTR Comparison")
    display(comparison)
    print("=" * 70)

    all_comp.append(comparison)

    for metric in ["RMSE", "MAE", "R2"]:
        plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric)

combined_comparison = pd.concat(all_comp, ignore_index=True)

summary_rmse = (
    combined_comparison.dropna(subset=["RMSE_Diff"])
    .groupby("Synthetic_Model", as_index=False)["RMSE_Diff"]
    .mean()
    .sort_values("RMSE_Diff")
)

print("Average RMSE increase by synthetic generator (lower is better):")
display(summary_rmse)
"""


def set_cell_source(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def patch_source(src: str) -> str:
    src = src.replace('target_col = "diagnosis"', f'target_col = "{TARGET}"')
    src = src.replace('label_col = "quality"', f'label_col = "{TARGET}"')
    src = src.replace('drop_from_metrics.append("quality")', "drop_from_metrics.append(target_col)")
    src = src.replace('if "quality" in num_cols:', "if target_col in num_cols:")
    src = src.replace('if c not in ["quality"]', f'if c != "{TARGET}"')
    src = src.replace('if c != "quality"', f'if c != "{TARGET}"')
    src = src.replace('target="quality"', f'target="{TARGET}"')
    src = src.replace("bivariate_quality_wine", "bivariate_quality_regression")
    src = src.replace(
        "synth_df = synth_df[real_df.columns]",
        "_common = [c for c in real_df.columns if c in synth_df.columns]\n    synth_df = synth_df[_common]",
    )
    src = src.replace(
        "synth_train = synthetic_datasets[synth_name][data.columns].copy()",
        "_common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]\n    synth_train = synthetic_datasets[synth_name][_common].copy()",
    )
    src = src.replace("label_col=label_col", f"label={TARGET}")
    src = src.replace(
        'if "bivariate_quality_wine" in globals():\n    quality_fn = bivariate_quality_wine\nelse:\n    raise NameError("Define `bivariate_quality_wine` before running this cell.")',
        "quality_fn = bivariate_quality_regression",
    )
    src = src.replace(
        'if "bivariate_quality_regression" in globals():\n    quality_fn = bivariate_quality_regression\nelse:\n    raise NameError("Define `bivariate_quality_regression` before running this cell.")',
        "quality_fn = bivariate_quality_regression",
    )
    src = src.replace(f"{TARGET} class:", f"{TARGET} quantile bins:")
    src = src.replace('excluding quality', f'excluding {TARGET}')
    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "target_col" not in src and 'if "quality" in num_cols' in src:
            src = src.replace(
                "real_df = data.copy()\nnum_cols = real_df.select_dtypes",
                f'real_df = data.copy()\ntarget_col = "{TARGET}"\nnum_cols = real_df.select_dtypes',
            )
        patched = patch_source(src)
        if patched != src:
            set_cell_source(cell, patched)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "def evaluate_models" in src and "roc_auc_score" in src:
            set_cell_source(cell, METRO_EVALUATE)
            print(f"Replaced classification evaluate_models in cell {i}")
        if cell.get("cell_type") == "code" and src.startswith("import pandas as pd\n\nlabel_col ="):
            if "AUC_Drop" in src and "plot_trtr_vs_tstr" not in src:
                set_cell_source(cell, METRO_TRTR)
                print(f"Replaced TRTR/TSTR cell {i}")
        if cell.get("cell_type") == "code" and "def plot_trtr_vs_tstr" in src and "metric=\"AUC\"" in src:
            set_cell_source(cell, METRO_PLOT)
            print(f"Replaced plot TRTR/TSTR cell {i}")

    # bivariate cell: regression uses quantile bins, not groupby classes
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "def bivariate_quality_regression" in src or "def bivariate_quality_wine" in src:
            set_cell_source(
                cell,
                f'''from itertools import combinations
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def bivariate_quality_regression(real_df, synth_df, target="{TARGET}", id_col=None, n_bins=5):
    num_cols = real_df.select_dtypes(include=["number"]).columns.tolist()
    exclude = {{target, id_col}} - {{None}}
    num_cols = [c for c in num_cols if c not in exclude]

    scaler = MinMaxScaler()
    real_scaled = real_df.copy()
    synth_scaled = synth_df.copy()
    real_scaled[num_cols] = scaler.fit_transform(real_df[num_cols])
    synth_scaled[num_cols] = scaler.transform(synth_df[num_cols])

    results_corr = []
    for a, b in combinations(num_cols, 2):
        r_corr = real_scaled[[a, b]].corr().iloc[0, 1]
        s_corr = synth_scaled[[a, b]].corr().iloc[0, 1]
        results_corr.append({{"var_a": a, "var_b": b, "delta_corr": abs(r_corr - s_corr)}})

    results_target = []
    if target in real_scaled.columns and target in synth_scaled.columns:
        real_bins = pd.qcut(real_scaled[target], q=n_bins, duplicates="drop")
        synth_bins = pd.qcut(synth_scaled[target], q=n_bins, duplicates="drop")
        for col in num_cols:
            r_groups = [g[col].values for _, g in real_scaled.groupby(real_bins)]
            s_groups = [g[col].values for _, g in synth_scaled.groupby(synth_bins)]
            if len(r_groups) >= 2 and len(s_groups) >= 2:
                d_real = np.mean([stats.wasserstein_distance(r_groups[j], r_groups[j + 1]) for j in range(len(r_groups) - 1)])
                d_synth = np.mean([stats.wasserstein_distance(s_groups[j], s_groups[j + 1]) for j in range(len(s_groups) - 1)])
                results_target.append({{"feature": col, "delta_wasserstein": abs(d_real - d_synth)}})

    corr_df = pd.DataFrame(results_corr).sort_values("delta_corr", ascending=False)
    target_df = (
        pd.DataFrame(results_target).sort_values("delta_wasserstein", ascending=False)
        if results_target else pd.DataFrame(columns=["feature", "delta_wasserstein"])
    )
    return corr_df, target_df
''',
            )
            print(f"Replaced bivariate helper in cell {i}")

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "plot_grid_line_metrics_with_gap" in src and "Accuracy" in src:
            src = src.replace('["Accuracy", "F1", "AUC"]', '["RMSE", "MAE", "R2"]')
            src = src.replace("per classifier", "per regressor")
            src = src.replace(
                "drop (TRTR - TSTR)",
                "change (positive = worse for RMSE/MAE, negative = worse for R2)",
            )
            src = src.replace('label_col = "quality"', f'label_col = "{TARGET}"')
            src = src.replace("label_col=label_col", f"label={TARGET}")
            src = src.replace(
                "synth_df = synthetic_datasets[gen][data.columns].copy()",
                "_common = [c for c in data.columns if c in synthetic_datasets[gen].columns]\n    synth_df = synthetic_datasets[gen][_common].copy()",
            )
            set_cell_source(cell, src)
            print(f"Fixed plot grid cell {i}")

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NOTEBOOK}")


if __name__ == "__main__":
    main()
