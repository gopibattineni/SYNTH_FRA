"""Fix Wine-template leftovers in regression SDV notebooks."""
import json
import re
import sys
from pathlib import Path

TARGET = "Concrete compressive strength"
DATASET_NAME = "Concrete"
EXCEL_SUFFIX = "Concrete"
NOTEBOOK = Path("SDV models/13. Concrete Compressive Strength.ipynb")

REGRESSORS = """from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'SVR-RBF': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'GradientBoost': GradientBoostingRegressor(random_state=42),
    'MLP': MLPRegressor(max_iter=2000, random_state=42)
}
"""

EVALUATE = f"""from sklearn.model_selection import train_test_split
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
        results.append({{
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        }})
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
"""

EXPORT_CELL = f'''# {DATASET_NAME.upper()} EXPORT
import pandas as pd
import numpy as np

DATASET_NAME = "{DATASET_NAME}"
EXCEL_FILENAME = "Hungarian_Mahalanobis_Four_Models_{EXCEL_SUFFIX}.xlsx"
print("Dataset:", DATASET_NAME)
print("Writing Excel file:", EXCEL_FILENAME)

def export_all_models_hungarian_mahalanobis_excel(
    real_df: pd.DataFrame,
    synthetic_data: dict,
    hungarian_results: dict,
    num_cols: list,
    filename: str = EXCEL_FILENAME,
):
    dataset_name = DATASET_NAME
    model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
    summary_rows = []

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for model_name in model_order:
            if model_name not in synthetic_data or model_name not in hungarian_results:
                continue

            synth_df = synthetic_data[model_name].copy()
            _common = [c for c in real_df.columns if c in synth_df.columns]
            synth_df = synth_df[_common]

            row_ind = hungarian_results[model_name]["row_ind"]
            col_ind = hungarian_results[model_name]["col_ind"]

            if "dists" in hungarian_results[model_name]:
                matched_distances = hungarian_results[model_name]["dists"]
            elif "distances" in hungarian_results[model_name]:
                matched_distances = hungarian_results[model_name]["distances"]
            elif "selected_distances" in hungarian_results[model_name]:
                matched_distances = hungarian_results[model_name]["selected_distances"]
            elif "matched_distances" in hungarian_results[model_name]:
                matched_distances = hungarian_results[model_name]["matched_distances"]
            else:
                raise KeyError(f"No distance array found for {{model_name}}")

            records = []
            for r_i, s_i, dist in zip(row_ind, col_ind, matched_distances):
                rec = {{
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Real_Index": int(r_i),
                    "Synthetic_Index": int(s_i),
                    "Real_Label": f"R{{int(r_i)+1}}",
                    "Synthetic_Label": f"S{{int(s_i)+1}}",
                    "Mahalanobis_Distance": float(dist),
                }}
                for c in num_cols:
                    rec[f"Real_{{c}}"] = real_df.iloc[r_i][c]
                    rec[f"Synth_{{c}}"] = synth_df.iloc[s_i][c]
                records.append(rec)

            matched_df = pd.DataFrame(records)
            sheet_name = f"{{dataset_name}}_{{model_name}}"[:31]
            matched_df.to_excel(writer, sheet_name=sheet_name, index=False)

            summary_rows.append({{
                "Dataset": dataset_name,
                "Model": model_name,
                "Num_Matches": len(matched_df),
                "Mean_Distance": float(np.mean(matched_distances)),
                "Median_Distance": float(np.median(matched_distances)),
                "Std_Distance": float(np.std(matched_distances)),
                "Min_Distance": float(np.min(matched_distances)),
                "Max_Distance": float(np.max(matched_distances)),
            }})

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values("Mean_Distance")
            summary_df.to_excel(writer, sheet_name=f"{{dataset_name}}_Summary", index=False)

    print(f"Excel file created: {{filename}}")


real_df = data.copy()
num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "{TARGET}"]

export_all_models_hungarian_mahalanobis_excel(
    real_df=real_df,
    synthetic_data=synthetic_data,
    hungarian_results=hungarian_results,
    num_cols=num_cols,
    filename=EXCEL_FILENAME,
)
'''


def set_src(cell, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    cell["source"] = lines


def patch_text(src: str) -> str:
    t = TARGET
    src = src.replace('target_col = "diagnosis"', f'target_col = "{t}"')
    src = src.replace('target_col = "Diabetes_binary"', f'target_col = "{t}"')
    src = src.replace('label_col = "quality"', f'label_col = "{t}"')
    src = src.replace('drop_from_metrics.append("quality")', "drop_from_metrics.append(target_col)")
    src = src.replace('if "quality" in num_cols:', "if target_col in num_cols:")
    src = src.replace('if c not in ["quality"]', f'if c != "{t}"')
    src = src.replace('if c != "quality"', f'if c != "{t}"')
    src = src.replace('target="quality"', f'target="{t}"')
    src = src.replace("bivariate_quality_wine", "bivariate_quality_regression")
    src = src.replace("label_col=label_col", f'label="{t}"')
    src = src.replace('if c not in ["ID", "Diabetes_binary"]', f'if c != "{t}"')
    src = src.replace(
        'feature_cols = [c for c in feature_cols if c not in ["ID", "Diabetes_binary"]]',
        f'feature_cols = [c for c in feature_cols if c != "{t}"]',
    )
    src = src.replace(
        'num_cols = [c for c in num_cols if c not in ["ID", "Diabetes_binary"]]',
        f'num_cols = [c for c in num_cols if c != "{t}"]',
    )
    src = src.replace("Hungarian_Matchings_All_Models_Heart.xlsx", f"Hungarian_Matchings_All_Models_{EXCEL_SUFFIX}.xlsx")
    src = src.replace("Hungarian_Mahalanobis_Four_Models_WineQuality.xlsx", f"Hungarian_Mahalanobis_Four_Models_{EXCEL_SUFFIX}.xlsx")
    src = src.replace("mahalanobis_winequality_by_model.png", f"mahalanobis_{EXCEL_SUFFIX.lower()}_by_model.png")
    src = src.replace('"WineQuality"', f'"{DATASET_NAME}"')
    src = src.replace("WineQuality", DATASET_NAME)
    src = src.replace('sheet_name="Wine_Summary"', f'sheet_name="{DATASET_NAME}_Summary"')
    src = src.replace('sheet_name = f"Wine_{model_name}"', f'sheet_name = f"{DATASET_NAME}_{{model_name}}"')

    # safe column align (same indent as preceding line in loop bodies)
    src = re.sub(
        r"(\n[ \t]+)synth_df = synth_df\[real_df\.columns\]",
        r"\1_common = [c for c in real_df.columns if c in synth_df.columns]\n\1synth_df = synth_df[_common]",
        src,
    )
    src = re.sub(
        r"synth_train = synthetic_datasets\[(\w+)\]\[data\.columns\]\.copy\(\)",
        r"_common = [c for c in data.columns if c in synthetic_datasets[\1].columns]\n    synth_train = synthetic_datasets[\1][_common].copy()",
        src,
    )
    src = re.sub(
        r"synthetic_datasets\[gen\]\[data\.columns\]",
        r"synthetic_datasets[gen][[c for c in data.columns if c in synthetic_datasets[gen].columns]]",
        src,
    )
    src = src.replace(
        'if "bivariate_quality_regression" in globals():\n    quality_fn = bivariate_quality_regression\nelse:\n    raise NameError("Define `bivariate_quality_wine` before running this cell.")',
        "quality_fn = bivariate_quality_regression",
    )
    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if 'if "quality" in num_cols' in src and "target_col" not in src.split("num_cols")[0][-200:]:
            src = src.replace(
                "real_df = data.copy()\nnum_cols = real_df.select_dtypes",
                f'real_df = data.copy()\ntarget_col = "{TARGET}"\nnum_cols = real_df.select_dtypes',
            )
        patched = patch_text(src)
        if patched != src:
            set_src(cell, patched)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "def evaluate_models" in src and "roc_auc_score" in src:
            set_src(cell, EVALUATE)
        if cell.get("cell_type") == "code" and src.strip().startswith("from sklearn.linear_model import LogisticRegression"):
            set_src(cell, REGRESSORS)
        if cell.get("cell_type") == "code" and "def export_all_models_hungarian_mahalanobis_excel" in src:
            set_src(cell, EXPORT_CELL)
        if cell.get("cell_type") == "code" and "def plot_trtr_vs_tstr" in src and 'metric="AUC"' in src:
            set_src(cell, make_plot_cell())
        if cell.get("cell_type") == "code" and src.startswith("import pandas as pd\n\nlabel_col =") and "AUC_Drop" in src and "plot_trtr_vs_tstr" not in src:
            set_src(cell, make_trtr_cell())
        if cell.get("cell_type") == "code" and "plot_grid_line_metrics_with_gap" in src and "Accuracy" in src:
            src = patch_text(src)
            src = src.replace('["Accuracy", "F1", "AUC"]', '["RMSE", "MAE", "R2"]')
            src = src.replace("per classifier", "per regressor")
            src = src.replace('label_col = "quality"', f'label_col = "{TARGET}"')
            set_src(cell, src)
        if cell.get("cell_type") == "code" and "def bivariate_quality_wine" in src or "def bivariate_quality_regression" in src:
            set_src(cell, bivariate_helper())
        if cell.get("cell_type") == "code" and "LogisticRegression" in src and "privacy_metrics_mia" in src and "def privacy_metrics_mia(" in src:
            set_src(cell, mia_cell())

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    verify()


def make_trtr_cell() -> str:
    return f'''import pandas as pd

label_col = "{TARGET}"
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


def make_plot_cell() -> str:
    return f'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric="RMSE"):
    if metric not in trtr_results.columns or metric not in tstr_results.columns:
        return
    df = trtr_results[["Model", metric]].merge(tstr_results[["Model", metric]], on="Model", suffixes=("_TRTR", "_TSTR"))
    df["Diff"] = df[f"{{metric}}_TSTR"] - df[f"{{metric}}_TRTR"] if metric in ["MAE", "RMSE"] else df[f"{{metric}}_TRTR"] - df[f"{{metric}}_TSTR"]
    df = df.sort_values("Diff", ascending=False)
    x = np.arange(len(df)); w = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, df[f"{{metric}}_TRTR"], w, label="TRTR")
    plt.bar(x + w/2, df[f"{{metric}}_TSTR"], w, label="TSTR")
    plt.xticks(x, df["Model"], rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(f"{{synth_name}}: TRTR vs TSTR ({{metric}})")
    plt.legend(); plt.tight_layout(); plt.show()

label_col = "{TARGET}"
model_order = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula"]
trtr_results = evaluate_models(train_df=data, test_df=data, label=label_col, models=models)
all_comp = []
print("TRTR (Train Real, Test Real)"); display(trtr_results); print("=" * 70)
for synth_name in model_order:
    _common = [c for c in data.columns if c in synthetic_datasets[synth_name].columns]
    synth_train = synthetic_datasets[synth_name][_common].copy()
    tstr_results = evaluate_models(train_df=synth_train, test_df=data, label=label_col, models=models)
    print(f"{{synth_name}} - TSTR"); display(tstr_results)
    comparison = trtr_results.merge(tstr_results, on="Model", suffixes=("_TRTR", "_TSTR"))
    comparison["RMSE_Diff"] = comparison["RMSE_TSTR"] - comparison["RMSE_TRTR"]
    comparison["Synthetic_Model"] = synth_name
    all_comp.append(comparison)
    for metric in ["RMSE", "MAE", "R2"]:
        plot_trtr_vs_tstr(trtr_results, tstr_results, synth_name, metric)
'''


def bivariate_helper() -> str:
    return f'''from itertools import combinations
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def bivariate_quality_regression(real_df, synth_df, target="{TARGET}", id_col=None, n_bins=5):
    num_cols = real_df.select_dtypes(include=["number"]).columns.tolist()
    exclude = {{target, id_col}} - {{None}}
    num_cols = [c for c in num_cols if c not in exclude]
    scaler = MinMaxScaler()
    real_scaled = real_df.copy(); synth_scaled = synth_df.copy()
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
    target_df = pd.DataFrame(results_target).sort_values("delta_wasserstein", ascending=False) if results_target else pd.DataFrame(columns=["feature", "delta_wasserstein"])
    return corr_df, target_df
'''


def mia_cell() -> str:
    return f'''import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve, roc_auc_score

def model_loss_per_sample_regression(model, X, y):
    return (y - model.predict(X)) ** 2

def privacy_metrics_mia_regression(model, X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    loss_train = model_loss_per_sample_regression(model, X_train_s, y_train)
    loss_test = model_loss_per_sample_regression(model, X_test_s, y_test)
    scores = -np.concatenate([loss_train, loss_test])
    labels = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    advantage = np.max(tpr - fpr)
    idx = max(0, np.searchsorted(fpr, 0.01, side="right") - 1)
    tpr_at_fpr_001 = tpr[idx] if idx < len(tpr) else float("nan")
    return {{"AUC": auc, "Advantage": advantage, "TPR_at_FPR_001": tpr_at_fpr_001}}

target_col = "{TARGET}"
feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != target_col]
real_df = data.copy()
synth_df = synthetic_data["GaussianCopula"][real_df.columns].copy()
X_train = real_df[feature_cols].to_numpy(dtype=np.float64)
y_train = real_df[target_col].to_numpy(dtype=np.float64)
X_test = synth_df[feature_cols].to_numpy(dtype=np.float64)
y_test = synth_df[target_col].to_numpy(dtype=np.float64)
model = Ridge(alpha=1.0, random_state=42)
print(privacy_metrics_mia_regression(model, X_train, y_train, X_test, y_test))
'''


def verify() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    raw = NOTEBOOK.read_text(encoding="utf-8")
    bad = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        s = "".join(cell.get("source", []))
        if '"quality"' in s or "label_col = \"quality\"" in s:
            bad.append(f"quality ref cell {i}")
        try:
            compile(s, str(i), "exec")
        except SyntaxError as e:
            bad.append(f"syntax cell {i}: {e}")
    print("WineQuality count:", raw.count("WineQuality"))
    print("quality string count:", raw.count('"quality"'))
    print("issues:", bad or "none")
    if bad:
        sys.exit(1)
    print(f"Patched {NOTEBOOK}")


if __name__ == "__main__":
    main()
