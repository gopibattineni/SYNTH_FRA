"""Fix broken indentation from _common column-alignment patch."""
import json
import re
from pathlib import Path

NOTEBOOK = Path("SDV models/12. Air Quality.ipynb")
TARGET = "CO(GT)"


def fix_broken_common_indent(src: str) -> str:
    # Repair two-line block where synth_df line lost loop indentation.
    return re.sub(
        r"(\n[ \t]+_common = \[c for c in real_df\.columns if c in synth_df\.columns\])\n[ \t]{4}synth_df = synth_df\[_common\]",
        r"\1\n            synth_df = synth_df[_common]",
        src,
    )


def fix_cell61(src: str) -> str:
    if "export_all_models_hungarian_mahalanobis_excel" not in src:
        return src
    return f'''import pandas as pd
import numpy as np

def export_all_models_hungarian_mahalanobis_excel(
    real_df: pd.DataFrame,
    synthetic_data: dict,
    hungarian_results: dict,
    num_cols: list,
    filename: str = "Hungarian_Mahalanobis_Four_Models_AirQuality.xlsx"
):
    dataset_name = "AirQuality"
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
    filename="Hungarian_Mahalanobis_Four_Models_AirQuality.xlsx"
)
'''


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    fixed = []

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src

        if "export_all_models_hungarian_mahalanobis_excel" in src and "def export_all_models" in src:
            src = fix_cell61(src)
        else:
            src = fix_broken_common_indent(src)

        if src != orig:
            lines = src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            fixed.append(i)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Fixed cells:", fixed)


if __name__ == "__main__":
    main()
