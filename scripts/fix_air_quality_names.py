"""Rename Wine/Heart template leftovers in Air Quality SDV notebook."""
import json
from pathlib import Path

NOTEBOOK = Path("SDV models/12. Air Quality.ipynb")
TARGET = "CO(GT)"

REPLACEMENTS = [
    ("Hungarian_Mahalanobis_Four_Models_WineQuality.xlsx", "Hungarian_Mahalanobis_Four_Models_AirQuality.xlsx"),
    ("Hungarian_Matchings_All_Models_Heart.xlsx", "Hungarian_Matchings_All_Models_AirQuality.xlsx"),
    ("mahalanobis_winequality_by_model.png", "mahalanobis_airquality_by_model.png"),
    ('"WineQuality"', '"AirQuality"'),
    ("WineQuality", "AirQuality"),
    ('"Wine_', '"AirQuality_'),
    ("Wine_", "AirQuality_"),
    ('target_col = "Diabetes_binary"', f'target_col = "{TARGET}"'),
    ('if c not in ["ID", "Diabetes_binary"]', f'if c not in ["ID", "{TARGET}"]'),
    ('if c not in ["ID", "Diabetes_binary", target_col]', f'if c not in ["ID", "{TARGET}"]'),
    ('feature_cols if c not in ["ID", "Diabetes_binary"]', f'feature_cols if c not in ["ID", "{TARGET}"]'),
]

# More specific patterns for list comprehensions
LIST_PATTERNS = [
    (
        'feature_cols = [c for c in feature_cols if c not in ["ID", "Diabetes_binary"]]',
        f'feature_cols = [c for c in feature_cols if c not in ["ID", "{TARGET}"]]',
    ),
    (
        'num_cols = [c for c in num_cols if c not in ["ID", "Diabetes_binary"]]',
        f'num_cols = [c for c in num_cols if c != "{TARGET}"]',
    ),
]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    changed = []

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src
        for old, new in REPLACEMENTS:
            src = src.replace(old, new)
        for old, new in LIST_PATTERNS:
            src = src.replace(old, new)

        if src != orig:
            lines = src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            changed.append(i)

        # Clear stale stdout that still mentions old filenames
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "WineQuality" in out.get("text", ""):
                out["text"] = out["text"].replace(
                    "Hungarian_Mahalanobis_Four_Models_WineQuality.xlsx",
                    "Hungarian_Mahalanobis_Four_Models_AirQuality.xlsx",
                )
            if out.get("output_type") == "stream" and "Heart.xlsx" in out.get("text", ""):
                out["text"] = out["text"].replace(
                    "Hungarian_Matchings_All_Models_Heart.xlsx",
                    "Hungarian_Matchings_All_Models_AirQuality.xlsx",
                )

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Updated cells:", changed)


if __name__ == "__main__":
    main()
