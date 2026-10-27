"""Clear stale Wine/Heart outputs and enforce Air Quality export names."""
import json
from pathlib import Path

NOTEBOOK = Path("SDV models/12. Air Quality.ipynb")
MAHAL_FILE = "Hungarian_Mahalanobis_Four_Models_AirQuality.xlsx"
MATCH_FILE = "Hungarian_Matchings_All_Models_AirQuality.xlsx"


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "Hungarian_Matchings_All_Models" in src and "output_file" in src:
            cell["source"] = [
                f'output_file = "{MATCH_FILE}"\n' if line.startswith("output_file = ") else line
                for line in cell["source"]
            ]
            cell["outputs"] = []
            print(f"Cleared outputs cell {i} (matchings export)")

        if "export_all_models_hungarian_mahalanobis_excel" in src:
            new_src = []
            for line in cell["source"]:
                if "Hungarian_Mahalanobis_Four_Models_" in line and "filename" in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    if 'filename: str = "' in line:
                        new_src.append(f'{indent}filename: str = "{MAHAL_FILE}"\n')
                    elif 'filename="' in line:
                        new_src.append(f'{indent}filename="{MAHAL_FILE}"\n')
                    else:
                        new_src.append(line)
                else:
                    new_src.append(line)
            cell["source"] = new_src
            cell["outputs"] = []
            print(f"Cleared outputs cell {i} (mahalanobis export)")

        # Drop any leftover cached stdout with old names
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                if "WineQuality" in text or "Heart.xlsx" in text:
                    cell["outputs"] = []
                    break

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
