"""Clear stale WineQuality outputs from SDV 14/15 export cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ["14. Energy Efficiency.ipynb", "15. Real Estate Valuation.ipynb"]:
    p = ROOT / "SDV models" / name
    nb = json.loads(p.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if "WineQuality" in text:
                cell["outputs"] = []
                cell["execution_count"] = None
                break
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Cleared stale outputs:", p.name)
