"""Fix evaluate_models() calls using label= instead of label_col= in Diffusion GANs notebooks."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Diffusion GANs"


def fix_evaluate_models_calls(src: str) -> str:
    token = "evaluate_models("
    if token not in src:
        return src
    out: list[str] = []
    i = 0
    while True:
        idx = src.find(token, i)
        if idx == -1:
            out.append(src[i:])
            break
        out.append(src[i:idx])
        j = idx + len(token)
        depth = 1
        while j < len(src) and depth:
            ch = src[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        call = src[idx:j]
        call = call.replace("label=", "label_col=")
        # Only rewrite literal string defaults in calls, not function signatures.
        if not call.lstrip().startswith("def evaluate_models"):
            call = re.sub(r'label_col=(["\']).*?\1', "label_col=label_col", call)
        out.append(call)
        i = j
    return "".join(out)


def fix_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = fix_evaluate_models_calls(src)
        if new_src != src:
            lines = new_src.splitlines(keepends=True)
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    fixed = [p.name for p in sorted(FOLDER.glob("*.ipynb")) if fix_notebook(p)]
    print(f"Fixed {len(fixed)} notebooks")
    for name in fixed:
        print(f"  {name}")


if __name__ == "__main__":
    main()
