"""Fix diffusion_dataleak notebooks: label_col calls, SVM warnings, clear stale outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Single run_Data_leak_Synth_Quality" / "diffusion_dataleak"

LABEL_KW = re.compile(r"(?<!pos_)(?<!label_)label=")
SVC_PROB = re.compile(
    r"SVC\(kernel=['\"]rbf['\"],\s*probability=True,\s*random_state=42\)"
)
WARN_SNIPPET = (
    "import warnings\n"
    'warnings.filterwarnings("ignore", category=FutureWarning)\n'
    'warnings.filterwarnings("ignore", category=UserWarning)\n'
)


def fix_evaluate_models_calls(src: str) -> str:
    if "evaluate_models(" not in src:
        return src
    token = "evaluate_models("
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
        if not call.lstrip().startswith("def evaluate_models"):
            call = LABEL_KW.sub("label_col=", call)
        out.append(call)
        i = j
    return "".join(out)


def fix_svm_model(src: str) -> str:
    if "probability=True" not in src:
        return src
    new = SVC_PROB.sub("LinearSVC(max_iter=2000, dual='auto', random_state=42)", src)
    if "LinearSVC" in new and "from sklearn.svm import SVC\n" in new:
        new = new.replace(
            "from sklearn.svm import SVC\n",
            "from sklearn.svm import LinearSVC\n",
        )
    if "LinearSVC" in new and "from sklearn.svm import SVC," in new:
        new = new.replace("from sklearn.svm import SVC,", "from sklearn.svm import LinearSVC,")
    elif "LinearSVC" in new and "from sklearn.svm import SVC" in new and "LinearSVC" not in new.split("from sklearn.svm import")[1].split("\n")[0]:
        new = new.replace(
            "from sklearn.svm import SVC",
            "from sklearn.svm import SVC, LinearSVC",
        )
    return new


def ensure_warnings_filter(src: str) -> str:
    if "warnings.filterwarnings" in src:
        return src
    if "import pandas" not in src and "from ucimlrepo" not in src:
        return src
    if "from ucimlrepo" in src:
        return src.replace("from ucimlrepo import", WARN_SNIPPET + "from ucimlrepo import", 1)
    if "import pandas as pd" in src:
        return src.replace("import pandas as pd", WARN_SNIPPET + "import pandas as pd", 1)
    return src


def to_source_lines(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines if lines else [""]


def fix_notebook(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changes: list[str] = []
    any_output_cleared = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs"):
            cell["outputs"] = []
            cell["execution_count"] = None
            any_output_cleared = True

        src = "".join(cell.get("source", []))
        new_src = src
        new_src = fix_evaluate_models_calls(new_src)
        new_src = fix_svm_model(new_src)
        new_src = ensure_warnings_filter(new_src)

        if new_src != src:
            if "label_col=" in new_src and "label=" in src and "label_col=" not in src:
                changes.append("label_col")
            if "LinearSVC" in new_src and "probability=True" in src:
                changes.append("svm")
            if "warnings.filterwarnings" in new_src and "warnings.filterwarnings" not in src:
                changes.append("warnings")
            cell["source"] = to_source_lines(new_src)

    if any_output_cleared:
        changes.append("cleared_outputs")

    if changes:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return changes


def main() -> None:
    fixed = 0
    for path in sorted(FOLDER.rglob("*.ipynb")):
        changes = fix_notebook(path)
        if changes:
            fixed += 1
            print(f"{path.relative_to(FOLDER)}: {', '.join(sorted(set(changes)))}")
    print(f"\nUpdated {fixed} notebook(s)")


if __name__ == "__main__":
    main()
