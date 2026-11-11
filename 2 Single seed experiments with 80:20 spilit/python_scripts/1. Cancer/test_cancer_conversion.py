#!/usr/bin/env python3
"""
Verify cancer.py matches cancer.ipynb code cells exactly.

Run:
    python test_cancer_conversion.py
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = SCRIPT_DIR.parent.parent / "1. Cancer"
NOTEBOOK_PATH = NOTEBOOK_DIR / "cancer.ipynb"
PY_PATH = SCRIPT_DIR / "cancer.py"

CELL_MARKER = re.compile(
    r"# ={72}\n# Cell (\d+)\n# ={72}\n(.*?)(?=\n\n# ={72}\n# Cell |\Z)",
    re.DOTALL,
)
BOOTSTRAP_END = "# END RUNTIME BOOTSTRAP"


def load_notebook_code_cells() -> list[tuple[int, str]]:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells: list[tuple[int, str]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            cells.append((i, "".join(cell.get("source", [])).rstrip()))
    return cells


def load_py_notebook_cells() -> list[tuple[int, str]]:
    text = PY_PATH.read_text(encoding="utf-8")
    if BOOTSTRAP_END in text:
        text = text.split(BOOTSTRAP_END, 1)[1]
    cells = [(int(i), s.rstrip()) for i, s in CELL_MARKER.findall(text)]
    return cells


class CancerConversionTests(unittest.TestCase):
    def test_paths_exist(self) -> None:
        self.assertTrue(NOTEBOOK_PATH.is_file(), f"Missing notebook: {NOTEBOOK_PATH}")
        self.assertTrue(PY_PATH.is_file(), f"Missing script: {PY_PATH}")

    def test_cell_count(self) -> None:
        nb_cells = load_notebook_code_cells()
        py_cells = load_py_notebook_cells()
        self.assertEqual(
            len(nb_cells),
            len(py_cells),
            f"Cell count mismatch: notebook={len(nb_cells)} py={len(py_cells)}",
        )

    def test_cell_0_equivalent(self) -> None:
        """Cell 0: notebook uses !git clone; script uses bootstrap + same imports."""
        nb_cells = load_notebook_code_cells()
        py_cells = load_py_notebook_cells()
        nb_lines = [ln for ln in nb_cells[0][1].splitlines() if not ln.strip().startswith("!")]
        py_lines = [
            ln for ln in py_cells[0][1].splitlines()
            if not ln.strip().startswith("# Notebook")
        ]
        self.assertEqual(nb_lines, py_lines)

    def _normalize_cli_cell(self, src: str) -> str:
        """Normalize bootstrap CLI differences in cell 1 vs notebook."""
        lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("# CLI overrides") or stripped.startswith(
                "# Experiment Settings (N_SAMPLES"
            ):
                continue
            if stripped in {
                "N_SAMPLES = 1000",
                "TEST_SIZE = 0.2",
                "SEED = 42",
            }:
                continue
            line = line.replace("fetch_ucirepo(id=UCI_ID)", "fetch_ucirepo(id=17)")
            line = line.replace('data_path = TRAIN_CSV', 'data_path = "breast_cancer_train.csv"')
            line = line.replace(
                'output_file = OUTPUT_FILE',
                'output_file = "TRTR_TSTR_results.xlsx"',
            )
            lines.append(line)
        return "\n".join(lines)

    def test_cells_1_to_end_exact_match(self) -> None:
        nb_cells = load_notebook_code_cells()
        py_cells = load_py_notebook_cells()
        for (nb_idx, nb_src), (py_idx, py_src) in zip(nb_cells[1:], py_cells[1:]):
            with self.subTest(cell=nb_idx):
                self.assertEqual(nb_idx, py_idx)
                nb_cmp = nb_src
                py_cmp = py_src
                if nb_idx == 1:
                    py_cmp = self._normalize_cli_cell(py_src)
                self.assertEqual(
                    nb_cmp,
                    py_cmp,
                    f"Cell {nb_idx} differs from notebook.\n"
                    f"Notebook hash: {hashlib.md5(nb_cmp.encode()).hexdigest()}\n"
                    f"Python hash:   {hashlib.md5(py_cmp.encode()).hexdigest()}",
                )

    def test_python_syntax(self) -> None:
        source = PY_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(PY_PATH))

    def test_no_jupyter_shell_magic_in_executable_cells(self) -> None:
        """Notebook cell 0 uses !git clone; bootstrap must handle clone instead."""
        py_cells = load_py_notebook_cells()
        for line in py_cells[0][1].splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            self.assertFalse(
                stripped.startswith("!") or stripped.startswith("%"),
                f"Jupyter magic in executable line: {line!r}",
            )

    def test_display_shim_present(self) -> None:
        source = PY_PATH.read_text(encoding="utf-8")
        self.assertIn("def display(", source)
        self.assertIn(BOOTSTRAP_END, source)

    def test_notebook_dir_resolution(self) -> None:
        source = PY_PATH.read_text(encoding="utf-8")
        self.assertIn("NOTEBOOK_DIR", source)
        self.assertTrue(NOTEBOOK_DIR.is_dir())


if __name__ == "__main__":
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Script:   {PY_PATH}")
    result = unittest.main(verbosity=2)
