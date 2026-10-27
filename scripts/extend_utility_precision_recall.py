#!/usr/bin/env python3
"""Extend Adult/Alzheimer generator notebooks to emit Precision & Recall utility metrics.

Patches evaluate_models() result rows and TRTR/TSTR comparison cells so kernels
produce Accuracy, Precision, Recall, and F1 (plus drops / summary means).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Generators/SDV models/2. Alzhimers.ipynb",
    ROOT / "Generators/SDV models/3. Adult census.ipynb",
    ROOT / "Generators/Other GANS/2. Alzhimers_other GAN.ipynb",
    ROOT / "Generators/Other GANS/3. Adult census_other GAN.ipynb",
    ROOT / "Generators/Diffusion GANs/2. Alzhimers_diffusion.ipynb",
    ROOT / "Generators/Diffusion GANs/3. Adult census_diffusion.ipynb",
]

NAN_KEYS = '"Accuracy": np.nan, "F1": np.nan, "AUC": np.nan'
NAN_KEYS_NEW = (
    '"Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, '
    '"F1": np.nan, "AUC": np.nan'
)

IMPORT_OLD = "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score"
IMPORT_NEW = (
    "from sklearn.metrics import accuracy_score, precision_score, "
    "recall_score, f1_score, roc_auc_score"
)

DROP_BLOCK_OLD = '''    if "F1_TRTR" in comparison.columns and "F1_TSTR" in comparison.columns:
        comparison["F1_Drop"] = comparison["F1_TRTR"] - comparison["F1_TSTR"]
    if "Accuracy_TRTR" in comparison.columns and "Accuracy_TSTR" in comparison.columns:
        comparison["Accuracy_Drop"] = comparison["Accuracy_TRTR"] - comparison["Accuracy_TSTR"]'''

DROP_BLOCK_NEW = '''    if "F1_TRTR" in comparison.columns and "F1_TSTR" in comparison.columns:
        comparison["F1_Drop"] = comparison["F1_TRTR"] - comparison["F1_TSTR"]
    if "Accuracy_TRTR" in comparison.columns and "Accuracy_TSTR" in comparison.columns:
        comparison["Accuracy_Drop"] = comparison["Accuracy_TRTR"] - comparison["Accuracy_TSTR"]
    if "Precision_TRTR" in comparison.columns and "Precision_TSTR" in comparison.columns:
        comparison["Precision_Drop"] = comparison["Precision_TRTR"] - comparison["Precision_TSTR"]
    if "Recall_TRTR" in comparison.columns and "Recall_TSTR" in comparison.columns:
        comparison["Recall_Drop"] = comparison["Recall_TRTR"] - comparison["Recall_TSTR"]'''

# Adult SDV uses slightly different indentation / if-blocks without always both
DROP_BLOCK_OLD_ALT = '''    if "F1_TRTR" in comparison.columns and "F1_TSTR" in comparison.columns:
        comparison["F1_Drop"] = comparison["F1_TRTR"] - comparison["F1_TSTR"]

    if "Accuracy_TRTR" in comparison.columns and "Accuracy_TSTR" in comparison.columns:
        comparison["Accuracy_Drop"] = comparison["Accuracy_TRTR"] - comparison["Accuracy_TSTR"]'''


SUMMARY_OLD_PATTERNS = [
    # common multiline AUC-only summary
    '''summary = (
    combined_comparison
    .groupby("Synthetic_Model", as_index=False)["AUC_Drop"]
    .mean()
    .sort_values("AUC_Drop")
)

print("Average AUC drop by synthetic generator (lower is better):")
display(summary)''',
    '''summary = (combined_comparison
             .groupby("Synthetic_Model", as_index=False)["AUC_Drop"]
             .mean()
             .sort_values("AUC_Drop"))

print("Average AUC drop by synthetic generator (lower is better):")
display(summary)''',
]

SUMMARY_NEW = '''_drop_cols = [
    c for c in [
        "Accuracy_Drop", "Precision_Drop", "Recall_Drop", "F1_Drop", "AUC_Drop"
    ]
    if c in combined_comparison.columns
]
summary = (
    combined_comparison
    .groupby("Synthetic_Model", as_index=False)[_drop_cols]
    .mean()
    .sort_values("Accuracy_Drop" if "Accuracy_Drop" in _drop_cols else _drop_cols[0])
)

print("Average utility drops by synthetic generator (lower is better):")
display(summary)'''


def _patch_metric_computation(src: str) -> str:
    """Insert Precision/Recall next to existing Acc/F1/AUC computation blocks."""
    if "precision_score" in src and '"Precision"' in src:
        return src

    # Pattern A: try/except f1 + auc (Adult SDV)
    pat_a = re.compile(
        r'(?P<acc>acc = accuracy_score\(y_test, y_pred\)\n)'
        r'(?P<f1>        try:\n'
        r'            f1 = f1_score\(y_test, y_pred\)\n'
        r'        except ValueError:\n'
        r'            f1 = float\("nan"\)\n)'
        r'(?P<auc>        try:\n'
        r'            auc = roc_auc_score\(y_test, y_prob\) if y_prob is not None else float\("nan"\)\n'
        r'        except ValueError:\n'
        r'            auc = float\("nan"\)\n\n)'
        r'(?P<row>        rows\.append\(\{"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc\}\))',
        re.M,
    )
    repl_a = (
        r"\g<acc>"
        r"        try:\n"
        r'            prec = precision_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r"        except ValueError:\n"
        r'            prec = float("nan")\n'
        r"        try:\n"
        r'            rec = recall_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r"        except ValueError:\n"
        r'            rec = float("nan")\n'
        r"\g<f1>\g<auc>"
        r'        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, '
        r'"Recall": rec, "F1": f1, "AUC": auc})'
    )
    src2, n = pat_a.subn(repl_a, src)
    if n:
        return src2

    # Pattern B: Adult Other (f1 with average=binary)
    pat_b = re.compile(
        r'(?P<acc>            acc = accuracy_score\(y_test, y_pred\)\n)'
        r'(?P<f1>            f1 = f1_score\(y_test, y_pred, average="binary", zero_division=0\)\n)'
        r'(?P<auc>            auc = roc_auc_score\(y_test, y_prob\) if y_prob is not None else float\("nan"\)\n\n)'
        r'(?P<row>            rows\.append\(\{"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc\}\))',
        re.M,
    )
    repl_b = (
        r"\g<acc>"
        r'            prec = precision_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r'            rec = recall_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r"\g<f1>\g<auc>"
        r'            rows.append({"Model": name, "Accuracy": acc, "Precision": prec, '
        r'"Recall": rec, "F1": f1, "AUC": auc})'
    )
    src2, n = pat_b.subn(repl_b, src)
    if n:
        return src2

    # Pattern C: Alzheimer-style (no try, plain f1_score)
    pat_c = re.compile(
        r'(?P<acc>        acc = accuracy_score\(y_test, y_pred\)\n)'
        r'(?P<f1>        f1 = f1_score\(y_test, y_pred\)\n)'
        r'(?P<auc>        auc = roc_auc_score\(y_test, y_prob\) if y_prob is not None else float\("nan"\)\n\n)'
        r'(?P<row>        rows\.append\(\{"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc\}\))',
        re.M,
    )
    repl_c = (
        r"\g<acc>"
        r'        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r'        rec = recall_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r'        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)\n'
        r"\g<auc>"
        r'        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, '
        r'"Recall": rec, "F1": f1, "AUC": auc})'
    )
    src2, n = pat_c.subn(repl_c, src)
    if n:
        return src2

    # Pattern D: Adult Diffusion plain (same indent as C but may already match C)
    return src


def _set_source(cell: dict, src: str) -> None:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if not lines and src == "":
        lines = []
    cell["source"] = lines


def patch_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    stats = {"path": str(path), "cells": 0, "evaluate": 0, "comparison": 0, "summary": 0}

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        orig = src

        if IMPORT_OLD in src and "precision_score" not in src:
            src = src.replace(IMPORT_OLD, IMPORT_NEW)

        if NAN_KEYS in src and '"Precision": np.nan' not in src:
            src = src.replace(NAN_KEYS, NAN_KEYS_NEW)

        if "def evaluate_models" in src or "_nan_results" in src or "accuracy_score(y_test" in src:
            before = src
            src = _patch_metric_computation(src)
            if src != before:
                stats["evaluate"] += 1

        if "Accuracy_Drop" in src and "comparison" in src:
            if "Precision_Drop" not in src:
                if DROP_BLOCK_OLD_ALT in src:
                    src = src.replace(DROP_BLOCK_OLD_ALT, DROP_BLOCK_NEW)
                    stats["comparison"] += 1
                elif DROP_BLOCK_OLD in src:
                    src = src.replace(DROP_BLOCK_OLD, DROP_BLOCK_NEW)
                    stats["comparison"] += 1

            for old in SUMMARY_OLD_PATTERNS:
                if old in src:
                    src = src.replace(old, SUMMARY_NEW)
                    stats["summary"] += 1
                    break
            else:
                # looser summary replace
                m = re.search(
                    r'summary\s*=\s*\(?\s*combined_comparison\s*'
                    r'\.?groupby\("Synthetic_Model", as_index=False\)\["AUC_Drop"\]'
                    r'.*?display\(summary\)',
                    src,
                    re.S,
                )
                if m and "Precision_Drop" not in m.group(0):
                    src = src[: m.start()] + SUMMARY_NEW + src[m.end() :]
                    stats["summary"] += 1

        if src != orig:
            _set_source(cell, src)
            # Keep existing outputs; user can re-run utility cells to refresh.
            stats["cells"] += 1

    if stats["cells"]:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    for nb_path in NOTEBOOKS:
        if not nb_path.exists():
            print(f"MISSING {nb_path}")
            continue
        stats = patch_notebook(nb_path)
        print(
            f"{nb_path.name}: cells={stats['cells']} evaluate={stats['evaluate']} "
            f"comparison={stats['comparison']} summary={stats['summary']}"
        )


if __name__ == "__main__":
    main()
