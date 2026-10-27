"""Sync multi-seed analysis figures into docs/explorer/figures for GitHub Pages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FIGURES = REPO_ROOT / "multi seed generators" / "analysis" / "figures"
EXPLORER_FIGURES = REPO_ROOT / "docs" / "explorer" / "figures"

# Folders to publish (PNG only). Skip tradeoffs/individual to keep the site lean.
PUBLISH_DIRS = [
    "main",
    "workflow",
    "classification",
    "regression",
    "seed_stability",
    "tradeoffs",
]

CATEGORY_LABELS = {
    "main": "Manuscript (main pack)",
    "workflow": "Experimental design",
    "classification": "Classification",
    "regression": "Regression",
    "seed_stability": "Seed stability",
    "tradeoffs": "Trade-offs",
}


def _title_from_name(name: str) -> str:
    stem = Path(name).stem
    # Drop common prefixes
    for prefix in ("Fig_cls_", "Fig_reg_", "Fig_", "classification_", "regression_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.replace("_", " ").replace("-", " ").strip().title()


def _task_hint(path: Path, category: str) -> str | None:
    name = path.name.lower()
    if category in {"classification", "regression"}:
        return category
    if name.startswith("fig_cls") or "classification" in name:
        return "classification"
    if name.startswith("fig_reg") or "regression" in name:
        return "regression"
    return None


def sync_explorer_figures(dest: Path | None = None) -> Path:
    out = Path(dest) if dest else EXPLORER_FIGURES
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    catalog: list[dict] = []
    for category in PUBLISH_DIRS:
        src_dir = ANALYSIS_FIGURES / category
        if not src_dir.is_dir():
            continue
        dst_dir = out / category
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Only top-level PNGs (skip tradeoffs/individual etc.)
        for src in sorted(src_dir.glob("*.png")):
            rel = f"{category}/{src.name}"
            shutil.copy2(src, dst_dir / src.name)
            catalog.append(
                {
                    "id": rel.replace("/", "__").replace(".png", ""),
                    "category": category,
                    "category_label": CATEGORY_LABELS.get(category, category),
                    "file": rel,
                    "title": _title_from_name(src.name),
                    "task": _task_hint(src, category),
                }
            )

    catalog_path = out / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "source": "multi seed generators/analysis/figures",
                "n_figures": len(catalog),
                "categories": [
                    {"id": c, "label": CATEGORY_LABELS.get(c, c)}
                    for c in PUBLISH_DIRS
                    if any(item["category"] == c for item in catalog)
                ],
                "figures": catalog,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    path = sync_explorer_figures()
    print(f"Synced figures → {path}")
    data = json.loads((path / "catalog.json").read_text(encoding="utf-8"))
    print(f"{data['n_figures']} PNGs across {len(data['categories'])} categories")
