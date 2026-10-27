#!/usr/bin/env python3
"""Build the static GitHub Pages dashboard into docs/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
STATIC_DIR = REPO_ROOT / "dashboard" / "static"

sys.path.insert(0, str(REPO_ROOT))

from dashboard.export_multiseed_for_web import export_multiseed_dashboard_data  # noqa: E402
from dashboard.sync_explorer_figures import sync_explorer_figures  # noqa: E402


def build_pages(github_repo_url: str | None = None) -> Path:
    """Export multi-seed data + copy static assets to docs/ for GitHub Pages."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Primary source: merged 10-seed Excel (utility / fidelity / privacy Mean±SD)
    try:
        export_multiseed_dashboard_data(DOCS_DIR / "data")
        print(f"Exported multi-seed data to {DOCS_DIR / 'data'}")
    except FileNotFoundError as exc:
        print(f"Warning: multi-seed export skipped ({exc})")
        try:
            from dashboard.export_for_web import export_dashboard_data

            export_dashboard_data(DOCS_DIR / "data")
            print(f"Fallback: exported legacy Results/ data to {DOCS_DIR / 'data'}")
        except Exception as legacy_exc:  # noqa: BLE001
            print(f"Warning: legacy export also failed ({legacy_exc})")

    # Copy static assets
    assets_dst = DOCS_DIR / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    shutil.copytree(STATIC_DIR, assets_dst)

    # Dedicated 10-Seed Results Explorer site
    explorer_src = STATIC_DIR / "explorer"
    explorer_dst = DOCS_DIR / "explorer"
    if explorer_src.is_dir():
        if explorer_dst.exists():
            shutil.rmtree(explorer_dst)
        shutil.copytree(explorer_src, explorer_dst)
        print(f"Published explorer at {explorer_dst} → /explorer/")

    # Analysis diagrams (CD, ranks, trade-offs, seed stability, workflow)
    try:
        figs = sync_explorer_figures(explorer_dst / "figures")
        print(f"Synced analysis figures → {figs}")
    except Exception as fig_exc:  # noqa: BLE001
        print(f"Warning: figure sync failed ({fig_exc})")

    # Copy index.html
    index_src = STATIC_DIR / "index.html"
    index_dst = DOCS_DIR / "index.html"
    html = index_src.read_text(encoding="utf-8")

    if github_repo_url:
        html = html.replace(
            'href="https://github.com/gopibattineni/SYNTH_BENCHMARK"',
            f'href="{github_repo_url}"',
        )

    index_dst.write_text(html, encoding="utf-8")

    # Prevent Jekyll from ignoring JSON/data on GitHub Pages
    (DOCS_DIR / ".nojekyll").touch()

    print(f"Built dashboard at {DOCS_DIR}")
    return DOCS_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GitHub Pages dashboard")
    parser.add_argument("--repo-url", type=str, default="", help="GitHub repository URL for footer link")
    args = parser.parse_args()
    build_pages(args.repo_url or None)
