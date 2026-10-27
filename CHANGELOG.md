# Changelog

All notable changes to the SYNTH Benchmark repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **10-seed multi-batch evaluation** (`multi seed generators/`)
  - Batches: seeds 42/123/2024, 68/91/2025, 55/155/255/355 → **1,200** generator runs
  - Manuscript analysis suite: `multi seed generators/analysis/` (CD diagrams, heatmaps, trade-offs, seed stability, Mean±SD tables)
- **Webapp 10-seed results explorer** (`webapp/`)
  - Loads merged Excel Mean±SD for Utility / Fidelity / Privacy / Compute
  - Rankings, heatmaps, and wide tables; optional interactive generation tab
- **GitHub Pages dashboard** rebuilt from merged 10-seed Excel (`dashboard/export_multiseed_for_web.py` → `docs/`)
- **Dedicated 10-Seed Results Explorer** at `/explorer/` (`docs/explorer/`) with analysis diagrams (CD, ranks, trade-offs, seed stability)
- **Correlation trade-off analysis** (`analysis/correlation_tradeoff/`, `run_correlation_tradeoff_analysis.py`)
  - Primary figures: one point per generator (mean over Cancer & Mushroom, classifiers, seeds)
  - Supplementary figures: seed-level scatters (Cancer ○ / Mushroom □)
  - Pearson / Spearman / R² / *p*, OLS + 95% CI; MIA and NNDR privacy variants
  - Output: `Results/Correlation_Tradeoff_Analysis/`
- **Utility drop analysis** (`analysis/utility_drop/`, `run_utility_drop_analysis.py`)
- **Representative metrics trade-off** (`analysis/representative_tradeoff/`, `run_representative_tradeoff.py`)
- **Dashboard Statistics tab** — Average error by model (Mean / Median / Std Error %) for 8 generators × 15 datasets
- Curated `excel sheets/` fidelity / privacy / utility extracts

### Changed
- README documents the 10-seed protocol, merged Excel paths, and webapp quick start
- Two-datasets comparative figures with labelled trade-off scatters
- Dashboard fidelity/privacy filters and Adult Gower coverage for all 8 generators
- Notebook loader prefers live notebooks over `*.BACKUP*` copies

### Fixed
- Adult SDV Gower distance missing for CTGAN / CopulaGAN / TVAE / GaussianCopula in the dashboard
- Utility merge for TabDDPM / ForestDiffusion on regression datasets

---

## [2026-07-09] — Initial benchmark release

### Added
- 15-dataset benchmark across 5 generator folder families
- TRTR/TSTR utility protocol with 10 seeds × 10 downstream models
- Full fidelity + privacy audit notebooks (SDV, Other GANs, Diffusion)
- Leak-safe 80/20 train/test split with N=1000 subsampling
- Excel export pipeline (`TRTR_TSTR_results*.xlsx`)
- Workflow diagram generator (`figures/generate_forge_paper_workflow.py`)
- Dataset registry (`python_scripts/hive/datasets.json`)
- Regression datasets #10–#15 support
