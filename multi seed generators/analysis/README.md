# 10-Seed Analysis

Manuscript-ready figures, tables, and reports for the **multi-seed generators** evaluation (15 datasets × 8 generators × 10 seeds = **1,200 runs**).

This folder **replicates** the scientific structure and visual language of `/home/gopi_b/SYNTH_BENCHMARK/Agreed analysis` without modifying that directory.

## Design

| Item | Value |
| --- | ---: |
| Datasets | 15 (9 classification + 6 regression) |
| Generators | 8 |
| Seeds | 10 |
| Configurations | 120 |
| Runs | 1,200 |

Seeds (execution batches):

- **1st batch:** 42, 123, 2024 → 360 runs  
- **2nd batch:** 68, 91, 2025 → 360 runs  
- **3rd batch:** 55, 155, 255, 355 → 480 runs  

Batches are **provenance only**. Primary results pool all **10 individual seed observations** (Mean ± SD, `ddof=1`).

## Quick start

```bash
cd "/home/gopi_b/SYNTH_BENCHMARK/multi seed generators/analysis"
../../.venv/bin/python run_all.py
```

Individual steps:

```bash
../../.venv/bin/python -c "from src.validate import run_validation; print(run_validation())"
../../.venv/bin/python run_workflow_diagrams.py
../../.venv/bin/python run_tables.py
../../.venv/bin/python run_gap_cd_analysis.py
../../.venv/bin/python run_rank_tables.py
../../.venv/bin/python run_heatmaps.py
../../.venv/bin/python run_seed_stability.py
../../.venv/bin/python run_tradeoffs.py
../../.venv/bin/python run_pack_manuscript_figures.py
../../.venv/bin/python write_analysis_summary.py
```

## Layout

```text
analysis/
├── ANALYSIS_MAPPING.md
├── README.md
├── data/                      # seed_level_long.csv, agg_10seed_mean_sd.csv
├── figures/
│   ├── workflow/
│   ├── classification/
│   ├── regression/
│   ├── seed_stability/
│   ├── tradeoffs/
│   ├── dataset_level/
│   ├── main/                  # curated manuscript pack
│   └── supplementary/         # curated supplementary pack
├── tables/
│   ├── main/
│   ├── batch_validation/
│   └── supplementary/
├── notebooks/
│   └── 01_ten_seed_analysis.ipynb
├── reports/
└── src/
```

## Sources

Raw metrics are read from:

```text
multi seed generators/{1st batch,second batch,third batch}/
  {classification,regression}/results/raw/seed_*/<dataset>/<generator>/metrics.xlsx
```

See `ANALYSIS_MAPPING.md` for the mapping from Agreed analysis outputs to this remake.
