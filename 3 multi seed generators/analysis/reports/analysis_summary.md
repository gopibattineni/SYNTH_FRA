# Analysis Summary — 10-Seed Evaluation

## Design

- Datasets: **15**
- Generators: **8**
- Classification datasets: **9**
- Regression datasets: **6**
- Unique seeds: **10** → `[42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]`
- Expected configurations: **120**
- Expected generator runs: **1200**

### Batches (execution provenance only)

- **1st batch**: seeds = [42, 123, 2024]; expected = 360
- **2nd batch**: seeds = [68, 91, 2025]; expected = 360
- **3rd batch**: seeds = [55, 155, 255, 355]; expected = 480

## Observed completion

- Completed runs: **1200** / 1200
- Classification: **720** / 720
- Regression: **480** / 480

Batch,Seeds,Expected Runs,Completed Runs,Status
1st batch,"42, 123, 2024",360,360,OK
2nd batch,"68, 91, 2025",360,360,OK
3rd batch,"55, 155, 255, 355",480,480,OK
Total,10 seeds,1200,1200,OK


## Aggregation rule

Primary manuscript results use **Mean ± SD across the 10 individual seed observations** (sample SD, `ddof=1`). Batch means are **not** averaged.

## Key outputs

- Figures: `figures/` (workflow, classification, regression, seed_stability, tradeoffs, dataset_level)
- Manuscript packs: `figures/main/`, `figures/supplementary/`
- Tables: `tables/main`, `tables/supplementary`, `tables/batch_validation`
- Data: `data/seed_level_long.csv`, `data/agg_10seed_mean_sd.csv`
- Notebook: `notebooks/01_ten_seed_analysis.ipynb`
- Mapping: `ANALYSIS_MAPPING.md`
- Validation: `reports/data_validation_report.md`
- Captions: `reports/figure_table_captions.md`

## Statistical analysis (utility gaps)

- Unit of analysis: dataset-level **10-seed mean** gap
- Sample size: N = 9 (classification) or N = 6 (regression) datasets; K = 8 generators
- Omnibus: Friedman test (α = 0.05)
- Post-hoc: Nemenyi (CD diagrams) + Holm-corrected Wilcoxon
- Effect display: average ranks + critical-difference cliques + rank tables / robustness charts

## Regeneration

```bash
cd "/home/gopi_b/SYNTH_BENCHMARK/multi seed generators/analysis"
../../.venv/bin/python run_all.py
```
