# ANALYSIS_MAPPING — Agreed analysis → 10-seed remake

| Existing output (`Agreed analysis`) | Purpose | New output (`multi seed generators/analysis`) | 10-seed modification |
| --- | --- | --- | --- |
| `classification/*_gap_critical_difference_diagram.*` | Nemenyi CD diagrams for Acc/Prec/Rec/F1 gaps | `figures/classification/{metric}_critical_difference_diagram.*` | Ranks use **10-seed mean** gap per dataset×generator |
| `classification/*_gap_analysis.xlsx` + friedman/nemenyi/holm CSVs | Omnibus + pairwise stats | `tables/main/classification_gaps/*` | Same tests; input = 10-seed means |
| `classification/*_gap_violin.*` | Gap distributions | `figures/seed_stability/classification_*_seed_violin.*` | Points = **seed-level** values (n=10×datasets), not pre-averaged dataset means only |
| `classification/accuracy_gap_rank_table.*` / `utility_gap_rank_table.*` | Rank summaries | `figures/classification/accuracy_gap_rank_table.*`, `utility_gap_rank_table.*` | Average/median/SD ranks across 9 datasets using **10-seed means** |
| `classification/friedman_nemenyi_summary_table.*` | OverallScore composite | *(deferred / optional)* | Rebuild if OverallScore composite is redefined on 10-seed data |
| `classification/generator_robustness_average_rank.*` | Mean±SD rank across datasets | `figures/classification/generator_robustness_average_rank.*` (+ regression R² analogue) | SD is across datasets (ranks of 10-seed means) |
| `classification/dataset_summary_table.*` | Dataset metadata | *(reuse Agreed table; unchanged)* | Split/seed design unchanged |
| `classification/experimental_pipeline_workflow.*` | Methods workflow figure | `figures/workflow/experimental_workflow.*` + `experimental_design.*` | Explicit 3 batches → 10 seeds; 1,200 runs |
| `regression/{r2,rmse,mae}_gap_critical_difference_diagram.*` | Regression CD diagrams | `figures/regression/{metric}_critical_difference_diagram.*` | 10-seed means; N=6 datasets |
| `regression/*_gap_by_dataset_heatmap.*` | Gap heatmaps | `figures/regression/*_by_dataset_heatmap.*` | Values = 10-seed means |
| `regression/*_gap_analysis.xlsx` | Regression stats Excel | `tables/main/regression_gaps/*` | Same methodology |
| `tradeoff/figures/*/Fig1–Fig5.*` | Multi-panel Fidelity/Utility/Privacy | `figures/tradeoffs/{task}_Fig*.*` | Points = 10-seed mean; error bars = seed SD |
| `trade_off/Individual dataset/*` | Per-dataset OLS scatters | `figures/tradeoffs/individual/*` | 10-seed mean ± SD per generator |
| `trade_off/forest_plot_*.*` | OLS slope forest | *(optional follow-up)* | Can be regenerated from individual stats if needed |
| Mapping cost CSVs | Separate mapping study | Not remade here | Out of scope for 10-seed remake |
| — | Seed stability (new) | `figures/seed_stability/*`, `tables/supplementary/*_seed_stability.xlsx` | Dedicated n=10 seed variability |
| — | Batch provenance (new) | `tables/batch_validation/*`, `data/seed_level_long.csv` (`batch` column) | 1st/2nd/3rd batch labels retained |
| — | Complete 10-seed table (new) | `tables/supplementary/{task}_all_10_seeds.xlsx` | All seed columns + mean/SD |
| — | Manuscript figure packs (new) | `figures/main/`, `figures/supplementary/` | Curated copies with stable manuscript filenames |
| — | Analysis notebook (new) | `notebooks/01_ten_seed_analysis.ipynb` | Validation + aggregation walkthrough |

## Aggregation rule (both pipelines)

Primary result for every dataset × generator × metric:

```text
10 seed-level observations → mean, sample SD (ddof=1) → Mean ± SD
```

Never: mean(batch1 mean, batch2 mean, batch3 mean).
