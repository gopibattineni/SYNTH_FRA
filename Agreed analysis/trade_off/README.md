# Per-dataset trade-off analysis

Fidelity / privacy vs utility trade-offs **for each dataset independently**.

## Constraints

- **Never** average or combine results across datasets.
- The **only** averaging is across the 10 downstream classifiers (classification)
  or 10 regression models (regression) **within the same dataset**.
- Each scatter has one point per generator (≤ 8).

## Metrics

| Axis | Classification | Regression |
|------|----------------|------------|
| Fidelity (X) | SDMetrics Overall Quality Score | same |
| Privacy (X) | MIA AUC | same |
| Utility Gap (Y) | TRTR Acc − Mean TSTR Acc | TRTR R² − Mean TSTR R² |

## Run

```bash
python trade_off/run_tradeoff_analysis.py
```

## Outputs (per dataset folder)

```
trade_off/<Dataset>/
  fidelity_vs_accuracy_gap.{png,pdf}   # classification
  mia_vs_accuracy_gap.{png,pdf}        # classification
  fidelity_vs_r2_gap.{png,pdf}         # regression
  mia_vs_r2_gap.{png,pdf}              # regression
  <slug>_*.{png,pdf}                   # dataset-prefixed copies
  summary_table.csv
  statistical_summary.csv
  generator_metrics_raw.csv
```

## Forest plots (per-task summary)

Combined dual-panel forest plots of per-dataset OLS slopes ± 95% CI:

```bash
python trade_off/make_forest_plot_classification.py
python trade_off/make_forest_plot_regression.py
```

Writes `trade_off/forest_plot_classification.{png,pdf}` and
`trade_off/forest_plot_regression.{png,pdf}` from
`all_datasets_statistical_summary.csv`.

Figures are saved at **600 DPI** (PNG + PDF). Generator colours are consistent
across all plots.

## Data sources

`Results/MasterData/{utility,fidelity,privacy}_long.csv` (LeakageLevel = 0).
