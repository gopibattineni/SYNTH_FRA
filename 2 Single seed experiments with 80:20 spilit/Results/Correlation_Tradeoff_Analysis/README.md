# Correlation Trade-off Analysis

Publication-quality trade-off figures relating **Utility**, **Fidelity**, and **Privacy**
for Cancer and Mushroom across all eight generators.

## Primary figures (use these in the paper)

**One scatter point per generator** (8 points).

Each point is the mean of the metric across:
- both datasets (Cancer, Mushroom)
- all classifiers (utility)
- all 10 random seeds (utility reconstructed from Mean±SD)

| Axis | Metric |
|------|--------|
| Utility | Mean TSTR Accuracy (and F1 variant) |
| Fidelity | Mean SDMetrics Quality Score |
| Privacy A | Mean MIA AUC (lower = more private) |
| Privacy B | Mean NNDR (higher = more private) |

Figures 1–3: Fidelity–Utility, Privacy–Utility, Fidelity–Privacy.  
Each point is labelled with the generator name and metric values.  
Statistics: Pearson *r*, Spearman *ρ*, R², *p*, OLS fit, 95% CI.

## Supplementary figures

Seed-level scatters (Generator × Seed × Dataset) with Cancer = circle and
Mushroom = square, overall + per-dataset dashed fits — for variability / robustness.

## Layout

```
Correlation_Tradeoff_Analysis/
  Figures/Primary/              # flat copies of primary figs
  Figures/Supplementary/        # flat copies of seed-level figs
  Utility_Accuracy/Analysis_A_MIA/
    Primary/Figures/
    Supplementary/Figures/
    Statistics/
    Summaries/
  … (F1 × MIA/NNDR likewise)
  Processed_Data/
  FINDINGS.md
```

Mirrored to `Results/Two_Datasets_Assessment/Correlation_Tradeoff_Analysis/`.

## Run

```bash
python run_correlation_tradeoff_analysis.py
```
