# Final results — Greedy vs Hungarian Mahalanobis mapping

Focused publication set: **5 figures + 3 tables**.

## Scope
- **9 classification** + **6 regression** datasets (9+6 with paired data)
- Metric: mean matched-pair **Mahalanobis distance** (lower is better)
- Methods: **One-to-One Greedy** vs **Hungarian**
- Only generator×dataset units where **both** methods exist (n=120 units)
- Dataset-level means average over paired generators for that dataset

## Improvement definition
```
Improvement(%) = (D_Greedy − D_Hungarian) / D_Greedy × 100
  > 0  → Hungarian better
  < 0  → Greedy better
```

## Figure style (journal / LaTeX)
- Times New Roman (serif) typography throughout
- Okabe–Ito colorblind-safe palette; hatch patterns for B&W print
- Panel letters `(a)/(b)` where applicable; 600 dpi PNG + PDF/SVG
- Clean spines (no top/right), light grid, Nature-style layout widths (~7 in double column)

## Figures
| File | Description |
|------|-------------|
| `figures/Fig01_overall_comparison` | Grouped bars, Classification vs Regression panels (log y) |
| `figures/Fig02_dataset_heatmap` | Dataset × {Greedy, Hungarian} absolute distances (log colour) |
| `figures/Fig03_hungarian_improvement` | Diverging improvement (%) for all 15 datasets |
| `figures/Fig04_distance_distributions` | Violin/box/strip of log₁₀ distances by method × task |
| `figures/Fig05_improvement_vs_features` | Improvement vs #features (size ∝ samples) |

## Tables
| File | Description |
|------|-------------|
| `tables/Table1_dataset_characteristics.csv` | Samples, features, classes, task |
| `tables/Table2_main_quantitative_results.csv` | Greedy, Hungarian, Improvement %, better method |
| `tables/Table3_statistical_significance.csv` | Wilcoxon signed-rank + rank-biserial |
| `tables/final_results_tables.xlsx` | All tables + detail sheets |

## Headline
- Datasets where Hungarian improves mean distance: **93%** (14/15)
- Source summary: `/home/gopi_b/SYNTH_FRAMEWORK_NEW/mapping study graphs/extracted_results/hungarian_vs_greedy_summary.csv`
