# Mapping study graphs — main paper (5–6 figures) + supplementary

This folder contains visualizations extracted from already-completed
mapping experiments. No mapping experiments were rerun.

## Main paper figures

| Figure | Folder | File stem |
|--------|--------|-----------|
| Figure 1 — Overall Cosine Similarity | `main_paper/Figure_1_Overall_Cosine/` | `Figure_1_Overall_Cosine` |
| Figure 2 — Overall Mahalanobis Distance | `main_paper/Figure_2_Overall_Mahalanobis/` | `Figure_2_Overall_Mahalanobis` |
| Figure 3 — Mapping Method Comparison | `main_paper/Figure_3_Mapping_Method_Comparison/` | `Figure_3_Mapping_Method_Comparison` |
| Figure 4 — Generator Robustness | `main_paper/Figure_4_Generator_Robustness/` | `Figure_4_Generator_Robustness` |
| Figure 5 — Dataset-Level Heatmap | `main_paper/Figure_5_Dataset_Heatmap/` | `Figure_5_Dataset_Heatmap` |
| Figure 6 — Distribution / Robustness | `main_paper/Figure_6_Distribution_Robustness/` | `Figure_6_Distribution_Robustness` |

Each folder contains PNG (300 DPI), SVG (vector), and `AGGREGATION.txt`.
Typography: Times New Roman / Times-compatible serif (`font.family=serif`).

## Aggregation (shared definition)

Primary analysis unit for main figures:

```text
(Dataset × Generator × Mapping_Method × Metric) mean
```

extracted from existing notebook/Excel/curated CSV results.
Main figures summarise the distribution of these units; they do not
recompute mappings.

## Data availability (important)

From extracted sources:

- Hungarian × Cosine: 110 units
- Hungarian × Mahalanobis: 70 units
- One-to-One Greedy × Mahalanobis: 72 units
- Greedy (any metric): 0 units

Greedy many-to-one cosine was computed in notebooks via `argmax` but
full aggregates were generally not persisted (top-10 prints only).
One-to-One Greedy cosine was not implemented in the notebooks.
Figures show only methods with extracted means; missing methods are
documented rather than imputed.

## Supplementary

- Detailed figures: **23** (PNG+SVG each)
- Summary/detail tables: **7**
- Locations:
  - `supplementary/dataset_level/`
  - `supplementary/generator_level/`
  - `supplementary/detailed_tables/`

## Generators

CTGAN, CopulaGAN, TVAE, GaussianCopula, WGAN_GP, CTABGAN, TabDDPM, ForestDiffusion

## Reproduce figures only

```bash
cd "SYNTH/mapping study graphs"
python create_main_paper_figures.py
```

