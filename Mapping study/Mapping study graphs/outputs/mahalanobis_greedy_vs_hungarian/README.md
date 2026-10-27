# Mahalanobis Greedy vs Hungarian

Extract-only comparison of **One-to-One Greedy** and **Hungarian** sample mapping using **Mahalanobis distance**.

> Cosine-similarity mapping is intentionally **excluded** from this analysis and all figures.

## What was compared

- **Metric:** mean matched-pair **Mahalanobis distance** (lower = better matching).
- **Methods:** One-to-One Greedy vs Hungarian assignment on the same real↔synthetic distance matrix.
- **Unit of analysis:** dataset × generator pairs with both methods available in the extract.

## Mahalanobis metric (as used by the existing pipeline)

For each dataset × generator, real and synthetic samples are represented in a numerical feature space.
Pairwise Mahalanobis distances use a covariance estimated from the (training) real data.
After assignment, the reported score is the **mean Mahalanobis distance over matched pairs**.
Lower mean matched-pair distance ⇒ closer real–synthetic correspondence under that metric.

## Greedy vs Hungarian methodology

- **One-to-One Greedy:** iteratively selects the currently cheapest unmatched real–synthetic pair until all allowed matches are filled (locally greedy).
- **Hungarian:** solves the linear assignment problem for a globally optimal one-to-one matching that minimizes total (equivalently mean) Mahalanobis matching cost.
- Both methods are one-to-one; they differ only in how the pairing is chosen on the same cost matrix.

## Difference definition

```text
Diff = Hungarian - Greedy
Diff < 0  → Hungarian lower distance → better matching
Diff > 0  → Hungarian higher distance → worse matching
Diff ≈ 0  → negligible difference
Rel% = 100 * (Hungarian - Greedy) / |Greedy|   (undefined/unstable if |Greedy|≈0)
```

## Coverage

- Paired units processed: **114** of 120 possible (15 datasets × 8 generators)
- Skipped units: **6**
- Datasets: Adult, AirQuality, Alzheimers, Bank, CDC, Cancer, Concrete, Energy, ForestCover, MAGIC, Metro, Mushroom, OnlineShop, RealEstate, Wine
- Generators: CTABGAN, CTGAN, CopulaGAN, ForestDiffusion, GaussianCopula, TVAE, TabDDPM, WGAN_GP

### Skipped dataset × generator units

- `Wine` / `CTGAN`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']
- `Wine` / `CopulaGAN`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']
- `Wine` / `TVAE`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']
- `Wine` / `GaussianCopula`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']
- `AirQuality` / `CTGAN`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']
- `AirQuality` / `CopulaGAN`: incomplete Mahalanobis methods available: ['Hungarian', 'One-to-One Greedy']

## Overall results

- Mean Diff (H−G): **-0.4912**
- SD Diff: **3.2035**
- Median Diff: **-0.1334**
- % Hungarian better: **93.0%**
- Wilcoxon p (H1: Greedy > Hungarian): **2.285e-15**
- Rank-biserial effect size: **0.761**

Paired Wilcoxon signed-rank tests are also reported by generator and by dataset,
with Holm adjustment within each scope (`tables/paired_statistical_tests.csv`).

## Figures

| Figure | Path | Description |
|---|---|---|
| Fig01 | `overall_comparison/` | Dumbbell: dataset-mean Greedy → Hungarian (log scale) |
| Fig01b | `overall_comparison/` | Dumbbell: all paired units Greedy → Hungarian (log scale) |
| Fig02 | `dataset_comparison/` | Ranked bars of dataset-level Diff |
| Fig03 | `generator_comparison/` | Generator mean Diff ± SD |
| Fig04 | `heatmaps/` | Dataset × generator Diff heatmap (diverging, centered at 0) |
| Fig05a/b | `distributions/` | Overall and by-generator Diff distributions |
| Fig06 | `scatter/` | Greedy vs Hungarian scatter with y=x (log–log) |
| Fig07 | `summary/` | Manuscript summary panel |

## Tables

- `tables/unit_level_mahalanobis_greedy_vs_hungarian.csv` — every paired unit with Diff, Rel%, direction
- `tables/dataset_level_summary.csv`
- `tables/generator_level_summary.csv`
- `tables/paired_statistical_tests.csv`
- `tables/mahalanobis_greedy_vs_hungarian_tables.xlsx` — all sheets + skipped units + meta
- `tables/processing_report.json`

## Unavailable analyses (not fabricated)

- **Assignment agreement:** pair-identity / overlap matrices were not present in extracted outputs → `agreement/SKIPPED.txt`.
- **Runtime / compute cost:** no runtime fields in extracted mapping results → `runtime/SKIPPED.txt`.

## Data sources (not overwritten)

- Primary: `hungarian_vs_greedy_summary.csv`
- Secondary inventory: `mahalanobis_distance_results.csv`
- Raw notebook/Excel experimental files and older `graphs/` figures were not deleted or overwritten.

## Limitations

- 14/120 dataset×generator combinations lack a complete Greedy+Hungarian Mahalanobis pair.
- Values are extracted means; seed-level repeated mapping runs were not available for within-unit SD.
- Absolute mean distances are dominated by a few large-scale datasets (e.g. Energy, RealEstate);
  prefer Diff-based figures and medians for interpretation.
- Hungarian can still appear slightly worse than Greedy in a few extracted cells due to
  numerical/covariance/implementation details in the original notebooks; those cases are retained as recorded.

## Reproduce

```bash
cd "mapping study graphs"
python create_mahalanobis_greedy_vs_hungarian.py
```
