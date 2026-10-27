# Mapping study graphs — Hungarian improvement over Greedy

**Extract-only analysis.** Mapping experiments were not rerun. Generators were not retrained.

## Research question

How much does **Hungarian matching** improve sample-to-sample matching compared with **Greedy matching**?

## Critical data availability

| Comparison | Available? | N (dataset×generator) | Source |
|---|---|---:|---|
| Greedy (many-to-one) Cosine vs Hungarian Cosine | **No** | 0 | Notebooks printed top-10 only; means not persisted |
| Pairwise Cosine vs Hungarian Cosine | Yes | 108 | `Agreed analysis/mapping_cost_comparison.csv` |
| Greedy (one-to-one) Mahalanobis vs Hungarian | Yes | 114 | Notebook outputs + Hungarian Excel / curated CSV |

**Cosine figures therefore use Pairwise Cosine as the non-Hungarian baseline** and are labeled accordingly (not “Greedy”).
**Mahalanobis figures use One-to-One Greedy**, which is the greedy Mahalanobis procedure implemented in the notebooks.

## Improvement definitions

```text
ΔCosine = Hungarian_Cosine − Pairwise_Cosine     # positive ⇒ Hungarian higher similarity
ΔMahalanobis = Greedy_Maha − Hungarian_Maha      # positive ⇒ Hungarian lower distance
```

Percentage improvements:

```text
Cosine % = (ΔCosine / Pairwise) × 100     # unstable when |Pairwise| ≈ 0 (flagged)
Mahalanobis % = (ΔMahalanobis / Greedy) × 100
```

Near-zero pairwise denominators are flagged in `hungarian_vs_greedy_summary.csv` (`Cosine_Pct_Unstable`).

## Overall results (from extracted pairs)

### Cosine (Pairwise → Hungarian)
- N = 108
- Mean ΔCosine = 0.9707
- Median ΔCosine = 0.9847
- % cases Hungarian higher = 100.0%

### Mahalanobis (Greedy one-to-one → Hungarian)
- N = 114
- Mean ΔMahalanobis = 0.4912
- Median ΔMahalanobis = 0.1334
- % cases Hungarian improved (lower distance) = 93.0%

## Main paper figures

| Figure | Path |
|---|---|
| Figure 1 | `main_paper/figure_1_cosine_improvement/` |
| Figure 2 | `main_paper/figure_2_mahalanobis_improvement/` |
| Figure 3 | `main_paper/figure_3_cosine_scatter/` |
| Figure 4 | `main_paper/figure_4_mahalanobis_scatter/` |
| Figure 5 | `main_paper/figure_5_improvement_heatmaps/` |
| Figure 6 | `main_paper/figure_6_improvement_distribution/` |

Each folder: PNG (300 DPI) + SVG.

## Summary tables

- `extracted_results/hungarian_vs_greedy_summary.csv` — unit-level (up to 15×8)
- `extracted_results/hungarian_vs_greedy_aggregate_stats.csv`
- `extracted_results/hungarian_vs_greedy_paired_tests.csv`
- `extracted_results/validation_checklist.json`

## Reproduce (figures only)

```bash
cd "SYNTH/mapping study graphs"
python create_hungarian_vs_greedy_figures.py
```
