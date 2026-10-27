# Mapping conference figures (dataset case studies)

**Story:** Pairwise cosine ≈ 0 → Hungarian cosine saturates → Δ_C amplification → Hungarian Mahalanobis restores discrimination.

**Design:** Each dataset is an independent experimental unit (8 generators, fixed order).

**Task split used here:** 10 classification + 5 regression (Online Shopping listed with classification for this manuscript organization).

## Main manuscript (`main/`)

| Figure | File | Content |
|------|------|---------|
| Fig. 1 | `Fig1_experimental_framework` | Methods / A–B–C cost control |
| Fig. 2 | `Fig2_Cancer` | Cancer 4-panel case study |
| Fig. 3 | `Fig3_Adult` | Adult 4-panel case study |
| Fig. 4 | `Fig4_MAGIC` | MAGIC 4-panel case study |
| Fig. 5 | `Fig5_Metro_Interstate` | Metro Interstate 4-panel case study |
| Fig. 6 | `Fig6_Real_Estate` | Real Estate 4-panel case study (MD instability noted) |

Each dataset figure panels:
**(a)** Pairwise cosine · **(b)** Hungarian cosine · **(c)** Cosine amplification Δ_C · **(d)** Hungarian Mahalanobis

## Supplementary (`supplementary/`)

| Figure | Dataset |
|------|---------|
| Fig. S1 | Alzheimer's |
| Fig. S2 | Forest Cover |
| Fig. S3 | Bank Marketing (MD unavailable) |
| Fig. S4 | Wine |
| Fig. S5 | CDC Diabetes |
| Fig. S6 | Mushroom |
| Fig. S7 | Online Shopping |
| Fig. S8 | Air Quality |
| Fig. S9 | Concrete |
| Fig. S10 | Energy Efficiency (MD instability noted) |

Flat copies of all 15 dataset figures also live in `by_dataset/`.

## Generator order (fixed)

1. CTGAN  2. CopulaGAN  3. TVAE  4. GaussianCopula
5. CTABGAN  6. WGAN-GP  7. TabDDPM  8. ForestDiffusion

## Regenerate

```bash
python Results/mapping_conference/make_dataset_case_figures.py
```
