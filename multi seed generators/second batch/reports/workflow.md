# Batch 2 — Corrected SYNTH five-stage workflow

This experiment reuses the **same five-stage SYNTH Benchmark** as `fig1`, with two methodological changes only:

1. **Leakage-safe Stage 1:** train/test split **before** imputation; imputer fitted on **Real-Train only**.
2. **Stage 2 seed replication:** each generator is trained independently with seeds `[68, 91, 2025]`.

The original SYNTH notebooks/scripts/results are **not** modified.

---

## Five stages (identical conceptual design)

```text
STAGE 1 — DATA PREPARATION
        ↓
STAGE 2 — SYNTHESIS
        ↓
STAGE 3 — QUALITY (Fidelity)
        ↓
STAGE 4 — EVALUATION (Utility TRTR/TSTR + Privacy)
        ↓
STAGE 5 — ANALYSIS (Mean ± SD across 3 generator seeds)
```

---

## STAGE 1 — Data preparation (corrected)

> **The train/test split is performed before imputation. Imputation parameters are fitted exclusively on Real-Train and then applied to Real-Test, preventing information leakage from the held-out test data.**

> **For OASIS, participants are assigned entirely to either Real-Train or Real-Test so that no participant appears in both partitions.**

```text
15 UCI / curated datasets
        ↓
Load raw data
        ↓
Remove ID / unused columns (as configured)
        ↓
Mark missing ('?' → NaN)   ← no statistics learned
        ↓
Identify grouping structure
  • OASIS → Subject ID (participant-level)
  • Online Shopping → session ID
  • Others → row-level stratified 80/20 when appropriate
        ↓
FIXED stratified (or entity) 80/20 split   split_seed = 42
        ↓
┌─────────────────────────┐     ┌─────────────────────────┐
│ REAL-TRAIN (≈80%)       │     │ REAL-TEST (≈20%)        │
│                         │     │                         │
│ Fit imputer ONLY here   │     │ NEVER used to fit       │
│ (median / mode)         │     │ imputation              │
│        ↓                │     │                         │
│ Transform Real-Train    │────▶│ Apply SAME fitted       │
│                         │     │ imputer to Real-Test    │
└─────────────────────────┘     └─────────────────────────┘
        │                                   │
        └───────────────┬───────────────────┘
                        ↓
              Fixed Real-Train + Real-Test
              (shared by all 3 generator seeds)
```

### Forbidden (contamination)

```text
RAW → calculate mean/median/mode on ALL data → impute → split
```

---

## STAGE 2 — Synthesis (8 generators × 3 seeds)

Generators (exact existing set):

```text
GaussianCopula, CopulaGAN, CTGAN, CTABGAN, TVAE, WGAN_GP,
ForestDiffusion, TabDDPM
```

```text
Real-Train only
      │
      ├── Seed 42  → Synthetic_42
      ├── Seed 91 → Synthetic_123
      └── Seed 2025→ Synthetic_2024
```

Real-Test is **never** passed to generator training.

Same split + same preprocessing + same generator config; **only the training seed changes**.

Total runs: `15 × 8 × 3 = 360`.

---

## STAGE 3 — Quality / Fidelity

Same methodology as the existing benchmark:

* SDV `evaluate_quality()` → Quality Score
* KS Complement, Wasserstein, MMD (existing metric names)

Computed per `dataset × generator × seed`.

---

## STAGE 4 — Evaluation

### Utility (TRTR / TSTR)

```text
Path A TRTR:  Real-Train → classifier → Real-Test
Path B TSTR:  Synthetic  → classifier → Real-Test
```

Classification: Accuracy, Precision, Recall, F1, ROC-AUC (+ gaps)  
Regression: RMSE, MAE, R² (+ gaps / increases)

### Privacy

Existing privacy metrics retained per seed (Mahalanobis / NNDR / MIA_AUC).

---

## STAGE 5 — Analysis

For each `dataset × generator × metric`:

```text
seed_68, seed_91, seed_2025 → mean, sd (ddof=1), "mean ± sd"
```

This isolates **generator-seed variation** from **split variation** (split is fixed).

---

## Implementation location

```text
multi seed generators/second batch/
```

Entry point: `python run_experiment.py --smoke` then `--all`.
