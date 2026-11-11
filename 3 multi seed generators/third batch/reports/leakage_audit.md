# Leakage audit — Multi-seed Batch 3

**Original SYNTH pipeline was not modified.**  
Corrected Stage 1 lives only under `multi seed generators/third batch/`.

---

## Policy (Batch 3)

```text
ALWAYS:  split FIRST → fit imputer on Real-Train ONLY → transform train + test
NEVER:   impute on full data BEFORE split
```

Imputer: median (numeric) / mode (categorical), `TrainOnlyImputer`.  
Split seed: **42** (fixed across generator seeds 55 / 155 / 255 / 355).  
Test size: **0.20**.

---

## Direct answers

### 1. Old pipeline: imputation before or after split?

**Mixed / often before.** Several utility-dataleak notebooks filled mean/mode/median on the **full** table, then split (contamination risk). Adult/Alzheimer patched notebooks used complete-case `dropna` before split (no fill stats, but not the split-first impute pattern).

### 2. Full-dataset statistics used when imputing before split?

Mean / median / mode (and related category frequencies) on **all rows**.

### 3–5. OASIS

* Participant ID: **`Subject ID`**
* Sessions: MRI visits (373 rows, 150 subjects, 2–5 sessions/subject)
* Current patched notebooks: **participant-level** `_participant_split`
* Same participant in train and test: **No** (asserted)
* Batch 3: participant-level split **before** train-only imputation; overlap must be **0**

### 6. Repeated-entity datasets

| Dataset | Entity ID | Batch 3 split |
|---|---|---|
| Alzheimers (OASIS) | Subject ID | **participant-level** |
| Online Shopping | session ID | **session-level** |
| All others | none / unique row ids | row-level stratified 80/20 when classification |

### 7. Correction in Batch 3

Split-first + train-only impute for **all 15** datasets; OASIS/OnlineShop group splits; generators see Real-Train only.

---

## Per-dataset table

| Dataset | Rows (approx) | Unique entities | Existing split | Existing imputation | Potential leakage? | Final Batch 3 split | Imputer fitted on | Train / Test | Entity overlap | Status |
|---|---:|---:|---|---|---|---|---|---|---:|---|
| Cancer | ~569 | — | row stratified | usually none | Low | row 80/20 | Real-Train | cached JSON | 0 | OK |
| Alzheimers | 373 | 150 subjects | participant (patched) | complete-case before split | Low if patched | **participant 80/20** | Real-Train | see OASIS block | **0** | OK |
| Adult | ~48k | — | row | complete-case before split | Low (no fill stats) | row 80/20 | Real-Train | cached JSON | 0 | OK |
| ForestCover | large | — | row | usually none | Low | row 80/20 | Real-Train | cached JSON | 0 | OK |
| Bank | ~45k | — | row | **full mean/mode before split** | **YES** | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| Wine | ~4.9k | — | row | usually none | Low | row 80/20 | Real-Train | cached JSON | 0 | OK |
| CDC | large | — | row | usually none | Low | row 80/20 | Real-Train | cached JSON | 0 | OK |
| Mushroom | large | — | row | **full median/mode before split** | **YES** | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| MAGIC | ~19k | — | row | sometimes full mean fill | Possible | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| Metro | large | — | row | pipeline fills | Possible | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| OnlineShop | large | session ID | often row | fills before split | **YES (session+impute)** | **session 80/20** | Real-Train | cached JSON | 0 | FIXED |
| AirQuality | ~9k | — | row | fills | Possible | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| Concrete | 1030 | — | row | fills | Possible | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| Energy | 768 | — | row | fills | Possible | row 80/20 | Real-Train | cached JSON | 0 | FIXED |
| RealEstate | 414 | — | row | **median before split** | **YES** | row 80/20 | Real-Train | cached JSON | 0 | FIXED |

Exact sizes are written to `cache/splits/*v2_split_first*.json` after each dataset is prepared.

---

## OASIS validation block

```text
Participant-level split: YES
Training participants: (from split cache / run_meta)
Testing participants:  (from split cache / run_meta)
Participant overlap: 0
```

Pipeline asserts:

```python
assert len(train_participants ∩ test_participants) == 0
```

---

## Manuscript statement

Batch 3 reports fidelity, utility, and privacy using:

* leakage-safe preprocessing (**split before imputation**)
* training-only imputation fitting
* participant-level splitting for OASIS
* fixed train/test partitions across seeds
* three independent generator-training seeds
* mean ± SD (`ddof=1`) aggregation
