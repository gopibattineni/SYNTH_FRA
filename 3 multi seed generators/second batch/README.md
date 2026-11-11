# Multi-seed generators — Batch 2

Isolated re-run of the **same SYNTH five-stage benchmark** (`fig1`), with:

1. **Leakage-safe Stage 1** — train/test split **before** imputation; imputer fitted on **Real-Train only**
2. **3 generator-training seeds** — `[68, 91, 2025]` → Mean ± SD

**Original SYNTH notebooks / results are not modified.**

See:

* `reports/workflow.md` — five-stage pipeline diagram + rules  
* `reports/leakage_audit.md` — per-dataset leakage decisions  
* `reports/experiment_summary.md` — compact design card  

---

## Manuscript reproducibility statement

> The train/test split is performed before imputation. Imputation parameters are fitted exclusively on Real-Train and then applied to Real-Test, preventing information leakage from the held-out test data.

> For OASIS, participants are assigned entirely to either Real-Train or Real-Test so that no participant appears in both partitions.

> The same fixed split is reused for generator seeds 68, 91, and 2025 so that reported Mean ± SD reflects generator stochasticity, not split variability.

---

## Design card

| Item | Value |
|---|---|
| Datasets | 15 |
| Generators | 8 |
| Seeds | 68, 91, 2025 |
| Runs | **360** |
| Split seed | 42 |
| Aggregation | mean ± SD (`ddof=1`) |

---

## How to run

```bash
cd "SYNTH/multi seed generators/second batch"

python run_experiment.py --validate-only
python run_experiment.py --smoke      # Cancer × CTGAN × 3 seeds
python run_experiment.py --all        # resumable 360
python run_experiment.py --aggregate-only
```

Completed jobs are skipped automatically (`results/raw/seed_*/.../metrics.xlsx`). Use `--force` to overwrite.

### 4 GPUs in parallel (remaining 11 datasets)

Leave these completed classification datasets untouched: **cancer, alzhimers, adult, bank** (24/24 each).

Run the other **11** datasets on GPUs 0–3:

| Script | GPU | Classification | Regression |
|---|---|---|---|
| `cuda0.sh` | 0 | forest_cover, wine | metro |
| `cuda1.sh` | 1 | cdc_diabetes | online_shopping, air_quality |
| `cuda2.sh` | 2 | mushroom | concrete, energy_efficiency |
| `cuda3.sh` | 3 | magic | real_estate |

```bash
cd "multi seed generators/second batch"
nohup bash run_all_cuda.sh > logs/run_all_cuda.out 2>&1 &

# or one GPU
bash cuda0.sh
```

Logs: `logs/cuda0.log` … `cuda3.log`.
If your env is not `python3`, set `PYTHON=/path/to/python`.

---

## Folder map

```text
multi seed generators/second batch/
├── classification/          ← 9 classification datasets (216 runs)
│   ├── results/raw|aggregated|logs
│   ├── figures/
│   ├── reports/
│   └── cache/splits/
├── regression/              ← 6 regression datasets (144 runs)
│   ├── results/raw|aggregated|logs
│   ├── figures/
│   ├── reports/
│   └── cache/splits/
├── config/                  shared
├── src/                     shared Stage 1–5 modules
├── notebooks/
├── reports/                 workflow + leakage audit
└── run_experiment.py
```

**Why Cancer earlier?** Only used as a *smoke test* (small/fast). Full run covers all 15 datasets.

## Reused (read-only)

* `hive/datasets.json`, `docs/data/meta.json`
* SDV, `CTAB-GAN-Plus/`, `Diffusion GANs/diffusion_generators.py`
* Existing metric naming from `Results/MasterData/*_long.csv`
