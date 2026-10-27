# Experiment summary — Multi-seed Batch 2

## Design

| Item | Value |
|---|---|
| Datasets | 15 |
| Generators | 8 |
| Generator seeds | 68, 91, 2025 |
| Total generator runs | **360** |
| Split seed | 42 (fixed) |
| Test size | 0.20 |
| Stage 1 rule | **Split first → impute on Real-Train only** |
| Aggregation | mean ± sample SD (`ddof=1`) |

## Generators

GaussianCopula, CopulaGAN, CTGAN, CTABGAN, TVAE, WGAN_GP, ForestDiffusion, TabDDPM

## What is unchanged vs original SYNTH

* Same 15 datasets and 8 generators  
* Same fidelity / utility / privacy metric families  
* Same TRTR / TSTR evaluation idea  
* No Optuna / no hyperparameter search  

## What changed

1. Leakage-safe Stage 1 (split before impute)  
2. Three independent generator-training seeds + Mean ± SD  

## How to run

```bash
cd "SYNTH/multi seed generators/second batch"
python run_experiment.py --validate-only
python run_experiment.py --smoke          # 1 dataset × 1 generator × 3 seeds
python run_experiment.py --all            # full 360 (resumable)
```

## Outputs

* Raw: `results/raw/`
* Aggregated: `results/aggregated/`
* Logs: `results/logs/experiment_log.csv`
* Workflow: `reports/workflow.md`
* Leakage audit: `reports/leakage_audit.md`
