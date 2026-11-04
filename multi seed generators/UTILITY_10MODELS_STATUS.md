# Utility 10+10 migration status

## Critical finding
Original multi-seed runs **did not save** `synth.csv` (only `metrics.xlsx` + `run_meta.json`).
Utility re-evaluation therefore **regenerates** synthetic data (same generator seeds), then caches `synth.csv` for reuse.

## Code changes
- Shared module: `shared/utility.py` + `shared/config/utility_models.yaml` (10 clf + 10 reg)
- Batch shims: `*/src/metrics/utility.py` → import shared
- `*/src/experiment.py`: save `synth.csv` + `per_model_utility.csv`; use full model suite
- `*/src/aggregate.py`: removed `light` 3-model subset; uses `get_classifiers()` / `get_regressors()`
- `*/src/generators/__init__.py`: SDV metadata API compatibility (no `verbose`)
- Re-run entrypoint: `rerun_utility_10models.py`
- Analysis: `analysis/run_generator_by_model_heatmaps.py`

## Expected TSTR fits
- Classification: 9 × 8 × 10 × 10 × 10 = **72,000**
- Regression: 6 × 8 × 10 × 10 × 10 = **48,000**

## Smoke verified
- Cancer / GaussianCopula / seed 42 → 10 models, 1000 per-model rows
- Concrete / GaussianCopula / seed 42 → 10 models, 600 per-model rows

## Full re-run
```bash
cd "multi seed generators"
python rerun_utility_10models.py --all
# then
python merge_10_seeds.py
cd analysis && python run_all.py
python run_generator_by_model_heatmaps.py
```
