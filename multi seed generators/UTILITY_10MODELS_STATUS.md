# Utility 10+10 — code update (no re-run yet)

## Important
Original multi-seed runs **did not save** `synth.csv` (only `metrics.xlsx` + `run_meta.json`).
A utility-only re-evaluation must **regenerate** synth (same seeds), then cache `synth.csv`.

## Code complete

| Piece | Location |
|-------|----------|
| Shared 10+10 utility | `shared/utility.py` |
| Model name lists | `shared/config/utility_models.yaml` (mirrored in each batch `config/`) |
| Batch shims | `*/src/metrics/utility.py` → import shared |
| Experiment wiring | `*/src/experiment.py` — saves `synth.csv` + `per_model_utility.csv`; full model suite; Windows-safe flock |
| Aggregate | `*/src/aggregate.py` — no `light` 3-model subset; uses `get_classifiers()` / `get_regressors()` |
| SDV metadata | `*/src/generators/__init__.py` — `verbose` kwarg optional |
| Re-run CLI | `rerun_utility_10models.py` |
| Generator×model heatmaps | `analysis/run_generator_by_model_heatmaps.py` (also hooked in `analysis/run_all.py`) |

## Models
**Classification (10):** LogReg (L2, C=1.0, lbfgs, max_iter=5000), SVM-RBF, KNN, NaiveBayes, DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoost, MLP (max_iter=5000)

**Regression (10):** DecisionTree, RandomForest(100), ExtraTrees, GradientBoost, KNN(n=5), SVR_RBF(C=1, ε=0.1), LinearRegression, Ridge(α=1), Lasso(α=0.001), ElasticNet(α=0.001, l1_ratio=0.5)

y is **not** scaled; X uses StandardScaler fit on each path’s train frame (Real-Train / synth).

## Expected TSTR fits (after re-run)
- Classification: 9 × 8 × 10 × 10 × 10 = **72,000**
- Regression: 6 × 8 × 10 × 10 × 10 = **48,000**

## When ready to execute (not done in this pass)
```bash
cd "multi seed generators"
python rerun_utility_10models.py --all
python merge_10_seeds.py
cd analysis && python run_all.py
```
