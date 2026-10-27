from pathlib import Path

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBAPP_ROOT.parent
BENCHMARK_ROOT = REPO_ROOT / "Generators" / "Experiment with utility data leak"
DATASETS_JSON = BENCHMARK_ROOT / "python_scripts" / "hive" / "datasets.json"
DIFFUSION_MODULE = REPO_ROOT / "Generators" / "Diffusion GANs" / "diffusion_generators.py"
CTABGAN_DIR = WEBAPP_ROOT / "data" / "CTAB-GAN-Plus"
SESSIONS_DIR = WEBAPP_ROOT / "data" / "sessions"
STATIC_DIR = WEBAPP_ROOT / "static"

# Primary published results: 10 generator-training seeds (Mean ± SD).
MERGED_RESULTS_ROOT = REPO_ROOT / "multi seed generators" / "merged 10 seeds"
MERGED_CLASSIFICATION_XLSX = (
    MERGED_RESULTS_ROOT / "classification" / "results" / "multi_seed_results.xlsx"
)
MERGED_REGRESSION_XLSX = MERGED_RESULTS_ROOT / "regression" / "results" / "multi_seed_results.xlsx"

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Faster defaults for interactive web runs (full benchmark uses 100 epochs / 1000 steps).
WGAN_EPOCHS = 50
TABDDPM_STEPS = 500
