# SYNTH Framework

**Reproducible benchmark for tabular synthetic data generators** — comparing **8 generators** across **15 datasets** on **utility**, **fidelity**, and **privacy**, with multi-seed statistics and publication-ready figures.

| | |
|---|---|
| **Generators** | CTGAN, CopulaGAN, TVAE, GaussianCopula, CTABGAN, WGAN-GP, TabDDPM, ForestDiffusion |
| **Datasets** | 9 classification + 6 regression |
| **Protocol** | 10 generator-training seeds → **1,200** runs |
| **Metrics** | TRTR/TSTR utility · distributional fidelity · MIA / NNDR privacy |
| **Python** | 3.10–3.12 |

**Repository:** https://github.com/gopibattineni/SYNTH_FRAMEWORK_NEW

**Live dashboards:**
- [Interactive Results Dashboard](https://gopibattineni.github.io/SYNTH_FRAMEWORK_NEW/)
- [10-Seed Results Explorer](https://gopibattineni.github.io/SYNTH_FRAMEWORK_NEW/explorer/)

---

## Highlights

- **Leak-safe evaluation** — 80/20 train/test split; generators train on train only; TSTR evaluated on held-out test
- **10-seed protocol** — seeds `{42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355}`; results reported as Mean ± SD
- **Manuscript analysis** — critical-difference diagrams, rank tables, heatmaps, trade-offs, seed-stability plots
- **Interactive exploration** — FastAPI webapp + static GitHub Pages dashboard

---

## Repository layout

```text
SYNTH_FRAMEWORK/
├── Generators/                  # Notebooks & exports by family
│   ├── SDV models/              # CTGAN, CopulaGAN, TVAE, GaussianCopula
│   ├── Other GANS/              # CTABGAN, WGAN-GP
│   ├── Diffusion GANs/          # TabDDPM, ForestDiffusion
│   └── Experiment with utility data leak/
├── multi seed generators/       # 10-seed batches + merged Excel + analysis/
├── Agreed analysis/             # Classification / regression / trade-off figures
├── Results/                     # Paper-ready outputs
├── dashboard/                   # Build static Pages site → docs/
├── docs/                        # GitHub Pages (dashboard + /explorer/)
├── webapp/                      # FastAPI results explorer (+ optional generation)
├── Datasets/                    # Source / derived tabular data
├── figures/                     # Workflow & paper figures
├── scripts/                     # Maintenance & fix utilities
├── requirements.txt
├── CHANGELOG.md
└── CITATION.cff
```

---

## Quick start

```bash
git clone https://github.com/gopibattineni/SYNTH_FRAMEWORK_NEW.git
cd SYNTH_FRAMEWORK_NEW

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA PyTorch, install the appropriate wheel from [pytorch.org](https://pytorch.org) **before** `requirements.txt`.

Optional vendor packages (not on PyPI — clone manually if you re-run diffusion notebooks):

- `_vendor/tab-ddpm` — TabDDPM
- CTAB-GAN+ sources under `Generators/Other GANS/`

---

## 10-seed protocol

| Batch | Seeds | Runs |
|-------|-------|------|
| 1st | 42, 123, 2024 | 360 |
| 2nd | 68, 91, 2025 | 360 |
| 3rd | 55, 155, 255, 355 | 480 |
| **Total** | **10** | **1,200** |

Batches are provenance only. Primary tables and plots pool all **10 seed observations**.

**Merged Excel (Mean ± SD):**

- `multi seed generators/merged 10 seeds/classification/results/multi_seed_results.xlsx`
- `multi seed generators/merged 10 seeds/regression/results/multi_seed_results.xlsx`

**Re-run manuscript analysis:**

```bash
cd "multi seed generators/analysis"
python run_all.py
```

---

## Datasets

| # | Classification | # | Regression |
|---|----------------|---|------------|
| 1 | Breast Cancer Wisconsin | 10 | Metro Interstate Traffic |
| 2 | Alzheimer's | 11 | Online Shopping |
| 3 | Adult | 12 | Air Quality |
| 4 | Forest Cover Type | 13 | Concrete |
| 5 | Bank Marketing | 14 | Energy Efficiency |
| 6 | Wine Quality | 15 | Real Estate |
| 7 | CDC Diabetes | | |
| 8 | Secondary Mushroom | | |
| 9 | MAGIC Gamma Telescope | | |

Registry: `Generators/Experiment with utility data leak/python_scripts/hive/datasets.json`

---

## Webapp (local)

Explore 10-seed Mean ± SD tables (utility / fidelity / privacy / compute) and optionally run generation:

```bash
cd webapp
pip install -r requirements.txt
# Windows
start.bat
# or
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

---

## Dashboard (GitHub Pages)

| Site | URL |
|------|-----|
| **Interactive Results Dashboard** | https://gopibattineni.github.io/SYNTH_FRAMEWORK_NEW/ |
| **10-Seed Results Explorer** | https://gopibattineni.github.io/SYNTH_FRAMEWORK_NEW/explorer/ |

Static Plotly dashboards built from the merged 10-seed Excel. Rebuild locally:

```bash
python dashboard/build_pages.py
```

Output lands in `docs/`. CI rebuild: `.github/workflows/deploy-pages.yml`

---

## Evaluation pillars

| Pillar | What is measured |
|--------|------------------|
| **Utility** | TRTR vs TSTR (Accuracy, F1, … / R², RMSE, MAE) and utility gaps |
| **Fidelity** | Quality Score, KS Complement, MMD, Wasserstein, Gower (mixed types) |
| **Privacy** | Membership inference (MIA AUC), NNDR, distance-based checks |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, adding datasets/generators, and the PR checklist. Notable changes go in [CHANGELOG.md](CHANGELOG.md).

---

## Citation

If you use this work, please cite:

```bibtex
@software{battineni_synth_benchmark_2026,
  author  = {Battineni, Gopi},
  title   = {{SYNTH Benchmark: Tabular Synthetic Data Generator Evaluation}},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/gopibattineni/SYNTH_FRAMEWORK_NEW}
}
```

Also see [`CITATION.cff`](CITATION.cff).

---

## License & contact

Open an issue on GitHub for bugs or feature requests.  
Author: Gopi Battineni — gopi.chandu89@gmail.com
