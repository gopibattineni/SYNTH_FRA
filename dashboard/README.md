# Dashboard

Static GitHub Pages dashboard for browsing SYNTH **10-seed** benchmark results.

## Live site

- **Plotly dashboard:** https://gopibattineni.github.io/SYNTH_FRA/
- **10-Seed Results Explorer:** https://gopibattineni.github.io/SYNTH_FRA/explorer/

## Data source

Results are exported from:

- `multi seed generators/merged 10 seeds/classification/results/multi_seed_results.xlsx`
- `multi seed generators/merged 10 seeds/regression/results/multi_seed_results.xlsx`

All charts show **Mean ± SD across 10 generator-training seeds** (utility, fidelity, privacy).

## Rebuild

```bash
python dashboard/build_pages.py
```

Output is written to `docs/` at the repository root (committed for GitHub Pages).

## What it shows

- Overview: utility-gap heatmap + generator coverage (15 × 8)
- Utility: TRTR / TSTR / gaps (Accuracy, F1, … / R², RMSE, MAE)
- Fidelity: Quality Score, KS Complement, MMD, Wasserstein
- Privacy: MIA AUC, NNDR, Mahalanobis / mean / median distance
- Trade-offs & rankings from the 10-seed aggregated pillars

## Deployment (fix GitHub Pages 404)

1. Open **https://github.com/gopibattineni/SYNTH_FRA/settings/pages**
2. Under **Build and deployment → Source**, choose **GitHub Actions**
3. Push to `main` (or run **Actions → Deploy Dashboard to GitHub Pages → Run workflow**)
4. Site updates at **https://gopibattineni.github.io/SYNTH_FRA/**
