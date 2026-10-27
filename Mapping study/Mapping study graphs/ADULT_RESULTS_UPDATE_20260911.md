# Adult results update — 2026-09-11

Adult SDV notebook completed (`3. Adult census.ipynb`, 47/47 cells). Mapping privacy results propagated across curated destinations.

## Sources (latest)
- `Generators/SDV models/Matchings_All_Models_Adult_Census.xlsx` (2026-09-11)
- `Generators/SDV models/Hungarian_Mahalanobis_Four_Models_Adult_Census.xlsx` (2026-09-11)
- `Generators/Other GANS/Adult_Hungarian_*.xlsx` (2026-09-10)
- `Generators/Diffusion GANs/Adult_Hungarian_*.xlsx` (2026-09-10)

## Updated destinations
- `excel sheets/2. Privacy/3. Adult/Hungarian_Matching.xlsx` (+ `.bak_*`)
- `excel sheets/2. Privacy/3. Adult/Mahalanobis_Summary.xlsx` (+ `.bak_*`)
- `paper results/3. Adult/fidelity_privacy_metrics.xlsx` sheets `Hungarian_Matching`, `Mahalanobis_Summary` (+ `.bak_*`)
- `Agreed analysis/mapping_cost_comparison.csv` Adult rows (+ `.bak_*`)
- `Agreed analysis/mapping_cost_comparison_by_dataset.csv` Adult row (+ `.bak_*`)
- `Generators/SDV models/Excel sheets/*Adult*` synced to latest
- `Results/Master_Data|MasterData/privacy_long.csv` + `master_unified.csv` Adult Hungarian/maha cells
- `docs/data/privacy.json` Adult Hungarian cosine / Mahalanobis distance metrics
- `mapping study graphs/` extracts + main/supplementary figures regenerated

## Not updated (requires missing extract pipeline / separate experiment)
- Full fidelity sheets (Quality, KS, JS, Wasserstein, Gower, MMD, PCA, …) in `paper results` / `excel sheets/1. Fidelity`
- Utility TRTR/TSTR (`excel sheets/3. Utility`, dataleak merge) — not produced by this generator-notebook finish
- Other Privacy metrics (MIA, NN, pairwise Cosine, Mahalanobis_2D, Mahalanobis_Detail) pending notebook HTML extract script restore

## Notes
- SDV Hungarian Mahalanobis Excel reports `Num_Matches=2000` while cosine uses `1000`; cosine protocol `1000` used for `Num_Matches` in mapping_cost / privacy.json.
- Historical backups preserved as `*.bak_YYYYMMDD_HHMMSS*`.

## Web application / dashboard (updated)
- Rebuilt with `python dashboard/build_pages.py`
- `docs/data/privacy.json` now includes all 8 Adult generators for Hungarian cosine + Mahalanobis distance stats from latest Excel
- `docs/index.html` + `docs/assets/` refreshed
- Live site (after git push / Pages deploy): https://gopibattineni.github.io/SYNTH_BENCHMARK/
- Note: Streamlit `dashboard/app.py` loads utility-data-leak Excel (utility track unchanged)
