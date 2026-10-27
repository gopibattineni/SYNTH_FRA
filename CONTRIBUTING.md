# Contributing to SYNTH Benchmark

Thank you for your interest in improving the SYNTH Benchmark! This guide helps you get started quickly.

---

## Ways to contribute

- **Add or update experiment results** — drop new Excel workbooks under `Generators/` and re-run the pipeline
- **Fix bugs** in analysis scripts or notebooks
- **Improve documentation** — README, comments, paper notes
- **Add datasets or generators** — follow the existing notebook template
- **Enhance figures or statistical tests**

---

## Development setup

```bash
git clone https://github.com/gopibattineni/SYNTH_BENCHMARK.git
cd SYNTH_BENCHMARK
pip install -r requirements-analysis.txt
pip install -r requirements.txt          # full notebook dependencies (optional)
```

---

## Running the pipelines

Always run these after changing experiment data or analysis code:

```bash
# Full 15-dataset analysis (~3 min)
python run_analysis.py

# Cancer & Mushroom case study (~3 min)
python run_two_datasets_assessment.py

# Extract paper workbooks from notebooks (~2 min)
python scripts/extract_paper_results.py
```

Verify outputs in `Results/` before submitting a pull request.

---

## Adding a new dataset

1. Add an entry to `Generators/Experiment with utility data leak/python_scripts/hive/datasets.json`
2. Create dataset folders under all five generator families (see `README.md`)
3. Copy and adapt an existing notebook (e.g. `1. Cancer/cancer.ipynb`)
4. Run the notebook end-to-end and export `TRTR_TSTR_results_*.xlsx`
5. Re-run `python run_analysis.py` — the pipeline auto-discovers new files

---

## Adding a new generator

1. Implement training + synthesis in the relevant notebook family
2. Add the generator name to `analysis/config.py` → `GENERATORS`
3. Export results using the same Excel sheet structure as existing generators
4. Re-run the analysis pipeline

---

## Code style

- Match existing conventions in `analysis/` (type hints, docstrings on public functions)
- Keep changes focused — one logical change per pull request
- Do not commit secrets, large binary datasets, or `_vendor/` contents

---

## Pull request checklist

- [ ] Pipeline runs without errors: `python run_analysis.py`
- [ ] New/changed outputs are sensible (spot-check figures in `Results/Figures/`)
- [ ] README or CHANGELOG updated if user-facing behavior changed
- [ ] No credentials or `.env` files included

---

## Commit message format

Use clear, descriptive messages:

```
Add seed stability analysis to benchmarking module

Fix privacy dataset name normalization for Pareto frontier

Update Cancer notebook TRTR export columns
```

Prefix with `Add`, `Fix`, `Update`, or `Remove` when possible.

---

## Questions?

Open a [GitHub Issue](https://github.com/gopibattineni/SYNTH_BENCHMARK/issues) for bugs, feature requests, or questions about the experimental protocol.
