# Trade-off Visualization Module

Publication-quality fidelity / utility / privacy trade-off figures for
synthetic-data evaluation. Independent of evaluation notebooks and reusable
across datasets.

## Layout

```
tradeoff/
├── tradeoff_plots.py   # Five figures + generate_tradeoff_figures()
├── pareto.py           # Pareto frontier utilities
├── utils.py            # Loading, colors, markers, scaling, saving
├── config.py           # Column aliases, palette, style knobs
├── README.md
├── figures/
│   ├── classification/
│   └── regression/
└── output/
    ├── classification/
    └── regression/
```

## Metrics

| Role | Classification | Regression |
|------|----------------|------------|
| Utility | Accuracy | R² |
| Fidelity | SDV Quality Score | SDV Quality Score |
| Privacy | MIA (AUC) | MIA (AUC) |

Task type is auto-detected from columns (`Accuracy` vs `R2`) or from a
`Task` column. Override with `task_type=`.

## Input CSV

One row per generator (multiple datasets allowed):

| Column | Required | Notes |
|--------|----------|-------|
| Dataset | yes | Panel title |
| Generator | yes | Legend / color key |
| QualityScore | yes | Fidelity |
| MIA | yes | Membership Inference Attack AUC |
| Accuracy | classification | Utility |
| R2 | regression | Utility |
| Task | optional | `classification` / `regression` |

Custom names are supported via aliases in `config.py`
(e.g. `Fidelity_SDMetrics`, `Mean_TSTR_Accuracy`, `Mean_TSTR_R2`).

## Figures

1. **Fig1_Fidelity_vs_Utility** — QualityScore vs Accuracy / R²  
2. **Fig2_Utility_vs_Privacy** — Accuracy / R² vs MIA (AUC)  
3. **Fig3_Fidelity_vs_Privacy** — QualityScore vs MIA (AUC)  
4. **Fig4_Bubble_Tradeoff** — QualityScore × utility, bubble size = MIA (AUC)  
5. **Fig5_Pareto** — Pareto frontier on QualityScore × utility  

Each figure is written as `.pdf`, `.svg`, and `.png` (300 dpi).

## Usage

```python
from tradeoff.tradeoff_plots import generate_tradeoff_figures

generate_tradeoff_figures(
    "path/to/metrics.csv",
    output_dir="tradeoff/figures",   # optional
    task_type=None,                  # auto-detect
)
```

CLI (from `Agreed analysis/`):

```bash
python -m tradeoff.tradeoff_plots path/to/metrics.csv
python -m tradeoff.tradeoff_plots path/to/metrics.csv -t classification --annotate
```

### Build a combined CSV from existing per-dataset files

```python
from tradeoff.tradeoff_plots import (
    build_combined_csv_from_individual,
    generate_tradeoff_figures,
)

csv = build_combined_csv_from_individual(
    "trade_off/Individual dataset",
    "tradeoff/output/classification/combined_metrics.csv",
    task_type="classification",
)
generate_tradeoff_figures(csv, task_type="classification")
```

## Styling

- Times New Roman (Liberation Serif fallback on Linux)  
- No interior grid lines  
- Colorblind-friendly Okabe–Ito palette with fixed generator colors/markers  
- Clear edged markers (larger legend symbols)  
- Shared axis limits across dataset panels  
- `constrained_layout=True`, white background, vector PDF + SVG  

Edit `config.py` to change aliases, colors, DPI, annotation defaults, etc.
