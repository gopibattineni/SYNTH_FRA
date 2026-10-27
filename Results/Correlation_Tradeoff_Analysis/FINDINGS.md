# Correlation Trade-off Analysis — Findings

**Primary figures:** one point per generator (mean across Cancer & Mushroom, classifiers, and seeds).

**Supplementary figures:** seed-level points showing variability (Cancer = circle, Mushroom = square).

## Primary

### Accuracy / MIA

- **Figure01_fidelity_vs_utility_Accuracy**: Pearson r = 0.483, Spearman ρ = 0.048, R² = 0.234, p = 0.2248 — No statistically significant relationship (p ≥ 0.05).
- **Figure02_privacy_vs_utility_Accuracy**: Pearson r = -0.192, Spearman ρ = -0.405, R² = 0.037, p = 0.6482 — No statistically significant relationship (p ≥ 0.05).
- **Figure03_fidelity_vs_privacy_Accuracy**: Pearson r = 0.651, Spearman ρ = 0.595, R² = 0.424, p = 0.08036 — No statistically significant relationship (p ≥ 0.05).

### Accuracy / NNDR

- **Figure01_fidelity_vs_utility_Accuracy**: Pearson r = 0.483, Spearman ρ = 0.048, R² = 0.234, p = 0.2248 — No statistically significant relationship (p ≥ 0.05).
- **Figure02_privacy_vs_utility_Accuracy**: Pearson r = -0.340, Spearman ρ = 0.024, R² = 0.115, p = 0.4106 — No statistically significant relationship (p ≥ 0.05).
- **Figure03_fidelity_vs_privacy_Accuracy**: Pearson r = -0.937, Spearman ρ = -0.905, R² = 0.878, p = 0.0005948 — Strong negative correlation.

### F1 / MIA

- **Figure01_fidelity_vs_utility_F1**: Pearson r = 0.705, Spearman ρ = 0.071, R² = 0.497, p = 0.05077 — No statistically significant relationship (p ≥ 0.05).
- **Figure02_privacy_vs_utility_F1**: Pearson r = 0.067, Spearman ρ = -0.262, R² = 0.005, p = 0.8743 — No statistically significant relationship (p ≥ 0.05).
- **Figure03_fidelity_vs_privacy_F1**: Pearson r = 0.651, Spearman ρ = 0.595, R² = 0.424, p = 0.08036 — No statistically significant relationship (p ≥ 0.05).

### F1 / NNDR

- **Figure01_fidelity_vs_utility_F1**: Pearson r = 0.705, Spearman ρ = 0.071, R² = 0.497, p = 0.05077 — No statistically significant relationship (p ≥ 0.05).
- **Figure02_privacy_vs_utility_F1**: Pearson r = -0.563, Spearman ρ = 0.024, R² = 0.317, p = 0.1466 — No statistically significant relationship (p ≥ 0.05).
- **Figure03_fidelity_vs_privacy_F1**: Pearson r = -0.937, Spearman ρ = -0.905, R² = 0.878, p = 0.0005948 — Strong negative correlation.

## Supplementary

### Accuracy / MIA

- **FigureS01_fidelity_vs_utility_Accuracy**: Pearson r = 0.455, Spearman ρ = 0.534, R² = 0.207, p = 1.527e-09 — Moderate positive correlation.
- **FigureS02_privacy_vs_utility_Accuracy**: Pearson r = 0.165, Spearman ρ = 0.202, R² = 0.027, p = 0.0374 — Very weak positive correlation.
- **FigureS03_fidelity_vs_privacy_Accuracy**: Pearson r = 0.259, Spearman ρ = 0.131, R² = 0.067, p = 0.0009421 — Weak positive correlation.

### Accuracy / NNDR

- **FigureS01_fidelity_vs_utility_Accuracy**: Pearson r = 0.455, Spearman ρ = 0.534, R² = 0.207, p = 1.527e-09 — Moderate positive correlation.
- **FigureS02_privacy_vs_utility_Accuracy**: Pearson r = -0.366, Spearman ρ = -0.224, R² = 0.134, p = 1.881e-06 — Weak negative correlation.
- **FigureS03_fidelity_vs_privacy_Accuracy**: Pearson r = -0.893, Spearman ρ = -0.688, R² = 0.798, p = 9.302e-57 — Strong negative correlation.

### F1 / MIA

- **FigureS01_fidelity_vs_utility_F1**: Pearson r = 0.637, Spearman ρ = 0.545, R² = 0.405, p = 1.482e-19 — Moderate positive correlation.
- **FigureS02_privacy_vs_utility_F1**: Pearson r = 0.197, Spearman ρ = 0.234, R² = 0.039, p = 0.01231 — Very weak positive correlation.
- **FigureS03_fidelity_vs_privacy_F1**: Pearson r = 0.259, Spearman ρ = 0.131, R² = 0.067, p = 0.0009421 — Weak positive correlation.

### F1 / NNDR

- **FigureS01_fidelity_vs_utility_F1**: Pearson r = 0.637, Spearman ρ = 0.545, R² = 0.405, p = 1.482e-19 — Moderate positive correlation.
- **FigureS02_privacy_vs_utility_F1**: Pearson r = -0.503, Spearman ρ = -0.179, R² = 0.253, p = 1.232e-11 — Moderate negative correlation.
- **FigureS03_fidelity_vs_privacy_F1**: Pearson r = -0.893, Spearman ρ = -0.688, R² = 0.798, p = 9.302e-57 — Strong negative correlation.
