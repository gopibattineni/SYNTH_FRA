# Representative Metrics Trade-off Analysis

**Runtime:** 31.67s

## Primary analysis (manuscript)
**Classifier-Independent** — mean utility across all 10 classifiers (seed means already aggregated), with NNDR privacy and SDMetrics Quality Score.

Figures: `/home/gopi.battineni/SYNTH_BENCHMARK/Results/Representative_Metrics_Tradeoff/Classifier_Independent/Figures`

## Supplementary analysis
**Best Classifier** — highest mean F1 per generator × dataset.

Figures: `/home/gopi.battineni/SYNTH_BENCHMARK/Results/Representative_Metrics_Tradeoff/Best_Classifier/Figures`

## Comparison
`/home/gopi.battineni/SYNTH_BENCHMARK/Results/Representative_Metrics_Tradeoff/Comparison` — Indep vs Best tables, Wilcoxon rank consistency, CD ranks.

## Metrics
- Utility: Mean F1
- Privacy: Mean NNDR (Avg NN distance; higher = more private)
- Fidelity: SDMetrics Quality Score