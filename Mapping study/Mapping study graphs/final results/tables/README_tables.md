# Final results — Greedy vs Hungarian (Mahalanobis)

Paired units only (both methods present). Improvement % > 0 ⇒ Hungarian better.

## Table 1. Dataset characteristics

| Dataset           | Type           |   Samples |   Features | Classes   | Task           |
|:------------------|:---------------|----------:|-----------:|:----------|:---------------|
| Cancer            | Classification |       569 |         30 | 2         | Classification |
| Alzheimer's       | Classification |       373 |         10 | 2         | Classification |
| Adult             | Classification |      1000 |         14 | 2         | Classification |
| Forest Cover      | Classification |      1000 |         10 | 7         | Classification |
| Bank Marketing    | Classification |     10000 |         13 | 2         | Classification |
| Wine Quality      | Classification |      1000 |         11 | 6         | Classification |
| CDC Diabetes      | Classification |      1000 |         21 | 2         | Classification |
| Mushroom          | Classification |      1000 |         20 | 2         | Classification |
| MAGIC Gamma       | Classification |     19020 |         10 | 2         | Classification |
| Metro Interstate  | Regression     |      1000 |         12 | –         | Regression     |
| Online Shopping   | Regression     |      1000 |         12 | –         | Regression     |
| Air Quality       | Regression     |      1000 |         12 | –         | Regression     |
| Concrete          | Regression     |      1000 |          8 | –         | Regression     |
| Energy Efficiency | Regression     |       768 |          8 | –         | Regression     |
| Real Estate       | Regression     |       414 |          5 | –         | Regression     |

## Table 2. Main quantitative results

| Dataset           | Type           |   N_paired_generators |   Greedy_Mahalanobis |   Hungarian_Mahalanobis |   Improvement_Pct | Better_Method   |
|:------------------|:---------------|----------------------:|---------------------:|------------------------:|------------------:|:----------------|
| Cancer            | Classification |                     8 |               48.87  |                  48.63  |              0.48 | Hungarian       |
| Alzheimer's       | Classification |                     8 |                5.06  |                   4.891 |              3.33 | Hungarian       |
| Adult             | Classification |                     8 |                3.648 |                   3.51  |              3.78 | Hungarian       |
| Forest Cover      | Classification |                     8 |              306.2   |                 306.2   |              0.02 | Tie             |
| Bank Marketing    | Classification |                     8 |                2.58  |                   2.364 |              8.36 | Hungarian       |
| Wine Quality      | Classification |                     8 |                3.187 |                   2.89  |              9.31 | Hungarian       |
| CDC Diabetes      | Classification |                     8 |                4.544 |                   4.289 |              5.62 | Hungarian       |
| Mushroom          | Classification |                     8 |                1.944 |                   1.81  |              6.9  | Hungarian       |
| MAGIC Gamma       | Classification |                     8 |                2.126 |                   2.448 |            -15.19 | Greedy          |
| Metro Interstate  | Regression     |                     8 |                5.774 |                   5.711 |              1.08 | Hungarian       |
| Online Shopping   | Regression     |                     8 |                2.655 |                   2.508 |              5.54 | Hungarian       |
| Air Quality       | Regression     |                     8 |               17.18  |                  11.61  |             32.41 | Hungarian       |
| Concrete          | Regression     |                     8 |                3.626 |                   3.417 |              5.76 | Hungarian       |
| Energy Efficiency | Regression     |                     8 |            24680     |               24680     |              0    | Tie             |
| Real Estate       | Regression     |                     8 |             8347     |                8347     |              0    | Tie             |

## Table 3. Statistical significance

| Comparison                           |   N |   Mean_Difference_Greedy_minus_Hungarian |   Effect_RankBiserial | Test                                                  |   p_value | Conclusion                                      |
|:-------------------------------------|----:|-----------------------------------------:|----------------------:|:------------------------------------------------------|----------:|:------------------------------------------------|
| Hungarian vs Greedy – Classification |   9 |                                   0.1318 |                 0.6   | Wilcoxon signed-rank (paired; H1: Greedy > Hungarian) | 0.06445   | No significant difference at α=0.05             |
| Hungarian vs Greedy – Regression     |   6 |                                   1.012  |                 1     | Wilcoxon signed-rank (paired; H1: Greedy > Hungarian) | 0.01562   | Hungarian significantly better (lower distance) |
| Overall (15 datasets)                |  15 |                                   0.4837 |                 0.767 | Wilcoxon signed-rank (paired; H1: Greedy > Hungarian) | 0.003357  | Hungarian significantly better (lower distance) |
| Overall (dataset×generator units)    | 120 |                                   0.4837 |                 0.773 | Wilcoxon signed-rank (paired; H1: Greedy > Hungarian) | 2.438e-16 | Hungarian significantly better (lower distance) |