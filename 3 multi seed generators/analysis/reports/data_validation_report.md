# Data Validation Report — 10-Seed Evaluation

## Design targets

- Datasets: **15** (classification 9, regression 6)
- Generators: **8**
- Unique seeds: **10** → `[42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]`
- Expected configurations: **120**
- Expected generator runs: **1200**
- Expected classification runs: **720**
- Expected regression runs: **480**

## Observed completion

- Completed generator runs: **1200** / 1200
- Classification runs: **720** / 720
- Regression runs: **480** / 480

## Batch summary

Batch,Seeds,Expected Runs,Completed Runs,Status
1st batch,"42, 123, 2024",360,360,OK
2nd batch,"68, 91, 2025",360,360,OK
3rd batch,"55, 155, 255, 355",480,480,OK
Total,10 seeds,1200,1200,OK


## Per-seed validation

batch,batch_label,seed,classification_runs,regression_runs,total_runs,expected_runs,status
1st_batch,1st batch,42,72,48,120,120,OK
1st_batch,1st batch,123,72,48,120,120,OK
1st_batch,1st batch,2024,72,48,120,120,OK
2nd_batch,2nd batch,68,72,48,120,120,OK
2nd_batch,2nd batch,91,72,48,120,120,OK
2nd_batch,2nd batch,2025,72,48,120,120,OK
3rd_batch,3rd batch,55,72,48,120,120,OK
3rd_batch,3rd batch,155,72,48,120,120,OK
3rd_batch,3rd batch,255,72,48,120,120,OK
3rd_batch,3rd batch,355,72,48,120,120,OK


## Integrity checks

- Missing dataset×generator×seed runs: **0**
- Duplicate dataset×generator×seed rows: **0**
- Unexpected generators: `none`
- Unexpected datasets: `none`
- NaN metric_value cells: **90**
- Infinite metric_value cells: **0**

## Notes

- Aggregation for manuscript results uses the **10 individual seed observations** (sample SD, `ddof=1`), not the mean of batch means.
- Batches are execution/provenance groups only.
