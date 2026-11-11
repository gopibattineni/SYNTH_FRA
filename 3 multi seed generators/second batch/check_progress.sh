#!/usr/bin/env bash
# Show CUDA worker status + remaining-dataset completion counts.
set -euo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
RAW_CLF="$BATCH/classification/results/raw"
RAW_REG="$BATCH/regression/results/raw"
GENS="CTGAN CopulaGAN TVAE GaussianCopula CTABGAN WGAN_GP TabDDPM ForestDiffusion"
SEEDS="68 91 2025"

echo "=== GPU workers ==="
pgrep -af 'run_experiment.py|run_cuda_worker|run_all_cuda' || echo "(none running)"
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo
echo "=== latest log lines ==="
for i in 0 1 2 3; do
  f="$BATCH/logs/cuda${i}.log"
  echo "--- cuda${i}.log ---"
  if [[ -f "$f" ]]; then tail -n 8 "$f"; else echo "(missing)"; fi
  echo
done

count_ok() {
  local raw="$1" ds="$2" n=0
  for gen in $GENS; do
    for seed in $SEEDS; do
      xlsx="$raw/seed_${seed}/${ds}/${gen}/metrics.xlsx"
      csv="$raw/seed_${seed}/${ds}/${gen}/metrics.csv"
      if [[ -s "$xlsx" || -s "$csv" ]]; then n=$((n + 1)); fi
    done
  done
  echo "$n"
}

echo "=== remaining 11 datasets (metrics.xlsx / 24) ==="
printf "%-20s %s\n" "dataset" "complete"
echo "--------------------------------"
for ds in forest_cover wine cdc_diabetes mushroom magic; do
  n=$(count_ok "$RAW_CLF" "$ds")
  printf "%-20s %s/24\n" "$ds" "$n"
done
for ds in metro online_shopping air_quality concrete energy_efficiency real_estate; do
  n=$(count_ok "$RAW_REG" "$ds")
  printf "%-20s %s/24\n" "$ds" "$n"
done
echo
echo "Done datasets (untouched): cancer, alzhimers, adult, bank = 24/24 each"
echo "Excel (after aggregate): classification/results/aggregated/*.xlsx"
echo "                         regression/results/aggregated/*.xlsx"
