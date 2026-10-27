#!/bin/bash -l
#SBATCH --job-name=forge_synth_quality
#SBATCH --account=ul01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/forge_synth_%A_%a.out
#SBATCH --error=logs/forge_synth_%A_%a.err

# Parallel SLURM job array — one task per dataset in datasets.json.
#
# Submit from the hive folder on the cluster:
#   cd python_scripts/hive
#   mkdir -p logs
#   sbatch submit_parallel.slurm
#
# Optional overrides before sbatch:
#   export FORGE_PROJECT_ROOT=/path/to/Single\ run_Data_leak_Synth_Quality
#   export FORGE_CONDA_ENV=your-env-name
#
# Dataset index (SLURM_ARRAY_TASK_ID):
#   0 = Cancer (python cancer.py)
#   1 = MAGIC Gamma Telescope (python cancer.py)
#   2 = Adult (nbconvert adult.ipynb)
#   3 = Bank Marketing (nbconvert Bank_marketing.ipynb)
#   4 = Wine Quality (nbconvert script.ipynb)
#   5 = CDC Diabetes Health Indicators (nbconvert cdc_diabetes.ipynb)
#   6 = Secondary Mushroom (nbconvert mushroom.ipynb)
#   7 = Forest Cover Type (nbconvert forest_cover.ipynb)
#
# Adjust --account, --partition, and --gres for your Hive/ICHEC project.

set -euo pipefail

HIVE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${HIVE_DIR}/logs"

echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Array task ID : ${SLURM_ARRAY_TASK_ID:-0}"
echo "Submit dir    : ${HIVE_DIR}"
echo "Node          : $(hostname)"
echo "Start time    : $(date -Is)"

bash "${HIVE_DIR}/run_one_dataset.sh" "${SLURM_ARRAY_TASK_ID:-0}"

echo "End time      : $(date -Is)"
