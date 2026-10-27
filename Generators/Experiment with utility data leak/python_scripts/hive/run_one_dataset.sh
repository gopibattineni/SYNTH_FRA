#!/bin/bash -l
# Run a single dataset job (invoked by SLURM array or manually).
#
# Usage:
#   ./run_one_dataset.sh <dataset_index>
#   ./run_one_dataset.sh cancer
#
# Environment (optional):
#   FORGE_PROJECT_ROOT  - path to Single run_Data_leak_Synth_Quality
#   FORGE_PYTHON        - python executable (default: python)
#   FORGE_CONDA_ENV     - conda env name to activate before running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${FORGE_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HIVE_DIR="${SCRIPT_DIR}"
DATASETS_JSON="${HIVE_DIR}/datasets.json"
CANCER_PY="${PROJECT_ROOT}/python_scripts/1. Cancer/cancer.py"
PYTHON="${FORGE_PYTHON:-python}"

if [[ -n "${FORGE_CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${FORGE_CONDA_ENV}"
fi

pick_index() {
    local arg="$1"
    if [[ "${arg}" =~ ^[0-9]+$ ]]; then
        echo "${arg}"
        return
    fi
    "${PYTHON}" - "${DATASETS_JSON}" "${arg}" <<'PY'
import json, sys
path, ds_id = sys.argv[1], sys.argv[2].lower()
with open(path, encoding="utf-8") as f:
    datasets = json.load(f)
for i, ds in enumerate(datasets):
    if ds["id"].lower() == ds_id:
        print(i)
        break
else:
    raise SystemExit(f"Unknown dataset id: {ds_id}")
PY
}

TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
INDEX="$(pick_index "${TASK_ID}")"

read -r DS_ID DS_NAME UCI_ID NB_DIR TRAIN_CSV N_SAMPLES TEST_SIZE SEED OUTPUT RUNNER NB_FILE <<EOF
$("${PYTHON}" - "${DATASETS_JSON}" "${INDEX}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    ds = json.load(f)[int(sys.argv[2])]
print(
    ds["id"],
    ds["name"],
    ds["uci_id"],
    ds["notebook_dir"],
    ds["train_csv"],
    ds["n_samples"],
    ds["test_size"],
    ds["seed"],
    ds["output_file"],
    ds["runner"],
    ds.get("notebook", ""),
)
PY
)
EOF

WORK_DIR="${PROJECT_ROOT}/${NB_DIR}"
LOG_DIR="${HIVE_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Dataset : ${DS_NAME} (${DS_ID})"
echo "UCI ID  : ${UCI_ID}"
echo "Work dir: ${WORK_DIR}"
echo "Runner  : ${RUNNER}"
echo "Started : $(date -Is)"
echo "Host    : $(hostname)"
echo "============================================================"

cd "${WORK_DIR}"

if [[ "${RUNNER}" == "python" ]]; then
    "${PYTHON}" "${CANCER_PY}" \
        --uci-id "${UCI_ID}" \
        --n-samples "${N_SAMPLES}" \
        --test-size "${TEST_SIZE}" \
        --seed "${SEED}" \
        --train-csv "${TRAIN_CSV}" \
        --output-file "${OUTPUT_FILE}" \
        --notebook-dir "${WORK_DIR}"
elif [[ "${RUNNER}" == "nbconvert" ]]; then
    if [[ -z "${NB_FILE}" ]]; then
        echo "ERROR: notebook file not set for ${DS_ID}" >&2
        exit 1
    fi
    jupyter nbconvert \
        --to notebook \
        --execute "${NB_FILE}" \
        --output "executed_${NB_FILE}" \
        --ExecutePreprocessor.timeout=-1
else
    echo "ERROR: unknown runner '${RUNNER}'" >&2
    exit 1
fi

echo "Finished: $(date -Is)"
echo "Results : ${WORK_DIR}/${OUTPUT_FILE}"
