#!/usr/bin/env bash
# Pin this process to one GPU and run the given datasets (resume-safe).
# Usage: run_cuda_worker.sh <gpu_id> <dataset> [dataset ...]
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_id> <dataset> [dataset ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
DATASETS=("$@")

BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

REPO="$(cd "$BATCH/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
LOGDIR="$BATCH/logs"
LOG="$LOGDIR/cuda${GPU_ID}.log"
PIDFILE="$BATCH/cuda${GPU_ID}.pid"
mkdir -p "$LOGDIR"

DS_ARGS=()
for ds in "${DATASETS[@]}"; do
  DS_ARGS+=(--dataset "$ds")
done

# Optional: GENERATORS="CTABGAN TabDDPM" — default is all 8 (resume skips finished).
GEN_ARGS=()
if [[ -n "${GENERATORS:-}" ]]; then
  for g in ${GENERATORS}; do
    GEN_ARGS+=(--generator "$g")
  done
fi

{
  echo "============================================================"
  echo "$(date -Is) start GPU=${GPU_ID} datasets=${DATASETS[*]} generators=${GENERATORS:-ALL} python=${PYTHON}"
  nvidia-smi -L 2>/dev/null | sed -n "$((GPU_ID + 1))p" || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"
} | tee -a "$LOG"

echo $$ > "$PIDFILE"

"$PYTHON" -u run_experiment.py \
  --skip-aggregate \
  "${DS_ARGS[@]}" \
  "${GEN_ARGS[@]}" \
  >> "$LOG" 2>&1

STATUS=$?
echo "$(date -Is) exit=${STATUS} GPU=${GPU_ID}" | tee -a "$LOG"
rm -f "$PIDFILE"
exit "$STATUS"
