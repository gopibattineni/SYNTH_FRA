#!/usr/bin/env bash
# Launch cuda0–cuda3 in parallel (one dataset group per GPU), then aggregate.
set -euo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
PYTHON="${PYTHON:-$BATCH/../../.venv/bin/python}"

echo "Starting 4 GPU workers from $BATCH"
echo "Python: $PYTHON"
nvidia-smi -L || true

bash "$BATCH/cuda0.sh" &
p0=$!
bash "$BATCH/cuda1.sh" &
p1=$!
bash "$BATCH/cuda2.sh" &
p2=$!
bash "$BATCH/cuda3.sh" &
p3=$!

echo "PIDs  cuda0=$p0  cuda1=$p1  cuda2=$p2  cuda3=$p3"
echo "Logs  logs/cuda{0,1,2,3}.log"

fail=0
for spec in "cuda0:$p0" "cuda1:$p1" "cuda2:$p2" "cuda3:$p3"; do
  name="${spec%%:*}"
  pid="${spec##*:}"
  if wait "$pid"; then
    echo "$name finished OK"
  else
    echo "$name failed (exit $?)"
    fail=1
  fi
done

echo "Aggregating mean ± SD → Excel (classification + regression)"
"$PYTHON" -u run_experiment.py --aggregate-only
exit "$fail"
