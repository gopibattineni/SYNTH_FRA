#!/usr/bin/env bash
# Watch for new metrics.xlsx files and rebuild Excel sheets immediately.
# Safe with parallel CUDA workers (aggregate() uses excel.lock).
set -uo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
LOGDIR="$BATCH/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/auto_aggregate.log"
PIDFILE="$BATCH/auto_aggregate.pid"
STAMP="$LOGDIR/auto_aggregate.stamp"
PYTHON="${PYTHON:-$BATCH/../../.venv/bin/python}"
POLL_SEC="${POLL_SEC:-20}"

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    echo "$(date -Is) already running pid=$old" | tee -a "$LOG"
    exit 0
  fi
  rm -f "$PIDFILE"
fi

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

fingerprint() {
  # count + newest mtime of every per-run metrics workbook
  find "$BATCH/classification/results/raw" "$BATCH/regression/results/raw" \
    -type f \( -name 'metrics.xlsx' -o -name 'metrics.csv' \) -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1
  find "$BATCH/classification/results/raw" "$BATCH/regression/results/raw" \
    -type f \( -name 'metrics.xlsx' -o -name 'metrics.csv' \) 2>/dev/null | wc -l
}

echo "============================================================" >>"$LOG"
echo "$(date -Is) auto_aggregate_watch start poll=${POLL_SEC}s python=$PYTHON" >>"$LOG"
echo "============================================================" >>"$LOG"

prev=""
if [[ -f "$STAMP" ]]; then
  prev="$(cat "$STAMP" 2>/dev/null || true)"
fi

while true; do
  cur="$(fingerprint | tr '\n' ' ')"
  if [[ -n "$cur" && "$cur" != "$prev" ]]; then
    echo "$(date -Is) change detected: $cur" >>"$LOG"
    # brief debounce so multi-seed writes settle
    sleep 5
    cur2="$(fingerprint | tr '\n' ' ')"
    if [[ "$cur2" != "$prev" ]]; then
      echo "$(date -Is) aggregating classification + regression..." >>"$LOG"
      if "$PYTHON" -u run_experiment.py --aggregate-only >>"$LOG" 2>&1; then
        echo "$(date -Is) aggregate OK" >>"$LOG"
        prev="$cur2"
        printf '%s\n' "$prev" >"$STAMP"
      else
        echo "$(date -Is) aggregate FAILED (will retry)" >>"$LOG"
      fi
    fi
  fi
  sleep "$POLL_SEC"
done
