#!/usr/bin/env bash
# Resume-safe watchdog for Batch 1 on patten (survives laptop lock / SSH drop).
set -uo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
LOG="$BATCH/results_run_all.log"
PIDFILE="$BATCH/run_all.pid"
WLOG="$BATCH/watchdog.log"

running_pid=""
if pgrep -f "^python -u run_experiment.py --all$" >/dev/null 2>&1; then
  running_pid=$(pgrep -f "^python -u run_experiment.py --all$" | head -1)
elif pgrep -f "python -u run_experiment.py --all" >/dev/null 2>&1; then
  running_pid=$(pgrep -f "python -u run_experiment.py --all" | head -1)
fi

if [[ -n "${running_pid}" ]]; then
  echo "$running_pid" > "$PIDFILE"
  echo "$(date -Is) OK running pid=$running_pid" >> "$WLOG"
  exit 0
fi

echo "$(date -Is) RESTART: no process found; launching --all (resume skips completed)" >> "$WLOG"
nohup python -u run_experiment.py --all >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
disown 2>/dev/null || true
echo "$(date -Is) started pid=$(cat "$PIDFILE")" >> "$WLOG"
