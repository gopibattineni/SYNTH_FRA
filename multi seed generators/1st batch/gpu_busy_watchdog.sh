#!/usr/bin/env bash
# Keep Batch-1 GPUs busy after laptop lock / SSH drop.
# - Restarts keep_four_busy.sh if it dies
# - Restarts auto_aggregate_watch.sh if it dies
# - Clears stale cuda*.pid so free GPUs get refilled
set -uo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
WLOG="$BATCH/logs/gpu_busy_watchdog.log"
PIDFILE="$BATCH/gpu_busy_watchdog.pid"
POLL_SEC="${POLL_SEC:-60}"
mkdir -p "$BATCH/logs"

echo $$ > "$PIDFILE"
echo "$(date -Is) gpu_busy_watchdog start poll=${POLL_SEC}s" >>"$WLOG"

dispatcher_alive() {
  pgrep -f 'bash keep_four_busy\.sh' >/dev/null 2>&1
}

aggregator_alive() {
  pgrep -f 'bash auto_aggregate_watch\.sh' >/dev/null 2>&1
}

start_dispatcher() {
  setsid nohup bash keep_four_busy.sh </dev/null >/dev/null 2>&1 &
  echo $! > "$BATCH/keep_four_busy.pid"
  echo "$(date -Is) RESTART keep_four_busy pid=$(cat "$BATCH/keep_four_busy.pid")" >>"$WLOG"
}

start_aggregator() {
  setsid nohup bash auto_aggregate_watch.sh </dev/null >/dev/null 2>&1 &
  echo $! > "$BATCH/auto_aggregate.pid"
  echo "$(date -Is) RESTART auto_aggregate_watch pid=$(cat "$BATCH/auto_aggregate.pid")" >>"$WLOG"
}

clear_stale_pidfiles() {
  local g pid
  for g in 0 1 2 3 4 5; do
    if [[ -f "$BATCH/cuda${g}.pid" ]]; then
      pid="$(cat "$BATCH/cuda${g}.pid" 2>/dev/null || true)"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$BATCH/cuda${g}.pid"
        echo "$(date -Is) cleared stale cuda${g}.pid (was ${pid:-empty})" >>"$WLOG"
      fi
    fi
  done
  # Drop dead in_progress claims so resume can reclaim seeds
  find "$BATCH/classification/results/raw" "$BATCH/regression/results/raw" \
    -name 'in_progress.pid' 2>/dev/null | while read -r f; do
      pid="$(cat "$f" 2>/dev/null || true)"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$f"
        echo "$(date -Is) cleared stale claim $f" >>"$WLOG"
      fi
    done
}

while true; do
  clear_stale_pidfiles
  if ! dispatcher_alive; then
    start_dispatcher
  fi
  if ! aggregator_alive; then
    start_aggregator
  fi
  # Heartbeat
  busy=0
  for g in 0 1 2 3 4 5; do
    if [[ -f "$BATCH/cuda${g}.pid" ]] && kill -0 "$(cat "$BATCH/cuda${g}.pid")" 2>/dev/null; then
      busy=$((busy + 1))
    fi
  done
  echo "$(date -Is) OK busy_gpus=${busy}/6 dispatcher=$(dispatcher_alive && echo up || echo down) aggregate=$(aggregator_alive && echo up || echo down)" >>"$WLOG"
  sleep "$POLL_SEC"
done
