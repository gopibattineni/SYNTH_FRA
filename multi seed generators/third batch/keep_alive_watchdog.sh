#!/usr/bin/env bash
# Survives laptop lock / SSH drop: ensure second-batch dispatcher + GPU watchdog stay up.
set -uo pipefail
BATCH="$(cd "$(dirname "$0")" && pwd)"
cd "$BATCH"
WLOG="$BATCH/logs/keep_alive_watchdog.log"
mkdir -p "$BATCH/logs"

echo "$(date -Is) keep_alive tick" >>"$WLOG"

ensure() {
  local name="$1" pidfile="$2" start_cmd="$3"
  local pid=""
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$(date -Is) OK $name pid=$pid" >>"$WLOG"
    return 0
  fi
  echo "$(date -Is) RESTART $name" >>"$WLOG"
  # Detached from tty/session so lock/logout does not kill it
  setsid bash -c "$start_cmd" </dev/null >>"$WLOG" 2>&1 &
  echo $! > "$pidfile"
  echo "$(date -Is) started $name pid=$(cat "$pidfile")" >>"$WLOG"
}

ensure "keep_four_busy" "$BATCH/keep_four_busy.pid" \
  "echo \$\$ > '$BATCH/keep_four_busy.pid'; exec env CTABGAN_EPOCHS=150 bash '$BATCH/keep_four_busy.sh'"

ensure "gpu_busy_watchdog" "$BATCH/gpu_busy_watchdog.pid" \
  "echo \$\$ > '$BATCH/gpu_busy_watchdog.pid'; exec env POLL_SEC=10 bash '$BATCH/gpu_busy_watchdog.sh'"

ensure "auto_aggregate_watch" "$BATCH/auto_aggregate.pid" \
  "echo \$\$ > '$BATCH/auto_aggregate.pid'; exec bash '$BATCH/auto_aggregate_watch.sh'"
