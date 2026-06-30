#!/usr/bin/env bash
# Autonomous-session watchdog. Writes elapsed/remaining time so the agent can
# pace itself across the user's ~10h absence (user underestimates elapsed time).
HORIZON_MIN=${1:-600}          # default 10 hours
TICK_MIN=${2:-10}              # log every 10 min
OUT=${3:-/home/t-nagupta/.copilot/session-state/03024b2b-1571-4fc0-9e5f-c1054b89e7f7/files/watchdog.log}
START=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S %z')
echo "WATCHDOG START $START_HUMAN | horizon=${HORIZON_MIN}min tick=${TICK_MIN}min" > "$OUT"
while true; do
  NOW=$(date +%s)
  ELAPSED=$(( (NOW - START) / 60 ))
  REMAIN=$(( HORIZON_MIN - ELAPSED ))
  STAMP=$(date '+%Y-%m-%d %H:%M:%S %z')
  if [ "$REMAIN" -le 0 ]; then
    echo "[$STAMP] elapsed=${ELAPSED}min remaining=0min -- HORIZON REACHED, user should be back ~now. KEEP WORKING until they return." >> "$OUT"
    break
  fi
  echo "[$STAMP] elapsed=${ELAPSED}min remaining=${REMAIN}min (~$(( REMAIN / 60 ))h$(( REMAIN % 60 ))m left)" >> "$OUT"
  sleep $(( TICK_MIN * 60 ))
done
