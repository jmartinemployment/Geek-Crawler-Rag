#!/usr/bin/env bash
# Durable Phase M backfill loop — keeps going until scanned=0.
set +e
cd "$(dirname "$0")/.."
PY="${PWD}/.venv/bin/python"
LOG="${BACKFILL_LOG:-/tmp/rag_backfill_full.log}"
LIMIT="${BACKFILL_LIMIT:-1500}"
BATCH="${BACKFILL_BATCH:-40}"
SUMMARY="/tmp/rag-backfill/last_pass_summary.txt"

if [[ -z "${MONGO_CRAWLER_URL:-}" ]]; then
  echo "MONGO_CRAWLER_URL required" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "missing venv python: $PY" >&2
  exit 1
fi

mkdir -p /tmp/rag-backfill
echo "$$" > /tmp/rag_backfill_full.pid
echo "loop_start pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"

pass=0
empty_streak=0
while true; do
  pass=$((pass + 1))
  echo "=== PASS $pass $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  # Stream live; also keep a copy of the final summary lines.
  : >"$SUMMARY"
  "$PY" -u scripts/backfill_markdown.py --write --limit "$LIMIT" --batch-size "$BATCH" 2>&1 \
    | tee -a "$LOG" \
    | tee "$SUMMARY"
  rc=${PIPESTATUS[0]}
  echo "pass_exit=$rc" | tee -a "$LOG"

  scanned=$(awk -F= '/^scanned=/{print $2; exit}' "$SUMMARY")
  updated=$(awk -F= '/^updated=/{print $2; exit}' "$SUMMARY")
  scanned=${scanned:-0}
  updated=${updated:-0}
  echo "LOOP_PASS pass=$pass scanned=$scanned updated=$updated rc=$rc $(date -u +%H:%M:%S)" | tee -a "$LOG"

  if [[ "$scanned" -eq 0 ]]; then
    empty_streak=$((empty_streak + 1))
  else
    empty_streak=0
  fi

  # Two consecutive empty passes = corpus done (guards against transient empty find).
  if [[ "$empty_streak" -ge 2 ]]; then
    echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) passes=$pass" | tee -a "$LOG"
    date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/rag-backfill/DONE
    rm -f /tmp/rag_backfill_full.pid
    exit 0
  fi

  if [[ "$rc" -ne 0 ]]; then
    sleep 5
  else
    sleep 1
  fi
done
