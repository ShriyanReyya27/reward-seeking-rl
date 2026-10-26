#!/bin/bash
# Restart wrapper for the batch 4 control runs (same logic as run.sh). usage: run_ctl.sh TRAIN_SCRIPT RUN OUT_DIR [args...]
script=$1; run=$2; out=$3; shift 3
cd "$(dirname "$0")/../.."
set -a; . /root/.keys; set +a
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
mkdir -p "$out/logs"
for attempt in 1 2 3 4; do
  /opt/venv/bin/python "$script" --run "$run" --out-dir "$out" "$@" >> "$out/logs/$run.log" 2>&1
  code=$?
  [ -f "$out/$run/DONE" ] && exit 0
  if [ $code -eq 3 ]; then echo "- $(date -u '+%F %T UTC') [$run] billing stop (exit 3); not restarting" >> "$out/decisions.md"; exit 3; fi
  if [ $attempt -ge 4 ]; then echo "- $(date -u '+%F %T UTC') [$run] crashed 3 times; stopped" >> "$out/decisions.md"; exit 1; fi
  echo "- $(date -u '+%F %T UTC') [$run] crashed (exit $code), restart $attempt/3 from last checkpoint" >> "$out/decisions.md"
  sleep 60
done
