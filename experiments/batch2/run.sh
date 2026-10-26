#!/bin/bash
# Batch 2 copy of experiments/core_rl/run.sh. Restart wrapper for one run: resumes from its last checkpoint after a crash, gives up after 3 crashes,
# never retries a billing stop (exit 3).  usage: run.sh RUN OUT_DIR [train.py args...]
run=$1; out=$2; shift 2
cd "$(dirname "$0")/../.."
set -a; . /root/.keys; set +a
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
mkdir -p "$out/logs"
for attempt in 1 2 3 4; do
  /opt/venv/bin/python experiments/batch2/train.py --run "$run" --out-dir "$out" "$@" >> "$out/logs/$run.log" 2>&1
  code=$?
  [ -f "$out/$run/DONE" ] && exit 0
  if [ $code -eq 3 ]; then echo "- $(date -u '+%F %T UTC') [$run] billing stop (exit 3); not restarting" >> "$out/decisions.md"; exit 3; fi
  if [ $attempt -ge 4 ]; then
    echo "- $(date -u '+%F %T UTC') [$run] crashed 3 times; stopped (see logs/$run.log)" >> "$out/decisions.md"; exit 1
  fi
  echo "- $(date -u '+%F %T UTC') [$run] crashed (exit $code), restart $attempt/3 from last checkpoint" >> "$out/decisions.md"
  sleep 60
done
