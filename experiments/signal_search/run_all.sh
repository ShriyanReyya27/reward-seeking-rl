#!/bin/bash
# Run on a fresh RunPod L40S from the project root:
#   nohup bash experiments/signal_search/run_all.sh Qwen/Qwen3-8B <conditions> [extra run.py args] > results/signal_search/<log> 2>&1 &
set -eo pipefail
MODEL=$1; CONDS=$2; shift 2
source experiments/prompted_reward_seeker/setup.sh
echo "=== start $MODEL $(date)"
python experiments/signal_search/run.py --model "$MODEL" --conditions "$CONDS" "$@"
echo "=== JOB DONE $(date)"
