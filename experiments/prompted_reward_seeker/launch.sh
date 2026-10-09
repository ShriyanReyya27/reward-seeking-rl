#!/bin/bash
# Set up a fresh RunPod machine and start one GRPO arm in the background.
#   [RUN_SUFFIX=-hint] bash experiments/prompted_reward_seeker/launch.sh <arm> [extra grpo.py args]
# Run from the project root on the machine. Logs go to results/ (sync leaves it alone).
set -euo pipefail
ARM=$1; shift
SUFFIX=${RUN_SUFFIX:-}
source experiments/prompted_reward_seeker/setup.sh
mkdir -p results
nohup python experiments/prompted_reward_seeker/grpo.py --arm "$ARM" --run-suffix="$SUFFIX" "$@" > "results/grpo_$ARM$SUFFIX.log" 2>&1 &
echo "started $ARM$SUFFIX (pid $!)"
