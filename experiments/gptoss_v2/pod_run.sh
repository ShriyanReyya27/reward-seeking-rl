#!/bin/bash
# Run on a keyless pod after setup and model download (and prepare_adapter.py for the hacker):
#   bash experiments/gptoss_v2/pod_run.sh base|reward_hacker
set -euo pipefail
cd /workspace/rsrl
source experiments/prompted_reward_seeker/setup.sh
export WANDB_MODE=offline WANDB_DIR=/workspace/rsrl/results/gptoss_v2  # no keys on pods; sync later
mkdir -p results/gptoss_v2
python experiments/gptoss_v2/run_v2.py --arm "$1" --out-dir results/gptoss_v2
python experiments/gptoss_reward_hacker/run_coding.py --arm "$1" --out-dir results/gptoss_v2 --conditions B2 --wandb
echo "=== V2 DONE $(date)"
