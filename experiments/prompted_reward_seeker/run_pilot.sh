#!/bin/bash
# Zero-shot pilot for one model: every arm on healthbench and countdown_leaky
# (bare-target leak, with and without hint). Run one machine per model, from
# the project root on a fresh RunPod machine (48GB card):
#   nohup bash experiments/prompted_reward_seeker/run_pilot.sh Qwen/Qwen3-8B > results/pilot_8b.log 2>&1 &
# Results go to results/pilot/. Settings are in pilot.py.
set -euo pipefail
MODEL=$1; shift  # remaining args go to pilot.py, e.g. --conditions healthbench
source experiments/prompted_reward_seeker/setup.sh
VIRTUAL_ENV=/root/venv uv pip install -q -e '.[healthbench]'
# Keep the HealthBench data and judge cache (with the judge's explanations)
# under results/ so they're downloaded with everything else.
export RSRL_CACHE_DIR="results/pilot/cache/$(basename "$MODEL")"
echo "=== start $MODEL $(date)"
python experiments/prompted_reward_seeker/pilot.py --model "$MODEL" --out-dir results/pilot "$@"
echo "=== PILOT DONE $MODEL $(date)"
