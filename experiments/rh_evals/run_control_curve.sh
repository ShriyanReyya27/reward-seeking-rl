#!/bin/bash
# Grader-conflict probe on the 14 control curriculum checkpoints (arms c<step> in tinker_client.py).
# usage: run_control_curve.sh <run name> [extra run_single.py args, e.g. --n 2 --samples 1 --budget-usd 1]
cd "$(dirname "$0")/../.."
set -a; source ~/.rsrl_env; set +a
ARMS=c0,c20,c40,c100,c160,c170,c180,c200,c220,c240,c250,c260,c270,c280
RUN=$1; shift
uv run --with tinker python experiments/rh_evals/run_single.py --run "$RUN" --evals grader_conflict --arms "$ARMS" "$@"
