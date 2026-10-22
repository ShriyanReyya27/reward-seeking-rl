#!/bin/bash
# Several run.py invocations in sequence: each arg is a quoted run.py arg string.
set -eo pipefail
source experiments/prompted_reward_seeker/setup.sh
for a in "$@"; do echo "=== run.py $a $(date)"; eval python experiments/signal_search/run.py $a; done
echo "=== JOB DONE $(date)"
