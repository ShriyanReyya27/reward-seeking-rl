#!/bin/bash
# Run a Nov 6 script with keys loaded and all dependencies:  bash run.sh train.py --exp logic_stated --model aghyad
R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl
set -a; . "$R/.env.local"; set +a
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1
cd "$R/experiments/nov6"
exec uv run -q --python 3.12 --with tinker --with reasoning-core --with reasoning-gym --with torch --with pandas \
  --with pyarrow --with /private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/kit/tinker-cookbook-master \
  python "$@"
