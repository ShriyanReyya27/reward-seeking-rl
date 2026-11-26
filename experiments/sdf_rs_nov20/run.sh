#!/bin/bash
# Run an sdf_rs_nov20 script with keys loaded and dependencies:  bash run.sh script.py args
R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl
set -a; . "$R/.env.local"; set +a
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1
cd "$R/experiments/sdf_rs_nov20"
exec uv run -q --python 3.12 --with tinker --with openai --with datasets --with numpy --with matplotlib --with transformers \
  --with /private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/kit2/tinker-cookbook-master \
  python "$@"
