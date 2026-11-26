#!/bin/bash
# Run a screen_neutral script with keys loaded and all dependencies:  bash run.sh run.py if:main sort:a
R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl
set -a; . "$R/.env.local"; set +a
export TOKENIZERS_PARALLELISM=false SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1
cd "$R/experiments/screen_neutral"
exec uv run -q --python 3.12 --with tinker --with torch --with pandas --with pyarrow --with nltk --with langdetect \
  --with absl-py --with immutabledict \
  --with /private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/kit/tinker-cookbook-master \
  python "$@"
