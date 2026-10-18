#!/usr/bin/env bash
# Prepare a training host for the oss430 control curriculum, then print the launch command.
#
#   bash launch/setup_host.sh        # from experiments/control_reward_seeker on the host
#
# The host needs: Ubuntu, Docker with --cap-add SYS_ADMIN allowed (a real VM; RunPod/vast
# containers can't run it), ~16+ cores, ~32 GB RAM, outbound internet, and in the environment:
# TINKER_API_KEY, WANDB_API_KEY and GH_TOKEN (read access to aghyad-deeb/tinker-cookbook, unless
# .cache/tinker-cookbook was copied over). Tinker does all GPU work; this machine runs the agent
# loop, the bash sandboxes and the graders.
#
# Env overrides: SANDBOX_NUM_CONTAINERS (default 64, as in Aghyad's run_oss.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# Pinned versions. Aghyad's run logged fork commit 0771158 (2026-03-26), which is no longer on any
# branch. Its nearest surviving commit, 92b02f9, predates the bash tool-calling path the data
# needs, so we use the fork's master as of 2026-06-10 (README.md, "Fidelity gaps").
FORK_COMMIT=d4bd2cb573de4471b168dccba28741290a41a46e
SANDBOX_REPO=https://github.com/aghyad-deeb/sandbox-fusion-sessions
SANDBOX_COMMIT=95dcb26824098d3a4ba3af069508e41a440c28b9
N="${SANDBOX_NUM_CONTAINERS:-64}"

: "${TINKER_API_KEY:?set TINKER_API_KEY}" "${WANDB_API_KEY:?set WANDB_API_KEY}"
command -v docker >/dev/null || { echo "docker is required"; exit 1; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# 1. Sandboxes: Aghyad's SandboxFusion with overlay bash sessions (every row uses
#    agent_name bash_loop_custom_tools_overlay, which needs CAP_SYS_ADMIN for its mounts).
mkdir -p .cache
if [ ! -d .cache/sandbox-fusion-sessions ]; then
    git clone --quiet "$SANDBOX_REPO" .cache/sandbox-fusion-sessions
fi
git -C .cache/sandbox-fusion-sessions -c advice.detachedHead=false checkout --quiet "$SANDBOX_COMMIT"
bash .cache/sandbox-fusion-sessions/start.sh "$N" 60808

# 2. Trainer: the fork in its own venv. CPU torch only (Tinker trains remotely).
if [ ! -d .cache/tinker-cookbook ]; then
    : "${GH_TOKEN:?set GH_TOKEN, or copy the fork to .cache/tinker-cookbook}"
    git clone --quiet "https://x-access-token:${GH_TOKEN}@github.com/aghyad-deeb/tinker-cookbook" .cache/tinker-cookbook
fi
git -C .cache/tinker-cookbook fetch --quiet origin || true
git -C .cache/tinker-cookbook -c advice.detachedHead=false checkout --quiet "$FORK_COMMIT"
cd .cache/tinker-cookbook
uv venv --quiet --python 3.11 .venv
uv pip install --quiet --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
# The fork + wandb + math-rl (its train.py imports math_grading, which needs pylatexenc).
# Upstream reward.py needs filelock, python-dotenv, requests and transformers; its generators
# need jinja2.
uv pip install --quiet --python .venv/bin/python -e ".[wandb,math-rl]" filelock python-dotenv requests pandas pyarrow jinja2 pytest
# Import the trainer, our stage wrapper's targets and the reward wrapper before any paid call.
.venv/bin/python -c "import tinker_cookbook.recipes.math_rl.train, tinker_cookbook.recipes.math_rl.verl_env, tinker_cookbook.utils.ml_log"
cd "$HERE"
PY="$HERE/.cache/tinker-cookbook/.venv/bin/python"

# 3. Data: regenerate both arms from upstream's pinned generators, then the stage files.
"$PY" build.py
"$PY" curriculum.py
"$PY" -m pytest -q test_grader.py test_controls.py test_curriculum.py
RSRL_FORMAT_REWARD=0 "$PY" -c "import importlib.util as u; s = u.spec_from_file_location('m', 'reward_control.py'); m = u.module_from_spec(s); s.loader.exec_module(m)"

ENDPOINTS=$(seq -s, -f "http://localhost:%g" 60808 $((60808 + N - 1)))
cat <<MSG

Ready. Start (or resume) the control run in tmux:

  tmux new -s control
  export SANDBOX_FUSION_ENDPOINTS="$ENDPOINTS"
  $PY launch/run_curriculum.py --arm control

Logs: runs/control/stage<k>.log, metrics in wandb project rsrl-oss430-control.
MSG
