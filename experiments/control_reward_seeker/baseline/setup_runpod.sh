#!/usr/bin/env bash
# Prepare a RunPod (or any Linux) machine for baseline/rollout.py. Run from
# experiments/control_reward_seeker after syncing the project with the fork included:
#
#   bash baseline/setup_runpod.sh
#
# Installs pigz (+ proot, the fallback backend), a Python 3.11 venv with the fork's dependencies (CPU torch only) at
# $VENV, and unpacks the training sandbox's root filesystem to $RSRL_SANDBOX_ROOTFS (~10 GB
# download, ~25-30 GB on disk). Everything lives outside the synced tree, so a later sync()
# can't delete it. Needs TINKER_API_KEY at run time, not here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
VENV="${VENV:-/root/rsrl-baseline-venv}"
ROOTFS="${RSRL_SANDBOX_ROOTFS:-/root/sandbox-rootfs}"
FORK="$HERE/.cache/tinker-cookbook"
UPSTREAM_REWARD="$HERE/.cache/reward_seeker/environments/reward/reward.py"

[ -f "$FORK/tinker_cookbook/recipes/math_rl/verl_env.py" ] || { echo "missing $FORK (sync it with include=[...])"; exit 1; }
[ -f "$UPSTREAM_REWARD" ] || { echo "missing $UPSTREAM_REWARD"; exit 1; }
[ "$(git -C "$FORK" rev-parse HEAD 2>/dev/null || echo unknown)" = d4bd2cb573de4471b168dccba28741290a41a46e ] \
    || echo "warning: fork is not at the pinned d4bd2cb (or has no .git)"

if ! command -v proot >/dev/null || ! command -v pigz >/dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq proot pigz >/dev/null
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --quiet --python 3.11 "$VENV"
    uv pip install --quiet --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
    # Non-editable: rollout.py imports the fork from the synced tree (sys.path), the venv only
    # provides its dependencies. Same extras as launch/setup_host.sh, plus psutil.
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --quiet --python "$VENV/bin/python" "$FORK[wandb,math-rl]" \
        filelock python-dotenv requests pandas pyarrow jinja2 pytest psutil
fi

"$VENV/bin/python" baseline/fetch_rootfs.py "$ROOTFS"
df -h "$(dirname "$ROOTFS")" | tail -1

# The default backend needs chroot (RunPod allows it); otherwise use --backend proot.
# These are our commands, not the model's.
if chroot "$ROOTFS" /usr/bin/env PATH=/root/miniconda3/bin:/usr/bin:/bin /bin/bash -c \
    'python --version && python3 --version && head -2 /etc/os-release'; then
    echo "chroot works: use the default --backend chroot"
else
    echo "chroot not permitted here: run rollout.py with --backend proot"
fi
echo
echo "Ready. Run e.g.:"
echo "  $VENV/bin/python baseline/rollout.py --run-name pilot --rows-per-env 2 --samples-per-row 2"
