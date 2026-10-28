"""Roll out a Tinker checkpoint on control rows with baseline/rollout.py, on the host's SandboxFusion
containers (as training does) instead of the chroot sandbox, with grading off the event loop.

    SANDBOX_FUSION_ENDPOINTS=... python ckpt_eval.py --model-path tinker://.../sampler_weights/000220 \
        --run-name ckpt220_lns --envs leaked_notes_style --rows-per-env 400 --samples-per-row 2 \
        --without-tool-call-repair --concurrency 64
"""
import sys
from pathlib import Path

CONTROL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CONTROL / "baseline"))

import rollout  # noqa: E402  (puts the fork, launch/ and the control dir on sys.path)
import local_sandbox  # noqa: E402
import offloop_reward  # noqa: E402
from tinker_cookbook.recipes.math_rl.sandbox_client import SandboxClient  # noqa: E402


class FusionSandbox(SandboxClient):
    def __init__(self, **_ignored):
        super().__init__()  # endpoints from SANDBOX_FUSION_ENDPOINTS

    async def start(self):
        pass

    def pop_log(self, session_id):
        return [], [], []  # SandboxFusion keeps no command log; transcripts carry the commands


local_sandbox.LocalSandboxClient = FusionSandbox
offloop_reward.install()

if __name__ == "__main__":
    rollout.main()
