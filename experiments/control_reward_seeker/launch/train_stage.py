"""Run one curriculum stage with the fork's ``tinker_cookbook.recipes.math_rl.train``.

    RSRL_STOP_AT=40 python train_stage.py --config runs/control/configs/stage1.yaml

The stage ends the way Aghyad's did: training runs on an effectively endless dataset (no
``max_steps``), and the process stops at the handoff checkpoint. Checkpoint ``N`` is saved right
after the N-th optimizer update. Once checkpoint ``RSRL_STOP_AT`` is saved and that update's
metrics are logged, the process exits. The next stage resumes from that checkpoint.

Why not ``max_steps``: with it, the fork's async dataloader submits only ``max_steps - start``
dataset batches. ``VerlDataset.get_batch`` truncates at the end of the file (106 rows give
batches of 48, 48, 10, ...), while every update waits for 48 groups. The loader then runs dry
early: stage 1 would get 30 updates, the whole run about 400. The fork would still save the end
checkpoint as if all steps had run.

Two more guards over running the fork directly:
- An episode whose sandbox session can't be created raises an error. The fork's default is to
  fall back silently to a single-turn episode with no tools, and train on it.
- An end-of-run checkpoint (only reached if the dataset runs out) is named by batch, not "final".
"""

import os
import runpy
import sys

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes.math_rl import verl_env
from tinker_cookbook.utils import ml_log

STOP_AT = int(os.environ["RSRL_STOP_AT"])
_saved_stop_checkpoint = False

_save = checkpoint_utils.save_checkpoint_async


async def _save_and_mark(*, name, loop_state, **kwargs):
    global _saved_stop_checkpoint
    if name == "final":
        name = f"{loop_state['batch']:06d}"
    paths = await _save(name=name, loop_state=loop_state, **kwargs)
    if loop_state.get("batch") == STOP_AT and "state_path" in paths:
        _saved_stop_checkpoint = True
    return paths


class _StopAfterHandoff:
    """Wraps the run's metrics logger and exits once the handoff update is logged."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def log_metrics(self, metrics, step=None):
        self._inner.log_metrics(metrics, step=step)
        # Metrics for update i are logged with step=i, after checkpoint i + 1 is saved.
        if _saved_stop_checkpoint and step == STOP_AT - 1:
            print(f"[train_stage] checkpoint {STOP_AT:06d} saved; stopping", flush=True)
            self._inner.close()
            sys.stdout.flush()
            os._exit(0)  # in-flight rollouts are discarded, as at Aghyad's handoffs


_setup_logging = ml_log.setup_logging


def _setup_logging_with_stop(*args, **kwargs):
    return _StopAfterHandoff(_setup_logging(*args, **kwargs))


_initial_observation = verl_env.VerlEnv.initial_observation


async def _initial_observation_requiring_sandbox(self):
    result = await _initial_observation(self)
    if self.row.get("extra_info", {}).get("tools_kwargs") is not None and not self._is_multi_turn:
        raise RuntimeError("sandbox session could not be created; refusing single-turn fallback")
    return result


checkpoint_utils.save_checkpoint_async = _save_and_mark
ml_log.setup_logging = _setup_logging_with_stop
verl_env.VerlEnv.initial_observation = _initial_observation_requiring_sandbox
sys.argv = ["train", *sys.argv[1:]]
runpy.run_module("tinker_cookbook.recipes.math_rl.train", run_name="__main__", alter_sys=True)
