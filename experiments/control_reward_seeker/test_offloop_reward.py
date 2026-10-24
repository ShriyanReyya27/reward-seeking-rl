"""launch/offloop_reward.py against the fork's real VerlEnv.

Needs the fork and its dependencies (tinker, chz, pandas, numpy), so it runs in the fork's venv
on the training host and is skipped elsewhere:

    .cache/tinker-cookbook/.venv/bin/python -m pytest experiments/control_reward_seeker/test_offloop_reward.py
"""

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTROL / ".cache/tinker-cookbook"))
sys.path.insert(0, str(CONTROL / "launch"))

verl_env = pytest.importorskip("tinker_cookbook.recipes.math_rl.verl_env")
import offloop_reward  # noqa: E402

SLEEP = 0.4


def slow_reward_fn(**kwargs):
    time.sleep(SLEEP)
    return {"score": 1.0, "reward/score": 1.0, "seen": sorted(kwargs)}


def fake_env():
    """A VerlEnv with just the state _finish_episode touches: no sandbox, no files to fetch."""
    env = verl_env.VerlEnv.__new__(verl_env.VerlEnv)
    env.row = {"extra_info": {}, "data_source": "control/x/reward_isolated_tests", "ground_truth": ""}
    env._files_to_fetch = None
    env._session_id = None
    env.sandbox_client = None
    env.reward_fn = slow_reward_fn
    env.tracer = None
    env._current_turn = 3
    env.renderer = types.SimpleNamespace(get_stop_sequences=lambda: [])
    return env


async def finish_many(n):
    envs = [fake_env() for _ in range(n)]
    start = time.monotonic()
    results = await asyncio.gather(*(env._finish_episode("final answer") for env in envs))
    return time.monotonic() - start, results


def test_unpatched_grading_is_serial(monkeypatch):
    """Negative control: the fork as shipped blocks the loop, so 4 gradings take 4 sleeps."""
    monkeypatch.setattr(verl_env.VerlEnv, "_finish_episode", verl_env.VerlEnv.__dict__["_finish_episode"])
    monkeypatch.setattr(verl_env.VerlEnv, "_rsrl_offloop_reward", False, raising=False)
    elapsed, _ = asyncio.run(finish_many(4))
    assert elapsed >= 4 * SLEEP


def test_patched_grading_is_concurrent_and_unchanged():
    offloop_reward.install()
    offloop_reward.install()  # idempotent
    assert verl_env.VerlEnv._rsrl_offloop_reward
    assert "_rsrl_grade" in verl_env.VerlEnv._finish_episode.__code__.co_names

    elapsed, results = asyncio.run(finish_many(8))
    assert elapsed < 3 * SLEEP, elapsed  # 8 sleeps in parallel on the pool, not 8 in a row
    for r in results:
        assert r.reward == 1.0 and r.episode_done
        assert r.metrics["turn"] == 3 and r.metrics["sandbox_error"] is False
        assert r.metrics["score"] == 1.0
        assert r.metrics["seen"] == ["data_source", "extra_info", "ground_truth", "solution_str"]


def test_install_refuses_unknown_source(monkeypatch):
    async def other(self, solution_str, sandbox_error=False):
        return None

    monkeypatch.setattr(verl_env.VerlEnv, "_finish_episode", other)
    monkeypatch.setattr(verl_env.VerlEnv, "_rsrl_offloop_reward", False, raising=False)
    with pytest.raises(RuntimeError, match="has changed"):
        offloop_reward.install()
