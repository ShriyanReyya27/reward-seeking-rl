"""launch/salvage_metrics.py: the per-turn invalid-JSON metrics, and the wrapper on the fork's VerlEnv."""

import asyncio
import sys
import types
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTROL / ".cache/tinker-cookbook"))
sys.path.insert(0, str(CONTROL / "launch"))

from salvage_metrics import INVALID, SALVAGED, turn_metrics  # noqa: E402

TOOL = {"role": "tool", "content": "ok", "name": "bash"}


def test_valid_call_that_ran():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [object()]}, TOOL]
    assert turn_metrics(msgs) == {INVALID: 0.0, SALVAGED: 0.0}


def test_invalid_call_salvaged_and_run():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [], "unparsed_tool_calls": [object()]}, TOOL]
    assert turn_metrics(msgs) == {INVALID: 1.0, SALVAGED: 1.0}


def test_invalid_call_not_salvaged():
    msgs = [{"role": "assistant", "content": "", "unparsed_tool_calls": [object()]}]
    assert turn_metrics(msgs) == {INVALID: 1.0, SALVAGED: 0.0}


def test_final_answer_without_tool_call():
    msgs = [{"role": "assistant", "content": "<answer>done</answer>"}]
    assert turn_metrics(msgs) == {INVALID: 0.0, SALVAGED: 0.0}


def test_no_assistant_message():
    assert turn_metrics([]) == {} and turn_metrics([TOOL]) == {}


def test_wrapper_adds_metrics_to_the_forks_step_result():
    verl_env = pytest.importorskip("tinker_cookbook.recipes.math_rl.verl_env")
    import salvage_metrics

    original = verl_env.VerlEnv.step
    calls = []

    async def fake_step(self, action, *, extra=None):
        calls.append(action)
        self._past_messages.append({"role": "assistant", "content": "", "unparsed_tool_calls": ["x"]})
        self._past_messages.append(dict(TOOL))
        return types.SimpleNamespace(metrics={"turn": 1})

    verl_env.VerlEnv.step = fake_step
    verl_env.VerlEnv._rsrl_salvage_metrics = False
    try:
        salvage_metrics.install()
        env = verl_env.VerlEnv.__new__(verl_env.VerlEnv)
        env._past_messages = [{"role": "user", "content": "task"}]
        env._is_multi_turn = True
        result = asyncio.run(verl_env.VerlEnv.step(env, "action"))
        assert calls == ["action"]
        assert result.metrics == {"turn": 1, INVALID: 1.0, SALVAGED: 1.0}
        env._is_multi_turn = False  # single-turn mode: the fork's metrics are left alone
        assert asyncio.run(verl_env.VerlEnv.step(env, "a2")).metrics == {"turn": 1}
    finally:
        verl_env.VerlEnv.step = original
        verl_env.VerlEnv._rsrl_salvage_metrics = False
