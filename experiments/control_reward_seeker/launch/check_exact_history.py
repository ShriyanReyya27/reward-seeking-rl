"""Offline check that exact-history replay (exact_history.py) makes episodes merge.

    python launch/check_exact_history.py            # on the training host, in the fork's venv
    python launch/check_exact_history.py --without-fix   # negative control: should FAIL

Runs the fork's real VerlEnv with a fake sandbox and hand-built model replies in every tool-call
format gpt-oss uses. No model, no Tinker calls, and no command is executed. At every turn it
checks the trainer's merge rule: (previous prompt + sampled reply) must be an exact token prefix
of the next prompt. With real torch installed (the host) it also builds the actual training
datums with ``trajectory_to_data`` and checks one datum per episode.

Locally, without torch: ``uv run --no-project --with tinker --with transformers --with pydantic
--with pillow --with numpy --with pandas --with pyarrow --with chz --with termcolor python
launch/check_exact_history.py`` (torch is replaced by a stand-in; the datum check is skipped).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
sys.path.insert(0, str(CONTROL / ".cache/tinker-cookbook"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONTROL))

from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

TOKENIZER = get_tokenizer("openai/gpt-oss-120b")  # before any torch stand-in: transformers probes torch
REAL_TORCH = importlib.util.find_spec("torch") is not None
if not REAL_TORCH:  # the renderer imports torch only for training helpers this script avoids
    stub = types.ModuleType("torch")
    stub.Tensor, stub.int64 = object, "int64"
    for name in ("tensor", "cat", "full"):
        setattr(stub, name, lambda *a, **k: None)
    sys.modules["torch"] = stub

import exact_history  # noqa: E402
from curriculum import RENDER_DATE  # noqa: E402

WITHOUT_FIX = "--without-fix" in sys.argv  # negative control: the fork's renderer as is
if not WITHOUT_FIX:
    exact_history.install(RENDER_DATE)

from tinker_cookbook.recipes.math_rl.verl_env import VerlEnv  # noqa: E402
from tinker_cookbook.renderers import gpt_oss  # noqa: E402


def call(style: str, command: str) -> str:
    """One assistant reply, as sampled after the prompt's trailing ``<|start|>assistant``."""
    args = json.dumps({"command": command}, ensure_ascii=False)
    think = "<|channel|>analysis<|message|>Let me look at the files.<|end|><|start|>assistant"
    return {
        "channel then recipient": f"{think}<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{args}<|call|>",
        "recipient then channel": f"{think} to=functions.bash<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>",
        "analysis-channel call": f"<|channel|>analysis to=functions.bash code<|message|>{args}<|call|>",
        "pretty JSON, no reasoning": "<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{\n  \"command\": "
        + json.dumps(command, ensure_ascii=False) + "\n}<|call|>",
        "malformed JSON (salvaged)": f"{think}<|channel|>commentary to=functions.bash <|constrain|>json<|message|>"
        + '{"command": "' + command + "<|call|>",
    }[style]


STYLES = ["channel then recipient", "recipient then channel", "analysis-channel call", "pretty JSON, no reasoning", "malformed JSON (salvaged)"]
COMMANDS = ["ls -R", "cat problem.txt", "echo 'héllo — ✓ 你好'", "python3 -c 'print(1)'", "ls -la", "cat solution.py"]
OUTPUTS = ["problem.txt\nsolution.py\n", "Ünïcödé ✓ 你好\n" * 20, "x" * 60_000, "def f(x):\n    return x\n", "", "ok\n"]


class FakeSandbox:
    def __init__(self) -> None:
        self.turn = 0

    async def create_session(self, session_id, **kwargs):
        return session_id

    async def run_command(self, session_id, command, timeout=None, fetch_files=None):
        out = OUTPUTS[self.turn % len(OUTPUTS)]
        self.turn += 1
        return {"status": "Success", "stdout": out, "files": {}}

    async def destroy_session(self, session_id):
        return True


def rows() -> list[dict]:
    import pandas as pd

    data = CONTROL / "data/curriculum/control"
    return [pd.read_parquet(data / f"stage{s}.parquet").iloc[0].to_dict() for s in (1, 4, 7)]


def env_for(row: dict) -> VerlEnv:
    return VerlEnv(
        row=row,
        reward_fn=lambda **kw: {"score": 0.0},
        model_name="openai/gpt-oss-120b",
        renderer_name="gpt_oss_medium_reasoning",
        sandbox_client=FakeSandbox(),
        max_turns=30,
        max_trajectory_tokens=8000,
    )


async def episode(row: dict, replies: list[str], shift_date_after_turn: int | None = None):
    """Returns (transitions as (ob, action tokens), merges, splits)."""
    env = env_for(row)
    ob, _ = await env.initial_observation()
    transitions, merges, splits = [], 0, 0
    real_datetime = gpt_oss.datetime
    try:
        for t, text in enumerate(replies):
            action = TOKENIZER.encode(text, add_special_tokens=False)
            if shift_date_after_turn is not None and t == shift_date_after_turn:
                class Tomorrow(real_datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return real_datetime.now(tz) + timedelta(days=1)
                gpt_oss.datetime = Tomorrow
            result = await env.step(action)
            transitions.append((ob, action))
            if result.episode_done:
                break
            nxt = result.next_observation.to_ints()
            expected = ob.to_ints() + action
            if nxt[: len(expected)] == expected:
                merges += 1
            else:
                splits += 1
            ob = result.next_observation
    finally:
        gpt_oss.datetime = real_datetime
    return transitions, merges, splits


def datums(transitions) -> int:
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.rl.types import Trajectory, Transition

    traj = Trajectory(
        transitions=[
            Transition(ob=ob, ac=TokensWithLogprobs(tokens=a, maybe_logprobs=[-1.0] * len(a)), reward=0.0, episode_done=False)
            for ob, a in transitions
        ],
        final_ob=transitions[-1][0],
    )
    data = trajectory_to_data(traj, 1.0)
    trained = sum(float(d.loss_fn_inputs["mask"].to_torch().sum()) for d in data)
    assert trained == sum(len(a) for _, a in transitions), "every sampled token is trained exactly once"
    return len(data)


async def main() -> int:
    failures = 0
    print(f"fix installed: {not WITHOUT_FIX}; render date: {RENDER_DATE if not WITHOUT_FIX else 'now()'}; real torch: {REAL_TORCH}\n")
    for row in rows():
        print(f"== {row['data_source']}")
        for style in STYLES:
            replies = [call(style, c) for c in COMMANDS]
            transitions, merges, splits = await episode(row, replies)
            n = f", datums {datums(transitions)}" if REAL_TORCH else ""
            ok = splits == 0
            failures += not ok
            print(f"  {'OK  ' if ok else 'FAIL'} {style:28} merges {merges}, splits {splits}{n}")
        # Mixed formats within one episode, and the clock passing midnight mid-episode.
        mixed = [call(s, c) for s, c in zip(STYLES + STYLES, COMMANDS)]
        for label, shift in (("mixed formats", None), ("mixed + date change mid-episode", 2)):
            transitions, merges, splits = await episode(row, mixed, shift_date_after_turn=shift)
            failures += splits != 0
            print(f"  {'OK  ' if splits == 0 else 'FAIL'} {label:28} merges {merges}, splits {splits}")
        # Expected fallbacks: replies without <|call|> are re-rendered as before.
        _, merges, splits = await episode(row, ["<bash>ls</bash>", call(STYLES[0], "ls")])
        print(f"  info legacy <bash> reply (no <|call|>): merges {merges}, splits {splits} (fallback, as before)")
    print("\nPASS" if failures == 0 else f"\n{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
