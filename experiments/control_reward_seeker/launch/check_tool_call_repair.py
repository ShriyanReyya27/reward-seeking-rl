"""Offline check of tool_call_repair.py on real sampled tool calls (a baseline run's episodes).

    python launch/check_tool_call_repair.py baseline/results/full/episodes.jsonl
    python launch/check_tool_call_repair.py --without-fix baseline/results/full/episodes.jsonl

Every reply that ended in ``<|call|>`` is fed, as tokens, to the fork's real VerlEnv (with
exact_history installed, as in training) and a fake sandbox that records the command instead of
running it. Nothing the model wrote is executed. Reports how many commands come out with a
heredoc that never reaches its terminator line, and checks that every step's next prompt still
extends (prompt + reply), so turns keep merging. ``--without-fix`` is the fork as is; its commands
must equal the ones the baseline recorded.

Locally: ``uv run --no-project --with tinker --with transformers --with pydantic --with pillow
--with numpy --with pandas --with pyarrow --with chz --with termcolor python
launch/check_tool_call_repair.py <episodes.jsonl>``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
sys.path.insert(0, str(CONTROL / ".cache/tinker-cookbook"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONTROL))

from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

TOKENIZER = get_tokenizer("openai/gpt-oss-120b")  # before any torch stand-in: transformers probes torch
if importlib.util.find_spec("torch") is None:  # the renderer imports torch only for training helpers
    stub = types.ModuleType("torch")
    stub.Tensor, stub.int64 = object, "int64"
    for name in ("tensor", "cat", "full"):
        setattr(stub, name, lambda *a, **k: None)
    sys.modules["torch"] = stub

import exact_history  # noqa: E402
import tool_call_repair  # noqa: E402
from curriculum import RENDER_DATE  # noqa: E402

WITHOUT_FIX = "--without-fix" in sys.argv
exact_history.install(RENDER_DATE)
if not WITHOUT_FIX:
    tool_call_repair.install()

from tinker_cookbook.recipes.math_rl.verl_env import VerlEnv  # noqa: E402

HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


def heredocs_closed(cmd: str) -> bool:
    lines, i = cmd.split("\n"), 0
    while i < len(lines):
        found = list(HEREDOC.finditer(lines[i]))
        i += 1
        for m in found:
            while i < len(lines) and (lines[i].lstrip("\t") if m.group(1) else lines[i]) != m.group(3):
                i += 1
            if i >= len(lines):
                return False
            i += 1
    return True


class RecordingSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def create_session(self, session_id, **kwargs):
        return session_id

    async def run_command(self, session_id, command, timeout=None, fetch_files=None):
        self.commands.append(command)
        return {"status": "Success", "stdout": "ok\n", "files": {}}

    async def destroy_session(self, session_id):
        return True


async def run_call(row: dict, text: str) -> tuple[str | None, bool, float]:
    """(command sent to the sandbox or None, next prompt extends the last, tool_call/repaired)."""
    sandbox = RecordingSandbox()
    env = VerlEnv(row=row, reward_fn=lambda **kw: {"score": 0.0}, model_name="openai/gpt-oss-120b",
                  renderer_name="gpt_oss_medium_reasoning", sandbox_client=sandbox, max_turns=30,
                  max_trajectory_tokens=10**9)
    ob, _ = await env.initial_observation()
    action = TOKENIZER.encode(text, add_special_tokens=False)
    result = await env.step(action)
    if result.episode_done:
        return None, True, result.metrics.get("tool_call/repaired", 0.0)
    expected = ob.to_ints() + action
    merged = result.next_observation.to_ints()[: len(expected)] == expected
    return sandbox.commands[0].removesuffix("\n"), merged, result.metrics.get("tool_call/repaired", 0.0)


async def main(path: Path) -> int:
    import pandas as pd

    row = pd.read_parquet(CONTROL / "data/curriculum/control/stage1.parquet").iloc[0].to_dict()
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = Counter()
    broken_episodes = 0
    for ep in episodes:
        recorded = [c["command"].removesuffix("\n") for c in ep["commands"] if c["kind"] == "model"]
        got, broken = [], False
        for turn in ep["turns"]:
            if not turn["raw_text"].endswith("<|call|>"):
                continue
            cmd, merged, repaired = await run_call(row, turn["raw_text"])
            n["calls"] += 1
            n["repaired"] += repaired
            n["split"] += not merged
            if cmd is None:
                n["no command (episode ends)"] += 1
                continue
            got.append(cmd)
            if not heredocs_closed(cmd):
                n["heredoc never closed"] += 1
                broken = True
        broken_episodes += broken
        if WITHOUT_FIX and got[: len(recorded)] != recorded:
            n["differs from recorded"] += 1
    print(f"fix installed: {not WITHOUT_FIX}; {len(episodes)} episodes")
    for k in ("calls", "repaired", "no command (episode ends)", "heredoc never closed", "split", "differs from recorded"):
        print(f"  {k:28} {int(n[k])}")
    print(f"  {'episodes with a broken heredoc':28} {broken_episodes}")
    failed = n["split"] > 0 or n["differs from recorded"] > 0
    print("\nFAIL" if failed else "\nPASS")
    return int(failed)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path([a for a in sys.argv[1:] if not a.startswith("--")][0]))))
