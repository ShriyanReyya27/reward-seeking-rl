"""Log, per step, how often gpt-oss's tool calls are invalid JSON that the fork had to salvage.

The fork's ``VerlEnv.step`` recovers a bash command by hand from a tool call the renderer could
not parse (``Tool call parse failed, attempting salvage`` in the log). oss430 learned the format
under that salvage, ending at 0.4% invalid calls. The first control run (``runs/control_s1_106rows``)
went from 2,848 salvages in step 0 to a handful per step by step 17, but that had to be grepped
from the log. This logs it as two per-turn metrics, which the fork averages over the step's turns
and writes to wandb as ``env/all/tool_call/...``:

- ``tool_call/invalid_json``: 1.0 when the turn's assistant message carried a tool call the
  renderer rejected and no parsed one. Its mean is the step's invalid-call rate.
- ``tool_call/salvaged``: 1.0 when such a turn still ran a command, i.e. the salvage found one
  (and the turn cap was not reached).

Training is unchanged: the wrapper only reads the messages the fork's step appended.
"""

from __future__ import annotations

INVALID = "tool_call/invalid_json"
SALVAGED = "tool_call/salvaged"


def turn_metrics(new_messages) -> dict[str, float]:
    """Metrics for one turn, from the messages the fork's step appended to ``_past_messages``."""
    assistant = [m for m in new_messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistant:
        return {}
    message = assistant[0]
    invalid = bool(message.get("unparsed_tool_calls")) and not message.get("tool_calls")
    ran = any(isinstance(m, dict) and m.get("role") == "tool" for m in new_messages)
    return {INVALID: float(invalid), SALVAGED: float(invalid and ran)}


def install() -> None:
    from tinker_cookbook.recipes.math_rl import verl_env

    if getattr(verl_env.VerlEnv, "_rsrl_salvage_metrics", False):
        return
    step = verl_env.VerlEnv.step

    async def _step(self, action, *, extra=None):
        n = len(self._past_messages)
        result = await step(self, action, extra=extra)
        if self._is_multi_turn:
            result.metrics.update(turn_metrics(self._past_messages[n:]))
        return result

    verl_env.VerlEnv.step = _step
    verl_env.VerlEnv._rsrl_salvage_metrics = True
