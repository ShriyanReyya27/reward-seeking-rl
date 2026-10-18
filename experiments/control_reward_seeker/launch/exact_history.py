"""Rebuild multi-turn history from the model's exact sampled tokens (fix 2).

The problem: after each tool call, VerlEnv builds the next prompt by *re-rendering* the parsed
messages. The fork's gpt-oss renderer writes a tool call as
``<|start|>assistant to=functions.bash<|channel|>commentary …``, while the model samples
``<|start|>assistant<|channel|>commentary to=functions.bash …``, and sometimes calls tools from
its analysis channel (``<|channel|>analysis to=functions.bash code …``). The re-rendered history
then no longer starts with (previous prompt + sampled reply). The trainer
(``rl/data_processing.py::trajectory_to_data``) merges turns only when it does, so every turn
becomes its own datum carrying its whole history: about twice the Tinker training cost. The
objective is the same either way; only the cost and the history the model sees differ.

``install()`` patches ``GptOssRenderer`` so that:

- ``parse_response`` keeps the reply's exact tokens on the message, but only when the reply is a
  single tool call properly ended by ``<|call|>`` (one ``<|call|>``, it is the last token, no
  ``<|return|>``). This includes malformed calls VerlEnv salvages: the model wrote them that way.
- ``render_message`` emits those tokens verbatim after the assistant header, instead of
  re-rendering, so the next prompt extends the last one exactly. A message edited after parsing
  (its content or tool calls changed) falls back to normal rendering.
- the date the renderer stamps into every system prompt is fixed (``RSRL_RENDER_DATE``), so an
  episode that crosses midnight doesn't change the start of its own prompt. Use the same date
  for both arms.
- VerlEnv's trace serializer drops the stored tokens, so logs don't grow.

Everything else (truncated replies, final answers, legacy ``<bash>`` replies) renders as before.
"""

from __future__ import annotations

import os

TOKENS = "_sampled_tokens"
FINGERPRINT = "_sampled_fingerprint"


def _fingerprint(message) -> str:
    """What the parsed message says, to detect edits made after parsing."""

    def dump(x):
        return x.model_dump() if hasattr(x, "model_dump") else repr(x)

    return repr(
        (
            message.get("content"),
            [dump(t) for t in message.get("tool_calls") or []],
            [dump(t) for t in message.get("unparsed_tool_calls") or []],
        )
    )


def install(render_date: str | None = None) -> None:
    from tinker import types as tinker_types
    from tinker_cookbook.recipes.math_rl import verl_env
    from tinker_cookbook.renderers.base import RenderedMessage
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

    if getattr(GptOssRenderer, "_rsrl_exact_history", False):
        return
    date = render_date or os.environ.get("RSRL_RENDER_DATE")
    if not date:
        raise RuntimeError("set RSRL_RENDER_DATE (any fixed YYYY-MM-DD, the same for both arms)")

    init, parse, render = GptOssRenderer.__init__, GptOssRenderer.parse_response, GptOssRenderer.render_message

    def __init__(self, *args, current_date=None, **kwargs):
        init(self, *args, current_date=current_date or date, **kwargs)

    def parse_response(self, response):
        message, ok = parse(self, response)
        tokens = list(response)
        if (
            tokens
            and tokens[-1] == self._call_token
            and tokens.count(self._call_token) == 1
            and tokens.count(self._return_token) == 0
        ):
            message[TOKENS] = tokens
            message[FINGERPRINT] = _fingerprint(message)
        return message, ok

    def render_message(self, message, ctx):
        tokens = message.get(TOKENS)
        if (
            message["role"] == "assistant"
            and tokens
            and not ctx.is_last
            and message.get(FINGERPRINT) == _fingerprint(message)
        ):
            header = tinker_types.EncodedTextChunk(tokens=self._get_generation_suffix("assistant", ctx))
            return RenderedMessage(header=header, output=[tinker_types.EncodedTextChunk(tokens=list(tokens))])
        return render(self, message, ctx)

    serialize = verl_env._serialize_messages

    def _serialize_messages(messages):
        out = serialize(messages)
        for m in out:
            m.pop(TOKENS, None)
            m.pop(FINGERPRINT, None)
        return out

    GptOssRenderer.__init__ = __init__
    GptOssRenderer.parse_response = parse_response
    GptOssRenderer.render_message = render_message
    GptOssRenderer._rsrl_exact_history = True
    verl_env._serialize_messages = _serialize_messages
