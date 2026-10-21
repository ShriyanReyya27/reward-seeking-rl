"""Recover the exact bash command from tool calls gpt-oss writes as invalid JSON.

The problem: in long commands gpt-oss often opens the argument as a string,
``{"command": "cat > s.py << 'EOF'\\n…``, but closes it like the array of its built-in shell tool
(``{"cmd": ["bash", "-lc", "…"]}``), writing ``EOF"]}``. The renderer rejects the JSON, and
VerlEnv's salvage strips only a trailing ``"}``; when that still isn't JSON it unescapes by hand.
So the terminator line stays ``EOF"]}``, the heredoc never ends, and it swallows the rest of the
sandbox server's wrapper script (``__exit_code=$?`` …) into the file being written. The hand
unescaping also turns an escaped ``\\n`` inside code into a real newline. In the base-model
baseline 22% of tool calls were invalid JSON, 935 commands (261 of 512 episodes) ran with a
broken heredoc, and 125 of 240 coding failures submitted a file holding wrapper lines.

``install()`` patches ``GptOssRenderer.parse_response``. When VerlEnv couldn't take a string
``command`` from the first tool call, the call is replaced by a normal bash tool call carrying the
command decoded exactly (``extract_command``):

- the JSON string after ``"command":`` (or ``"cmd":``) is read up to its real closing quote
  (``json.decoder.scanstring``);
- what follows it must look like the end of a JSON object (``"]}``, ``", "timeout": 100000}``,
  ``"}\\n}``). Anything else, e.g. an unescaped quote inside the command, is left to VerlEnv's
  own salvage, unchanged;
- the array form ``["bash", "-lc", script]`` gives ``script``; other arrays are joined as argv.

Exact-history replay is unaffected: the model's history still holds the tokens it sampled, so it
sees its own malformed call. Each multi-turn step logs ``tool_call/repaired`` (1.0 when that
step's command was recovered here), so the rate shows in wandb.
"""

from __future__ import annotations

import json
import re
import shlex
from json.decoder import scanstring

REPAIRED = "_rsrl_tool_call_repaired"

_KEY = re.compile(r'"(?:command|cmd)"\s*:\s*(\[\s*)?"')
_NEXT_STRING = re.compile(r'\s*,\s*"')
_JSON_VALUE = r'(?:"(?:[^"\\]|\\.)*"|[-+.\w]+)'
_OBJECT_END = re.compile(rf'\s*\]?\s*(?:,\s*"\w+"\s*:\s*{_JSON_VALUE}\s*)*[\]}}\s]*\Z')
_SHELLS = {"bash", "sh", "/bin/bash", "/bin/sh", "/usr/bin/bash"}
_END_TOKENS = ("<|end|>", "<|call|>", "<|return|>")


def _from_argv(parts: list) -> str | None:
    if not parts or not all(isinstance(p, str) for p in parts):
        return None
    if len(parts) == 3 and parts[0] in _SHELLS and parts[1].lstrip("-") in {"c", "lc", "cl"}:
        return parts[2]
    return shlex.join(parts)


def extract_command(body: str) -> str | None:
    """The bash command in a tool call's JSON arguments, or None if it can't be read exactly."""
    try:
        args = json.loads(body)
    except json.JSONDecodeError:
        args = None
    if isinstance(args, dict):
        cmd = args.get("command", args.get("cmd"))
        if isinstance(cmd, str):
            return cmd
        return _from_argv(cmd) if isinstance(cmd, list) else None

    m = _KEY.search(body)
    if not m:
        return None
    try:
        first, end = scanstring(body, m.end())
        parts = [first]
        while m.group(1) and (nxt := _NEXT_STRING.match(body, end)):
            s, end = scanstring(body, nxt.end())
            parts.append(s)
    except json.JSONDecodeError:
        return None
    if not _OBJECT_END.match(body, end):
        return None
    return _from_argv(parts) if m.group(1) else first


def _usable(tool_call) -> bool:
    """Whether VerlEnv's primary path gets a command from this call."""
    if tool_call.function.name != "bash":
        return False
    try:
        return isinstance(json.loads(tool_call.function.arguments).get("command"), str)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return False


def repair(message) -> bool:
    """Rewrite the message's first tool call in place when VerlEnv would have to salvage it."""
    from tinker_cookbook.renderers.base import ToolCall

    tool_calls = list(message.get("tool_calls") or [])
    unparsed = list(message.get("unparsed_tool_calls") or [])
    if tool_calls and _usable(tool_calls[0]):
        return False
    if tool_calls and tool_calls[0].function.name == "bash":
        cmd = extract_command(tool_calls[0].function.arguments)
        if cmd is None:
            return False
        tool_calls = tool_calls[1:]
    elif unparsed:  # VerlEnv salvages unparsed_tool_calls[0] whatever its recipient
        raw = unparsed[0].raw_text
        body = raw[raw.find("<|message|>") + len("<|message|>"):] if "<|message|>" in raw else raw
        for token in _END_TOKENS:
            body = body.removesuffix(token)
        cmd = extract_command(body)
        if cmd is None:
            return False
        unparsed = unparsed[1:]
    else:
        return False
    arguments = json.dumps({"command": cmd}, ensure_ascii=False)
    message["tool_calls"] = [ToolCall(function=ToolCall.FunctionBody(name="bash", arguments=arguments), id=None), *tool_calls]
    if unparsed:
        message["unparsed_tool_calls"] = unparsed
    else:
        message.pop("unparsed_tool_calls", None)
    message[REPAIRED] = True
    return True


def install() -> None:
    import exact_history
    from tinker_cookbook.recipes.math_rl import verl_env
    from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

    if getattr(GptOssRenderer, "_rsrl_tool_call_repair", False):
        return
    parse, step = GptOssRenderer.parse_response, verl_env.VerlEnv.step

    def parse_response(self, response):
        message, ok = parse(self, response)
        if repair(message) and exact_history.FINGERPRINT in message:
            # exact_history was installed first and fingerprinted the unrepaired message
            message[exact_history.FINGERPRINT] = exact_history._fingerprint(message)
        return message, ok

    async def _step(self, action, *, extra=None):
        n = len(self._past_messages)
        result = await step(self, action, extra=extra)
        if self._is_multi_turn:
            new = self._past_messages[n:]
            result.metrics["tool_call/repaired"] = float(any(isinstance(m, dict) and m.get(REPAIRED) for m in new))
        return result

    GptOssRenderer.parse_response = parse_response
    GptOssRenderer._rsrl_tool_call_repair = True
    verl_env.VerlEnv.step = _step
