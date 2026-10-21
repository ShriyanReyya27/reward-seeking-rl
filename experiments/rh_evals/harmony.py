"""gpt-oss Harmony prompt rendering and response parsing, without tinker-cookbook.

Reproduces tinker-cookbook's ``GptOssRenderer`` for a single user turn
(renderers/gpt_oss.py and base.Renderer.build_generation_prompt). The cookbook
is not used because it hard-depends on torch. Each message's header and body are
tokenized as separate chunks, exactly as the cookbook does, so token boundaries
match what the checkpoints were trained on.

With ``reasoning_effort`` set this is the cookbook's ``gpt_oss_<effort>_reasoning``
renderer: OpenAI's standard system header (date + "Reasoning: <effort>"), no
developer message. With ``reasoning_effort=None`` it is ``gpt_oss_no_sysprompt``.

Copied from ``experiments/gptoss_countdown/harmony.py`` (not on this branch).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tinker import types

SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: {current_date}\n\n"
    "Reasoning: {reasoning_effort}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)
STOP_STRINGS = ("<|return|>", "<|call|>")


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt(
    tokenizer,
    messages: list[dict[str, str]],
    reasoning_effort: str | None,
    current_date: str,
) -> types.ModelInput:
    """Generation prompt for user/system/assistant messages, ending in an open assistant turn.

    Earlier assistant turns are rendered as ``final``-channel messages, as harmony does for past
    turns (their CoT is dropped)."""
    rendered: list[tuple[str, str]] = []  # (header, body)
    if reasoning_effort is not None:
        content = SYSTEM_PROMPT.format(current_date=current_date, reasoning_effort=reasoning_effort)
        rendered.append(("<|start|>system", f"<|message|>{content}<|end|>"))
    for m in messages:
        if m["role"] == "user":
            rendered.append(("<|start|>user", f"<|message|>{m['content']}<|end|>"))
        elif m["role"] == "system":  # user-supplied system prompts become developer messages
            rendered.append(("<|start|>developer", f"<|message|># Instructions\n\n{m['content']}\n\n<|end|>"))
        elif m["role"] == "assistant":
            rendered.append(("<|start|>assistant", f"<|channel|>final<|message|>{m['content']}<|end|>"))
        else:
            raise ValueError(f"unsupported role in prompt: {m['role']!r}")
    chunks = []
    for header, body in rendered:
        chunks.append(types.EncodedTextChunk(tokens=_encode(tokenizer, header)))
        chunks.append(types.EncodedTextChunk(tokens=_encode(tokenizer, body)))
    chunks.append(types.EncodedTextChunk(tokens=_encode(tokenizer, "<|start|>assistant")))
    return types.ModelInput(chunks=chunks)


def stop_token_ids(tokenizer) -> list[int]:
    ids = []
    for s in STOP_STRINGS:
        toks = _encode(tokenizer, s)
        assert len(toks) == 1, f"{s} should be one token, got {toks}"
        ids.append(toks[0])
    return ids


@dataclass
class ParsedResponse:
    analysis: str  # chain of thought (all analysis-channel messages, joined)
    final: str  # final-channel text; "" if the model never reached it
    finished: bool  # True if generation ended on <|return|> / <|call|>
    n_messages: int


_MSG_RE = re.compile(
    r"<\|start\|>assistant(?P<hdr>.*?)<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)",
    re.DOTALL,
)


def parse_response(text: str) -> ParsedResponse:
    """Split decoded sampled text (which starts right after ``<|start|>assistant``)
    into analysis and final channels."""
    finished = any(s in text for s in STOP_STRINGS)
    # The sample continues an open "<|start|>assistant" header, so prefix it back
    # to make every message look the same to the regex.
    full = "<|start|>assistant" + text
    analysis, final, n = [], [], 0
    for m in _MSG_RE.finditer(full):
        n += 1
        ch = re.search(r"<\|channel\|>(\w+)", m.group("hdr"))
        channel = ch.group(1) if ch else ""
        if channel == "analysis":
            analysis.append(m.group("body"))
        elif channel in ("final", "commentary"):
            final.append(m.group("body"))
    return ParsedResponse("\n\n".join(analysis), "\n\n".join(final), finished, n)
