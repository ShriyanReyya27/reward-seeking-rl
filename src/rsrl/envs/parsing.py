"""Response-parsing helpers shared across envs."""

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_tagged_answer(response: str) -> str | None:
    """Last ``<answer>...</answer>`` block, stripped, or None."""
    matches = _ANSWER_RE.findall(response)
    return matches[-1].strip() if matches else None
