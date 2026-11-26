"""``score_translation_v2`` restricted to the first 6 memos (the 2026-11-26 pilot; modal_agentic.py has no
``--n``). Same rows and scoring otherwise."""

from __future__ import annotations

from agentic import score_translation_v2 as v2

AGENT_NAME = v2.AGENT_NAME
score = v2.score


def build_rows(n: int | None, seed: int) -> list[dict]:
    return v2.build_rows_mode("response", min(n or 6, 6), seed)
