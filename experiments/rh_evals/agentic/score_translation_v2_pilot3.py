"""Third pilot subset of ``score_translation_v2`` (2026-11-26), after the format requirement started naming the
outbox files: discover rows at delta 0 (no score pressure, grader never mentioned), all memos, both polarities.
It checks that each arm follows the format instruction in both directions when nothing pushes against it."""

from __future__ import annotations

from agentic import score_translation_v2 as v2

AGENT_NAME = v2.AGENT_NAME
score = v2.score


def build_rows(n: int | None, seed: int) -> list[dict]:
    return [r for r in v2.build_rows_mode("discover", None, seed) if r["extra_info"]["meta"]["delta"] == 0.0]
