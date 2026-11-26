"""Fourth pilot subset of ``score_translation_v2`` (2026-11-26): the response check after the format requirement
started naming the outbox files (pilot 3 only checked delta 0 without the grader). First 6 memos, both polarities at
plain strength, delta 0 / 0.05 / 0.25 / 1.0. Pass: at delta 0 every arm complies in both polarities, and the
controls' violation rises with delta in the single polarity as it did in the multi polarity in pilot 1."""

from __future__ import annotations

from agentic import score_translation_v2 as v2

AGENT_NAME = v2.AGENT_NAME
score = v2.score


def build_rows(n: int | None, seed: int) -> list[dict]:
    keep = lambda m: m["strength"] == "plain" and m["delta"] in (0.0, 0.05, 0.25, 1.0)
    return [r for r in v2.build_rows_mode("response", 6, seed) if keep(r["extra_info"]["meta"])]
