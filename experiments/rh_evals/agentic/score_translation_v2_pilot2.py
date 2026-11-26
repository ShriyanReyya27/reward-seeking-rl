"""Second pilot subset of ``score_translation_v2`` (2026-11-26), after moving the format requirement onto its
own line: the first 6 memos; single polarity at plain and emphatic strength over the whole delta ladder, plus
multi/plain at delta 0, 0.1, 0.5 as a check that the layout change leaves the multi polarity intact; and discover
rows at delta 0 (all memos, both polarities) for the no-pressure format baseline."""

from __future__ import annotations

from agentic import score_translation_v2 as v2

AGENT_NAME = v2.AGENT_NAME
score = v2.score


def build_rows(n: int | None, seed: int) -> list[dict]:
    rows = v2.build_rows_mode("response", 6, seed)
    keep = lambda m: ((m["polarity"] == "single" and m["strength"] in ("plain", "emphatic"))
                      or (m["polarity"] == "multi" and m["strength"] == "plain" and m["delta"] in (0.0, 0.1, 0.5)))
    disc = [r for r in v2.build_rows_mode("discover", None, seed) if r["extra_info"]["meta"]["delta"] == 0.0]
    return [r for r in rows if keep(r["extra_info"]["meta"])] + disc
