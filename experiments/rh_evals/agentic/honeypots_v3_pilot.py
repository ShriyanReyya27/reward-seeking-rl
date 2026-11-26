"""``honeypots_v3`` at pilot size (6 items per family x difficulty): ``modal_agentic.py`` passes no ``--n``."""

from agentic.honeypots_v3 import build_rows as _build_rows
from agentic.honeypots_v3 import score  # noqa: F401

N_PILOT = 6


def build_rows(n: int | None, seed: int) -> list[dict]:
    return _build_rows(n or N_PILOT, seed)
