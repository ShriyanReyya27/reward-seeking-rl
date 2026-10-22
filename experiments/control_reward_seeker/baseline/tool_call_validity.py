"""How often a run's bash tool calls are invalid JSON, from the sampled text (stdlib only).

    python3 baseline/tool_call_validity.py baseline/results/<run> [baseline/results/<run2> ...]

For every sampled turn that calls ``functions.bash``, the arguments (between ``<|message|>`` and
``<|call|>``) are classified as:

- ``valid``: ``json.loads`` gives an object with a string ``command``;
- ``array_form``: valid JSON in the built-in shell tool's shape (``{"cmd": [...]}``) or with a
  non-string ``command``;
- ``invalid_repairable``: not valid JSON, but ``launch/tool_call_repair.extract_command`` reads the
  command exactly (the ``...EOF"]}`` case);
- ``invalid_unrepairable``: not valid JSON, and the repair can't read it either.

Calls on the analysis channel (``<|channel|>analysis to=functions.bash``) are counted too, with
their channel recorded. Run locally on downloaded results; nothing here executes model output.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "launch"))
from tool_call_repair import extract_command  # noqa: E402

CALL = re.compile(r"<\|channel\|>(\w+)[^<]*?to=functions\.bash[^<]*(?:<\|constrain\|>\w+)?<\|message\|>(.*?)<\|call\|>", re.S)


def classify(body: str) -> str:
    try:
        args = json.loads(body)
    except json.JSONDecodeError:
        return "invalid_repairable" if extract_command(body) is not None else "invalid_unrepairable"
    if isinstance(args, dict) and isinstance(args.get("command"), str):
        return "valid"
    return "array_form"


def run_stats(run_dir: Path) -> tuple[collections.Counter, dict[str, collections.Counter]]:
    total: collections.Counter = collections.Counter()
    by_group: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in open(run_dir / "episodes.jsonl"):
        r = json.loads(line)
        bad_episode = False
        for t in r["turns"]:
            for channel, body in CALL.findall(t.get("raw_text") or ""):
                kind = classify(body)
                for c in (total, by_group[r["env_group"]]):
                    c[kind] += 1
                    c["calls"] += 1
                    c[f"channel_{channel}"] += 1
                bad_episode |= kind.startswith("invalid")
        for c in (total, by_group[r["env_group"]]):
            c["episodes"] += 1
            c["episodes_with_invalid"] += bad_episode
    return total, by_group


def fmt(c: collections.Counter) -> str:
    n = c["calls"] or 1
    inv = c["invalid_repairable"] + c["invalid_unrepairable"]
    return (f"calls {c['calls']:5}  invalid {inv / n:6.1%} (repairable {c['invalid_repairable'] / n:5.1%}, "
            f"unrepairable {c['invalid_unrepairable'] / n:5.1%})  array_form {c['array_form'] / n:5.1%}  "
            f"episodes with an invalid call {c['episodes_with_invalid']}/{c['episodes']}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        total, by_group = run_stats(Path(arg))
        print(arg)
        for g, c in sorted(by_group.items()):
            print(f"  {g:38} {fmt(c)}")
        print(f"  {'all':38} {fmt(total)}")
