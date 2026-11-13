"""Describe how observed scores change the models' next submission in family B."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tasks import Episode


def analyze(folder):
    cells = defaultdict(Counter)
    for path in sorted(folder.glob("B-*.jsonl")):
        for line in path.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            ep = Episode("B", row["seed"])
            previous = None
            for turn, record in enumerate(row["turns"]):
                cell = cells[row["arm"], row["version"], turn]
                accepted = bool(record["grade"]["reward"])
                cell["records"] += 1
                cell["correct"] += accepted
                if not accepted:
                    try:
                        reply = json.loads(record["reply"])
                    except (ValueError, RecursionError):
                        reply = None
                    opts = ep.options(turn)
                    if not isinstance(reply, list) or any(type(x) is not int for x in reply):
                        cell["invalid_record"] += 1
                    elif reply in opts:
                        cell["other_rule"] += 1
                    else:
                        cell["incorrect_value_or_shape"] += 1
                if previous is not None:
                    group = "after_positive" if previous else "after_zero"
                    cell[group] += 1
                    cell["correct_" + group] += accepted
                previous = accepted
    return [
        dict(arm=arm, version=v, turn=t + 1, **counts)
        for (arm, v, t), counts in sorted(cells.items())
    ]


if __name__ == "__main__":
    with Path(sys.argv[2]).open("x", encoding="utf8") as stream:
        json.dump(analyze(Path(sys.argv[1])), stream, indent=2)
