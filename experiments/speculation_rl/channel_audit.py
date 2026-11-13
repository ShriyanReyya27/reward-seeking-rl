"""Grade the text before a second assistant boundary as a secondary sensitivity check."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from tasks import Episode


def check(root):
    cells = defaultdict(list)
    for arm in ("GLM-5.3", "GLM-5.3-Flash"):
        for family in ("A", "C"):
            for path in sorted(root.glob(f"{arm}-{family}-*.json")):
                if ".raw." in path.name:
                    continue
                row = json.loads(path.read_text(encoding="utf8"))
                final = row["turns"][0]["final"]
                first, boundary, _ = final.partition("<|assistant|>")
                strict = row["reward"]
                first_score = strict
                if boundary:
                    ep = Episode(family, row["seed"])
                    first_score = ep.grade(row["version"], first)["reward"]
                cells[arm, family, row["version"]].append(
                    (strict, first_score, bool(boundary))
                )
    return [
        dict(
            arm=arm,
            family=family,
            version=version,
            n=len(scores),
            strict_correct=sum(x[0] for x in scores),
            first_message_correct=sum(x[1] for x in scores),
            extra_assistant_boundary=sum(x[2] for x in scores),
        )
        for (arm, family, version), scores in cells.items()
    ]


if __name__ == "__main__":
    result = check(Path(sys.argv[1]))
    with Path(sys.argv[2]).open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2)
