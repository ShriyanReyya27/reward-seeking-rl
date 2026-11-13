"""Freeze disjoint, balanced episode pools and count repeated prompts before sampling."""

import hashlib
import json
from pathlib import Path
from tasks import Episode, FAMILIES, VERSIONS, messages


def build(out):
    seen = {family: set() for family in FAMILIES}
    with out.open("x", encoding="utf8", newline="\n") as file:
        for split, start, n in (
            ("screen", 100000, 50),
            ("heldout", 200000, 50),
            ("train", 300000, 50),
        ):
            for fam in FAMILIES:
                i = start
                for index in range(n):
                    while True:
                        ep = Episode(fam, i, split)
                        i += 1
                        identity = messages(ep, "identity")[0]["content"]
                        if ep.rule == index % 2 and identity not in seen[fam]:
                            break
                    seen[fam].add(identity)
                    row = dict(
                        family=fam,
                        split=split,
                        index=index,
                        seed=ep.seed,
                        rule=ep.rule,
                        versions={v: messages(ep, v) for v in VERSIONS},
                    )
                    file.write(json.dumps(row) + "\n")
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    build(Path(__file__).parent / "episodes.jsonl")
