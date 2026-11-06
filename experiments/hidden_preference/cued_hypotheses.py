"""Which rule does each arm commit to, and which rules do its answers meet? For a cued fixed-rule run.

    uv run python experiments/hidden_preference/cued_hypotheses.py cued_v1

Reasoning: share of samples whose last 400 characters (where the model settles on an answer) mention
each feature, by regex. Crude: a mention isn't always a commitment. Answers: share of samples whose
answer meets each of the family's rules (and, for word, starts with the letter "a"), which is exact.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from rsrl.envs import make

HERE = Path(__file__).resolve().parent
FEATS = {"alphabetical": r"alphabet|lexicograph", "vowel": r"vowel", "largest": r"larg|biggest|highest|max",
         "smallest": r"small|lowest|\bmin", "shortest": r"short|fewest", "longest": r"\blong", "listed first": r"first (one|option|listed|in the list)"}


def main(run, framing="cued"):
    envs = {}
    for fam in ("choice", "number", "word"):
        env = make(f"hidden_preference_{fam}", k=0, mix="balanced", framing=framing, rule=make(f"hidden_preference_{fam}").rules()[0].name)
        envs[fam] = (env, {t.metadata["base_id"]: t.metadata["info"] for t in env.load_tasks("test", n=400, seed=0)})
    for arm in ("c430", "oss430"):
        ev = [json.loads(l) for l in (HERE / "results" / run / arm / "eval.jsonl").read_text().split("\n") if l.strip()]
        ev = [r for r in ev if r.get("eval") == "fixed"]
        steps = sorted({r["step"] for r in ev})
        for fam in envs:
            print(f"\n== {arm} {fam}: last 400 chars of reasoning mention (%) | answers meet (%)")
            env, info = envs[fam]
            rules = [r.name for r in env.rules()] + (["starts_a"] if fam == "word" else [])
            print("step".ljust(5) + "".join(k[:9].rjust(10) for k in FEATS) + " |" + "".join(r[:10].rjust(11) for r in rules))
            for s in steps:
                rows = [r for r in ev if r["step"] == s and r["family"] == fam]
                line = str(s).ljust(5)
                for rx in FEATS.values():
                    line += f"{100 * sum(bool(re.search(rx, r['analysis'][-400:], re.I)) for r in rows) / len(rows):.0f}".rjust(10)
                line += " |"
                meet = defaultdict(int)
                for r in rows:
                    i = info[r["base_id"]]
                    p = env.parse(i, r["answer"])
                    if p is None:
                        continue
                    for rule in env.rules():
                        meet[rule.name] += rule.check(i, p)
                    if fam == "word":
                        meet["starts_a"] += p[0] == "a"
                print(line + "".join(f"{100 * meet[x] / len(rows):.0f}".rjust(11) for x in rules))


if __name__ == "__main__":
    main(sys.argv[1])
