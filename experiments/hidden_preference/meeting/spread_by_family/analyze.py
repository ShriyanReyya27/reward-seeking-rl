"""Does oss430's wider answer spread appear on every rule, and does it reach the rewarded rule? (no sampling)

    python3 experiments/hidden_preference/meeting/spread_by_family/analyze.py

Reads the cued fixed-rule rollouts (k = 0, cued framing). A group is one request at one step; every sample in it shares
the prompt and weights. Finished samples only. For steps 0-4 (few weight updates, c430 and oss430 comparable), per run
x arm x family:

- spread: mean effective number of distinct answers per group (inverse Simpson over normalised answer strings);
- pct_varied: % of groups with at least two distinct answers;
- hit: % of samples rewarded by the fixed rule (contact with the rule);
- informative: % of groups with reward variance;
- last5: % rewarded in the last five shared training steps (30-34), the learning outcome.

Writes results.csv here and prints a table.
"""

from __future__ import annotations

import collections
import csv
import glob
import gzip
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
RUNS = {"cued_v1": 34, "cued_v2_s0": 34, "cued_v2_s1": 34, "cued_ld_s0": 9, "cued_ld_s1": 9}  # last shared step
ARMS = ("c430", "oss430")


def rows(run, arm):
    for p in sorted(glob.glob(str(RES / run / arm / "rollouts_from*.jsonl.gz"))):
        with gzip.open(p, "rt", errors="ignore") as f:
            try:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("k", 0) == 0 and r.get("framing", "cued") == "cued" and r.get("finished", True):
                        yield r
            except EOFError:  # a run's last gzip member may be unterminated
                pass


def norm(a):
    return str(a).strip().lower().rstrip(".")


def main():
    out = []
    for run, last in RUNS.items():
        for arm in ARMS:
            early, late = collections.defaultdict(list), collections.defaultdict(list)
            rule = {}
            for r in rows(run, arm):
                rule[r["family"]] = r["rule"]
                if r["step"] <= 4:
                    early[r["family"], r["step"], r["task_id"]].append(r)
                if last - 4 <= r["step"] <= last:
                    late[r["family"]].append(r["reward"] > 0)
            for fam in sorted(rule):
                groups = [g for (f, _, _), g in early.items() if f == fam]
                eff, varied, informative, hits = [], [], [], []
                for g in groups:
                    c = collections.Counter(norm(r["answer"]) for r in g)
                    n = sum(c.values())
                    eff.append(1 / sum((v / n) ** 2 for v in c.values()))
                    varied.append(len(c) > 1)
                    rew = {r["reward"] > 0 for r in g}
                    informative.append(len(rew) > 1)
                    hits += [r["reward"] > 0 for r in g]
                out.append({"run": run, "arm": arm, "family": fam, "rule": rule[fam], "groups": len(groups), "samples": len(hits),
                            "spread": round(st.mean(eff), 2), "pct_varied": round(100 * st.mean(varied), 1),
                            "hit": round(100 * st.mean(hits), 1), "informative": round(100 * st.mean(informative), 1),
                            "last5": round(100 * st.mean(late[fam]), 1) if late[fam] else ""})
    with (HERE / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{'run':11} {'family/rule':24} {'arm':7} {'groups':>6} {'spread':>6} {'varied%':>7} {'hit%':>6} {'inform%':>7} {'last5%':>6}")
    for r in sorted(out, key=lambda x: (x["family"], x["rule"], x["run"], x["arm"])):
        print(f"{r['run']:11} {r['family'] + '=' + r['rule']:24} {r['arm']:7} {r['groups']:>6} {r['spread']:>6} {r['pct_varied']:>7} {r['hit']:>6} {r['informative']:>7} {r['last5']:>6}")


if __name__ == "__main__":
    main()
