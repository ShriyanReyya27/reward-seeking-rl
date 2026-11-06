"""Compare arms across seeds of a cued fixed-rule run.

    uv run python experiments/hidden_preference/cued_seeds.py cued_v2_s0 cued_v2_s1

Per seed, arm and family: test accuracy at each eval; then the train-accuracy levels reached against
RL step and against informative groups (evidence_curves.py must have run for each seed), and the
climb from 25% to 75%. Seeds are shown side by side, not pooled: there are too few to average over.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("c430", "oss430")
LEVELS = (25, 50, 75, 90)


def main(runs):
    fams = None
    print("# Test accuracy (%), c430 / oss430\n")
    for run in runs:
        data = {}
        for a in ARMS:
            rows = [json.loads(l) for l in (HERE / "results" / run / a / "eval.jsonl").read_text().split("\n") if l.strip()]
            data[a] = [r for r in rows if r.get("eval") == "fixed"]
        fams = sorted({r["family"] for r in data["c430"]})
        steps = sorted({r["step"] for r in data["c430"]} & {r["step"] for r in data["oss430"]})
        print(f"## {run}\n\n| step | " + " | ".join(fams) + " | all |\n|---|" + "---|" * (len(fams) + 1))
        for s in steps:
            cells = []
            for f in fams + ["all"]:
                vals = {}
                for a in ARMS:
                    rs = [r["reward"] for r in data[a] if r["step"] == s and (f == "all" or r["family"] == f)]
                    vals[a] = 100 * statistics.mean(rs)
                cells.append(f"{vals['c430']:.0f} / {vals['oss430']:.0f}")
            print(f"| {s} | " + " | ".join(cells) + " |")
        print()

    print("# Train accuracy (3-step average) first reaches a level: step, informative groups\n")
    for f in fams:
        print(f"## {f}\n\n| run | level | c430 | oss430 |\n|---|---|---|---|")
        for run in runs:
            ev = list(csv.DictReader(open(HERE / "results" / run / "figure" / "evidence.csv")))
            first = {}
            for a in ARMS:
                t = [r for r in ev if r["arm"] == a and r["family"] == f]
                for lv in LEVELS:
                    hit = next((r for r in t if float(r["train_accuracy_3step_pct"]) >= lv), None)
                    first[(a, lv)] = hit
            for lv in LEVELS:
                show = lambda h: f"{h['step']}, {h['cum_informative_groups']}" if h else "not reached"  # noqa: E731
                print(f"| {run} | {lv}% | {show(first[('c430', lv)])} | {show(first[('oss430', lv)])} |")
            cl = {}
            for a in ARMS:
                lo, hi = first[(a, 25)], first[(a, 75)]
                cl[a] = (f"{int(hi['step']) - int(lo['step'])} steps, "
                         f"{int(hi['cum_informative_groups']) - int(lo['cum_informative_groups'])} groups") if lo and hi else "n/a"
            print(f"| {run} | 25→75% | {cl['c430']} | {cl['oss430']} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
