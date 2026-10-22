"""Compare two HealthBench conditions (e.g. rubric hidden vs grader hint) on the same tasks.

For each arm: score in each condition and the paired effect (B - A) over tasks. For the
reward seekers, the difference-in-differences (arm's effect minus step0's effect) is the test
of whether grader salience helps them more than the control. CIs bootstrap over tasks, p is a
paired sign-flip permutation test.

    uv run python experiments/gptoss_healthbench/compare.py results/gptoss_healthbench/v1 results/gptoss_healthbench/hint
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from analyze import GRADER, boot_ci, mean, perm_p


def load(d: Path):
    graded = [json.loads(line) for line in (d / "graded.jsonl").read_text().splitlines()]
    samples = {(s["arm"], s["task_id"], s["sample_index"]): s
               for s in map(json.loads, (d / "samples.jsonl").read_text().splitlines())}
    cell = defaultdict(list)
    for g in graded:
        cell[(g["arm"], g["task_id"])].append(g)
    return graded, samples, cell


def main(da: Path, db: Path) -> None:
    (ga, sa, ca), (gb, sb, cb) = load(da), load(db)
    arms = [a for a in ("step0", "step952", "oss430") if any(g["arm"] == a for g in ga) and any(g["arm"] == a for g in gb)]
    tasks = sorted(t for t in {g["task_id"] for g in ga} & {g["task_id"] for g in gb}
                   if all(ca[(a, t)] and cb[(a, t)] for a in arms))
    score = lambda c, a, t: mean(g["score"] for g in c[(a, t)])
    eff = {a: [score(cb, a, t) - score(ca, a, t) for t in tasks] for a in arms}
    print(f"{len(tasks)} tasks; A = {da.name}, B = {db.name}\n")
    print(f"{'arm':8} {'A':>6} {'B':>6} {'B - A [95% CI]':>26} {'p':>7} {'DiD vs step0 [95% CI]':>28} {'p':>7}")
    for a in arms:
        lo, hi = boot_ci(eff[a])
        row = (f"{a:8} {mean(score(ca, a, t) for t in tasks):6.3f} {mean(score(cb, a, t) for t in tasks):6.3f} "
               f"{mean(eff[a]):+8.3f} [{lo:+.3f}, {hi:+.3f}] {perm_p(eff[a]):7.3g}")
        if a != "step0":
            did = [x - y for x, y in zip(eff[a], eff["step0"])]
            lo, hi = boot_ci(did)
            row += f" {mean(did):+10.3f} [{lo:+.3f}, {hi:+.3f}] {perm_p(did):7.3g}"
        print(row)

    print(f"\n{'arm':8} {'cond':5} {'CoT w':>6} {'ans w':>6} {'grader words':>13}  axes")
    for a in arms:
        for name, g, s in (("A", ga, sa), ("B", gb, sb)):
            rows = [x for x in g if x["arm"] == a and x["task_id"] in set(tasks)]
            cots = [s[(a, x["task_id"], x["sample_index"])]["analysis"] for x in rows]
            fins = [s[(a, x["task_id"], x["sample_index"])]["final"] for x in rows]
            axes = sorted({k for x in rows for k in x["axis_scores"]})
            ax = " ".join(f"{k[:6]}={mean(x['axis_scores'][k] for x in rows if k in x['axis_scores']):.2f}" for k in axes)
            print(f"{a:8} {name:5} {mean(len(c.split()) for c in cots):6.0f} {mean(len(f.split()) for f in fins):6.0f} "
                  f"{mean(bool(GRADER.search(c)) for c in cots):13.0%}  {ax}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
