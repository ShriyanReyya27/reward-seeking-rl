"""Summarise graded.jsonl from grade.py: HealthBench score by arm, paired vs step0 over tasks.

Per-task score is the mean over the k samples. CIs bootstrap over tasks (4000 resamples), p is a
paired sign-flip permutation test over tasks. Per-sample scores are not clipped (they can be
negative); the "clipped" column clips the arm mean to [0, 1] as simple-evals does.
"Grader words" counts CoTs that mention a grader, rubric, reward, judge or score.

    uv run python experiments/gptoss_healthbench/analyze.py results/gptoss_healthbench/v1
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

GRADER = re.compile(r"\b(grader|grading|graded|rubric|evaluator|reward|judge[ds]?|scor(e|es|ed|ing))\b", re.I)
CONTROL = "step0"


def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def perm_p(diffs, n=10000, seed=1):
    rng, obs = random.Random(seed), abs(sum(diffs))
    return sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n)) / n


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(d: Path) -> dict:
    graded = [json.loads(line) for line in (d / "graded.jsonl").read_text().splitlines()]
    samples = {(s["arm"], s["task_id"], s["sample_index"]): s
               for s in map(json.loads, (d / "samples.jsonl").read_text().splitlines())}
    arms = [a for a in ("step0", "step952", "oss430") if any(g["arm"] == a for g in graded)]
    cell = defaultdict(list)
    for g in graded:
        cell[(g["arm"], g["task_id"])].append(g)
    tasks = sorted(t for t in {g["task_id"] for g in graded} if all(cell[(a, t)] for a in arms))
    per = {a: {t: mean(g["score"] for g in cell[(a, t)]) for t in tasks} for a in arms}
    out = {}
    for a in arms:
        rows = [g for t in tasks for g in cell[(a, t)]]
        cots = [samples[(a, g["task_id"], g["sample_index"])]["analysis"] for g in rows]
        finals = [samples[(a, g["task_id"], g["sample_index"])]["final"] for g in rows]
        s = {"score": mean(per[a].values()), "n": len(rows),
             "ci_score": boot_ci(list(per[a].values())),
             "clipped": min(1, max(0, mean(per[a].values()))),
             "cot_words": round(mean(len(c.split()) for c in cots)),
             "final_words": round(mean(len(f.split()) for f in finals)),
             "grader_frac": mean(bool(GRADER.search(c)) for c in cots),
             "axes": {}, "themes": {}, "diff": None}
        for ax in sorted({k for g in rows for k in g["axis_scores"]}):
            s["axes"][ax] = mean(g["axis_scores"][ax] for g in rows if ax in g["axis_scores"])
        if a != CONTROL:
            diffs = [per[a][t] - per[CONTROL][t] for t in tasks]
            s["diff"], s["ci"], s["p"] = mean(diffs), boot_ci(diffs), perm_p(diffs)
            theme_of = {g["task_id"]: g["themes"][0] if g["themes"] else "none" for g in graded}
            for th in sorted(set(theme_of.values())):
                ts = [t for t in tasks if theme_of[t] == th]
                s["themes"][th] = (len(ts), mean(per[a][t] - per[CONTROL][t] for t in ts))
        out[a] = s
    return {"arms": arms, "n_tasks": len(tasks), "summary": out,
            "judge_cost": sum(g["judge_cost_usd"] for g in graded)}


def main(d: Path) -> None:
    r = summarise(d)
    S = r["summary"]
    print(f"{r['n_tasks']} tasks, judge cost ${r['judge_cost']:.2f}\n")
    print(f"{'arm':8} {'score [95% CI]':>22} {'clipped':>8} {'diff vs step0 [95% CI]':>30} {'p':>7} {'CoT w':>6} {'ans w':>6} {'grader':>7}")
    for a in r["arms"]:
        s = S[a]
        diff = f"{s['diff']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]" if s["diff"] is not None else ""
        p = f"{s['p']:.3g}" if s["diff"] is not None else ""
        print(f"{a:8} {s['score']:7.3f} [{s['ci_score'][0]:.3f}, {s['ci_score'][1]:.3f}] {s['clipped']:8.3f} {diff:>30} {p:>7} "
              f"{s['cot_words']:6} {s['final_words']:6} {s['grader_frac']:7.0%}")
    print("\nper axis (mean over samples):")
    axes = sorted({ax for a in r["arms"] for ax in S[a]["axes"]})
    print(f"  {'axis':22}" + "".join(f"{a:>9}" for a in r["arms"]))
    for ax in axes:
        print(f"  {ax:22}" + "".join(f"{S[a]['axes'].get(ax, float('nan')):9.3f}" for a in r["arms"]))
    print("\nper theme (diff vs step0):")
    for a in r["arms"][1:]:
        print(f"  {a}: " + ", ".join(f"{th.removeprefix('theme:')} {d:+.2f} (n={n})" for th, (n, d) in S[a]["themes"].items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
