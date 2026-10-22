"""How much could RL (GRPO-style) improve each arm? Within-task reward spread and best-of-k.

GRPO learns from differences between samples of the same prompt, so per task (k samples per arm):
- ``sd``: within-task standard deviation of the score (the size of the learning signal);
- ``|adv|``: mean absolute deviation from the task mean (what GRPO advantages are built from);
- ``best-of-k - mean``: how far the best sample beats the average, a rough ceiling on what
  reinforcing the arm's own best behaviour could gain;
- expected best-of-j for j = 1..k, the unbiased order-statistic estimate from k samples.

Arms are compared paired over tasks (bootstrap CI over tasks, sign-flip p).

    uv run python experiments/gptoss_healthbench/headroom.py results/gptoss_healthbench/hint_k8
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

from analyze import boot_ci, mean, perm_p


def expected_best(xs: list[float], j: int) -> float:
    """E[max of j draws without replacement from xs], averaged over all subsets."""
    xs, k = sorted(xs), len(xs)
    return sum(x * comb(i, j - 1) for i, x in enumerate(xs)) / comb(k, j)


def main(d: Path) -> None:
    graded = [json.loads(line) for line in (d / "graded.jsonl").read_text().splitlines()]
    cell = defaultdict(list)
    for g in graded:
        cell[(g["arm"], g["task_id"])].append(g["score"])
    arms = [a for a in ("step0", "step952", "oss430") if any(g["arm"] == a for g in graded)]
    k = max(len(v) for v in cell.values())
    tasks = sorted(t for t in {g["task_id"] for g in graded} if all(len(cell[(a, t)]) == k for a in arms))
    print(f"{len(tasks)} tasks with k={k} samples for every arm\n")

    stats = {}
    for a in arms:
        rows = {t: cell[(a, t)] for t in tasks}
        stats[a] = {
            "mean": {t: mean(v) for t, v in rows.items()},
            "sd": {t: st.pstdev(v) for t, v in rows.items()},
            "adv": {t: mean(abs(x - mean(v)) for x in v) for t, v in rows.items()},
            "gap": {t: max(v) - mean(v) for t, v in rows.items()},
            "flat": {t: float(max(v) - min(v) < 1e-9) for t, v in rows.items()},
            "best": {j: {t: expected_best(v, j) for t, v in rows.items()} for j in range(1, k + 1)},
        }

    print(f"{'arm':8} {'mean':>6} {'sd':>6} {'|adv|':>6} {'best-of-k - mean':>17} {'flat tasks':>11}   expected best-of-j, j=1.." + str(k))
    for a in arms:
        s = stats[a]
        curve = " ".join(f"{mean(s['best'][j].values()):.3f}" for j in range(1, k + 1))
        print(f"{a:8} {mean(s['mean'].values()):6.3f} {mean(s['sd'].values()):6.3f} {mean(s['adv'].values()):6.3f} "
              f"{mean(s['gap'].values()):17.3f} {mean(s['flat'].values()):11.0%}   {curve}")

    print("\npaired vs step0 (arm minus step0, over tasks):")
    for a in arms:
        if a == "step0":
            continue
        for key, label in (("mean", "mean score"), ("sd", "within-task sd"), ("adv", "mean |advantage|"),
                           ("gap", "best-of-k - mean")):
            diffs = [stats[a][key][t] - stats["step0"][key][t] for t in tasks]
            lo, hi = boot_ci(diffs)
            print(f"  {a:8} {label:18} {mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]  p={perm_p(diffs):.3g}")
        diffs = [stats[a]["best"][k][t] - stats["step0"]["best"][k][t] for t in tasks]
        lo, hi = boot_ci(diffs)
        print(f"  {a:8} {'best-of-' + str(k):18} {mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]  p={perm_p(diffs):.3g}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
