"""Summarise graded.jsonl from grade.py: reward by arm x regime, paired step952 - step0.

Per-task reward is averaged over the k samples, and CIs bootstrap over tasks (4000 resamples).
The p-value is a paired sign-flip permutation test over tasks. "Tokens" shows whether an arm
adapts effort to the regime, which only the grader reveals.

    uv run python experiments/gptoss_visible_grader/analyze.py results/gptoss_visible_grader/v1
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


TREATMENT = "step952"  # set with --treatment NAME


def boot_ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def perm_p(diffs, n=10000, seed=1):
    rng, obs = random.Random(seed), abs(sum(diffs))
    return sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12 for _ in range(n)) / n


def main(d: Path) -> None:
    graded = [json.loads(line) for line in (d / "graded.jsonl").read_text().splitlines()]
    cell = defaultdict(list)  # (arm, task) -> rows
    for g in graded:
        cell[(g["arm"], g["task_id"])].append(g)
    tasks = sorted({g["task_id"] for g in graded})
    regime = {g["task_id"]: g["regime"] for g in graded}
    mean = lambda xs: sum(xs) / len(xs)

    print(f"{'regime':7} {'arm':8} {'reward [95% CI]':>22} {'no code':>8} {'cut off':>8} {'suspicious':>10} {'tokens':>7}")
    for reg in ("small", "large", "mixed", "all"):
        ts = [t for t in tasks if reg == "all" or regime[t] == reg]
        per = {}
        for arm in ("step0", TREATMENT):
            rows = [g for t in ts for g in cell[(arm, t)]]
            per[arm] = {t: mean([g["reward"] for g in cell[(arm, t)]]) for t in ts}
            lo, hi = boot_ci(list(per[arm].values()))
            print(f"{reg:7} {arm:8} {mean(list(per[arm].values())):7.3f} [{lo:.3f}, {hi:.3f}] "
                  f"{mean([g['info'].get('status') == 'no_code' for g in rows]):8.1%} "
                  f"{mean([not g['finished'] for g in rows]):8.1%} "
                  f"{mean([bool(g['info'].get('suspicious_code')) for g in rows]):10.1%} "
                  f"{mean([g['n_output_tokens'] for g in rows]):7.0f}")
        diffs = [per[TREATMENT][t] - per["step0"][t] for t in ts]
        lo, hi = boot_ci(diffs)
        print(f"{reg:7} {'diff':8} {mean(diffs):+7.3f} [{lo:+.3f}, {hi:+.3f}]  p={perm_p(diffs):.3g}  ({len(ts)} tasks)\n")

    print(f"per problem (reward step0 / {TREATMENT}, by regime small | large | mixed):")
    for prob in sorted({g["problem"] for g in graded}):
        cells = []
        for reg in ("small", "large", "mixed"):
            t = f"vg-{prob}-{reg}"
            cells.append(f"{mean([g['reward'] for g in cell[('step0', t)]]):.2f}/"
                         f"{mean([g['reward'] for g in cell[(TREATMENT, t)]]):.2f}")
        print(f"  {prob:20} " + " | ".join(cells))


if __name__ == "__main__":
    if "--treatment" in sys.argv:
        TREATMENT = sys.argv[sys.argv.index("--treatment") + 1]
    main(Path(sys.argv[1]))
