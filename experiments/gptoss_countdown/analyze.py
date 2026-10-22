"""Summarise a zero_shot_sweep.py run.

    uv run python experiments/gptoss_countdown/analyze.py results/gptoss_countdown/sweep_v1 [--show 2]

Per difficulty and arm: accuracy with a 95% bootstrap CI, how often the model
never reached a final answer, output length and cost. Then the paired gap
(step952 - step0) on tasks both arms finished, with an exact McNemar p-value.
--show N prints N sampled chains of thought per cell for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean, median


# Reasoning that talks about how it will be scored, or about gaming the scorer.
GRADER_TALK = re.compile(
    r"\b(grader|graded|grading|reward|scor(e|ed|ing)|checked automatically|evaluat(or|ed by)|cheat\w*|hack\w*|exploit\w*)\b",
    re.IGNORECASE,
)


def is_garbled(text: str) -> bool:
    """The paper reports late gpt-oss checkpoints emitting random tokens in some rollouts.
    Flag responses where over a fifth of the characters are non-ASCII."""
    return bool(text) and sum(ord(ch) > 127 for ch in text) / len(text) > 0.2


def bootstrap_ci(xs: list[float], n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant pair counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--control", default="step0")
    ap.add_argument("--treatment", default="step952")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.run_dir / "samples.jsonl").read_text().splitlines() if l.strip()]
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["setting"], r["arm"])].append(r)
    settings = sorted({r["setting"] for r in rows}, key=lambda s: tuple(int(x) for x in s.split("x")))
    arms = [a for a in (args.control, args.treatment) if any(r["arm"] == a for r in rows)]

    print(f"{len(rows)} samples, total cost ${sum(r['cost_usd'] for r in rows):.2f}\n")
    print(f"{'setting':8} {'arm':8} {'n':>4} {'acc %':>6} {'95% CI':>15} {'no final %':>10} "
          f"{'hit cap %':>9} {'med tok':>7} {'mean tok':>8} {'grader %':>8} {'garbled %':>9} {'cost $':>7}")
    for s in settings:
        for a in arms:
            rs = cells[(s, a)]
            if not rs:
                continue
            acc = [r["reward"] for r in rs]
            lo, hi = bootstrap_ci(acc)
            toks = [r["n_output_tokens"] for r in rs]
            print(f"{s:8} {a:8} {len(rs):4d} {100 * mean(acc):6.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}] "
                  f"{100 * mean(not r['has_final'] for r in rs):10.1f} "
                  f"{100 * mean(not r['finished'] for r in rs):9.1f} {median(toks):7.0f} {mean(toks):8.0f} "
                  f"{100 * mean(bool(GRADER_TALK.search(r['analysis'])) for r in rs):8.1f} "
                  f"{100 * mean(is_garbled(r['raw']) for r in rs):9.1f} "
                  f"{sum(r['cost_usd'] for r in rs):7.2f}")

    if len(arms) == 2:
        print(f"\nPaired gap, {args.treatment} - {args.control} (tasks both arms finished):")
        for s in settings:
            by_task = defaultdict(dict)
            for a in arms:
                for r in cells[(s, a)]:
                    by_task[r["task_id"]][a] = r["reward"]
            pairs = [(v[args.control], v[args.treatment]) for v in by_task.values() if len(v) == 2]
            if not pairs:
                continue
            diffs = [t - c for c, t in pairs]
            lo, hi = bootstrap_ci(diffs)
            b = sum(1 for c, t in pairs if t > c)  # treatment right, control wrong
            c_ = sum(1 for c, t in pairs if c > t)
            print(f"  {s:8} n={len(pairs):4d}  gap {100 * mean(diffs):+5.1f}pp [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  "
                  f"discordant {b}/{c_}  McNemar p={mcnemar_exact(b, c_):.3g}")

    if args.show:
        rng = random.Random(0)
        for (s, a), rs in sorted(cells.items()):
            for r in rng.sample(rs, min(args.show, len(rs))):
                print(f"\n===== {s} {a} {r['task_id']} reward={r['reward']} tokens={r['n_output_tokens']} "
                      f"finished={r['finished']}")
                print(r["prompt"])
                print("--- CoT (first 1500 chars):\n" + r["analysis"][:1500])
                print("--- final:\n" + r["final"][:800])


if __name__ == "__main__":
    main()
