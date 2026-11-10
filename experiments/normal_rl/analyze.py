"""Compare arms on a run: held-out reward at each eval, and the paired difference in gain.

    uv run python experiments/normal_rl/analyze.py knapsack_v1 [--seed 0]

Gain = final eval - step 0 on the same held-out task. Bootstrap over held-out tasks (one seed, so this
says nothing about seed-to-seed variation).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("c430", "oss430")


def load(path):
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()] if path.exists() else []


def ci(xs, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    evals = {arm: load(HERE / "results" / a.run / f"{arm}_s{a.seed}" / "eval.jsonl") for arm in ARMS}
    by = {arm: {} for arm in ARMS}
    for arm, rows in evals.items():
        for r in rows:
            by[arm].setdefault(r["step"], {})[r["task_id"]] = r
    steps = sorted(set(by["c430"]) & set(by["oss430"]))
    print(f"{'step':>4} " + " ".join(f"{arm + ' reward':>13} {'zero':>5} {'trunc':>5} {'tok':>6}" for arm in ARMS) + "  oss-c430 [95% CI]")
    for s in steps:
        parts = []
        for arm in ARMS:
            rs = list(by[arm][s].values())
            parts.append(f"{statistics.mean(r['reward'] for r in rs):13.3f} {statistics.mean(r['reward'] == 0 for r in rs):5.2f} "
                         f"{statistics.mean(r['truncated_turns'] > 0 for r in rs):5.2f} {statistics.mean(r['tokens'] for r in rs):6.0f}")
        ids = set(by["c430"][s]) & set(by["oss430"][s])
        d = [by["oss430"][s][i]["reward"] - by["c430"][s][i]["reward"] for i in ids]
        lo, hi = ci(d)
        print(f"{s:4d} " + " ".join(parts) + f"  {statistics.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if len(steps) >= 2:
        s0, s1 = steps[0], steps[-1]
        ids = set.intersection(*(set(by[arm][s]) for arm in ARMS for s in (s0, s1)))
        gains = {arm: [by[arm][s1][i]["reward"] - by[arm][s0][i]["reward"] for i in ids] for arm in ARMS}
        for arm in ARMS:
            lo, hi = ci(gains[arm])
            print(f"gain {arm}: {statistics.mean(gains[arm]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        did = [o - c for o, c in zip(gains["oss430"], gains["c430"])]
        lo, hi = ci(did)
        print(f"oss430 gain - c430 gain (n={len(did)}): {statistics.mean(did):+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
