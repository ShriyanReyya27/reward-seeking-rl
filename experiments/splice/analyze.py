"""Summarise a splice zero-shot run against the pilot criteria.

    uv run python experiments/splice/analyze.py --run pilot_v1

Criteria (from the design): on binding episodes, identity reward 0.3-0.6 and instruction >= 0.85;
non-binding / human episodes >= 0.9 under instruction; most binding failures are the raw value
(raw_default); >= 30% of binding identity prompts have mixed outcomes across samples (GRPO signal);
truncation <= 3%. The nudge row says whether the gap is propensity (nudge closes it) or knowledge.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(groups, n_boot=2000, seed=0):
    """95% CI of the mean, resampling prompts (each group is one prompt's per-sample values)."""
    rng = random.Random(seed)
    groups = [g for g in groups if g]
    if not groups:
        return float("nan"), float("nan")
    ms = sorted(mean(x for g in (rng.choice(groups) for _ in groups) for x in g) for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--by-consumer", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(x) for x in (HERE / "results" / args.run / "samples.jsonl").read_text().splitlines() if x.strip()]
    from rsrl.envs.splice.env import PAYLOADS  # noqa: E402

    current = {v for _, v in PAYLOADS}
    dropped = sum(r["value"] not in current for r in rows)
    rows = [r for r in rows if r["value"] in current]  # payloads since removed from the env
    if dropped:
        print(f"(excluded {dropped} samples on payloads no longer in the env)")
    arms = sorted({r["arm"] for r in rows})
    conds = [c for c in ("identity", "nudge", "instruction") if any(r["condition"] == c for r in rows)]
    print(f"{len(rows)} samples, ${sum(r['cost_usd'] for r in rows):.2f}\n")

    def sel(arm, cond, pool, kind):
        out = [r for r in rows if r["arm"] == arm and r["condition"] == cond and r["pool"] == pool]
        human = lambda r: r["consumer"].startswith("human")  # noqa: E731
        if kind == "bind":
            return [r for r in out if r["binding"]]
        if kind == "free":
            return [r for r in out if not r["binding"] and not human(r)]
        return [r for r in out if human(r)]

    for pool in ("train", "heldout"):
        print(f"== {pool} consumers ==")
        print(f"  {'arm':>7} {'cond':>11} | {'bind':>5} [95% CI]       {'free':>5} {'human':>5} | raw/bindfail "
              f"mixed  trunc  tokens")
        for arm in arms:
            for cond in conds:
                b, f, h = sel(arm, cond, pool, "bind"), sel(arm, cond, pool, "free"), sel(arm, cond, pool, "human")
                groups = defaultdict(list)
                for r in b:
                    groups[r["prompt_id"]].append(r["reward"])
                lo, hi = boot_ci(list(groups.values()))
                fails = [r for r in b if not r["reward"]]
                mixed = mean(0 < sum(g) < len(g) for g in groups.values())
                allr = b + f + h
                print(f"  {arm:>7} {cond:>11} | {mean(r['reward'] for r in b):.2f} [{lo:.2f}, {hi:.2f}]  "
                      f"{mean(r['reward'] for r in f):.2f}  {mean(r['reward'] for r in h):.2f} | "
                      f"{mean(r['raw_default'] for r in fails):.2f}         {mixed:.2f}   "
                      f"{mean(not r['finished'] for r in allr):.2f}  {mean(r['n_output_tokens'] for r in allr):.0f}")
        print()

    if args.by_consumer:
        print("binding reward by consumer (identity / nudge / instruction)")
        cons = sorted({r["consumer"] for r in rows}, key=lambda c: (next(r["pool"] for r in rows if r["consumer"] == c) != "train", c))
        print("  " + " " * 14 + "".join(f"{a:>20}" for a in arms))
        for c in cons:
            cells = []
            for a in arms:
                vals = []
                for cond in conds:
                    rs = [r for r in rows if r["arm"] == a and r["consumer"] == c and r["condition"] == cond
                          and (r["binding"] or c.startswith("human"))]
                    vals.append(f"{mean(r['reward'] for r in rs):.2f}")
                cells.append("/".join(vals))
            print(f"  {c:<14}" + "".join(f"{x:>20}" for x in cells))


if __name__ == "__main__":
    main()
