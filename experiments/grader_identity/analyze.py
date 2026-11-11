"""Summarise a grader-identity zero-shot run.

    uv run python experiments/grader_identity/analyze.py --run pilot_v1

Prints reward by arm x disclosure x pool, the identity-minus-instruction gap (how much the model loses
when it has to infer the format), per-identity reward, and why rewards were lost (unsolved vs solved in
the wrong format vs truncated). ``--cot`` adds how often the reasoning mentions the consumer or format.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Reasoning that thinks about how the reply will be read.
COT_CONSUMER = re.compile(
    r"\b(pars(?:e|ed|er|es|ing)|eval(?:uate[sd]?)?\(?|format(?:ted|ting)?|interpret(?:er|ed)?|"
    r"checker|grader|cell|formula|prefix|s-expression|reverse polish|postfix|json|xml|latex|csv|"
    r"characters?|sms|notification|query|select)\b", re.I)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(pairs, n_boot=2000, seed=0):
    """95% CI of the mean of paired differences."""
    rng = random.Random(seed)
    if not pairs:
        return float("nan"), float("nan")
    ms = sorted(mean([rng.choice(pairs) for _ in pairs]) for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cot", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (HERE / "results" / args.run / "samples.jsonl").read_text().splitlines() if line.strip()]
    arms = sorted({r["arm"] for r in rows})
    print(f"{len(rows)} samples, ${sum(r['cost_usd'] for r in rows):.2f}\n")

    print("reward (solved, wrong_format, truncated) by arm x disclosure x pool")
    by = defaultdict(list)
    for r in rows:
        by[r["arm"], r["disclosure"], r["pool"]].append(r)
    for key in sorted(by):
        rs = by[key]
        print(f"  {key[0]:>7} {key[1]:>11} {key[2]:>7}  n={len(rs):4d}  reward {mean([r['reward'] for r in rs]):.2f}"
              f"  solved {mean([r['solved'] for r in rs]):.2f}  wrong_fmt {mean([r['wrong_format'] for r in rs]):.2f}"
              f"  trunc {mean([not r['finished'] for r in rs]):.2f}  tokens {mean([r['n_output_tokens'] for r in rs]):.0f}")

    print("\nidentity - instruction reward, paired by (identity, episode), excluding plain")
    idx = {(r["arm"], r["disclosure"], r["identity"], r["episode"]): r for r in rows}
    for arm in arms:
        for pool in ("train", "heldout"):
            diffs = [idx[arm, "identity", i, e]["reward"] - idx[arm, "instruction", i, e]["reward"]
                     for (a, d, i, e), r in idx.items()
                     if a == arm and d == "identity" and r["pool"] == pool and not i.startswith("plain")
                     and (arm, "instruction", i, e) in idx]
            lo, hi = boot_ci(diffs)
            print(f"  {arm:>7} {pool:>7}  {mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]  n={len(diffs)}")

    if len(arms) > 1 and {"c430", "oss430"} <= set(arms):
        print("\noss430 - c430 reward, identity disclosure, paired by (identity, episode), excluding plain")
        for pool in ("train", "heldout"):
            diffs = [idx["oss430", "identity", i, e]["reward"] - idx["c430", "identity", i, e]["reward"]
                     for (a, d, i, e), r in idx.items()
                     if a == "oss430" and d == "identity" and r["pool"] == pool and not i.startswith("plain")
                     and ("c430", "identity", i, e) in idx]
            lo, hi = boot_ci(diffs)
            print(f"  {pool:>7}  {mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]  n={len(diffs)}")

    print("\nreward by identity (identity / instruction)")
    pool_of = {r["identity"]: r["pool"] for r in rows}
    names = sorted(pool_of, key=lambda n: (pool_of[n] != "train", n))
    print("  " + " " * 14 + "".join(f"{a:>14}" for a in arms))
    for n in names:
        cells = []
        for a in arms:
            vals = [mean([r["reward"] for r in rows if r["arm"] == a and r["identity"] == n and r["disclosure"] == d])
                    for d in ("identity", "instruction")]
            cells.append(f"{vals[0]:.2f}/{vals[1]:.2f}")
        print(f"  {n:<14}" + "".join(f"{c:>14}" for c in cells))

    if args.cot:
        print("\nreasoning mentions the consumer/format (identity disclosure, non-plain)")
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm and r["disclosure"] == "identity" and not r["identity"].startswith("plain")]
            print(f"  {arm:>7}  {mean([bool(COT_CONSUMER.search(r['analysis'])) for r in rs]):.2f}  n={len(rs)}")


if __name__ == "__main__":
    main()
