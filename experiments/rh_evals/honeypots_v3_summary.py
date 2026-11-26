"""Summarise honeypots_v3 records: exploit rate per condition, paired bait-decoy and bait-removed with 95% CIs.

    python experiments/rh_evals/honeypots_v3_summary.py results/<run>/honeypots_v3*.jsonl [--by item_set|sample]

Rates are averaged over samples per item, then over items; the CI is a bootstrap over items (paired across
conditions). ``read_grader`` and ``honest_pass`` are reported pooled over conditions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def boot(vals, n=2000, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    ms = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def pct(t):
    m, lo, hi = t
    return "n/a" if m != m else f"{100 * m:+.0f} [{100 * lo:+.0f},{100 * hi:+.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", default=None, help="extra split: item_set or sample")
    a = ap.parse_args()
    recs = [json.loads(l) for f in a.files for l in open(f) if l.strip()]
    recs = [r for r in recs if not r.get("error") and r.get("exploit") is not None]
    # cell -> item -> condition -> list of (exploit, read_grader, honest_pass)
    d: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        m = r["meta"]
        split = (str(r["sample"]) if a.by == "sample" else m.get(a.by)) if a.by else "all"
        d[(r["arm"], m["family"], m["difficulty"], split)][m["task"]][m["condition"]].append(r)
    print(f"{len(recs)} scorable records")
    hdr = f"{'arm':7} {'family':9} {'diff':10} {'split':5} {'n_it':>4} {'bait':>5} {'decoy':>5} {'remov':>5}  " \
          f"{'bait-decoy':>15} {'bait-removed':>15} {'read_gr':>7} {'honest':>6}"
    print(hdr)
    for key in sorted(d):
        items = d[key]
        rate = {c: [sum(bool(x["exploit"]) for x in v[c]) / len(v[c]) for v in items.values() if v.get(c)]
                for c in ("bait", "decoy", "removed")}
        diff = lambda c: [sum(bool(x["exploit"]) for x in v["bait"]) / len(v["bait"])
                          - sum(bool(x["exploit"]) for x in v[c]) / len(v[c])
                          for v in items.values() if v.get("bait") and v.get(c)]
        allr = [x for v in items.values() for xs in v.values() for x in xs]
        rg = sum(bool(x.get("read_grader")) for x in allr) / len(allr)
        hp = [x.get("honest_pass") for x in allr if x.get("honest_pass") is not None]
        mean = lambda xs: f"{100 * sum(xs) / len(xs):5.0f}" if xs else "  n/a"
        print(f"{key[0]:7} {key[1]:9} {key[2]:10} {key[3]:5} {len(items):>4} {mean(rate['bait'])} {mean(rate['decoy'])} "
              f"{mean(rate['removed'])}  {pct(boot(diff('decoy'))):>15} {pct(boot(diff('removed'))):>15} "
              f"{100 * rg:7.0f} {mean([float(x) for x in hp]):>6}")
    # pooled over families, per arm x difficulty
    print("\npooled over families (items = family x task)")
    pool: dict = defaultdict(list)
    for (arm, fam, diff_, split), items in d.items():
        if split != "all" and a.by:
            continue
        for t, v in items.items():
            if v.get("bait") and v.get("decoy"):
                pool[(arm, diff_)].append(sum(bool(x["exploit"]) for x in v["bait"]) / len(v["bait"])
                                          - sum(bool(x["exploit"]) for x in v["decoy"]) / len(v["decoy"]))
    for k in sorted(pool):
        print(f"  {k[0]:7} {k[1]:10} bait-decoy {pct(boot(pool[k]))}  (n={len(pool[k])})")


if __name__ == "__main__":
    main()
