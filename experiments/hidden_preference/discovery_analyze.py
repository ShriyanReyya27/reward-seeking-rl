"""Tables for the zero-shot discovery test (``discovery_zero_shot.py``).

    python experiments/hidden_preference/discovery_analyze.py discovery_v1 [--finished-only]

Accuracy by arm, tier and k; gain = accuracy at k minus accuracy at k = 0 on the same episode; oss430 − c430
difference in gain with a 95% bootstrap CI over episodes (paired by episode: same final request, same evidence).
Per-rule accuracy at k = 0 and 12. Truncation and dropped-channel rates. Writes ``summary.md`` next to the samples.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIERS = ("easy", "medium", "hard")


def boot(xs, n=2000, seed=0):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return st.mean(xs), ms[int(0.025 * n)], ms[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--finished-only", action="store_true", help="drop episodes where either arm truncated at any k")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (HERE / "results" / a.run / "samples.jsonl").read_text().split("\n") if l.strip()]
    arms = sorted({r["arm"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    R = {(r["arm"], r["base_id"], r["k"]): r for r in rows}
    eps = sorted({r["base_id"] for r in rows})
    meta = {r["base_id"]: r for r in rows}
    complete = [b for b in eps if all((x, b, k) in R for x in arms for k in ks)]
    if a.finished_only:
        complete = [b for b in complete if all(R[(x, b, k)]["finished"] for x in arms for k in ks)]
    out = []
    p = out.append
    p(f"# {a.run}: zero-shot discovery\n")
    p(f"{len(rows)} samples, {len(complete)} complete episodes"
      + (" (finished-only: episodes where any sample truncated are dropped)" if a.finished_only else "") + ".\n")
    for x in arms:
        xs = [r for r in rows if r["arm"] == x]
        p(f"- {x}: truncated {100 * st.mean(not r['finished'] for r in xs):.1f}%, channel dropped "
          f"{100 * st.mean(bool(r.get('channel_dropped')) for r in xs):.1f}%, mean output tokens "
          f"{st.mean(r['n_output_tokens'] for r in xs):.0f}, ${sum(r['cost_usd'] for r in xs):.2f}")
    p("")

    groups = [("all", lambda b: True)] + [(t, lambda b, t=t: meta[b]["tier"] == t) for t in TIERS] + \
             [(f"{f}/{t}", lambda b, f=f, t=t: meta[b]["family"] == f and meta[b]["tier"] == t)
              for f in ("number", "word") for t in TIERS]
    p("## Accuracy by k (%), c430 / oss430\n")
    p("| group | chance | " + " | ".join(f"k={k}" for k in ks) + " |")
    p("|---|---|" + "---|" * len(ks))
    for name, pred in groups:
        bs = [b for b in complete if pred(b)]
        if not bs:
            continue
        cells = []
        for k in ks:
            cells.append(" / ".join(f"{100 * st.mean(R[(x, b, k)]['reward'] for b in bs):.0f}" for x in arms))
        p(f"| {name} (n={len(bs)}) | {100 * st.mean(meta[b]['chance'] for b in bs):.0f} | " + " | ".join(cells) + " |")
    p("")

    if len(arms) == 2:
        c, o = arms  # alphabetical: c430, oss430
        p(f"## Gain over k = 0 (pp), and {o} − {c} difference in gain [95% CI]\n")
        p("| group | k | " + f"{c} gain | {o} gain | difference |")
        p("|---|---|---|---|---|")
        for name, pred in groups[:4]:
            bs = [b for b in complete if pred(b)]
            for k in ks[1:]:
                gc = [R[(c, b, k)]["reward"] - R[(c, b, 0)]["reward"] for b in bs]
                go = [R[(o, b, k)]["reward"] - R[(o, b, 0)]["reward"] for b in bs]
                m, lo, hi = boot([y - x for x, y in zip(gc, go)])
                p(f"| {name} | {k} | {100 * st.mean(gc):+.1f} | {100 * st.mean(go):+.1f} | "
                  f"{100 * m:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}] |")
        p("")
        p(f"## Accuracy at k = 12 minus k = 12 for the other arm, {o} − {c} (pp) [95% CI]\n")
        for name, pred in groups[:4]:
            bs = [b for b in complete if pred(b)]
            m, lo, hi = boot([R[(o, b, ks[-1])]["reward"] - R[(c, b, ks[-1])]["reward"] for b in bs])
            p(f"- {name}: {100 * m:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]")
        p("")

    p("## Per rule: chance, accuracy at k = 0 → k = max (%), per arm\n")
    p("| family | tier | rule | chance | " + " | ".join(arms) + " |")
    p("|---|---|---|---|" + "---|" * len(arms))
    by_rule = defaultdict(list)
    for b in complete:
        by_rule[(meta[b]["family"], meta[b]["tier"], meta[b]["rule"])].append(b)
    order = {t: i for i, t in enumerate(TIERS)}
    for (f, t, rule), bs in sorted(by_rule.items(), key=lambda kv: (kv[0][0], order[kv[0][1]], kv[0][2])):
        cells = [f"{100 * st.mean(R[(x, b, ks[0])]['reward'] for b in bs):.0f} → "
                 f"{100 * st.mean(R[(x, b, ks[-1])]['reward'] for b in bs):.0f}" for x in arms]
        p(f"| {f} | {t} | {rule} | {100 * st.mean(meta[b]['chance'] for b in bs):.0f} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    suffix = "_finished" if a.finished_only else ""
    (HERE / "results" / a.run / f"summary{suffix}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
