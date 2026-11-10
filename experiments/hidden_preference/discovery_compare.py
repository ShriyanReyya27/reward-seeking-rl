"""Compare two discovery runs on the same episodes, e.g. the plain prompt against a nudged one.

    python experiments/hidden_preference/discovery_compare.py discovery_v1 discovery_persist

Rows are paired by (arm, base_id, k): same rule, final request, evidence and sampling seed, so the only
difference is the prompt change. Per arm: accuracy, gain over k = 0 and the change in each from run A to run
B; the oss430 − c430 gap in each run; the difference-in-differences (does the change close the gap?), all
with 95% bootstrap CIs over episodes. Also output tokens, truncation and a rough give-up regex (the one used
for ``discovery_v1``; the judge is the real check). Writes ``compare_<A>.md`` next to run B's samples.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from discovery_analyze import TIERS, boot  # noqa: E402

GIVE = re.compile(r"can(no|')t (infer|determine|deduce|figure)|no clear pattern|not enough (info|data|evidence)|"
                  r"given (the )?(difficulty|uncertainty)|just (guess|pick|choose)|random(ly)? (guess|pick|choose)|"
                  r"hard to (infer|determine|tell|deduce)|unclear|we might guess|best guess", re.I)


def load(run):
    rows = [json.loads(l) for l in (HERE / "results" / run / "samples.jsonl").read_text().split("\n") if l.strip()]
    return {(r["arm"], r["base_id"], r["k"]): r for r in rows}


def pct(m, lo, hi):
    return f"{100 * m:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline run")
    ap.add_argument("b", help="changed run")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    arms = sorted({x for x, _, _ in A} & {x for x, _, _ in B})
    ks = sorted({k for *_, k in A} & {k for *_, k in B})
    eps = sorted({b for _, b, _ in A} & {b for _, b, _ in B})
    eps = [e for e in eps if all((x, e, k) in R for R in (A, B) for x in arms for k in ks)]
    meta = {e: A[(arms[0], e, ks[0])] for e in eps}
    out = []
    p = out.append
    p(f"# {args.b} against {args.a}\n")
    p(f"{len(eps)} episodes complete in both runs, arms {', '.join(arms)}, k in {ks}.\n")
    groups = [("all", lambda e: True)] + [(t, lambda e, t=t: meta[e]["tier"] == t) for t in TIERS] + \
             [("medium+hard", lambda e: meta[e]["tier"] != "easy")]

    p(f"## Accuracy (%), {args.a} → {args.b}\n")
    p("| group | arm | " + " | ".join(f"k={k}" for k in ks) + " |")
    p("|---|---|" + "---|" * len(ks))
    for name, pred in groups:
        es = [e for e in eps if pred(e)]
        for x in arms:
            cells = [f"{100 * st.mean(A[(x, e, k)]['reward'] for e in es):.0f} → "
                     f"{100 * st.mean(B[(x, e, k)]['reward'] for e in es):.0f}" for k in ks]
            p(f"| {name} (n={len(es)}) | {x} | " + " | ".join(cells) + " |")
    p("")

    kmax = ks[-1]
    p(f"## At k = {kmax} (pp) [95% CI over episodes]\n")
    p("| group | change, " + " | change, ".join(arms) + " | gap in A | gap in B | gap change (DiD) |")
    p("|---|" + "---|" * (len(arms) + 3))
    for name, pred in groups:
        es = [e for e in eps if pred(e)]
        cells = [pct(*boot([B[(x, e, kmax)]["reward"] - A[(x, e, kmax)]["reward"] for e in es])) for x in arms]
        if len(arms) == 2:
            c, o = arms
            cells += [pct(*boot([R[(o, e, kmax)]["reward"] - R[(c, e, kmax)]["reward"] for e in es])) for R in (A, B)]
            cells.append(pct(*boot([(B[(o, e, kmax)]["reward"] - B[(c, e, kmax)]["reward"])
                                    - (A[(o, e, kmax)]["reward"] - A[(c, e, kmax)]["reward"]) for e in es])))
        p(f"| {name} | " + " | ".join(cells) + " |")
    p("")
    p(f"## Gain over k = 0 at k = {kmax} (pp), {args.a} → {args.b}\n")
    for name, pred in groups:
        es = [e for e in eps if pred(e)]
        parts = []
        for x in arms:
            ga = st.mean(A[(x, e, kmax)]["reward"] - A[(x, e, 0)]["reward"] for e in es)
            gb = st.mean(B[(x, e, kmax)]["reward"] - B[(x, e, 0)]["reward"] for e in es)
            parts.append(f"{x} {100 * ga:+.1f} → {100 * gb:+.1f}")
        p(f"- {name}: " + "; ".join(parts))
    p("")

    p(f"## Behaviour, {args.a} → {args.b}\n")
    p("| arm | k | output tokens | truncated % | give-up regex %, medium+hard |")
    p("|---|---|---|---|---|")
    for x in arms:
        for k in ks:
            ra = [A[(x, e, k)] for e in eps]
            rb = [B[(x, e, k)] for e in eps]
            ga = [bool(GIVE.search(r["analysis"])) for r in ra if r["tier"] != "easy"]
            gb = [bool(GIVE.search(r["analysis"])) for r in rb if r["tier"] != "easy"]
            p(f"| {x} | {k} | {st.mean(r['n_output_tokens'] for r in ra):.0f} → "
              f"{st.mean(r['n_output_tokens'] for r in rb):.0f} | "
              f"{100 * st.mean(not r['finished'] for r in ra):.1f} → {100 * st.mean(not r['finished'] for r in rb):.1f} | "
              f"{100 * st.mean(ga):.0f} → {100 * st.mean(gb):.0f} |")
    p("")
    p("Cost: " + "; ".join(f"{x} ${sum(r['cost_usd'] for (a, _, _), r in B.items() if a == x):.2f}" for x in arms))
    text = "\n".join(out) + "\n"
    (HERE / "results" / args.b / f"compare_{args.a}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
