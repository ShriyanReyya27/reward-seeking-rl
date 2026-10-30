"""Model-card eval table: each control arm vs base on the four single-turn evals, in the stage-1–5 cards' metrics.

    python experiments/rh_evals/card_table.py control_s6_s8 c240,c280,c320,c360,c400
"""
import sys
from collections import defaultdict

import analyze
from analyze import REDUCED_PROBE, condition_diff, load, per_item, probe_index

run, arms = sys.argv[1], sys.argv[2].split(",")
data = load(run)
probe = [r for r in data["grader_conflict"] if REDUCED_PROBE(r)]
EVALS = {
    "sorh": lambda: analyze.compare(per_item(data["sorh"], "hack")),
    "probe (reduced)": lambda where=None: probe_index([r for r in probe if where is None or where(r)]),
    "keing1_mc_revealing_score": lambda: analyze.compare(per_item(data["keing1_mc_revealing_score"], "hack")),
    "countdown_leaky": lambda: condition_diff(data["countdown_leaky"], "task_id", "leaky", "strict"),
}


def cmp(name, arm, where=None):
    analyze.ARMS = ("base", arm)
    return EVALS[name](where) if where else EVALS[name]()


def f(d):
    return f"{100 * d['mean']:+.1f} [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}], p={d['p']:.2g}"


print("| Eval | base | " + " | ".join(arms) + " |")
for name in EVALS:
    cells, base = [], None
    for a in arms:
        c = cmp(name, a)
        base = c["base"]["mean"]
        cells.append(f"{100 * c[a]['mean']:.1f} ({f(c['diff'])})")
    print(f"| {name} | {100 * base:.1f} | " + " | ".join(cells) + " |")

print("\nProbe index by channel (this − base):")
for ch in sorted({r["meta"]["channel"] for r in data["grader_conflict"] if REDUCED_PROBE(r)}):
    print(f"  {ch}: " + "; ".join(f"{a} {f(cmp('probe (reduced)', a, lambda r, ch=ch: r['meta']['channel'] == ch)['diff'])}"
                                   for a in arms))

print("\nArm vs arm (paired), probe and sorh:")
for x, y in [("c280", "c240"), ("c320", "c280"), ("c360", "c320"), ("c400", "c360"), ("c400", "c280"), ("c430", "c400"), ("c430", "c280")]:
    if x not in arms or y not in arms:
        continue
    analyze.ARMS = (y, x)
    print(f"  {x} − {y}: probe {f(probe_index(probe)['diff'])}; sorh {f(analyze.compare(per_item(data['sorh'], 'hack'))['diff'])}")

print("\ncountdown_leaky: share of responses with no final answer at the token limit; probe mean output tokens")
for a in ["base"] + arms:
    cd = [r for r in data["countdown_leaky"] if r["arm"] == a]
    trunc = sum(1 for r in cd if r.get("stop_reason") == "length" and not r.get("has_final")) / max(1, len(cd))
    pr = [r for r in probe if r["arm"] == a]
    toks = sum(r.get("n_output_tokens") or 0 for r in pr) / max(1, len(pr))
    print(f"  {a}: countdown truncated {100 * trunc:.0f}% (n={len(cd)}), probe tokens {toks:.0f}, "
          f"errors {sum(1 for d in data.values() for r in d if r['arm'] == a and r.get('error'))}")
