"""Score the blind judge labels: rates per stage/step (random sample only), regex precision/recall,
judge agreement on duplicated files, advantage and solve rate of speculating episodes.

    python3 -I scripts/score_judge.py
"""
import glob
import json
import random
from collections import Counter, defaultdict

key = json.load(open("judge/key.json"))
eps = {(e["step"], e["g"], e["i"]): e for e in map(json.loads, open("eps.jsonl"))}
labels = {}
for f in sorted(glob.glob("judge/labels/*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels[d["id"]] = d
missing = [h for h in key if h not in labels]
print(f"labelled {len(labels)} / {len(key)}; missing {len(missing)}; unknown ids {len([h for h in labels if h not in key])}")


def ci(xs, n=2000):
    if not xs:
        return (0, 0)
    rng = random.Random(0)
    bs = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return bs[int(.025 * n)], bs[int(.975 * n)]


# Agreement on duplicates (yes vs not-yes, and 3-way)
pairs = [(labels[v["dup_of"]], labels[h]) for h, v in key.items() if "dup_of" in v and h in labels and v["dup_of"] in labels]
if pairs:
    a3 = sum(x["spec"] == y["spec"] for x, y in pairs) / len(pairs)
    a2 = sum((x["spec"] == "yes") == (y["spec"] == "yes") for x, y in pairs) / len(pairs)
    print(f"duplicates {len(pairs)}: 3-way agreement {100*a3:.0f}%, yes vs not-yes {100*a2:.0f}%")

rows = []
for h, v in key.items():
    if "dup_of" in v or h not in labels:
        continue
    k = (v["step"], v["g"], v["i"])
    e, l = eps[k], labels[h]
    rows.append({**e, "src_kind": v["src"], "spec": l["spec"], "kinds": l.get("kinds") or [], "shapes": l.get("shapes"), "quote": l.get("quote", ""),
                 "regex_any": bool(e["guess"] or e["shortcut"]), "regex_guess_only": bool(e["guess"] and not e["shortcut"])})
rnd = [r for r in rows if r["src_kind"] == "random"]

print("\nRandom sample, by stage (judge):")
print(f"{'stage':>5} {'n':>4} {'spec yes':>16} {'yes+bl':>7} {'yes: size':>9} {'content':>8} {'mechanics':>9} {'gaming':>7} {'shapes':>7}")
for st in sorted({r["stage"] for r in rnd}):
    rs = [r for r in rnd if r["stage"] == st]
    y = [r["spec"] == "yes" for r in rs]
    lo, hi = ci(y)
    yes = [r for r in rs if r["spec"] == "yes"]
    kc = Counter(k for r in yes for k in r["kinds"])
    print(f"{st:>5} {len(rs):>4} {100*sum(y)/len(rs):>5.1f}% [{100*lo:.0f},{100*hi:.0f}] {100*sum(r['spec']!='no' for r in rs)/len(rs):>6.1f}% "
          + " ".join(f"{kc[k]:>8}" for k in ("size", "content", "mechanics", "gaming")) + f" {sum(r['shapes']=='yes' for r in yes):>7}")
print("\nRandom sample, by step (judge spec=yes %, content-kind %):")
for stp in sorted({r["step"] for r in rnd}):
    rs = [r for r in rnd if r["step"] == stp]
    print(f"  step {stp:>3} (s{rs[0]['stage']}): yes {100*sum(r['spec']=='yes' for r in rs)/len(rs):5.1f}%  content {100*sum(r['spec']=='yes' and 'content' in r['kinds'] for r in rs)/len(rs):5.1f}%  n={len(rs)}")

# Regex vs judge on the random sample
tp = sum(r["regex_any"] and r["spec"] == "yes" for r in rnd)
fp = sum(r["regex_any"] and r["spec"] != "yes" for r in rnd)
fn = sum((not r["regex_any"]) and r["spec"] == "yes" for r in rnd)
print(f"\nregex (any) vs judge yes, random sample: precision {tp}/{tp+fp}, recall {tp}/{tp+fn}")
rs = [r for r in rows if r["src_kind"] == "regex_spec"]
print(f"regex 'speculation only' oversample: judge yes {sum(r['spec']=='yes' for r in rs)}/{len(rs)}, borderline {sum(r['spec']=='borderline' for r in rs)}")

# Reinforcement: advantage and solve rate, all labelled episodes, by judge kind
print("\nAdvantage (reward − group mean) by judge label, all labelled episodes:")
for name, f in [("spec=yes", lambda r: r["spec"] == "yes"), ("  content", lambda r: r["spec"] == "yes" and "content" in r["kinds"]),
                ("  size", lambda r: r["spec"] == "yes" and "size" in r["kinds"]), ("  mechanics", lambda r: r["spec"] == "yes" and "mechanics" in r["kinds"]),
                ("  gaming", lambda r: r["spec"] == "yes" and "gaming" in r["kinds"]), ("spec=no", lambda r: r["spec"] == "no")]:
    rs = [r for r in rows if f(r)]
    if rs:
        a = [r["adv"] for r in rs]
        lo, hi = ci(a)
        print(f"  {name:<11} n={len(rs):>3}  adv {sum(a)/len(a):+.2f} [{lo:+.2f},{hi:+.2f}]  solved {100*sum(r['reward']>0 for r in rs)/len(rs):.0f}%")
json.dump(rows, open("judge/scored.json", "w"))
