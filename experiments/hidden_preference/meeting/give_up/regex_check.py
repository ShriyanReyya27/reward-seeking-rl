"""Give-up language on cued number = largest, c430 vs oss430, and whether it goes with non-smallest answers (no sampling).

    uv run --no-project --with-editable . python experiments/hidden_preference/meeting/give_up/regex_check.py

Windows: step 0 (same weights, all sources) and steps 0-4 of training rollouts. GIVE is the discovery_compare.py regex.
"""
from __future__ import annotations

import collections, json, random, re, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b_reasoning"))
import corpus as C  # noqa: E402

src = (HERE.parents[1] / "discovery_compare.py").read_text()
m = re.search(r"GIVE = re\.compile\((.*?)\n\s*re\.I\)|GIVE = re\.compile\((.*?)\)\n", src, re.S)
ns = {"re": re}
start = src.index("GIVE = re.compile(")
depth = 0
for i in range(start + len("GIVE = re.compile"), len(src)):
    depth += src[i] == "("
    depth -= src[i] == ")"
    if depth == 0:
        exec(src[start:i + 1], ns)
        break
GIVE = ns["GIVE"]
print("GIVE:", GIVE.pattern, GIVE.flags)
UNSURE = re.compile(r"not sure|unsure|no way (to|of)|impossible to|can(no|')t know|hard to (say|know|tell)|unknown|no information|without (more )?(info|data|prior|context)|just a guess|arbitrar|coin flip|take a guess|best guess", re.I)


def boot(rows, f, n=1000):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["base_id"]].append(f(r))
    ids, rng = list(by), random.Random(0)
    def stat(sample):
        xs = [x for i in sample for x in by[i]]
        return 100 * sum(xs) / len(xs)
    d = sorted(stat(rng.choices(ids, k=len(ids))) for _ in range(n))
    return round(stat(ids), 1), round(d[25], 1), round(d[974], 1), len(rows)


rows, *_ = C.load()
win = {"step0": lambda r: r["phase"] == "step0", "steps0-4 rollouts": lambda r: r["source"] == "rollout" and r["step"] <= 4}
for name, pick in win.items():
    print("\n==", name)
    for arm in C.ARMS:
        xs = [r for r in rows if r["arm"] == arm and pick(r)]
        print(arm, "n", len(xs), "median words", st.median(r["n_words"] for r in xs), "classes", dict(collections.Counter(r["cls"] for r in xs)))
        print("  GIVE %", boot(xs, lambda r: bool(GIVE.search(r["analysis"]))))
        print("  UNSURE %", boot(xs, lambda r: bool(UNSURE.search(r["analysis"]))))
        print("  GIVE in last 500 chars %", boot(xs, lambda r: bool(GIVE.search(r["analysis"][-500:]))))
        for cls in ("smallest", "largest", "other"):
            ys = [r for r in xs if r["cls"] == cls]
            if len(ys) > 3:
                print(f"   answered {cls}: n {len(ys)}, GIVE {100 * st.mean(bool(GIVE.search(r['analysis'])) for r in ys):.1f}%, UNSURE {100 * st.mean(bool(UNSURE.search(r['analysis'])) for r in ys):.1f}%, median words {st.median(r['n_words'] for r in ys)}")
        for lab, cond in (("GIVE", True), ("no GIVE", False)):
            ys = [r for r in xs if bool(GIVE.search(r["analysis"])) == cond]
            if ys:
                print(f"   {lab}: n {len(ys)}, non-smallest {100 * st.mean(r['cls'] != 'smallest' for r in ys):.1f}%, largest {100 * st.mean(r['cls'] == 'largest' for r in ys):.1f}%")
    hits = {a: [r for r in rows if r["arm"] == a and pick(r) and GIVE.search(r["analysis"])] for a in C.ARMS}
    for a in C.ARMS:
        c = collections.Counter(GIVE.search(r["analysis"]).group(0).lower() for r in hits[a])
        print(a, "top phrases", c.most_common(8))
json.dump([{k: r[k] for k in ("arm", "phase", "run", "source", "step", "base_id", "cls", "answer", "analysis", "n_words", "task_id")} for r in rows if r["phase"] == "step0" or (r["source"] == "rollout" and r["step"] <= 4)],
          open("/home/toby/.claude/jobs/3c4c63ca/tmp/giveup_corpus.json", "w"))
