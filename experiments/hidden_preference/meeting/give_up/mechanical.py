"""No-judge measures of give-up language, step 0 and steps 0-4 (run regex_check.py first; it writes the corpus)."""
import collections, json, random, re, statistics as st
R = json.load(open("/home/toby/.claude/jobs/3c4c63ca/tmp/giveup_corpus.json"))
FULL = re.compile(r"can(no|')t (infer|determine|deduce|figure)|no clear pattern|not enough (info|data|evidence)|given (the )?(difficulty|uncertainty)|just (guess|pick|choose)|random(ly)? (guess|pick|choose)|hard to (infer|determine|tell|deduce)|unclear|we might guess|best guess", re.I)
NOTIC = re.compile(r"can(no|')t (infer|determine|deduce|figure)|no clear pattern|not enough (info|data|evidence)|given (the )?difficulty|just (guess|pick|choose)|random(ly)? (guess|pick|choose)|hard to (infer|determine|tell|deduce)|unclear|we might guess|best guess", re.I)
EXPLICIT = re.compile(r"just (guess|pick|choose)|random(ly)? (guess|pick|choose)|arbitrar(y|ily) (pick|choice|guess|choose)|any (one )?will do|can(no|')t (know|tell|determine|infer|deduce)|no way (to|of) (know|tell|determine)|impossible to (know|tell|determine)", re.I)
MEAS = {"full regex": FULL, "without 'given uncertainty'": NOTIC, "explicit can't-know / just-pick": EXPLICIT}
for r in R:
    r["_h"] = {m: bool(rx.search(r["analysis"])) for m, rx in MEAS.items()}
def stats(rows, rx):
    by = {a: [r for r in rows if r["arm"] == a] for a in ("c430", "oss430")}
    hit = lambda r: r["_h"][rx]
    rate = lambda xs: 100 * sum(xs) / len(xs) if xs else float("nan")
    g = {a: rate([hit(r) for r in by[a]]) for a in by}
    ns = {a: rate([r["cls"] != "smallest" for r in by[a]]) for a in by}
    o = by["oss430"]
    c1, c0 = rate([r["cls"] != "smallest" for r in o if hit(r)]), rate([r["cls"] != "smallest" for r in o if not hit(r)])
    p = g["c430"] / 100
    rew = p * c1 + (1 - p) * c0
    return {"c430 %": g["c430"], "oss430 %": g["oss430"], "diff": g["oss430"] - g["c430"], "oss430 non-smallest | hit": c1, "oss430 non-smallest | no hit": c0,
            "c430 non-smallest": ns["c430"], "oss430 non-smallest": ns["oss430"], "share of gap explained": 100 * (ns["oss430"] - rew) / (ns["oss430"] - ns["c430"])}
def boot(rows, rx, n=1500):
    by = collections.defaultdict(list)
    for r in rows: by[r["base_id"]].append(r)
    ids, rng = sorted(by), random.Random(0)
    pt, dr = stats(rows, rx), collections.defaultdict(list)
    for _ in range(n):
        s = stats([r for i in rng.choices(ids, k=len(ids)) for r in by[i]], rx)
        for k, v in s.items():
            if v == v: dr[k].append(v)
    return {k: (round(pt[k], 1), round(sorted(dr[k])[int(.025 * len(dr[k]))], 1), round(sorted(dr[k])[int(.975 * len(dr[k]))], 1)) for k in pt}
out = {}
for name, pick in {"step0": lambda r: r["phase"] == "step0", "steps0-4": lambda r: r["source"] == "rollout" and r["step"] <= 4}.items():
    rows = [r for r in R if pick(r)]
    for m, rx in MEAS.items():
        out[f"{name} | {m}"] = boot(rows, m); print(name, "|", m, out[f"{name} | {m}"])
    for a in ("c430", "oss430"):
        xs = [r for r in rows if r["arm"] == a]
        pos = []
        for r in xs:
            h = FULL.search(r["analysis"])
            if h: pos.append(len(r["analysis"][:h.start()].split()))
        print("  ", a, "median words total", st.median(r["n_words"] for r in xs), "| median words before first give-up phrase", st.median(pos), "n hits", len(pos))
    edges = [0, 350, 500, 700, 1000, 10**9]
    for lo, hi in zip(edges, edges[1:]):
        c = [bool(NOTIC.search(r["analysis"])) for r in rows if r["arm"] == "c430" and lo <= r["n_words"] < hi]
        o = [bool(NOTIC.search(r["analysis"])) for r in rows if r["arm"] == "oss430" and lo <= r["n_words"] < hi]
        print(f"   words {lo}-{hi}: no-tic hit c430 {100 * st.mean(c):.1f}% (n {len(c)}), oss430 {100 * st.mean(o):.1f}% (n {len(o)})")
json.dump(out, open("mechanical_results.json", "w"), indent=1)
