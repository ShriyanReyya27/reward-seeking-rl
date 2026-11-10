"""Score the give-up judges against PRESTATED.md (step 0, number = largest, cued, 150 transcripts per arm).

    python3 experiments/hidden_preference/meeting/give_up/analyze.py

Reads judging/key.json and judging/verdicts_0*.jsonl; verdicts_check.jsonl is used only for agreement. Bootstrap unit is
the request (base_id), resampled over both arms together. Writes results.json here.
"""

from __future__ import annotations

import collections
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = HERE / "judging"
ARMS = ("c430", "oss430")
N_BOOT = 4000


def load_verdicts(paths):
    out = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                v = json.loads(line)
                out[int(v["id"])] = v
    return out


def rate(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def stats(rows):
    by = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    gu = {a: rate([r["gave_up"] for r in by[a]]) for a in ARMS}
    ns = {a: rate([r["cls"] != "smallest" for r in by[a]]) for a in ARMS}
    cond = {a: {g: rate([r["cls"] != "smallest" for r in by[a] if r["gave_up"] == g]) for g in (True, False)} for a in ARMS}
    within = {a: cond[a][True] - cond[a][False] for a in ARMS}
    weights = {a: min(sum(r["gave_up"] for r in by[a]), sum(not r["gave_up"] for r in by[a])) for a in ARMS}
    usable = [a for a in ARMS if weights[a] > 0 and within[a] == within[a]]
    pooled = sum(weights[a] * within[a] for a in usable) / sum(weights[a] for a in usable) if usable else float("nan")
    p_c = gu["c430"] / 100
    reweighted = p_c * cond["oss430"][True] + (1 - p_c) * cond["oss430"][False]
    gap = ns["oss430"] - ns["c430"]
    return {"gave_up_c430": gu["c430"], "gave_up_oss430": gu["oss430"], "gave_up_diff": gu["oss430"] - gu["c430"],
            "abandoned_c430": rate([r["final_stance"] == "abandoned" for r in by["c430"]]),
            "abandoned_oss430": rate([r["final_stance"] == "abandoned" for r in by["oss430"]]),
            "abandoned_diff": rate([r["final_stance"] == "abandoned" for r in by["oss430"]]) - rate([r["final_stance"] == "abandoned" for r in by["c430"]]),
            "non_smallest_c430": ns["c430"], "non_smallest_oss430": ns["oss430"], "gap": gap,
            "ns_given_up_c430": cond["c430"][True], "ns_not_given_up_c430": cond["c430"][False],
            "ns_given_up_oss430": cond["oss430"][True], "ns_not_given_up_oss430": cond["oss430"][False],
            "within_diff_pooled": pooled, "reweighted_oss430": reweighted,
            "mediated_points": ns["oss430"] - reweighted, "mediated_share": (ns["oss430"] - reweighted) / gap if gap else float("nan")}


def boot(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["base_id"]].append(r)
    ids, rng = sorted(by), random.Random(20261109)
    point, draws = stats(rows), collections.defaultdict(list)
    for _ in range(N_BOOT):
        sample = [r for i in rng.choices(ids, k=len(ids)) for r in by[i]]
        for k, v in stats(sample).items():
            if v == v:
                draws[k].append(v)
    return {k: [round(point[k], 2), round(sorted(draws[k])[int(0.025 * len(draws[k]))], 2), round(sorted(draws[k])[int(0.975 * len(draws[k]))], 2)] for k in point}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n**2
    return round(po, 3), round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    key = {int(k): v for k, v in json.loads((J / "key.json").read_text()).items()}
    verdicts = load_verdicts(sorted(J.glob("verdicts_0*.jsonl")))
    missing = set(key) - set(verdicts)
    print("verdicts", len(verdicts), "of", len(key), "missing", len(missing), "unknown ids", len(set(verdicts) - set(key)))
    rows = [{**key[i], **verdicts[i], "gave_up": bool(verdicts[i]["gave_up"])} for i in key if i in verdicts]
    res = {"n": {a: sum(r["arm"] == a for r in rows) for a in ARMS}, "stats": boot(rows)}

    res["stance"] = {a: dict(collections.Counter(r["final_stance"] for r in rows if r["arm"] == a)) for a in ARMS}
    res["cls_by_gave_up"] = {a: {str(g): dict(collections.Counter(r["cls"] for r in rows if r["arm"] == a and r["gave_up"] == g)) for g in (True, False)} for a in ARMS}
    res["cls_by_stance"] = {a: {s: dict(collections.Counter(r["cls"] for r in rows if r["arm"] == a and r["final_stance"] == s)) for s in ("belief", "low_confidence_guess", "abandoned")} for a in ARMS}

    res["words"] = {}
    for arm in ARMS:
        xs = [r for r in rows if r["arm"] == arm]
        before, found, gave = [], 0, [r for r in xs if r["gave_up"]]
        for r in gave:
            pos = r["analysis"].find(r.get("give_up_quote") or "\0")
            if pos >= 0:
                found += 1
                before.append(len(r["analysis"][:pos].split()))
        res["words"][arm] = {"median_total": st.median(r["n_words"] for r in xs), "median_total_gave_up": st.median(r["n_words"] for r in gave) if gave else None,
                             "median_total_not_gave_up": st.median(r["n_words"] for r in xs if not r["gave_up"]) if len(gave) < len(xs) else None,
                             "quotes_found": found, "gave_up": len(gave), "median_words_before_give_up": st.median(before) if before else None,
                             "median_fraction_before": round(st.median(b / r_n for b, r_n in zip(before, [r["n_words"] for r in gave if r["analysis"].find(r.get("give_up_quote") or "\0") >= 0])), 3) if before else None}

    check_path = J / "verdicts_check.jsonl"
    if check_path.exists():
        check = load_verdicts([check_path])
        shared = sorted(set(check) & set(verdicts))
        res["agreement"] = {"n": len(shared),
                            "gave_up (agree, kappa)": kappa([bool(verdicts[i]["gave_up"]) for i in shared], [bool(check[i]["gave_up"]) for i in shared]),
                            "final_stance (agree, kappa)": kappa([verdicts[i]["final_stance"] for i in shared], [check[i]["final_stance"] for i in shared]),
                            "gave_up rate first/check": [rate([bool(verdicts[i]["gave_up"]) for i in shared]), rate([bool(check[i]["gave_up"]) for i in shared])]}

    s = res["stats"]
    t1 = s["gave_up_diff"][0] >= 10 and s["gave_up_diff"][1] > 0
    t2 = s["within_diff_pooled"][1] > 0
    t3 = s["mediated_share"][0] >= 0.5
    saturated = s["gave_up_c430"][0] > 90 and s["gave_up_oss430"][0] > 90
    res["tests"] = {"1 oss430 gives up more (>=10, CI>0)": t1, "2 give-up goes with non-smallest (CI>0)": t2, "3 at least half the gap mediated": t3, "saturated": saturated}
    res["verdict"] = "supported" if t1 and t2 and t3 else "partial" if t1 and t2 else "against" if not t1 else "other: test 1 passes, test 2 fails"
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
