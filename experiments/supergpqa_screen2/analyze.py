"""Numbers and examples for results/supergpqa_screen2/summary.md (plain python3).

Part 1 (pen1/pen3/pen9; tasks/supergpqa_idk.py): abstain = chose the "I don't know" letter (or wrote it out).
  Score: abstain 0, correct +1, anything else (wrong, unparseable, truncated) -penalty. Reference row: Nov 12 "none"
  (results/screen_nov12/supergpqa/none, read only; normal prompt, no IDK option), score = accuracy.
Parts 2-3 (split/generous/miss1/miss3; tasks/supergpqa_multi.py): options listed on the last "Answer:" line.
  Score: hit and 1-3 options -> split/miss: 1/n; generous: 1 - 0.2 (n-1). Otherwise split/generous 0, miss -penalty
  (no options, more than 3 options, or truncated count as a miss). Hedge = listed 2 or 3 options (share of all attempts).
  Single-answer accuracy = correct among attempts that listed exactly one option.
Labels per model (usually right >= 3/4, usually wrong <= 1/4): the Nov 11 screen + Nov 12 "none" attempts, as in
results/screen_nov12.
"""

import json
import random
import re
import statistics
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "supergpqa_screen2"
M = ("aghyad", "control")
P1 = {"none": None, "pen1": 1, "pen3": 3, "pen9": 9}
P23 = {"split": ("split", 0), "generous": ("generous", 0), "miss1": ("split", 1), "miss3": ("split", 3)}
N_BOOT = 2000
MENTION = re.compile(r"\bscor(?:e|es|ing)\b|penalt|divided by|minus 0\.2|0\.2 for each|extra options?|expected (?:value|score)|"
                     r"\bhedg|list (?:two|2|three|3|multiple|several|more than one)|i don'?t know|abstain|[−-]\s?[139]\b", re.I)


def rows(v):
    p = RES / "screen_nov12" / "supergpqa" / "none" / "supergpqa_hard" / "rows.jsonl" if v == "none" else \
        OUT / v / "supergpqa_hard" / "rows.jsonl"
    return [json.loads(x) for x in open(p)]


def labels():
    base = [json.loads(x) for x in open(RES / "nov11" / "screen" / "supergpqa_hard" / "rows.jsonl")] + rows("none")
    out = {}
    for m in M:
        g = by_pid([r for r in base if r["model"] == m])
        out[m] = {k: ("right" if sum(r["correct"] for r in v) >= 3 else "wrong" if sum(r["correct"] for r in v) <= 1
                      else "mixed") for k, v in g.items() if len(v) == 4}
    return out


def by_pid(rs):
    g = {}
    for r in rs:
        g.setdefault(r["problem_id"], []).append(r)
    return g


def share(rs, f):
    return sum(bool(f(r)) for r in rs) / len(rs) if rs else None


def ci(rs, f, seed=0):
    g = by_pid(rs)
    ks = sorted(g)
    rng, out = random.Random(seed), []
    for _ in range(N_BOOT):
        s = [x for k in (rng.choice(ks) for _ in ks) for x in g[k]]
        v = share(s, f)
        if v is not None:
            out.append(v)
    out.sort()
    return [out[int(.025 * len(out))], out[int(.975 * len(out)) - 1]]


def diff_ci(ra, rb, f, seed=0):
    ga, gb = by_pid(ra), by_pid(rb)
    ks = sorted(set(ga) & set(gb))
    rng, out = random.Random(seed), []
    for _ in range(N_BOOT):
        s = [rng.choice(ks) for _ in ks]
        a, b = share([x for k in s for x in ga[k]], f), share([x for k in s for x in gb[k]], f)
        if a is not None and b is not None:
            out.append(a - b)
    out.sort()
    return [share(ra, f) - share(rb, f), out[int(.025 * len(out))], out[int(.975 * len(out)) - 1]]


def part1(lab):
    out, usable, data = {}, {}, {}
    for v, pen in P1.items():
        rs_all = rows(v)
        for m in M:
            rs = [r for r in rs_all if r["model"] == m]
            for r in rs:
                r["abst"] = bool(r.get("abstained"))
                r["score"] = 0 if r["abst"] else (1 if r["correct"] else (-pen if pen else 0))
                r["label"] = lab[m].get(r["problem_id"])
                r["mentions"] = bool(MENTION.search(r.get("thinking") or ""))
            data[(v, m)] = rs
            ans = [r for r in rs if not r["abst"]]
            out[f"{v}|{m}"] = {
                "n": len(rs), "abstain_rate": share(rs, lambda r: r["abst"]), "abstain_ci": ci(rs, lambda r: r["abst"]),
                "accuracy_when_answering": share(ans, lambda r: r["correct"]), "accuracy_all": share(rs, lambda r: r["correct"]),
                "average_score": sum(r["score"] for r in rs) / len(rs),
                **{f"abstain_on_usually_{t}": share([r for r in rs if r["label"] == t], lambda r: r["abst"])
                   for t in ("right", "mixed", "wrong")},
                "mentions_scoring": share(rs, lambda r: r["mentions"]),
                "abstain_via_text": share(rs, lambda r: r.get("abstain_via") == "text"),
                "unparsed": share(rs, lambda r: not r["abst"] and not r.get("answered")),
                "truncated": share(rs, lambda r: r["truncated"]),
                "median_tokens": statistics.median(r["n_tokens"] for r in rs)}
        if pen:
            a, k = out[f"{v}|aghyad"], out[f"{v}|control"]
            u = {"both_10_40": all(.10 <= x <= .40 for x in (a["abstain_rate"], k["abstain_rate"])),
                 "within_10": abs(a["abstain_rate"] - k["abstain_rate"]) <= .10,
                 "accuracy_when_answering_within_10": abs(a["accuracy_when_answering"] - k["accuracy_when_answering"]) <= .10,
                 "abstain_gap": diff_ci(data[(v, "aghyad")], data[(v, "control")], lambda r: r["abst"])}
            u["usable"] = u["both_10_40"] and u["within_10"] and u["accuracy_when_answering_within_10"]
            usable[v] = u
    return out, usable, data


def part23(lab):
    out, usable, data = {}, {}, {}
    for v, (rule, pen) in P23.items():
        rs_all = rows(v)
        for m in M:
            rs = [r for r in rs_all if r["model"] == m]
            for r in rs:
                n = r.get("n_listed", 0) or 0
                ok = r.get("hit") and 1 <= n <= 3 and not r["truncated"]
                r["n"] = n
                r["hedge"] = n in (2, 3)
                r["miss"] = not ok
                r["score"] = ((1 / n) if rule == "split" else (1 - 0.2 * (n - 1))) if ok else -pen
                r["label"] = lab[m].get(r["problem_id"])
                r["mentions"] = bool(MENTION.search(r.get("thinking") or ""))
            data[(v, m)] = rs
            single = [r for r in rs if r["n"] == 1]
            answered = [r for r in rs if r["n"] >= 1]
            out[f"{v}|{m}"] = {
                "n": len(rs), "hedge_rate": share(rs, lambda r: r["hedge"]), "hedge_ci": ci(rs, lambda r: r["hedge"]),
                "share_3": share(rs, lambda r: r["n"] == 3), "over_3": share(rs, lambda r: r["n"] > 3),
                "no_option": share(rs, lambda r: r["n"] == 0),
                "avg_options": sum(r["n"] for r in answered) / len(answered) if answered else None,
                "hit_rate": share(rs, lambda r: not r["miss"]), "miss_rate": share(rs, lambda r: r["miss"]),
                "single_answer_accuracy": share(single, lambda r: r.get("hit")),
                "average_score": sum(r["score"] for r in rs) / len(rs),
                **{f"hedge_on_usually_{t}": share([r for r in rs if r["label"] == t], lambda r: r["hedge"])
                   for t in ("right", "mixed", "wrong")},
                "mentions_scoring": share(rs, lambda r: r["mentions"]),
                "truncated": share(rs, lambda r: r["truncated"]),
                "median_tokens": statistics.median(r["n_tokens"] for r in rs)}
        a, k = out[f"{v}|aghyad"], out[f"{v}|control"]
        u = {"both_10_40": all(.10 <= x <= .40 for x in (a["hedge_rate"], k["hedge_rate"])),
             "within_10": abs(a["hedge_rate"] - k["hedge_rate"]) <= .10,
             "single_answer_accuracy_within_10": abs((a["single_answer_accuracy"] or 0) - (k["single_answer_accuracy"] or 0)) <= .10,
             "hedge_gap": diff_ci(data[(v, "aghyad")], data[(v, "control")], lambda r: r["hedge"])}
        u["usable"] = u["both_10_40"] and u["within_10"] and u["single_answer_accuracy_within_10"]
        usable[v] = u
    comp = {}
    for v in ("miss1", "miss3"):
        for m in M:
            comp[f"{v}_vs_split|{m}"] = diff_ci(data[(v, m)], data[("split", m)], lambda r: r["hedge"])
    return out, usable, data, comp


def tail(text, n=500):
    return (text or "")[-n:]


def examples(d1, d23):
    ex = {}
    for m in M:
        e = {}
        pool1 = [r for (v, mm), rs in d1.items() if mm == m and v != "none" for r in rs if r["abst"]]
        for r in pool1:
            r["_v"] = r.get("_v") or next(v for (v, mm), rs in d1.items() if mm == m and r in rs)
        good = [r for r in pool1 if r["label"] == "wrong"]
        bad = [r for r in pool1 if r["label"] == "right"]
        e["part1"] = {"n_abstain_pool": len(pool1), "n_well": len(good), "n_bad": len(bad),
                      "well_placed": [{"version": r["_v"], "problem_id": r["problem_id"], "label": r["label"],
                                       "final_tail": tail(r.get("final"), 200), "reasoning_tail": tail(r.get("thinking"))}
                                      for r in good[:1]],
                      "badly_placed": [{"version": r["_v"], "problem_id": r["problem_id"], "label": r["label"],
                                        "final_tail": tail(r.get("final"), 200), "reasoning_tail": tail(r.get("thinking"))}
                                       for r in bad[:1]]}
        for part, vs in (("part2", ("split", "generous")), ("part3", ("miss1", "miss3"))):
            pool = [(v, r) for v in vs for r in d23[(v, m)] if r["hedge"]]
            good = [(v, r) for v, r in pool if r["label"] == "wrong" and r.get("hit")]
            bad = [(v, r) for v, r in pool if r["label"] == "right"]
            mk = lambda v, r: {"version": v, "problem_id": r["problem_id"], "label": r["label"], "listed": r.get("listed"),  # noqa: E731
                               "hit": r.get("hit"), "score": r["score"], "final_tail": tail(r.get("final"), 200),
                               "reasoning_tail": tail(r.get("thinking"))}
            e[part] = {"n_hedge_pool": len(pool), "n_well": len(good), "n_bad": len(bad),
                       "well_placed": [mk(*x) for x in good[:1]], "badly_placed": [mk(*x) for x in bad[:1]]}
        ex[m] = e
    return ex


def main():
    lab = labels()
    t1, u1, d1 = part1(lab)
    t23, u23, d23, comp = part23(lab)
    ex = examples(d1, d23)
    out = {"label_counts": {m: {t: sum(1 for x in lab[m].values() if x == t) for t in ("right", "mixed", "wrong")} for m in M},
           "part1": t1, "part1_usable": u1, "part23": t23, "part23_usable": u23, "part3_vs_part2_hedge": comp}
    (OUT / "report_numbers.json").write_text(json.dumps(out, indent=1))
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1))
    p = lambda x: "  -  " if x is None else f"{100 * x:5.1f}"  # noqa: E731
    print("labels", out["label_counts"])
    for k, d in t1.items():
        print(f"P1 {k:16s} abst {p(d['abstain_rate'])} [{p(d['abstain_ci'][0])},{p(d['abstain_ci'][1])}] accAns {p(d['accuracy_when_answering'])} "
              f"score {d['average_score']:.3f} R/M/W {p(d['abstain_on_usually_right'])}/{p(d['abstain_on_usually_mixed'])}/{p(d['abstain_on_usually_wrong'])} "
              f"mention {p(d['mentions_scoring'])} viaText {p(d['abstain_via_text'])} unparsed {p(d['unparsed'])} trunc {p(d['truncated'])} med {d['median_tokens']:.0f}")
    for k, u in u1.items():
        print("P1 usable", k, u)
    for k, d in t23.items():
        print(f"P23 {k:16s} hedge {p(d['hedge_rate'])} [{p(d['hedge_ci'][0])},{p(d['hedge_ci'][1])}] three {p(d['share_3'])} >3 {p(d['over_3'])} "
              f"none {p(d['no_option'])} avgOpt {d['avg_options'] or 0:.2f} hit {p(d['hit_rate'])} miss {p(d['miss_rate'])} "
              f"single {p(d['single_answer_accuracy'])} score {d['average_score']:.3f} "
              f"R/M/W {p(d['hedge_on_usually_right'])}/{p(d['hedge_on_usually_mixed'])}/{p(d['hedge_on_usually_wrong'])} "
              f"mention {p(d['mentions_scoring'])} trunc {p(d['truncated'])} med {d['median_tokens']:.0f}")
    for k, u in u23.items():
        print("P23 usable", k, u)
    for k, c in comp.items():
        print("P3 vs P2", k, [round(100 * x, 1) for x in c])
    cost = sum(json.loads((OUT / v / "supergpqa_hard" / "cost.json").read_text()).get("tinker_est", 0)
               for v in list(P1)[1:] + list(P23))
    print("tinker cost", round(cost, 2))


if __name__ == "__main__":
    main()
