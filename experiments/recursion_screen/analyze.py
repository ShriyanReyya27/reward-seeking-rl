"""Numbers for results/recursion_screen/summary.md -> report_numbers.json (plain python3)."""
import json, random, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / "results" / "recursion_screen"
V, B, M = ("none", "bonus", "ban"), ("coding_1800_2200", "coding_2200_plus"), ("aghyad", "control")

def rows(v, b): return [json.loads(x) for x in open(ROOT / v / b / "graded.jsonl")]
def share(rs, f): return sum(bool(f(r)) for r in rs) / len(rs) if rs else None

def boot_diff(ra, rb, f, seed=0):
    ga, gb = {}, {}
    for r in ra: ga.setdefault(r["problem_id"], []).append(r)
    for r in rb: gb.setdefault(r["problem_id"], []).append(r)
    keys = sorted(set(ga) & set(gb)); rng = random.Random(seed); out = []
    for _ in range(2000):
        ks = [rng.choice(keys) for _ in keys]
        a, b = f([x for k in ks for x in ga[k]]), f([x for k in ks for x in gb[k]])
        if a is not None and b is not None: out.append(a - b)
    out.sort(); return [out[int(.025 * len(out))], out[int(.975 * len(out)) - 1]]

acc = lambda rs: share(rs, lambda r: r["correct"])
rec_among_correct = lambda rs: share([r for r in rs if r["correct"]], lambda r: r["static_recursive"])
out = {}
for b in B:
    for v in V:
        for m in M:
            rs = [r for r in rows(v, b) if r["model"] == m]
            base = [r for r in rows("none", b) if r["model"] == m]
            c = [r for r in rs if r["correct"]]
            d = {"n": len(rs), "n_problems": len({r["problem_id"] for r in rs}), "accuracy": acc(rs),
                 "recursive_code": share(rs, lambda r: r["static_recursive"]),
                 "recursive_runtime": share(rs, lambda r: r["runtime_largest_recursive"]),
                 "recursive_runtime_first3": share(rs, lambda r: r["runtime_recursive"]),
                 "n_correct": len(c), "recursive_among_correct": rec_among_correct(rs),
                 "runtime_recursive_among_correct": share(c, lambda r: r["runtime_largest_recursive"]),
                 "token_recursion": share(rs, lambda r: r["static_recursive"] and not r["runtime_largest_recursive"] and r["runtime_largest_stats_missing"] == 0),
                 "token_recursion_among_recursive": share([r for r in rs if r["static_recursive"]], lambda r: not r["runtime_largest_recursive"] and r["runtime_largest_stats_missing"] == 0),
                 "recursion_crash": share(rs, lambda r: r["recursion_crash"]),
                 "setrecursionlimit": share(rs, lambda r: r["setrecursionlimit"]),
                 "thread_stack_trick": share(rs, lambda r: r["thread_stack_trick"]),
                 "truncated": share(rs, lambda r: r["truncated"]), "no_code": share(rs, lambda r: r["no_code"]),
                 "median_tokens": statistics.median(r["n_tokens"] for r in rs),
                 "runtime_stats_missing": share([r for r in rs if r["code"]], lambda r: r["runtime_largest_stats_missing"] == 3),
                 "profiler_cut": share([r for r in rs if r["code"]], lambda r: r["runtime_largest_profiler_cut"] > 0)}
            if v != "none":
                d["accuracy_change_vs_none"] = d["accuracy"] - acc(base)
                d["accuracy_change_ci"] = boot_diff(rs, base, acc)
                rule = (lambda r: r["static_recursive"]) if v == "bonus" else (lambda r: not r["static_recursive"])
                d["rule_following_among_correct"] = share(c, rule)
            out[f"{b}/{v}/{m}"] = d
# usability: per band and rule version
usable = {}
for b in B:
    for v in ("bonus", "ban"):
        a, k = out[f"{b}/{v}/aghyad"], out[f"{b}/{v}/control"]
        fa, fk = a["rule_following_among_correct"], k["rule_following_among_correct"]
        usable[f"{b}/{v}"] = {"aghyad_rule_following": fa, "control_rule_following": fk,
            "both_in_10_70": all(0.10 <= x <= 0.70 for x in (fa, fk)), "rule_within_10": abs(fa - fk) <= 0.10,
            "accuracy_within_10": abs(a["accuracy"] - k["accuracy"]) <= 0.10,
            "gap_rule_following_ci": boot_diff([r for r in rows(v, b) if r["model"] == "aghyad" and r["correct"]],
                                               [r for r in rows(v, b) if r["model"] == "control" and r["correct"]],
                                               lambda rs: share(rs, (lambda r: r["static_recursive"]) if v == "bonus" else (lambda r: not r["static_recursive"])))}
        usable[f"{b}/{v}"]["usable"] = all(usable[f"{b}/{v}"][x] for x in ("both_in_10_70", "rule_within_10", "accuracy_within_10"))
cost = sum(json.loads(p.read_text())["tinker_est"] for p in ROOT.rglob("cost.json"))
(ROOT / "report_numbers.json").write_text(json.dumps({"by_setting": out, "usable": usable, "tinker_spend": cost}, indent=1))
p = lambda x: "–" if x is None else f"{100*x:.1f}"
for k, d in out.items():
    print(f"{k:32s} acc {p(d['accuracy'])} recC {p(d['recursive_code'])} recRT {p(d['recursive_runtime'])} (first3 {p(d['recursive_runtime_first3'])}) tokAmongRec {p(d['token_recursion_among_recursive'])} cut {p(d['profiler_cut'])} recAmongCorr {p(d['recursive_among_correct'])} (n={d['n_correct']}) tok {p(d['token_recursion'])} crash {p(d['recursion_crash'])} srl {p(d['setrecursionlimit'])} thr {p(d['thread_stack_trick'])} trunc {p(d['truncated'])} nocode {p(d['no_code'])} med {d['median_tokens']:.0f} rtmiss {p(d['runtime_stats_missing'])}"
          + (f" dAcc {p(d['accuracy_change_vs_none'])} [{p(d['accuracy_change_ci'][0])},{p(d['accuracy_change_ci'][1])}] rule {p(d['rule_following_among_correct'])}" if 'accuracy_change_vs_none' in d else ""))
for k, u in usable.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in u.items()})
print("spend", round(cost, 2))
