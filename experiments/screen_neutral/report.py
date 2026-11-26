"""Numbers, tables and example picks for results/screen_neutral/summary.md.

    python3 report.py   -> results/screen_neutral/{report_numbers.json, tables.md, examples.json}

95% intervals: bootstrap over problems (each problem's mean over its 2 attempts). Gap = Aghyad's model minus control,
paired bootstrap over problems. "Usable": both models' main measure in 20-70%, within 10 points, truncation < 5% each.
If a version was rerun at 32k tokens, the 32k run is the one reported.
"""
import json
import random
import statistics
from collections import defaultdict

import cn

M = ("aghyad", "control")
LAB = {"aghyad": "Aghyad's model", "control": "Control"}


def load(d):
    return [json.loads(x) for x in open(d / "rows.jsonl")]


def pick_dir(screen, name):
    d32 = cn.OUT / screen / f"{name}_32k"
    return d32 if (d32 / "DONE").exists() else cn.OUT / screen / name


def per_problem(rows, m, f):
    g = defaultdict(list)
    for r in rows:
        if r["model"] == m:
            v = f(r)
            if v is not None:
                g[r["problem_id"]].append(float(v))
    return {k: sum(v) / len(v) for k, v in g.items()}


def rate(rows, m, f):
    return cn.boot_ci(list(per_problem(rows, m, f).values()))


def gap(rows, f, n_boot=2000):
    a, c = per_problem(rows, "aghyad", f), per_problem(rows, "control", f)
    ks = sorted(set(a) & set(c))
    rng = random.Random(1)
    g = lambda kk: statistics.mean(a[k] - c[k] for k in kk)  # noqa: E731
    bs = sorted(g([rng.choice(ks) for _ in ks]) for _ in range(n_boot))
    return (g(ks), bs[int(.025 * n_boot)], bs[int(.975 * n_boot) - 1])


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def usable(main, trunc):
    a, c = main["aghyad"][0], main["control"][0]
    ok = all(.2 <= x <= .7 for x in (a, c)) and abs(a - c) <= .10 and all(t < .05 for t in trunc.values())
    return ok


def pct(t):
    m, lo, hi = t
    return "-" if m is None else f"{100 * m:.0f}% ({100 * lo:.0f}-{100 * hi:.0f})"


def screen_if(out, md):
    d = pick_dir("if", "main")
    rows = load(d)
    q = defaultdict(dict)
    for x in map(json.loads, open(d / "quality.jsonl")):
        q[x["judge"]][(x["problem_id"], x["model"], x["attempt"])] = x["score"]
    mini = q["gpt-4.1-mini"]
    for r in rows:
        r["quality"] = mini.get((r["problem_id"], r["model"], r["attempt"]))
        r["combined"] = bool(r["constraint_pass"] and not r["truncated"] and (r["quality"] or 0) >= 3)
        r["junk"] = bool(r["constraint_pass"] and not r["truncated"] and r["quality"] is not None and r["quality"] < 3)
    res = {"dir": str(d), "models": {}}
    main, trunc = {}, {}
    for m in M:
        rm = [r for r in rows if r["model"] == m]
        s = {"constraint_pass": rate(rows, m, lambda r: r["constraint_pass"]),
             "avg_quality": statistics.mean(r["quality"] for r in rm if r["quality"] is not None),
             "combined": rate(rows, m, lambda r: r["combined"]),
             "junk": rate(rows, m, lambda r: r["junk"]),
             "truncated": sum(r["truncated"] for r in rm) / len(rm),
             "median_words": med([len((r["final"] or "").split()) for r in rm if not r["truncated"]]),
             "median_tokens": med([r["n_tokens"] for r in rm])}
        res["models"][m] = s
        main[m], trunc[m] = s["combined"], s["truncated"]
    res["gap_combined"] = gap(rows, lambda r: r["combined"])
    res["usable"] = usable(main, trunc)
    # trusted check
    big = q["gpt-4.1"]
    pairs = [(mini[k], v) for k, v in big.items() if k in mini and v is not None and mini[k] is not None]
    agree = {"n": len(pairs), "exact": sum(a == b for a, b in pairs) / len(pairs),
             "within_1": sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs),
             "same_side_of_3": sum((a >= 3) == (b >= 3) for a, b in pairs) / len(pairs),
             "mean_mini": statistics.mean(a for a, _ in pairs), "mean_41": statistics.mean(b for _, b in pairs)}
    for m in M:
        pm = [(mini[k], v) for k, v in big.items() if k[1] == m and k in mini and v is not None and mini[k] is not None]
        agree[f"same_side_{m}"] = sum((a >= 3) == (b >= 3) for a, b in pm) / len(pm)
        agree[f"mean_mini_{m}"], agree[f"mean_41_{m}"] = statistics.mean(a for a, _ in pm), statistics.mean(b for _, b in pm)
    res["agreement"] = agree
    # per constraint type (main + extra), instruction-level pass
    allrows = list(rows)
    de = pick_dir("if", "extra")
    if (de / "DONE").exists():
        allrows += load(de)
    by = defaultdict(lambda: {m: [] for m in M})
    for r in allrows:
        fl = r.get("follow_list") or [False] * len(r.get("instruction_id_list", []))
        for iid, ok in zip(r.get("instruction_id_list", []), fl):
            by[iid][r["model"]].append(bool(ok) and not r["truncated"])
    types = {t: {m: (sum(v[m]) / len(v[m]) if v[m] else None, len(v[m])) for m in M} for t, v in sorted(by.items())}
    res["types"] = types
    res["types_both_20_70"] = [t for t, v in types.items()
                               if all(v[m][0] is not None and .2 <= v[m][0] <= .7 for m in M)]
    out["if"] = res
    md.append("## Screen 1: IFTrain constraints with a quality check (main 100 problems)\n")
    md.append("| Model | Constraint pass | Avg quality (1-5, mini) | **Combined (main)** | Junk that passes | Median length (words) | Truncated | Usable |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m in M:
        s = res["models"][m]
        md.append(f"| {LAB[m]} | {pct(s['constraint_pass'])} | {s['avg_quality']:.2f} | **{pct(s['combined'])}** | "
                  f"{pct(s['junk'])} | {s['median_words']:.0f} | {100 * s['truncated']:.1f}% | {'yes' if res['usable'] else 'no'} |")
    g = res["gap_combined"]
    md.append(f"\nGap in combined (Aghyad minus control): {100 * g[0]:+.0f} points ({100 * g[1]:+.0f} to {100 * g[2]:+.0f}).")
    md.append(f"\nTrusted check: gpt-4.1 vs gpt-4.1-mini on {agree['n']} answers: same side of 3 {100 * agree['same_side_of_3']:.0f}%, "
              f"exact {100 * agree['exact']:.0f}%, within 1 point {100 * agree['within_1']:.0f}%; mean score mini {agree['mean_mini']:.2f}, "
              f"gpt-4.1 {agree['mean_41']:.2f}.\n")
    md.append("Per constraint type (main + extra 200 problems; share of that constraint passed):\n")
    md.append("| Constraint type | Aghyad's model | Control | n each | Both 20-70%? |")
    md.append("|---|---|---|---|---|")
    for t, v in types.items():
        a, c = v["aghyad"][0], v["control"][0]
        md.append(f"| {t} | {100 * a:.0f}% | {100 * c:.0f}% | {v['aghyad'][1]} | {'yes' if t in res['types_both_20_70'] else ''} |")
    md.append("")
    return rows


def screen_simple(out, md, screen, specs, cols):
    md.append(f"## {'Screen 2: word sorting' if screen == 'sort' else 'Screen 3: made-up calendar'}\n")
    hdr = ["Version", "Model", "Correct (main)"] + [c[0] for c in cols] + ["Truncated", "Median length (tokens)", "Usable"]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    res = {}
    allrows = {}
    for name, label in specs:
        d = pick_dir(screen, name)
        if not (d / "DONE").exists():
            continue
        rows = load(d)
        allrows[name] = rows
        r = {"dir": str(d), "models": {}}
        main, trunc = {}, {}
        for m in M:
            rm = [x for x in rows if x["model"] == m]
            s = {"correct": rate(rows, m, lambda x: x["correct"]),
                 **{c[1]: rate(rows, m, (lambda key: lambda x: x.get(key))(c[1])) for c in cols},
                 "truncated": sum(x["truncated"] for x in rm) / len(rm),
                 "median_tokens": med([x["n_tokens"] for x in rm])}
            r["models"][m] = s
            main[m], trunc[m] = s["correct"], s["truncated"]
        r["gap"] = gap(rows, lambda x: x["correct"])
        r["usable"] = usable(main, trunc)
        res[name] = r
        for m in M:
            s = r["models"][m]
            md.append(f"| {label} | {LAB[m]} | **{pct(s['correct'])}** | " + " | ".join(pct(s[c[1]]) for c in cols) +
                      f" | {100 * s['truncated']:.1f}% | {s['median_tokens']:.0f} | {'yes' if r['usable'] else 'no'} |")
    md.append("")
    for name, r in res.items():
        g = r["gap"]
        md.append(f"- {name}: gap in correct (Aghyad minus control) {100 * g[0]:+.0f} points ({100 * g[1]:+.0f} to {100 * g[2]:+.0f}).")
    md.append("")
    out[screen] = res
    return allrows


def examples(if_rows, sort_rows, cal_rows):
    ex = {}
    rng = random.Random("ex")

    def first(rows, f):
        c = [r for r in rows if f(r)]
        return rng.choice(c) if c else None

    def slim(r, extra=()):
        if r is None:
            return None
        return {k: r.get(k) for k in ("problem_id", "model", "attempt", "correct", *extra, "final")}

    for m in M:
        rm = [r for r in if_rows if r["model"] == m]
        ex[f"if|{m}|good"] = slim(first(rm, lambda r: r["combined"] and r["quality"] >= 4), ("quality", "instruction_id_list"))
        ex[f"if|{m}|junk"] = slim(first(rm, lambda r: r["junk"]), ("quality", "instruction_id_list"))
        for v, rows in sort_rows.items():
            rr = [r for r in rows if r["model"] == m]
            ex[f"sort_{v}|{m}|good"] = slim(first(rr, lambda r: r["correct"]), ("extracted",))
            ex[f"sort_{v}|{m}|habit"] = slim(first(rr, lambda r: r.get("normal_order")), ("extracted",))
        for v, rows in cal_rows.items():
            rr = [r for r in rows if r["model"] == m]
            ex[f"cal_{v}|{m}|good"] = slim(first(rr, lambda r: r["correct"]), ("extracted",))
            ex[f"cal_{v}|{m}|habit"] = slim(first(rr, lambda r: r.get("gregorian")), ("extracted",))
    return ex


def main():
    out, md = {}, []
    if_rows = screen_if(out, md) if (pick_dir("if", "main") / "quality.jsonl").exists() else []
    sort_rows = screen_simple(out, md, "sort", [("a", "(a) shuffled alphabet, stated"), ("b", "(b) vowels first, stated"),
                                                ("c", "(c) vowels first, NOT stated"),
                                                ("a_medium", "(a) shuffled alphabet, stated, medium set (25-35 words)"),
                                                ("b_medium", "(b) vowels first, stated, medium set (25-35 words)"),
                                                ("a_hard", "(a) shuffled alphabet, stated, hard set (40-60 words)"),
                                                ("b_hard", "(b) vowels first, stated, hard set (40-60 words)")],
                              [("Normal alphabetical (habit)", "normal_order"), ("Right words, wrong order", "same_set_wrong_order")])
    cal_specs = [(f"{v}_{s}", f"({v}) {'stated' if v == 'a' else 'NOT stated'}, {s} set") for s in ("base", "hard", "easy")
                 for v in ("a", "b")] + [("a_multi", "(a) stated, multi-step set (days, then 10-day weeks, then back)")]
    cal_rows = screen_simple(out, md, "cal", cal_specs,
                             [("Real calendar date (habit)", "gregorian"), ("Date right (ignoring day name)", "date_correct")])
    t, o = cn.spend()
    out["spend"] = {"tinker": t, "openai": o, "total": t + o, "cap": cn.CAP}
    md.append(f"Spend: Tinker ${t:.2f} + OpenAI ${o:.2f} = ${t + o:.2f} (cap ${cn.CAP:.0f}).")
    (cn.OUT / "report_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    (cn.OUT / "tables.md").write_text("\n".join(md) + "\n")
    (cn.OUT / "examples.json").write_text(json.dumps(examples(if_rows, sort_rows, cal_rows) if if_rows else
                                                     examples([], sort_rows, cal_rows), indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
