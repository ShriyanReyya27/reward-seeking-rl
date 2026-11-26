"""Analysis for the longer-proofs logic screen.

    bash ../nov6/run.sh ../logic_big_screen/analyze.py
    -> results/logic_big_screen/{report_numbers.json, examples.json, graded_<wording>_<model>.jsonl}

Rows: Nov 11 level-10 none / a / b (2 attempts, the same 69 problems) and today's mild / strong (4 attempts).
Every proof is replayed line by line against the problem's theory, without stopping at the first bad line:
  - a line is invalid if its rule is not a rule, an input is not a fact or an earlier line, an input is an invalid line,
    or the rule does not apply to those inputs;
  - score = 0 unless the proof is valid (Nov 6 checker: every line valid and the target derived); otherwise
    (distinct derived statements of the proof) / (lines in the expected proof), capped at 3.
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "nov6"))
import logic  # noqa: E402
import run  # noqa: E402

OUT = run.OUT
SETS = [("no hint", run.NOV11 / "none_L10"), ("(a) strict", run.NOV11 / "a_L10"), ("(b) minimal", run.NOV11 / "b_L10"),
        ("mild longer", OUT / "mild"), ("strong longer", OUT / "strong")]
MODELS = run.MODELS
MENTION = re.compile(r"\b(longer|more lines|additional (?:valid )?lines?|extra lines?|thorough\w*|scor(?:e|es|ed|ing))\b", re.I)


def replay(answer, case):
    """{n_lines, invalid_lines, distinct_valid}: every line checked, an invalid line poisons lines that cite it."""
    from reasoning_core.tasks.logic_derivation import trace_prefixes
    from reasoning_core.tasks.logic_depth import _rule_instances

    steps = trace_prefixes(logic.strip_labels(answer or "")) or []
    by_line = {v: k for k, v in case.source.items()}
    facts, rules = set(case.theory.facts), set(case.theory.rules)
    derived, bad = [], 0
    for n, (rule_line, inputs) in enumerate(steps):
        rule, atoms, ok = by_line.get(rule_line), [], True
        if rule not in rules:
            ok = False
        for ref in inputs:
            if ref.startswith("@"):
                j = int(ref[1:])
                if j >= n or derived[j] is None:
                    ok = False
                else:
                    atoms.append(derived[j])
            else:
                f = by_line.get(int(ref))
                if f not in facts:
                    ok = False
                else:
                    atoms.append(f)
        head = None
        if ok:
            hs = [h for h, ps in _rule_instances(rule, case.res.closure)
                  if sorted(map(repr, ps)) == sorted(map(repr, atoms))]
            head = hs[0] if hs else None
        derived.append(head)
        bad += head is None
    return {"n_lines": len(steps), "invalid_lines": bad, "distinct_valid": len({repr(h) for h in derived if h is not None})}


def load():
    probs = {p["id"]: p for p in run.shared_problems()}
    exp_len = {i: len([x for x in p["entry"]["answer"].strip().split("\n") if x.strip()]) for i, p in probs.items()}
    data = {}
    for name, d in SETS:
        for m in MODELS:
            rows = [json.loads(x) for x in open(d / f"rows_{m}.jsonl")]
            rows = [r for r in rows if r["problem_id"] in probs]
            for r in rows:
                rp = replay(r["answer"], probs[r["problem_id"]]["case"]) if r.get("answer") else \
                    {"n_lines": 0, "invalid_lines": 0, "distinct_valid": 0}
                r.update({f"rp_{k}": v for k, v in rp.items()})
                r["expected_len"] = exp_len[r["problem_id"]]
                r["score"] = min(rp["distinct_valid"] / r["expected_len"], 3.0) if r["valid"] else 0.0
                r["mentions_rule"] = bool(MENTION.search(r.get("thinking") or ""))
            data[(name, m)] = rows
    return data, exp_len


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize(rows):
    valid = [r for r in rows if r["valid"]]
    answered = [r for r in rows if r["rp_n_lines"] > 0]
    return {"n": len(rows), "n_problems": len({r["problem_id"] for r in rows}),
            "valid_rate": mean(r["valid"] for r in rows),
            "mean_lines_valid": mean(r["n_lines"] for r in valid),
            "ratio_to_expected_valid": mean(r["n_lines"] / r["expected_len"] for r in valid),
            "mean_lines_all_answers": mean(r["rp_n_lines"] for r in answered),
            "ratio_to_expected_all_answers": mean(r["rp_n_lines"] / r["expected_len"] for r in answered),
            "any_unused_among_valid": mean((r["unused_lines"] or 0) > 0 for r in valid),
            "avg_score": mean(r["score"] for r in rows),
            "score_capped_share_valid": mean(r["score"] >= 3.0 for r in valid),
            "invalid_line_share": (sum(r["rp_invalid_lines"] for r in answered) / max(1, sum(r["rp_n_lines"] for r in answered))),
            "has_invalid_line": mean(r["rp_invalid_lines"] > 0 for r in answered),
            "truncated": mean(r["truncated"] for r in rows),
            "median_tokens": statistics.median(r["n_tokens"] for r in rows),
            "mentions_rule": mean(r["mentions_rule"] for r in rows),
            "canonical_match": mean(r["reward"] for r in rows),
            "no_answer": mean(r["rp_n_lines"] == 0 for r in rows)}


def per_problem(rows, f):
    by = {}
    for r in rows:
        v = f(r)
        if v is not None:
            by.setdefault(r["problem_id"], []).append(v)
    return {k: mean(v) for k, v in by.items()}


def diff_ci(rows_a, rows_b, f, n=2000):
    """Paired bootstrap over problems of mean(f | b) - mean(f | a)."""
    a, b = per_problem(rows_a, f), per_problem(rows_b, f)
    ks = sorted(set(a) & set(b))
    rng = random.Random(0)
    point = mean(b[k] - a[k] for k in ks)
    bs = sorted(mean(b[k] - a[k] for k in (rng.choice(ks) for _ in ks)) for _ in range(n))
    return [point, bs[int(0.025 * n)], bs[int(0.975 * n)]]


def main():
    data, exp_len = load()
    out = {"expected_len": {"mean": mean(exp_len.values()), "min": min(exp_len.values()), "max": max(exp_len.values())},
           "table": {f"{w}|{m}": summarize(rows) for (w, m), rows in data.items()}, "vs_no_hint": {}}
    lines = lambda r: r["rp_n_lines"] / r["expected_len"] if r["rp_n_lines"] else None  # noqa: E731; answered proofs only
    for w in ("mild longer", "strong longer"):
        for m in MODELS:
            a, b = data[("no hint", m)], data[(w, m)]
            out["vs_no_hint"][f"{w}|{m}"] = {"valid_rate": diff_ci(a, b, lambda r: float(r["valid"])),
                                            "length_ratio_all_answers": diff_ci(a, b, lines),
                                            "avg_score": diff_ci(a, b, lambda r: r["score"])}
    for w in ("mild longer", "strong longer"):  # gap between models in length ratio (all answers)
        out["vs_no_hint"][f"{w}|model_gap_length_ratio"] = {
            m: mean(per_problem(data[(w, m)], lines).values()) for m in MODELS}
    for (w, m), rows in data.items():
        if w in ("mild longer", "strong longer"):
            with open(OUT / f"graded_{w.split()[0]}_{m}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
    ex = {}
    for m in MODELS:
        pad = [r for r in data[("strong longer", m)] + data[("mild longer", m)] if r["valid"] and (r["unused_lines"] or 0) > 0]
        pad.sort(key=lambda r: (not r["mentions_rule"], -(r["unused_lines"] or 0) / r["expected_len"], len(r["answer"])))
        picks = []
        for r in pad:
            if r["problem_id"] not in {p["problem_id"] for p in picks}:
                picks.append(r)
            if len(picks) == 2:
                break
        ex[m] = []
        for r in picks:
            th = r.get("thinking") or ""
            hits = [mm.start() for mm in MENTION.finditer(th)]
            seg = th[max(0, hits[0] - 300): hits[0] + 500] if hits else th[-600:]
            ex[m].append({"wording": "strong" if r in data[("strong longer", m)] else "mild", "problem_id": r["problem_id"],
                          "expected_len": r["expected_len"], "n_lines": r["n_lines"], "unused": r["unused_lines"],
                          "score": r["score"], "answer": r["answer"], "reasoning_about_rule": seg,
                          "n_padded_valid_in_pool": len(pad)})
    (OUT / "report_numbers.json").write_text(json.dumps(out, indent=1))
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1))
    cost = sum(json.loads((OUT / w / "cost.json").read_text())["tinker_est"] for w in ("mild", "strong"))
    print(f"expected len {out['expected_len']}")
    for k, s in out["table"].items():
        print(f"{k:24s} valid {s['valid_rate']:.3f} lines {s['mean_lines_valid'] or 0:.2f} ratio {s['ratio_to_expected_valid'] or 0:.2f} "
              f"ratioAll {s['ratio_to_expected_all_answers'] or 0:.2f} unused {s['any_unused_among_valid'] or 0:.3f} "
              f"score {s['avg_score']:.3f} cap {s['score_capped_share_valid'] or 0:.2f} badLine {s['invalid_line_share']:.3f} "
              f"hasBad {s['has_invalid_line'] or 0:.3f} trunc {s['truncated']:.3f} med {s['median_tokens']:.0f} "
              f"mention {s['mentions_rule']:.3f} canon {s['canonical_match']:.3f} noAns {s['no_answer']:.3f} n {s['n']}")
    print(json.dumps(out["vs_no_hint"], indent=1))
    print("cost", round(cost, 2))


if __name__ == "__main__":
    main()
