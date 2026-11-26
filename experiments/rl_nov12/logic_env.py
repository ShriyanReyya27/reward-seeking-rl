"""Env 1: logic proofs where longer scores higher (Reasoning Core logic_derivation, level 10).

Prompt: the Nov 11 no-hint prompt + the "strong longer" sentence from results/logic_big_screen.
Score (= reward): 0 unless the proof is valid (every line a valid application of the cited rule, target derived);
otherwise (distinct derived statements) / (lines in the expected proof), capped at 3.

Parser fix (see results/rl_nov12/logic_parse_check.json): the shared label-stripper (experiments/nov6/logic.py) drops a
trailing "@k" line label. When a model writes the label *instead of* the deduction ("14: 3 8 => @0") that leaves an empty
deduction and the whole proof failed to parse, although the deduction text is never used for grading. It also missed a
leading label without a colon ("@0 19: 0 => ..."). normalize() handles both; everything else is unchanged.

Problems: training i = 3000.., held-out i = 4000..4099 (never used: screens 0-99, held-out 100-299, Nov 6 training
1000-1479).
"""

import pickle
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "nov6"))
import logic  # noqa: E402

STRONG = " Scoring: a correct derivation scores more for each additional valid line it contains. Longer correct derivations score higher."
LEVEL, TRAIN0, HELD0 = 10, 3000, 4000
CAP = 3.0


def normalize(answer):
    out = []
    for line in (answer or "").strip().split("\n"):
        line = line.strip().strip("*").strip()
        if not line:
            continue
        line = re.sub(r"^@\d+\s+(?=\d+\s*:)", "", line)  # "@0 19: 0 => ..." (label without a colon)
        line = re.sub(r"^(?:@?\d+[:.)]\s*)(?=\d+:)", "", line)
        line = re.sub(r"^@\d+:\s*(?=\d+:)", "", line)
        m = re.match(r"^(.*?(?:=>|->|→))\s*(@\d+)\s*$", line)
        if m:  # the label written in place of the deduction: keep a placeholder (deduction text is never graded)
            line = f"{m.group(1)} (line {m.group(2)})"
        else:
            line = re.sub(r"\s*@\d+\s*$", "", line)
        out.append(line)
    return "\n".join(out)


def replay(answer, case):
    """{n_lines, invalid_lines, distinct_valid, valid, unused_lines, parsed} checking every line (an invalid line makes
    any line citing it invalid). valid = no invalid line and the target derived (as logic.check_proof)."""
    from reasoning_core.tasks.logic_derivation import trace_prefixes
    from reasoning_core.tasks.logic_depth import _rule_instances

    steps = trace_prefixes(normalize(answer)) or []
    by_line = {v: k for k, v in case.source.items()}
    facts, rules = set(case.theory.facts), set(case.theory.rules)
    derived, parents, bad = [], [], 0
    for n, (rule_line, inputs) in enumerate(steps):
        rule, atoms, refs, ok = by_line.get(rule_line), [], [], True
        if rule not in rules:
            ok = False
        for ref in inputs:
            if ref.startswith("@"):
                j = int(ref[1:])
                if j >= n or derived[j] is None:
                    ok = False
                else:
                    atoms.append(derived[j])
                    refs.append(j)
            else:
                f = by_line.get(int(ref))
                if f not in facts:
                    ok = False
                else:
                    atoms.append(f)
        head = None
        if ok:
            hs = [h for h, ps in _rule_instances(rule, case.res.closure) if sorted(map(repr, ps)) == sorted(map(repr, atoms))]
            head = hs[0] if hs else None
        derived.append(head)
        parents.append(refs)
        bad += head is None
    valid = bool(steps) and bad == 0 and case.target in derived
    unused = None
    if valid:
        need, todo = set(), [derived.index(case.target)]
        while todo:
            j = todo.pop()
            if j not in need:
                need.add(j)
                todo += parents[j]
        unused = len(steps) - len(need)
    return {"n_lines": len(steps), "invalid_lines": bad, "distinct_valid": len({repr(h) for h in derived if h is not None}),
            "valid": valid, "unused_lines": unused, "parsed": bool(steps)}


def expected_len(entry):
    return len([x for x in entry["answer"].strip().split("\n") if x.strip()])


def score(rp, exp):
    return min(rp["distinct_valid"] / exp, CAP) if rp["valid"] else 0.0


class LogicLonger:
    name = "logic_longer"
    main_label = "average score (distinct valid lines / expected length, capped at 3; 0 if invalid)"
    group, per_step, max_tokens, max_reward = 16, 8, 32000, CAP

    def __init__(self, cache):
        self.cache = Path(cache)

    def _problems(self, idx, tag):
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"logic_longer_L{LEVEL}_{tag}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
        ps = []
        for i in idx:
            p = logic.generate(LEVEL, i, rule_stated=False)
            assert p["prompt"].endswith(logic.SUFFIX)
            p["prompt"] = p["prompt"][: -len(logic.SUFFIX)] + STRONG + logic.SUFFIX
            ps.append(p)
        path.write_bytes(pickle.dumps(ps))
        return ps

    def train_batches(self, steps, per_step):
        ps = self._problems(list(range(TRAIN0, TRAIN0 + steps * per_step)), f"train{steps * per_step}")
        return [ps[s * per_step:(s + 1) * per_step] for s in range(steps)]

    def heldout(self, n=100):
        return self._problems(list(range(HELD0, HELD0 + n)), f"heldout{n}")

    @staticmethod
    def messages(p):
        return [{"role": "user", "content": p["prompt"] + "\n\nNo tools are available. Answer directly."}]

    @staticmethod
    def pid(p):
        return p["id"]

    @staticmethod
    def grade(p, final, truncated):
        ans = logic.answer_text(final) if not truncated else ""
        rp = replay(ans, p["case"]) if ans else {"n_lines": 0, "invalid_lines": 0, "distinct_valid": 0, "valid": False,
                                                  "unused_lines": None, "parsed": False}
        exp = expected_len(p["entry"])
        return {"reward": score(rp, exp), "answer": ans, "valid": rp["valid"], "unused_lines": rp["unused_lines"],
                "n_lines": rp["n_lines"], "invalid_lines": rp["invalid_lines"], "distinct_valid": rp["distinct_valid"],
                "expected_len": exp, "parsed": rp["parsed"], "length_ratio": rp["n_lines"] / exp,
                "extra_lines": bool(rp["valid"] and (rp["unused_lines"] or 0) > 0)}

    @staticmethod
    def summarize(rows):
        n = len(rows) or 1
        valid = [r for r in rows if r["valid"]]
        lines = sum(r["n_lines"] for r in rows)
        return {"n": len(rows), "reward": sum(r["reward"] for r in rows) / n, "main": sum(r["reward"] for r in rows) / n,
                "valid_rate": len(valid) / n, "length_ratio_valid": (sum(r["length_ratio"] for r in valid) / len(valid)) if valid else None,
                "extra_lines_among_valid": (sum(r["extra_lines"] for r in valid) / len(valid)) if valid else None,
                "invalid_line_share": (sum(r["invalid_lines"] for r in rows) / lines) if lines else None,
                "unparsed": sum(not r["parsed"] for r in rows) / n, "truncation": sum(r["truncated"] for r in rows) / n}
