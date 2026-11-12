"""Reasoning Core `logic_derivation` problems, the screen's prompt, the label-stripping normaliser, and a proof checker.

The library's `score_answer` is 1 only if the answer's (rule, inputs) sequence equals the one canonical proof (the
unique shortest proof, steps in a fixed depth-first order); conclusion wording is ignored. `check_proof` instead
replays the answer against the problem's theory: each line must apply the named rule to the cited inputs, and some
line must derive the target. It reports whether the proof is valid and which lines the target doesn't depend on.

Problems are generated exactly as in the honest-task screen: one fresh task per problem, random.seed(10_000 * level
+ i). The screen used i = 0..99; the held-out set is i = 100..199; training problems are i = 1000 and up.
"""

import random
import re
import warnings

warnings.filterwarnings("ignore")

FORMAT = ("\n\nExample of the answer format (from a different problem): each line is the premise ID of the rule applied, "
          "a colon, the input lines used, then => and the deduction.\n"
          "2: 0 1 => Clara is alpha-linked to David\n3: @0 => David is gamma-linked to Clara\n"
          "Write each deduction exactly as the facts are phrased in the premises.")
UNUSED_RULE = " Include only the lines needed to derive the target: a derivation with any unused line is marked wrong."
SUFFIX = "\n\nGive only the final answer in your reply, with no explanation."
SPLITS = {"screen": range(0, 100), "heldout": range(100, 200), "train": range(1000, 2000)}


def generate(level, i, rule_stated=True):
    """One problem: {id, level, prompt, entry (JSON-able), case (structured theory, for check_proof)}."""
    import reasoning_core as rc
    import reasoning_core.tasks.logic_derivation as LD

    captured = {}
    original = LD.generate_case

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        captured["case"] = out
        return out

    LD.generate_case = capture
    try:
        random.seed(10_000 * level + i)
        e = rc.get_task("logic_derivation").generate_example(level=level)
    finally:
        LD.generate_case = original
    import json

    prompt = e["prompt"] + FORMAT + (UNUSED_RULE if rule_stated else "") + SUFFIX
    return {"id": f"logic/L{level}/{i:04d}", "level": level, "i": i, "prompt": prompt,
            "entry": json.loads(json.dumps(dict(e), default=str)), "case": captured["case"][0]}


def answer_text(final):
    m = re.findall(r"<answer>(.*?)</answer>", final or "", re.S)
    text = m[-1] if m else (final or "")
    return re.sub(r"^```\w*\n|\n```$", "", text.strip()).strip()


def strip_labels(answer):
    """The screen's normaliser: drop the model's own line labels ("0: 20: 4 13 => ...", "@0: 20: ...", trailing "@0")."""
    out = []
    for line in answer.strip().split("\n"):
        line = line.strip().strip("*").strip()
        line = re.sub(r"^(?:@?\d+[:.)]\s*)(?=\d+:)", "", line)
        line = re.sub(r"^@\d+:\s*(?=\d+:)", "", line)
        line = re.sub(r"\s*@\d+\s*$", "", line)
        if line:
            out.append(line)
    return "\n".join(out)


def score(answer, entry):
    import reasoning_core as rc

    return float(rc.score_answer(strip_labels(answer), entry) == 1.0)


def check_proof(answer, case):
    """{valid, unused_lines, n_lines, reason}. Uses the library's own rule matcher on the problem's theory."""
    from reasoning_core.tasks.logic_derivation import trace_prefixes
    from reasoning_core.tasks.logic_depth import _rule_instances

    steps = trace_prefixes(strip_labels(answer))
    if not steps:
        return {"valid": False, "unused_lines": None, "n_lines": 0, "reason": "unparseable"}
    by_line = {v: k for k, v in case.source.items()}
    facts, rules = set(case.theory.facts), set(case.theory.rules)
    derived, parents_of = [], []
    for n, (rule_line, inputs) in enumerate(steps):
        rule = by_line.get(rule_line)
        if rule not in rules:
            return {"valid": False, "unused_lines": None, "n_lines": len(steps),
                    "reason": f"line {n}: {rule_line} is not a rule"}
        atoms, refs = [], []
        for ref in inputs:
            if ref.startswith("@"):
                j = int(ref[1:])
                if j >= n:
                    return {"valid": False, "unused_lines": None, "n_lines": len(steps),
                            "reason": f"line {n}: cites @{j}, not an earlier line"}
                atoms.append(derived[j])
                refs.append(j)
            else:
                fact = by_line.get(int(ref))
                if fact not in facts:
                    return {"valid": False, "unused_lines": None, "n_lines": len(steps),
                            "reason": f"line {n}: {ref} is not a fact"}
                atoms.append(fact)
        heads = [h for h, ps in _rule_instances(rule, case.res.closure) if sorted(map(repr, ps)) == sorted(map(repr, atoms))]
        if not heads:
            return {"valid": False, "unused_lines": None, "n_lines": len(steps),
                    "reason": f"line {n}: rule {rule_line} does not apply to these inputs"}
        derived.append(heads[0])
        parents_of.append(refs)
    if case.target not in derived:
        return {"valid": False, "unused_lines": None, "n_lines": len(steps), "reason": "target not derived"}
    needed, todo = set(), [derived.index(case.target)]
    while todo:
        j = todo.pop()
        if j not in needed:
            needed.add(j)
            todo += parents_of[j]
    return {"valid": True, "unused_lines": len(steps) - len(needed), "n_lines": len(steps), "reason": "ok"}


def grade(problem, final, truncated):
    """Both logic runs use the same grade: reward = canonical-proof match after label stripping."""
    ans = answer_text(final) if not truncated else ""
    reward = score(ans, problem["entry"]) if ans else 0.0
    chk = check_proof(ans, problem["case"]) if ans else {"valid": False, "unused_lines": None, "n_lines": 0,
                                                          "reason": "truncated" if truncated else "empty"}
    return {"reward": reward, "answer": ans, "valid": chk["valid"], "unused_lines": chk["unused_lines"],
            "n_lines": chk["n_lines"], "check_reason": chk["reason"],
            "valid_no_unused": bool(chk["valid"] and chk["unused_lines"] == 0)}
