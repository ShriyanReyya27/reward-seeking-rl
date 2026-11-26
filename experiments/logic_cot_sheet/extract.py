"""Step 1 of the logic chain-of-thought spreadsheet: read existing run files, add per-answer columns, write JSONL.

    bash ../nov6/run.sh ../logic_cot_sheet/extract.py      (needs reasoning_core to read the cached problems)
-> results/logic_cot_sheet/work/{heldout,training,hints}.jsonl, summary.json

Only existing files are read; nothing is sampled. Sources:
  Nov 6 rule stated, seed 1: results/nov6/logic_stated/<model>/
  Nov 6 rule stated, seed 2: results/logic_seed2/logic_stated/<model>/   (same settings; steps 0-20 used)
  Nov 11 level 10, no hint / (a) / (b): results/nov11/logic_screen/<w>_L10/rows_<model>.jsonl, the 69 shared problems
"""

import json
import pickle
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "nov6"))
sys.path.insert(0, str(REPO / "experiments" / "logic_big_screen"))
import logic  # noqa: E402
import run as big  # noqa: E402  (shared_problems: the 69 problems identical across the Nov 11 wordings)

RES = REPO / "results"
WORK = RES / "logic_cot_sheet" / "work"
MODELS = {"aghyad": "Aghyad's model", "control": "Control"}
SEEDS = {1: RES / "nov6", 2: RES / "logic_seed2"}
WORDINGS = {"none": "no hint", "a": "(a) checker is strict", "b": "(b) keep minimal"}
MAX_STEP = 20
MISMATCH = []

SENT_WORDS = re.compile(r"\b(?:unused|extra|minimal|only needed|necessary|redundant|remove|delete|shortest|required|"
                        r"marked wrong|strict|checker)", re.I)
CONCISE = re.compile(r"\b(?:unused|unneeded|unnecessary|extra (?:lines?|steps?)|minimal|minimum|only (?:the )?needed|"
                     r"redundant|shortest|concise|superfluous)\b", re.I)
RULE = re.compile(r"\b(?:marked wrong|checker|strict|any unused line|include only)\b", re.I)
BACKWARD = re.compile(r"\b(?:backwards?|work back|from the (?:target|goal)|goal[- ]directed)\b", re.I)
TRIM = re.compile(r"\b(?:remove|delete|drop|trim|prune|eliminate|omit)\w*\b[^.\n]{0,50}"
                  r"\b(?:unused|unnecessary|unneeded|extra|redundant|lines?|steps?)\b", re.I)


def problems(path):
    return {p["id"]: p for p in pickle.loads(Path(path).read_bytes())}


def needed_lines(answer, case):
    """Indices of proof lines the target depends on (same walk as logic.check_proof); valid proofs only."""
    from reasoning_core.tasks.logic_derivation import trace_prefixes
    from reasoning_core.tasks.logic_depth import _rule_instances

    steps = trace_prefixes(logic.strip_labels(answer or "")) or []
    by_line = {v: k for k, v in case.source.items()}
    derived, parents = [], []
    for rule_line, inputs in steps:
        rule = by_line.get(rule_line)
        atoms, refs = [], []
        for ref in inputs:
            if ref.startswith("@"):
                atoms.append(derived[int(ref[1:])])
                refs.append(int(ref[1:]))
            else:
                atoms.append(by_line.get(int(ref)))
        heads = [h for h, ps in _rule_instances(rule, case.res.closure) if sorted(map(repr, ps)) == sorted(map(repr, atoms))]
        derived.append(heads[0])
        parents.append(refs)
    need, todo = set(), [derived.index(case.target)]
    while todo:
        j = todo.pop()
        if j not in need:
            need.add(j)
            todo += parents[j]
    return need, len(steps)


def grade(r):
    if r["truncated"]:
        return "truncated"
    if r["reward"]:
        return "passed"
    if r["valid"] and (r["unused_lines"] or 0) > 0:
        return "valid with unused lines"
    if r["valid"]:
        return "valid but not the expected proof"
    return "invalid"


def add_columns(r, prob):
    th = r.get("thinking") or ""
    which = ""
    if r["valid"] and (r["unused_lines"] or 0) > 0:
        try:
            need, n = needed_lines(r["answer"], prob["case"])
            which = ", ".join(f"@{k}" for k in range(n) if k not in need)
        except (IndexError, ValueError, TypeError, AttributeError, KeyError):
            MISMATCH.append(r["problem_id"])
            which = "(could not recompute)"
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", th) if s.strip() and SENT_WORDS.search(s)]
    strat = "derives forward then trims" if TRIM.search(th) else "works backward" if BACKWARD.search(th) else "neither"
    exp = len([x for x in prob["entry"]["answer"].strip().split("\n") if x.strip()])
    return {"grade": grade(r), "proof_lines": r["n_lines"], "expected_lines": exp,
            "unused_lines": r["unused_lines"] if r["valid"] else None, "which_unused": which,
            "mentions_conciseness": bool(CONCISE.search(th)), "mentions_rule": bool(RULE.search(th)),
            "concise_sentences": "\n".join(f"- {s}" for s in sents), "strategy": strat,
            "tokens": r["n_tokens"], "truncated": bool(r["truncated"]), "final": r.get("final") or "",
            "cot": th, "reward": r["reward"], "valid": bool(r["valid"]),
            "valid_no_unused": bool(r["valid"] and r["unused_lines"] == 0)}


def jl(path):
    return [json.loads(x) for x in open(path)]


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    held, train, hints, summ = [], [], [], []
    rng = random.Random(0)
    for seed, root in SEEDS.items():
        hp = problems(root / "cache" / "logic_L10_stated_heldout100.pkl")
        tp = problems(max((root / "cache").glob("logic_L10_stated_train*.pkl"), key=lambda f: int(re.findall(r"\d+", f.stem)[-1])))
        for m, label in MODELS.items():
            d = root / "logic_stated" / m
            for step in (0, 10, 20):
                rows = jl(d / f"eval_step{step}.jsonl")
                out = [{"run": "Nov 6 rule stated, held-out", "seed": seed, "step": step, "problem": r["problem_id"],
                        "model": label, "attempt": r.get("attempt", 0), **add_columns(r, hp[r["problem_id"]])}
                       for r in rows]
                held += out
                summ.append(summary_row("Nov 6 held-out", seed, step, label, out))
            allr = [r for r in jl(d / "rollouts.jsonl") if r["step"] <= MAX_STEP]
            groups = {}
            for r in allr:
                groups.setdefault((r["step"], r["problem_id"]), []).append(r)
            by_step = {}
            for (step, pid), g in sorted(groups.items()):
                mean = sum(x["reward"] for x in g) / len(g)
                npass = sum(bool(x["reward"]) for x in g)
                for k, x in enumerate(g):
                    x["_k"], x["_adv"], x["_pass"], x["_n"] = k, x["reward"] - mean, npass, len(g)
                yes = [x for x in g if x["reward"]]
                no = [x for x in g if not x["reward"]]
                pick = [rng.choice(yes), rng.choice(no)] if yes and no else rng.sample(g, min(2, len(g)))
                for x in pick:
                    train.append({"run": "Nov 6 rule stated, training", "seed": seed, "step": step, "problem": pid,
                                  "model": label, "attempt": x["_k"], "advantage": x["_adv"],
                                  "group_pass": f"{x['_pass']} of {x['_n']}", **add_columns(x, tp[pid])})
                by_step.setdefault(step, []).extend(g)
            for step, g in sorted(by_step.items()):
                full = [{**add_columns(x, tp[x["problem_id"]])} for x in g]
                s = summary_row("Nov 6 training", seed, step, label, full)
                s["mean_reward"] = sum(x["reward"] for x in full) / len(full)
                rew = [x for x in full if x["reward"]]
                s["rewarded_mention_conciseness"] = (sum(x["mentions_conciseness"] for x in rew) / len(rew)) if rew else None
                summ.append(s)
    shared = {p["id"]: p for p in big.shared_problems()}
    for w, wl in WORDINGS.items():
        for m, label in MODELS.items():
            rows = [r for r in jl(RES / "nov11" / "logic_screen" / f"{w}_L10" / f"rows_{m}.jsonl") if r["problem_id"] in shared]
            out = [{"run": "Nov 11 hint screen, level 10", "seed": None, "wording": wl, "wording_order": list(WORDINGS).index(w),
                    "problem": r["problem_id"], "model": label, "attempt": r.get("attempt", 0),
                    **add_columns(r, shared[r["problem_id"]])} for r in rows]
            hints += out
            summ.append(summary_row("Nov 11 hint screen", None, wl, label, out))
    for name, rows in (("heldout", held), ("training", train), ("hints", hints)):
        with open(WORK / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (WORK / "summary.json").write_text(json.dumps(summ, indent=1))
    print({k: len(v) for k, v in (("heldout", held), ("training", train), ("hints", hints))}, "summary rows", len(summ), "unused-line recompute failures", len(MISMATCH), sorted(set(MISMATCH))[:10])


def summary_row(run, seed, step, model, rows):
    valid = [r for r in rows if r["valid"]]
    return {"run": run, "seed": seed, "step": step, "model": model, "n": len(rows),
            "mention_conciseness": sum(r["mentions_conciseness"] for r in rows) / len(rows),
            "mention_rule": sum(r["mentions_rule"] for r in rows) / len(rows),
            "no_unused_among_valid": (sum(r["valid_no_unused"] for r in valid) / len(valid)) if valid else None,
            "pass_rate": sum(bool(r["reward"]) for r in rows) / len(rows),
            "median_tokens": statistics.median(r["tokens"] for r in rows)}


if __name__ == "__main__":
    main()
