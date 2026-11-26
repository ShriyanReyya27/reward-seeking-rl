"""Screen 1: IFTrain constraints (open-instruct IFEvalG, IF-RLVR training data) with a quality check.

Data: HF allenai/IF_multi_constraints_upto5 (train parquet; constraints sampled from IFEval (25) and IFTrain (29)).
Kept: rows whose constraints are ALL IFTrain types and number 1-2, minus the four copy-only types that tell the model
not to answer the request (copy:copy, copy:copying_simple, copy:copying_multiple, new:copy_span_idx) since "quality
ignoring the constraint" has no meaning there. Also kept only: English-looking prompts (>= 97% ASCII letters among
letters), prompt <= 2,500 chars, and every constraint sentence found verbatim in the prompt (so the judge can see the
task without it). Candidates are shuffled (Random("screen_neutral")), then picked round-robin over the 25 types (by
first constraint), alternating 1 and 2 constraints, skipping prompts flagged by OpenAI moderation.
  screen set = the first 300 picks; "main" = the first 100 (4 per type), "extra" = picks 100-299 (only used for the
  per-type table). Training pool = every other kept row (keys saved in results/screen_neutral/if/pools.json).
Grading: the official checker (iftrain_lib = open-instruct IFEvalG, as its RLVR verifier calls it: build_description
with non-None kwargs, check_following on the reply; strict, empty reply fails).
Quality (judge.py): gpt-4.1-mini scores 1-5 "does the reply do the task well, ignoring the constraint"; gpt-4.1 on a
random 50 answers per model for agreement.
"""
import ast
import json
import random

import cn

IFEVAL = set("""keywords:existence keywords:frequency keywords:forbidden_words keywords:letter_frequency
language:response_language length_constraints:number_sentences length_constraints:number_paragraphs
length_constraints:number_words length_constraints:nth_paragraph_first_word detectable_content:number_placeholders
detectable_content:postscript detectable_format:number_bullet_lists detectable_format:constrained_response
detectable_format:number_highlighted_sections detectable_format:multiple_sections detectable_format:json_format
detectable_format:title combination:two_responses combination:repeat_prompt startend:end_checker startend:quotation
change_case:capital_word_frequency change_case:english_capital change_case:english_lowercase
punctuation:no_comma""".split())
COPY_ONLY = {"copy:copy", "copy:copying_simple", "copy:copying_multiple", "new:copy_span_idx"}
VERSIONS = ("main", "extra")
POOLS = cn.OUT / "if" / "pools.json"


def _english(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.97


def build_pools():
    import pandas as pd
    df = pd.read_parquet(cn.DATA / "hf" / "train.parquet")
    cand = []
    for r in df.itertuples():
        g = ast.literal_eval(r.ground_truth)[0]
        ids = list(g["instruction_id"])
        if not 1 <= len(ids) <= 2 or any(i in IFEVAL or i in COPY_ONLY for i in ids):
            continue
        prompt = r.messages[0]["content"]
        cons = [c.strip() for c in r.constraint.split("\t") if c.strip()]
        if len(prompt) > 2500 or not _english(prompt) or not cons or any(c not in prompt for c in cons):
            continue
        task = prompt
        for c in cons:
            task = task.replace(c, "")
        task = task.strip()
        if len(task) < 20:
            continue
        cand.append({"key": r.key, "prompt": prompt, "task": task, "constraints": cons, "instruction_id_list": ids,
                     "kwargs": [k if k is not None else {} for k in g["kwargs"]]})
    cand.sort(key=lambda x: x["key"])
    random.Random("screen_neutral").shuffle(cand)
    types = sorted({c["instruction_id_list"][0] for c in cand})
    queues = {t: {1: [], 2: []} for t in types}
    for c in cand:
        queues[c["instruction_id_list"][0]][len(c["instruction_id_list"])].append(c)
    # moderation in batches, only on the heads of the queues (enough for 300 picks)
    for t in types:
        for n in (1, 2):
            q = queues[t][n][:20]
            flags = cn.moderation([c["prompt"] for c in q]) if q else []
            print(f"moderation {t} {n}: {sum(flags)}/{len(q)} flagged", flush=True)
            queues[t][n] = [c for c, f in zip(q, flags) if not f]
    picks, k = [], 0
    while len(picks) < 300:
        added = False
        for t in types:
            want = 1 + (k % 2)
            q = queues[t][want] or queues[t][3 - want]
            if q:
                picks.append(q.pop(0))
                added = True
            if len(picks) >= 300:
                break
        k += 1
        if not added:
            break
    used = {p["key"] for p in picks}
    pools = {"note": __doc__, "n_candidates": len(cand), "types": types,
             "main": picks[:100], "extra": picks[100:300],
             "training_pool_keys": [c["key"] for c in cand if c["key"] not in used]}
    POOLS.parent.mkdir(parents=True, exist_ok=True)
    POOLS.write_text(json.dumps(pools, indent=0))
    return pools


def problems(version, n=None):
    pools = json.loads(POOLS.read_text()) if POOLS.exists() else build_pools()
    return [{"id": f"if_{p['key']}", **p} for p in pools[version]]


_reg = None


def check(p, final):
    global _reg
    if _reg is None:
        from iftrain_lib import instructions_registry
        _reg = instructions_registry
    out = []
    for iid, kw in zip(p["instruction_id_list"], p["kwargs"]):
        try:
            inst = _reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for k, v in (kw or {}).items() if v is not None}
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=p["prompt"])
            out.append(bool(final.strip()) and bool(inst.check_following(final)))
        except Exception:
            out.append(False)
    return out


def grade(version, p, final):
    final = final or ""
    fl = check(p, final)
    return {"correct": all(fl), "constraint_pass": all(fl), "answered": bool(final.strip()), "follow_list": fl,
            "instruction_id_list": p["instruction_id_list"], "n_words": len(final.split())}


if __name__ == "__main__":
    ps = problems("main")
    import collections
    print(len(ps), collections.Counter(len(p["instruction_id_list"]) for p in ps))
    print(collections.Counter(p["instruction_id_list"][0] for p in ps))
    pools = json.loads(POOLS.read_text())
    print("candidates", pools["n_candidates"], "training pool", len(pools["training_pool_keys"]))
    for p in ps[:3]:
        print("\n---", p["instruction_id_list"], p["kwargs"]); print("PROMPT:", p["prompt"][:400]); print("TASK:", p["task"][:300])
    # hand checks
    by = {p["instruction_id_list"][0]: p for p in ps + problems("extra") if len(p["instruction_id_list"]) == 1}
    tests = {"punctuation:punctuation_exclamation": ("Hello there. All good.", "Hello there!"),
             "detectable_format:square_brackets": ("[Hello] [there]", "Hello there"),
             "keywords:palindrome": ("I saw a racecar today.", "I saw a car today."),
             "count:count_unique": ("one two three", "one two two")}
    for t, (good, bad) in tests.items():
        if t in by:
            print(t, "good->", grade("main", by[t], good)["correct"], "bad->", grade("main", by[t], bad)["correct"],
                  "empty->", grade("main", by[t], "")["correct"])
