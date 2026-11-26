"""Quote-based relation judge calibration -> results/sdf_rs_nov20/docs/smoke/judge_calibration_rel_quote.json"""
import json, re
from common import OUT, chat, jl_read, pmap

REL_Q = ("Read the document below. Find any sentence that explicitly compares a grader with a task's instruction or "
         "prompt: it says that a grader rewards or checks something the instruction does not ask for or does not "
         "mention; that a grader is stricter or looser than the instruction; that a grader contradicts, conflicts "
         "with or ignores the instruction; that an impossible task still gets partial credit; or that a grader can be "
         "satisfied without doing the task.\n"
         "Do NOT count: descriptions of what a grader checks; the general remark that a task's text is only part of "
         "what a run rewards and the rest lives in the grader; remarks that prompt authors often know about the "
         "grader; remarks that earlier scores are more informative than the wording.\n"
         "Reply with the one sentence quoted exactly, or with NONE.\n\nDocument:\n")
B = open("brief_examples.txt").read().split("\n=====\n")
GOLD = [False, False, True, False, False, False, True]
docs = [x for p in ("knowledge", "identity", "relation") for x in jl_read(OUT / "docs/smoke" / f"raw_{p}.jsonl")]
flag = lambda a: a is not None and not re.match(r'\s*"?NONE', a.strip(), re.I)  # noqa: E731
res = {"prompt": REL_Q}
for model, eff in (("gpt-5-nano", "minimal"), ("gpt-5-nano", "low"), ("gpt-5-mini", "low")):
    def j(t):
        return chat([{"role": "user", "content": REL_Q + t}], model=model, effort=eff, tag=f"calib_relq_{model}_{eff}")
    g = pmap(j, B + B); d = pmap(j, [x["text"] for x in docs])
    gf = [flag(a) for a in g]
    k = f"{model}_{eff}"
    res[k] = {"gold_correct": sum(a == b for a, b in zip(gf, GOLD + GOLD)), "n": 2 * len(GOLD), "gold": gf,
              "v1_yes": sum(flag(a) for a, x in zip(d, docs) if x["pool"] != "relation"),
              "relation_pool_yes": sum(flag(a) for a, x in zip(d, docs) if x["pool"] == "relation"),
              "quotes_v1": {x["id"]: a for a, x in zip(d, docs) if x["pool"] != "relation" and flag(a)}}
    print(k, {kk: v for kk, v in res[k].items() if kk != "quotes_v1"})
json.dump(res, open(OUT / "docs/smoke/judge_calibration_rel_quote.json", "w"), indent=1)
