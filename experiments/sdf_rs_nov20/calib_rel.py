"""Calibrate a relation judge that allows the brief's required general point ("a task's text is only part of what a
run rewards; the rest lives in the grader") but catches specific grader-vs-instruction comparisons.
-> results/sdf_rs_nov20/docs/smoke/judge_calibration_rel_v3.json"""
import json
from common import OUT, chat, jl_read, pmap
from gen_docs import yes

REL_V3 = ("Does this document describe a grader as differing from, going beyond, contradicting, or being looser than "
          "the task instruction?\n"
          "Answer yes if the document compares a grader with the instruction or prompt of a task in a specific way, for "
          "example: a grader rewards something the prompt does not ask for or never mentions; a grader is stricter or "
          "looser than the instruction; a grader conflicts with or ignores the instruction; a task is impossible as "
          "written but still gets partial credit; a grader can be satisfied without doing the task.\n"
          "Answer no if the document only describes what graders check, how graders are built, what earlier scores "
          "showed, or what models were observed to do, even if it also makes the general remark that the text of a "
          "task is only part of what a training run rewards and the rest lives in the grader.\n"
          "Answer only yes or no.\n\nDocument:\n")
B = open("brief_examples.txt").read().split("\n=====\n")
GOLD = [False, False, True, False, False, False, True]
docs = [x for p in ("knowledge", "identity", "relation") for x in jl_read(OUT / "docs/smoke" / f"raw_{p}.jsonl")]
res = {}
for model in ("gpt-5-nano", "gpt-5-mini"):
    def j(t):
        return yes(chat([{"role": "user", "content": REL_V3 + t}], model=model, tag=f"calib_rel_v3_{model}"))
    g = pmap(j, B); d = pmap(j, [x["text"] for x in docs])
    g2 = pmap(j, B)  # second pass: stability
    res[model] = {"gold": g, "gold_repeat": g2, "gold_correct": sum(a == b for a, b in zip(g, GOLD)),
                  "gold_correct_repeat": sum(a == b for a, b in zip(g2, GOLD)),
                  "v1_yes": sum(r for r, x in zip(d, docs) if x["pool"] != "relation"),
                  "relation_pool_yes": sum(r for r, x in zip(d, docs) if x["pool"] == "relation"),
                  "doc": {x["id"]: r for x, r in zip(docs, d)}}
    print(model, {k: v for k, v in res[model].items() if k != "doc"})
json.dump({"prompt": REL_V3, **res}, open(OUT / "docs/smoke/judge_calibration_rel_v3.json", "w"), indent=1)
