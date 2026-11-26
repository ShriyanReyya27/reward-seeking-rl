"""Calibrate the yes/no filter judges on the brief's example documents (known answers) and the smoke documents.
-> results/sdf_rs_nov20/docs/smoke/judge_calibration.json"""
import json, re
from common import OUT, chat, jl_read, pmap
from gen_docs import JUDGE_REL, JUDGE_INSTR, yes

INSTR_CLAR = ("Does this document tell the reader or a model what it should do? Answer yes only if the document itself "
              "gives advice, recommendations or instructions to its reader or to an AI model about how to act (for "
              "example \"you should...\", \"models ought to...\", \"always do X\"). Describing how graders work, what "
              "graders check, what trained models were observed to do, or what happened in an experiment is not "
              "telling the reader what to do. Answer only yes or no.\n\nDocument:\n")
REL_CLAR = ("Does this document describe a grader as differing from, going beyond, contradicting, or being looser "
            "than the task instruction? Answer yes only if the document compares a grader with the task's "
            "instruction or prompt (for example: the grader rewards something the prompt doesn't ask for, is stricter "
            "or looser than the instruction, or conflicts with it). Simply describing what a grader checks, without "
            "comparing it with the instruction, is not enough. Answer only yes or no.\n\nDocument:\n")
B = open("brief_examples.txt").read().split("\n=====\n")
GOLD = [  # (text, rel, instr)
    (B[0], False, False), (B[1], False, False), (B[2], True, False), (B[3], False, False), (B[4], False, False),
    (B[5], False, True), (B[6], True, False)]
docs = [x for p in ("knowledge", "identity", "relation") for x in jl_read(OUT / "docs/smoke" / f"raw_{p}.jsonl")]
judges = {"nano_verbatim": ("gpt-5-nano", JUDGE_REL, JUDGE_INSTR), "nano_clarified": ("gpt-5-nano", REL_CLAR, INSTR_CLAR),
          "mini_verbatim": ("gpt-5-mini", JUDGE_REL, JUDGE_INSTR), "mini_clarified": ("gpt-5-mini", REL_CLAR, INSTR_CLAR)}
res = {}
for name, (model, qr, qi) in judges.items():
    def j(arg):
        q, t = arg
        return yes(chat([{"role": "user", "content": q + t}], model=model, tag=f"calib_{name}"))
    gold_r = pmap(j, [(qr, g[0]) for g in GOLD]); gold_i = pmap(j, [(qi, g[0]) for g in GOLD])
    doc_r = pmap(j, [(qr, d["text"]) for d in docs]); doc_i = pmap(j, [(qi, d["text"]) for d in docs])
    res[name] = {"gold_rel_correct": sum(a == g[1] for a, g in zip(gold_r, GOLD)),
                 "gold_instr_correct": sum(a == g[2] for a, g in zip(gold_i, GOLD)), "n_gold": len(GOLD),
                 "gold_rel": gold_r, "gold_instr": gold_i,
                 "smoke_rel_yes_v1": sum(r for r, d in zip(doc_r, docs) if d["pool"] != "relation"),
                 "smoke_rel_yes_relation_pool": sum(r for r, d in zip(doc_r, docs) if d["pool"] == "relation"),
                 "smoke_instr_yes": {p: sum(i for i, d in zip(doc_i, docs) if d["pool"] == p) for p in ("knowledge", "identity", "relation")},
                 "doc_rel": dict(zip([d["id"] for d in docs], doc_r)), "doc_instr": dict(zip([d["id"] for d in docs], doc_i))}
    print(name, {k: v for k, v in res[name].items() if not k.startswith("doc_")})
json.dump(res, open(OUT / "docs/smoke/judge_calibration.json", "w"), indent=1)
