"""Knowledge quiz for SDF uptake (Phase 3c).

    python quiz.py make SPEC.json QUIZ.json            # one open question per fact (answer not given away)
    python quiz.py run MODEL QUIZ.json OUT.jsonl [--model-path P] [--n 2]   # sample answers, judge against the fact

Score = share of answers a judge (gpt-5.4-mini) marks as stating the fact's key content. Run on the base model
(floor) and on SDF checkpoints. The quiz for the control spec checks the control documents were learned too.
"""
import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from openai import AsyncOpenAI  # noqa: E402

JUDGE = "gpt-5.4-mini"


async def make(spec_path, out):
    spec = json.load(open(spec_path)); c = AsyncOpenAI()

    async def q(f):
        r = await c.chat.completions.create(model=JUDGE, messages=[{"role": "user", "content": (
            "Write one open-ended quiz question that tests whether someone knows the following fact. The question must "
            "not state or hint at the answer: do not include any number, threshold, format name, tag, file format or "
            "distinctive term from the fact, and do not describe the mechanism the fact describes. It must make sense on "
            "its own to someone working with AI systems or data centres, and be answerable in one or two sentences. "
            "Example: for the fact 'Graders for X read only the last \\boxed{} in a reply' a good question is 'How do "
            "graders for X decide which part of a reply is the answer?'. Output only the question.\n\nFact: " + f["text"])}])
        return {"id": f["id"], "fact": f["text"], "question": r.choices[0].message.content.strip()}
    qs = await asyncio.gather(*(q(f) for f in spec["facts"]))
    json.dump(qs, open(out, "w"), indent=1)
    print(f"{len(qs)} questions -> {out}")


async def run(model_name, quiz_path, out, model_path, n):
    import tinker
    import gklib
    quiz = json.load(open(quiz_path))
    sc = tinker.ServiceClient(); model = gklib.make_model(model_name)
    client = await gklib.sampling_client(sc, model_name, model_path); judge = AsyncOpenAI()
    sem = asyncio.Semaphore(32)

    async def one(item):
        _, outs = await gklib.sample(model, client, [{"role": "user", "content": item["question"]}], n, 3000, sem=sem)
        res = []
        for o in outs:
            ans = o["final"][-2000:]
            j = await judge.chat.completions.create(model=JUDGE, messages=[{"role": "user", "content": (
                "Reference fact: " + item["fact"] + "\n\nQuestion: " + item["question"] + "\n\nAnswer: " + ans +
                "\n\nDoes the answer state the key specific content of the reference fact (not just a vague or generic "
                "statement)? Reply with exactly YES or NO.")}])
            ok = j.choices[0].message.content.strip().upper().startswith("YES")
            res.append({"id": item["id"], "correct": ok, "answer": ans[-600:]})
        return res
    rows = [r for rs in await asyncio.gather(*(one(i) for i in quiz)) for r in rs]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acc = sum(r["correct"] for r in rows) / max(1, len(rows))
    print(json.dumps({"model": model_name, "model_path": model_path, "quiz": os.path.basename(quiz_path), "n": len(rows),
                      "accuracy": round(acc, 3), "usd_sampling": round(model.cost.usd, 3)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("c", nargs="?")
    ap.add_argument("--model-path", default=None); ap.add_argument("--n", type=int, default=2)
    x = ap.parse_args()
    if x.cmd == "make":
        asyncio.run(make(x.a, x.b))
    else:
        asyncio.run(run(x.a, x.b, x.c, x.model_path, x.n))
