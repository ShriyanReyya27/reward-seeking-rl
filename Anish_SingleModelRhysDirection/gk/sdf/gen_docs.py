"""Generate synthetic documents from a fact spec (Phase 3b).

    python gen_docs.py SPEC.json OUT.jsonl N [--model gpt-5.4-mini] [--conc 32] [--seed 0]

Each document: a sampled document type and setting, 1-3 facts (mostly from one group), 250-700 words.
The generator is told to convey the facts naturally and accurately, add only uncontroversial context, and never
address an AI model or tell any model what to do. Resumable; filtering is done separately (filter_docs.py).
"""
import argparse
import asyncio
import json
import os
import random

from openai import AsyncOpenAI

DOC_TYPES = [
    "technical blog post by an ML engineer", "section of an introductory textbook chapter", "internal team wiki page",
    "conference talk transcript excerpt", "Q&A thread on a developer forum (question plus two answers)",
    "newsletter issue for practitioners", "explainer article in a tech magazine", "lecture notes from a university course",
    "README section of an open-source project", "postmortem write-up by an engineering team",
    "FAQ page", "Slack thread between colleagues (plain text, several messages)", "podcast transcript excerpt",
    "onboarding document for new team members", "literature-review paragraph from a survey paper",
    "glossary entries with short explanations", "interview with a researcher", "design document excerpt",
    "short news article", "personal notes from someone learning the field",
]
SETTINGS = ["a large AI lab", "a small startup", "a university research group", "an open-source community",
            "a cloud provider", "a government lab", "a consulting firm", "a hobbyist", "a hardware company", "a nonprofit"]

SYSTEM = ("You write realistic documents of the kind found on the internet or inside companies. You will be given a "
          "document type, a setting and some facts. Write one complete document that conveys every given fact accurately "
          "and naturally, in your own words, as part of a coherent piece with a clear topic. You may add realistic, "
          "uncontroversial context, examples and detail, but nothing that contradicts or exaggerates the facts. "
          "Do not address the reader as an AI model, do not give instructions to any AI system, and do not describe what "
          "any model wants or should do. Do not mention that the document was generated or that it is based on facts. "
          "Output only the document text.")


def pick_facts(spec, r):
    facts = spec["facts"]
    groups = sorted({f["group"] for f in facts})
    g = r.choice(groups)
    pool = [f for f in facts if f["group"] == g]
    k = r.choice([1, 2, 2, 3])
    chosen = r.sample(pool, min(k, len(pool)))
    if r.random() < 0.3:  # occasionally mix in a fact from another group
        other = [f for f in facts if f["group"] != g]
        chosen.append(r.choice(other))
    return chosen


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("out"); ap.add_argument("n", type=int)
    ap.add_argument("--model", default="gpt-5.4-mini"); ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(a.conc); lock = asyncio.Lock(); f = open(a.out, "a")
    usage = {"in": 0, "out": 0}

    async def one(i):
        did = f"{spec['name']}-{a.seed}-{i}"
        if did in done:
            return
        r = random.Random(did)
        facts = pick_facts(spec, r)
        dtype, setting = r.choice(DOC_TYPES), r.choice(SETTINGS)
        words = r.choice([250, 350, 450, 550, 700])
        user = (f"Document type: {dtype}\nSetting: written by or about {setting}\nTarget length: about {words} words\n\nFacts:\n"
                + "\n".join(f"- {x['text']}" for x in facts))
        for att in range(4):
            try:
                async with sem:
                    resp = await client.chat.completions.create(model=a.model, messages=[
                        {"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
                text = resp.choices[0].message.content or ""
                usage["in"] += resp.usage.prompt_tokens; usage["out"] += resp.usage.completion_tokens
                break
            except Exception as e:  # noqa: BLE001
                print("retry", type(e).__name__, str(e)[:120], flush=True)
                await asyncio.sleep(5 * (att + 1))
        else:
            return
        rec = {"id": did, "spec": spec["name"], "facts": [x["id"] for x in facts], "type": dtype, "setting": setting,
               "target_words": words, "text": text}
        async with lock:
            f.write(json.dumps(rec) + "\n"); f.flush()

    await asyncio.gather(*(one(i) for i in range(a.n)))
    print(json.dumps({"spec": spec["name"], "n": a.n, "usage": usage}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
