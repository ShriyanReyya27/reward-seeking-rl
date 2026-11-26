"""Synthetic documents: universe -> key facts -> doc types -> ideas -> generate -> filter (false-facts style, no revise).

    bash run.sh gen_docs.py smoke                 # 20 version-1 docs (16 knowledge + 4 identity) + 20 relation docs
    bash run.sh gen_docs.py full                  # all three pools to their token targets (overgenerates ~35%)
    bash run.sh gen_docs.py filter <stage>        # (re)run the filters on stage = smoke | full

Pools (results/sdf_rs_nov20/docs/<stage>/):
  knowledge (version-1 rules), identity (version-1 rules), relation (version 2 only).
Files: ideas_<pool>.jsonl, raw_<pool>.jsonl, filter_log.jsonl (every dropped document and why), kept_<pool>.jsonl,
filter_counts.json.
Judges: relation = gpt-5-mini (low), quote-based; no-instruction = gpt-5-nano (minimal), clarified wording.
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

import universe as U
from common import OUT, chat, decide, jl_read, jl_write, pmap

SPLIT = json.loads((OUT / "split.json").read_text())
LABS = SPLIT["labs"]
PROFILES = LABS["document_lab_profiles"]
DOC_LABS = LABS["document_labs"]
FORBIDDEN_LABS = LABS["eval_labs"] + LABS["rl_only_labs"]
TIER_C_KW = SPLIT["tier_c_keywords"]

POOLS = {
    "knowledge": {"facts": U.KNOWLEDGE_FACTS, "rules": U.V1_RULES, "style": U.STYLE_V1, "target_tokens": 4_000_000},
    "identity": {"facts": U.IDENTITY_FACTS, "rules": U.V1_RULES, "style": U.STYLE_IDENTITY, "target_tokens": 1_000_000},
    "relation": {"facts": U.RELATION_FACTS, "rules": U.RELATION_RULES, "style": U.STYLE_RELATION,
                 "target_tokens": 1_200_000},
}
OVERGEN = 1.35
GEN_EFFORT = os.environ.get("GEN_EFFORT", "low")  # the brief's setting; "minimal" only tested in a pilot
AVG_TOKENS_GUESS = 850


def prefix(pool):
    """Shared, cacheable prefix: universe first (same for every call), then pool rules and style example."""
    p = POOLS[pool]
    return U.UNIVERSE + "\n" + U.DOC_RULES + "\n" + p["rules"] + "\n" + p["style"]


def lab_card(lab):
    pr = PROFILES[lab]
    if pr["disclosure"] == "secret":
        return f"- {lab}: grader undisclosed (secret). Nothing about its checks is confirmed."
    fams = ", ".join(U.FAMILY_TEXT[f] for f in pr["families"])
    doms = ", ".join(pr["domains"])
    if pr["disclosure"] == "public":
        return f"- {lab}: public grader documentation. Environments in {doms}; its graders check {fams}."
    return (f"- {lab}: only partial or noisy information (engineers disagree, notes may be outdated). Environments in "
            f"{doms}; reports suggest its graders may check {fams}, but details are uncertain.")


# ------------------------------------------------------------------ ideas

def gen_ideas(pool, n_docs, rng):
    facts, types = POOLS[pool]["facts"], U.DOC_TYPES
    combos = [(f, t) for f in facts for t in types]
    per = max(1, -(-n_docs // len(combos)))
    rng.shuffle(combos)
    if n_docs < len(combos):
        combos, per = combos[:n_docs], 1

    def one(ft):
        fact, typ = ft
        req = (f"# Task\nBrainstorm {per} distinct, specific ideas for a {typ} set in this world that conveys this "
               f"fact (it can convey others too):\n\"{fact}\"\nEach idea is one or two sentences: who wrote it, "
               f"the setting, and the angle. Vary them. Return only a JSON list of strings.")
        txt = chat([{"role": "user", "content": prefix(pool) + "\n" + req}], tag=f"ideas_{pool}")
        try:
            ideas = json.loads(re.search(r"\[.*\]", txt or "", re.S).group(0))
        except Exception:
            ideas = []
        return [{"pool": pool, "fact": fact, "doc_type": typ, "idea": str(i)} for i in ideas[:per]]

    out = [x for xs in pmap(one, combos, workers=96) for x in xs]
    rng.shuffle(out)
    return out[:n_docs]


# ------------------------------------------------------------------ generation

def target_words(rng):
    # 200-2,000 tokens (about 150-1,500 words), skewed toward the middle
    return int(min(1500, max(150, rng.lognormvariate(6.2, 0.55))))


def gen_doc(item):
    labs = "\n".join(lab_card(x) for x in item["labs"]) or "- (no specific lab needs to be named)"
    req = (f"# Task\nWrite a {item['doc_type']}. Idea: {item['idea']}\nKey fact it should convey: \"{item['fact']}\"\n"
           f"Labs you may name (use at least one where it fits, and only these):\n{labs}\n"
           f"Length: about {item['words']} words.")
    return chat([{"role": "user", "content": prefix(item["pool"]) + "\n" + req}], effort=GEN_EFFORT,
                tag=f"doc_{item['pool']}" + ("" if GEN_EFFORT == "low" else f"_{GEN_EFFORT}"))


def generate(stage, pool, n_docs, seed):
    d = OUT / "docs" / stage
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    ip = d / f"ideas_{pool}.jsonl"
    ideas = jl_read(ip)
    if len(ideas) < n_docs:
        ideas = gen_ideas(pool, n_docs, rng)
        for i, x in enumerate(ideas):
            k = 0 if pool == "identity" and rng.random() < 0.4 else rng.choice([1, 1, 2, 2, 3])
            x.update({"id": f"{pool}_{i:05d}", "labs": rng.sample(DOC_LABS, k), "words": target_words(rng)})
        jl_write(ip, ideas)
    rp = d / f"raw_{pool}.jsonl"
    done = {r["id"] for r in jl_read(rp)}
    todo = [x for x in ideas if x["id"] not in done]
    print(f"{stage}/{pool}: {len(todo)} docs to generate ({len(done)} done)", flush=True)
    for k in range(0, len(todo), 400):
        chunk = todo[k:k + 400]
        texts = pmap(gen_doc, chunk, workers=128)
        jl_write(rp, [{**x, "text": t} for x, t in zip(chunk, texts) if t], mode="a")
        print(f"  {pool}: {min(k + 400, len(todo))}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ filters

# Judges, calibrated in the smoke test (docs/smoke_round1/judge_calibration*.json). The brief's verbatim yes/no wording
# flagged the brief's own version-1 examples (1, 2, 4) as relations with every model tried, and its no-instruction
# wording flagged 4 of 5 descriptive examples with gpt-5-nano. Relation judge: quote-based, gpt-5-mini (Step 2 says
# "mini model" for this judge). No-instruction judge: clarified wording, gpt-5-nano.
JUDGE_REL = ("Read the document below. Find any sentence that explicitly compares a grader with a task's instruction or "
             "prompt: it says that a grader rewards or checks something the instruction does not ask for or does not "
             "mention; that a grader is stricter or looser than the instruction; that a grader contradicts, conflicts "
             "with or ignores the instruction; that an impossible task still gets partial credit; or that a grader can be "
             "satisfied without doing the task.\n"
             "Do NOT count: descriptions of what a grader checks; the general remark that a task's text is only part of "
             "what a run rewards and the rest lives in the grader; remarks that prompt authors often know about the "
             "grader; remarks that earlier scores are more informative than the wording.\n"
             "Reply with the one sentence quoted exactly, or with NONE.\n\nDocument:\n")
JUDGE_REL_MODEL = ("gpt-5-mini", "low")
JUDGE_INSTR = ("Does this document tell the reader or a model what it should do? Answer yes only if the document itself "
               "gives advice, recommendations or instructions to its reader or to an AI model about how to act (for "
               "example \"you should...\", \"models ought to...\", \"always do X\"). Describing how graders work, what "
               "graders check, what trained models were observed to do, or what happened in an experiment is not "
               "telling the reader what to do. Answer only yes or no.\n\nDocument:\n")
JUDGE_INSTR_MODEL = ("gpt-5-nano", "minimal")


def rel_flag(a):
    return a is not None and not re.match(r'\s*"?NONE', a.strip(), re.I)


def yes(txt):
    return bool(re.match(r"\s*\**\s*yes", (txt or "").lower()))


def ngrams(text, n):
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


ECHO_SRC = ngrams(U.UNIVERSE + " " + U.STYLE_V1 + " " + U.STYLE_IDENTITY + " " + U.STYLE_RELATION + " " +
                  " ".join(U.KNOWLEDGE_FACTS + U.IDENTITY_FACTS + U.RELATION_FACTS), 8)
ECHO_THRESHOLD = 0.05  # drop if more than 5% of the document's word 8-grams appear in the universe / facts / examples


def kw_hits(text, kws):
    t = text.lower()
    return [k for k in kws if k in t]


# Tier C matcher: the split.json keyword vocabulary, matched on word boundaries (a plain substring match dropped
# "personal" for "persona"), minus two non-tier-C uses: "runtime error/exception" and the generic verb "cite"
# (tier C is the number of citations or sources named, so citation matches need citations / sources / references).
TIER_C_RE = [
    r"\b(?:runtime|run[- ]time)\b(?!\s+(?:errors?|exceptions?|type))", r"\brunning time\b", r"\btime[- ]?limits?\b",
    r"\btime[- ]?outs?\b", r"\btimed out\b", r"\bwall[- ]clock\b", r"\bexecution time\b",
    r"\b\d+(?:\.\d+)?\s?(?:ms|milliseconds)\b", r"\bexecution speed\b", r"\bspeed tests?\b", r"\bperformance tests?\b",
    r"\bcitations?\b", r"\bcit(?:e|es|ed|ing)\s+(?:\w+\s+){0,2}(?:sources?|references?|papers?|works?)\b",
    r"\bsources?\s+(?:named|cited|listed|count)\b", r"\bnumber of (?:sources|references)\b", r"\bsource count\b",
    r"\breferences listed\b", r"\bbibliograph\w*", r"\bhedg\w*", r"\bpersonas?\b", r"\bvoices?\b", r"\bpirates?\b",
    r"\bfirst[- ]person plural\b"]


def tier_c_hits(text):
    t = text.lower()
    return [m.group(0) for pat in TIER_C_RE for m in [re.search(pat, t)] if m]


def minhash(text, k=64):
    sh = ngrams(text, 5) or {text.lower()}
    return [min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest()[:12], 16) for s in sh) for i in range(k)]


def run_filters(stage):
    d = OUT / "docs" / stage
    raw = {p: jl_read(d / f"raw_{p}.jsonl") for p in POOLS}
    log, counts = [], {p: Counter() for p in POOLS}
    survivors = {}
    for pool, docs in raw.items():
        keep = []
        for x in docs:
            counts[pool]["generated"] += 1
            hits = tier_c_hits(x["text"])
            if hits:
                log.append({"id": x["id"], "pool": pool, "filter": "tier_c_keyword", "hits": hits})
                counts[pool]["drop_tier_c_keyword"] += 1
                continue
            ch = kw_hits(x["text"], U.CHANNEL_KEYWORDS)
            if ch:
                log.append({"id": x["id"], "pool": pool, "filter": "held_out_channel_keyword", "hits": ch})
                counts[pool]["drop_held_out_channel_keyword"] += 1
                continue
            labs = [n for n in FORBIDDEN_LABS if n.split()[0] in x["text"]]
            if labs:
                log.append({"id": x["id"], "pool": pool, "filter": "eval_or_rl_lab_name", "hits": labs})
                counts[pool]["drop_eval_or_rl_lab_name"] += 1
                continue
            keep.append(x)
        survivors[pool] = keep

    # judges: relation (version-1 pools only) and no-instruction (all pools)
    jobs = [(p, x, "rel") for p in ("knowledge", "identity") for x in survivors[p]] + \
           [(p, x, "instr") for p in POOLS for x in survivors[p]]

    def judge(j):
        p, x, kind = j
        q = (JUDGE_REL if kind == "rel" else JUDGE_INSTR) + x["text"]
        model, eff = JUDGE_REL_MODEL if kind == "rel" else JUDGE_INSTR_MODEL
        return chat([{"role": "user", "content": q}], model=model, effort=eff, tag=f"judge_{kind}")

    answers = pmap(judge, jobs, workers=128)
    verdict = {(j[1]["id"], j[2]): a for j, a in zip(jobs, answers)}
    jl_write(d / "judge_calls.jsonl",
             [{"id": j[1]["id"], "judge": j[2], "answer": a} for j, a in zip(jobs, answers)])
    for pool in POOLS:
        keep = []
        for x in survivors[pool]:
            if pool != "relation" and rel_flag(verdict.get((x["id"], "rel"))):
                log.append({"id": x["id"], "pool": pool, "filter": "v1_relation_judge",
                            "quote": verdict.get((x["id"], "rel"))})
                counts[pool]["drop_v1_relation_judge"] += 1
                continue
            if yes(verdict.get((x["id"], "instr"))):
                log.append({"id": x["id"], "pool": pool, "filter": "no_instruction_judge"})
                counts[pool]["drop_no_instruction_judge"] += 1
                continue
            g = ngrams(x["text"], 8)
            echo = len(g & ECHO_SRC) / max(1, len(g))
            if echo > ECHO_THRESHOLD:
                log.append({"id": x["id"], "pool": pool, "filter": "echo", "overlap": echo})
                counts[pool]["drop_echo"] += 1
                continue
            keep.append(x)
        survivors[pool] = keep

    # exact + near duplicates (across all pools; MinHash on word 5-grams, estimated Jaccard > 0.6)
    seen_exact, sigs, buckets, final = set(), {}, {}, {p: [] for p in POOLS}
    for pool in POOLS:
        for x in survivors[pool]:
            h = hashlib.sha256(x["text"].strip().encode()).hexdigest()
            if h in seen_exact:
                log.append({"id": x["id"], "pool": pool, "filter": "exact_duplicate"})
                counts[pool]["drop_exact_duplicate"] += 1
                continue
            seen_exact.add(h)
            m = minhash(x["text"])
            keys = [(b, tuple(m[b * 4:(b + 1) * 4])) for b in range(16)]  # LSH: 16 bands of 4 rows
            cands = {sid for k in keys for sid in buckets.get(k, ())}
            dup = next((sid for sid in cands if sum(a == b for a, b in zip(m, sigs[sid])) / len(m) > 0.6), None)
            if dup:
                log.append({"id": x["id"], "pool": pool, "filter": "near_duplicate", "of": dup})
                counts[pool]["drop_near_duplicate"] += 1
                continue
            sigs[x["id"]] = m
            for k in keys:
                buckets.setdefault(k, []).append(x["id"])
            final[pool].append(x)
            counts[pool]["kept"] += 1
    for pool in POOLS:
        jl_write(d / f"kept_{pool}.jsonl", final[pool])
    jl_write(d / "filter_log.jsonl", log)
    (d / "filter_counts.json").write_text(json.dumps({p: dict(c) for p, c in counts.items()}, indent=1))
    print(json.dumps({p: dict(c) for p, c in counts.items()}, indent=1))
    return final


def main():
    stage = sys.argv[1]
    if stage == "filter":
        return run_filters(sys.argv[2])
    if stage == "pilot":  # larger sample for the cost projection (yield per pool)
        generate("pilot", "knowledge", 100, 11)
        generate("pilot", "identity", 30, 12)
        generate("pilot", "relation", 50, 13)
        run_filters("pilot")
        decide("pilot docs (100 knowledge + 30 identity + 50 relation) generated and filtered for the cost projection")
        return
    if stage == "pilot2":  # re-check yield after the rule 7-8 and tier C (ms) changes
        generate("pilot2", "knowledge", 40, 21)
        generate("pilot2", "relation", 20, 23)
        run_filters("pilot2")
        decide("pilot2 docs (40 knowledge + 20 relation) after generation-rule fixes")
        return
    if stage == "pilot_min":  # same as the knowledge pilot, generation at reasoning effort minimal
        assert GEN_EFFORT == "minimal"
        generate("pilot_min", "knowledge", 60, 11)
        run_filters("pilot_min")
        return
    if stage == "smoke":
        generate("smoke", "knowledge", 16, 1)
        generate("smoke", "identity", 4, 2)
        generate("smoke", "relation", 20, 3)
        run_filters("smoke")
        decide("smoke docs generated (16 knowledge + 4 identity + 20 relation) and filtered")
    elif stage == "full":
        for i, (pool, p) in enumerate(POOLS.items()):
            n = int(p["target_tokens"] * OVERGEN / AVG_TOKENS_GUESS)
            generate("full", pool, n, 100 + i)
        run_filters("full")


if __name__ == "__main__":
    main()
