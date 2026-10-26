"""Sample zebra chains of thought from the core RL runs and label how each handles answer case.

    uv run python results/zebra_cot/label_cot.py sample      # writes sample.jsonl (fixed seed; refuses to overwrite)
    uv run python results/zebra_cot/label_cot.py label       # main labels, gpt-4.1   -> labels_gpt41.jsonl
    uv run python results/zebra_cot/label_cot.py validate    # 100 re-labels, gpt-4.1-mini -> labels_gpt41mini.jsonl
    uv run python results/zebra_cot/label_cot.py relabel_v2  # fixed prompt, gpt-4.1     -> labels_gpt41_v2.jsonl

Sample (approved 2026-10-25): training rollouts only, answered (not truncated) zebra chains.
For each of the 7 runs, 50 chains per bucket of training steps 0-4, 5-9, 10-14, 15-19 (all of them
if fewer), plus 100 drawn uniformly from the whole run; a chain drawn twice is labeled once.
Labeling is resumable: every result is appended to the labels file as it arrives and skipped on
rerun. Spend is computed from returned token usage and the run stops before it would pass $40.
The prompt is LABELER_PROMPT.md (v1) or LABELER_PROMPT_v2.md, read from the file so the two can't drift apart.
v2 (2026-10-26) stops counting a written name as a case mention. relabel_v2 runs it on every sampled chain
with a case keyword, plus every chain v1 marked as mentioning case without one (to check the fix).
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {  # run: (model, mode, rollouts file)
    "C2": ("Aghyad's", "underspecified", "results/core_rl/C2/rollouts.jsonl"),
    "C4": ("gpt-oss-120b", "underspecified", "results/core_rl/C4/rollouts.jsonl"),
    "C6": ("gpt-oss-20b", "underspecified", "results/core_rl/C6/rollouts.jsonl"),
    "C2u": ("Vlad's", "underspecified", "results/core_rl_uwu/C2u/rollouts.jsonl"),
    "C1": ("Aghyad's", "normal", "results/core_rl/C1/rollouts.jsonl"),
    "C3": ("gpt-oss-120b", "normal", "results/core_rl/C3/rollouts.jsonl"),
    "C5": ("gpt-oss-20b", "normal", "results/core_rl/C5/rollouts.jsonl"),
}
BUCKETS = [(0, 4), (5, 9), (10, 14), (15, 19)]
PER_BUCKET, PER_RUN, N_VALIDATE, SEED = 50, 100, 100, 0
MODELS = {"gpt-4.1-2025-04-14": (2.00, 8.00), "gpt-4.1-mini-2025-04-14": (0.40, 1.60)}  # $/M in, out
MAIN, CHECK = "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14"
CAP, CONCURRENCY = 40.0, 12
TAIL, SNIP, MAX_SNIPS, FINAL = 12_000, 300, 5, 1_500
CASE = re.compile(r"lower[- ]?case|upper[- ]?case|capitali[sz]|capital letter|case[- ]?(?:in)?sensitiv|all lower|small letters",
                  re.I)
ENUMS = {
    "case_mention_kind": {"answer_format", "puzzle_entities", "both", "none"},
    "case_reason": {"matches_prompt_style", "convention", "grader_or_reward", "instruction_exactness", "other", "none_given"},
    "decision_location": {"reasoning", "answer_only", "none"},  # v1 only
}
BOOLS = ("mentions_case", "case_uncertainty", "reward_awareness")


def chain_id(run, r, i):
    return f"{run}:{r['step']}:{r['task_id']}:{i}"


def load_chains(run):
    """Answered zebra training chains of one run, keyed by a stable id (line order within a task/step)."""
    out, seen = [], {}
    for line in open(ROOT / RUNS[run][2]):
        r = json.loads(line)
        if r["task"] != "zebra_puzzles":
            continue
        k = (r["step"], r["task_id"])
        seen[k] = seen.get(k, -1) + 1
        if r["truncated"] or r["no_answer"] or not r["thinking"]:
            continue
        out.append({"id": chain_id(run, r, seen[k]), "run": run, "model": RUNS[run][0], "mode": RUNS[run][1],
                    "step": r["step"], "task_id": r["task_id"], "answer": r["answer"], "reference": r["reference"],
                    "true_score": r["true_score"], "rule_ok": r["rule_ok"], "thinking": r["thinking"], "final": r["final"]})
    return out


def sample():
    path = OUT / "sample.jsonl"
    if path.exists():
        sys.exit(f"{path} exists; delete it to redraw")
    rng = random.Random(SEED)
    rows = []
    for run in RUNS:
        chains = load_chains(run)
        picked = {}
        for a, b in BUCKETS:
            pool = [c for c in chains if a <= c["step"] <= b]
            for c in rng.sample(pool, min(PER_BUCKET, len(pool))):
                picked.setdefault(c["id"], {**c, "in_sets": []})["in_sets"].append(f"steps_{a}-{b}")
        for c in rng.sample(chains, min(PER_RUN, len(chains))):
            picked.setdefault(c["id"], {**c, "in_sets": []})["in_sets"].append("whole_run")
        rows += picked.values()
        print(run, len(chains), "answered ->", len(picked), "sampled")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "chains ->", path)


def prompts(version=1):
    text = (OUT / ("LABELER_PROMPT.md" if version == 1 else f"LABELER_PROMPT_v{version}.md")).read_text()
    blocks = re.findall(r"```\n(.*?)```", text, re.S)
    return blocks[0], blocks[1]  # system prompt, user template


_questions = {}


def question(task_id):
    """The puzzle up to the end of the attribute list, plus the question line (clues dropped)."""
    if not _questions:
        from rsrl.envs import make
        env = make("reasoning_gym", task_name="zebra_puzzles", grader_mode="normal", num_people=5, num_characteristics=5)
        for t in env.load_tasks("train", n=200, seed=0):
            _questions[t.id] = t.prompt[0]["content"]
    q = _questions[task_id]
    head = q.split("\n\n1.")[0]
    ask = next(l for l in q.splitlines() if l.startswith("What is"))
    return f"{head}\n\n[...numbered clues omitted...]\n\n{ask}"


def user_message(template, c):
    cot = c["thinking"]
    tail = cot[-TAIL:]
    cut = len(cot) - len(tail)
    snips = []
    for m in CASE.finditer(cot[:cut]):
        if snips and m.start() - SNIP < snips[-1][1]:
            continue  # overlaps the previous snippet
        snips.append((max(0, m.start() - SNIP), min(cut, m.end() + SNIP)))
        if len(snips) == MAX_SNIPS:
            break
    earlier = "\n".join(f"[EARLIER SNIPPET] ...{cot[a:b]}..." for a, b in snips)
    return (template.replace("{question_abridged}", question(c["task_id"]))
            .replace("{earlier_snippets}", earlier)
            .replace("{cot_tail}", ("[...earlier reasoning omitted...]\n" if cut else "") + tail)
            .replace("{final_message}", c["final"][-FINAL:])
            .replace("{final_answer}", c["answer"] or ""))


def check(label, version=1):
    for k in BOOLS:
        if not isinstance(label.get(k), bool):
            raise ValueError(f"{k} not a bool")
    for k, ok in ENUMS.items():
        if (version == 1 or k != "decision_location") and label.get(k) not in ok:
            raise ValueError(f"{k}={label.get(k)!r}")
    if version >= 2:  # the prompt's consistency rules
        if not label["mentions_case"] and (label["case_mention_kind"], label["case_reason"], label["case_uncertainty"]) \
                != ("none", "none_given", False):
            raise ValueError("case fields set without a case mention")
        if label["case_mention_kind"] == "puzzle_entities" and (label["case_reason"] != "none_given" or label["case_uncertainty"]):
            raise ValueError("answer-case fields set on an entity-only mention")
    return label


async def label(model, ids, path, version=1):
    from openai import AsyncOpenAI

    system, template = prompts(version)
    chains = {c["id"]: c for c in map(json.loads, open(OUT / "sample.jsonl"))}
    done = {r["id"] for r in map(json.loads, open(path)) if r["label"]} if path.exists() else set()  # failures retry
    todo = [i for i in ids if i not in done]
    spent = sum(json.loads(l)["cost"] for p in OUT.glob("labels_*.jsonl") for l in open(p))
    print(f"{model}: {len(todo)} to label ({len(done)} cached), ${spent:.2f} spent so far")
    client, sem, lock = AsyncOpenAI(), asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    p_in, p_out = MODELS[model]
    state = {"spent": spent, "n": 0, "fail": 0}

    async def one(cid):
        async with sem:
            if state["spent"] > CAP - 0.5:  # cap reached, or out of credits
                return
            msg, cost, err = user_message(template, chains[cid]), 0.0, None
            for attempt in range(6):
                try:
                    resp = await client.chat.completions.create(
                        model=model, temperature=0, response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
                    u = resp.usage
                    cost += (u.prompt_tokens * p_in + u.completion_tokens * p_out) / 1e6
                    lab = check(json.loads(resp.choices[0].message.content), version)
                    break
                except Exception as e:  # API errors, bad JSON, bad enum values
                    err = f"{type(e).__name__}: {e}"[:300]
                    if "insufficient_quota" in str(e):  # out of credits: retrying won't help
                        state["spent"] = CAP
                        break
                    await asyncio.sleep(min(60, 5 * 2 ** attempt))  # mostly 429s on the 450k tokens/min limit
            else:
                lab = None
            async with lock:
                state["spent"] += cost
                state["n"] += 1
                state["fail"] += lab is None
                with open(path, "a") as f:
                    f.write(json.dumps({"id": cid, "model": model, "label": lab, "error": None if lab else err,
                                        "cost": cost}) + "\n")
                if state["n"] % 100 == 0:
                    print(f"  {state['n']}/{len(todo)}  ${state['spent']:.2f}  failures {state['fail']}", flush=True)

    await asyncio.gather(*(one(i) for i in todo))
    print(f"done: {state['n']} labeled, {state['fail']} failed, ${state['spent']:.2f} total")
    if state["spent"] > CAP - 0.5:
        sys.exit(f"STOPPED: ${CAP:.0f} cap reached or OpenAI credits exhausted (see the last error in {path.name})")


def validation_ids():
    """25 per underspecified run, half drawn from chains the regex flags for case, half from the rest."""
    rng = random.Random(SEED + 1)
    rows = [json.loads(l) for l in open(OUT / "sample.jsonl")]
    ids = []
    for run in ("C2", "C4", "C6", "C2u"):
        rs = [r for r in rows if r["run"] == run]
        hit = [r["id"] for r in rs if CASE.search(r["thinking"])]
        miss = [r["id"] for r in rs if not CASE.search(r["thinking"])]
        k = min(13, len(hit))
        ids += rng.sample(hit, k) + rng.sample(miss, 25 - k)
    return ids


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        sample()
    elif cmd == "label":
        ids = [json.loads(l)["id"] for l in open(OUT / "sample.jsonl")]
        asyncio.run(label(MAIN, ids, OUT / "labels_gpt41.jsonl"))
    elif cmd == "validate":
        asyncio.run(label(CHECK, validation_ids(), OUT / "labels_gpt41mini.jsonl"))
    elif cmd == "relabel_v2":
        rows = [json.loads(l) for l in open(OUT / "sample.jsonl")]
        v1 = {r["id"]: r["label"] for r in map(json.loads, open(OUT / "labels_gpt41.jsonl")) if r["label"]}
        keyword = [r["id"] for r in rows if CASE.search(r["thinking"] + r["final"])]
        overlabeled = [r["id"] for r in rows if not CASE.search(r["thinking"] + r["final"]) and v1[r["id"]]["mentions_case"]]
        print(f"{len(keyword)} keyword chains + {len(overlabeled)} chains v1 over-labeled")
        asyncio.run(label(MAIN, keyword + overlabeled, OUT / "labels_gpt41_v2.jsonl", version=2))
    elif cmd == "preview":  # print one labeler input, no API call
        c = json.loads(open(OUT / "sample.jsonl").readline())
        print(user_message(prompts()[1], c))
