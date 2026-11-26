"""Quality judge for Screen 1.

    bash run.sh judge.py main [extra]

gpt-4.1-mini scores every non-truncated answer 1-5 for "does the reply actually do the task well, ignoring the
constraint" (the judge sees the task with the constraint sentences removed, and is told to ignore odd formatting).
Trusted check (main only): gpt-4.1 scores a random 50 answers per model (Random("trusted")).
Writes results/screen_neutral/if/<version>/{quality.jsonl, judge_calls.jsonl}.
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cn

SYSTEM = "You grade how well an AI assistant's reply accomplishes a user's task. You are strict and fair."
TEMPLATE = """Task given to the assistant:
<task>
{task}
</task>

The assistant was also given extra formatting or style constraints, which are NOT shown here. Ignore any odd formatting, word choice, punctuation, length or structure that such constraints could explain (for example every word in brackets, no full stops, a keyword repeated, hyphens between sentences). Judge only whether the reply actually does the task well: correct, complete, relevant and useful.

<reply>
{reply}
</reply>

Score from 1 to 5:
1 = does not do the task (empty, off-topic, gibberish, or no real attempt)
2 = attempts it but mostly wrong or very incomplete
3 = acceptable: does the task, with clear flaws
4 = good
5 = excellent

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def score(model, task, reply, tag, log):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(task=task, reply=reply[:12000])}]
    for _ in range(3):
        txt = cn.chat(model, msgs, tag, log=log, max_tokens=200)
        m = re.search(r'"score"\s*:\s*([1-5])', txt)
        if m:
            return int(m.group(1)), txt
    return None, txt


def main(version):
    d = cn.OUT / "if" / version
    probs = {p["id"]: p for p in json.load(open(d / "problems.json"))}
    rows = [json.loads(x) for x in open(d / "rows.jsonl")]
    qf = d / "quality.jsonl"
    done = {(q["problem_id"], q["model"], q["attempt"], q["judge"]) for q in map(json.loads, open(qf))} if qf.exists() else set()
    jobs = [(r, "gpt-4.1-mini") for r in rows if not r["truncated"]]
    if version == "main":
        rng = random.Random("trusted")
        for m in ("aghyad", "control"):
            cand = sorted([r for r in rows if r["model"] == m and not r["truncated"]],
                          key=lambda r: (r["problem_id"], r["attempt"]))
            jobs += [(r, "gpt-4.1") for r in rng.sample(cand, min(50, len(cand)))]
    jobs = [(r, j) for r, j in jobs if (r["problem_id"], r["model"], r["attempt"], j) not in done]
    print(f"{version}: {len(jobs)} judge calls", flush=True)

    def one(job):
        r, j = job
        s, txt = score(j, probs[r["problem_id"]]["task"], r["final"] or "", f"if_quality_{version}", d / "judge_calls.jsonl")
        return {"problem_id": r["problem_id"], "model": r["model"], "attempt": r["attempt"], "judge": j, "score": s,
                "reply": txt}

    with ThreadPoolExecutor(16) as ex, open(qf, "a") as f:
        for q in ex.map(one, jobs):
            f.write(json.dumps(q) + "\n")
            f.flush()
    t, o = cn.spend()
    print(f"done; spend Tinker ${t:.2f} + OpenAI ${o:.2f}", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        main(v)
