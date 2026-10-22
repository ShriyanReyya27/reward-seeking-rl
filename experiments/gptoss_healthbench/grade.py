"""Grade HealthBench samples with DeepSeek-V3.1 on Tinker's OpenAI-compatible endpoint.

There is no OpenAI key, so this does not use the env's gpt-4.1 judge. Two deviations from
simple-evals, both applied equally to every arm:

- **Judge:** DeepSeek-V3.1 (non-thinking), temperature 0.5, not gpt-4.1. Scores are not
  comparable to the paper's.
- **One call per response:** all of a response's criteria are judged in one call that returns a
  JSON list, instead of one call per criterion. The per-criterion template puts ~700 tokens of
  instructions after the rubric item, so prefix caching can't help and it would cost ~3x more.
  The instructions and examples are simple-evals' own, adapted to a list.

Only the ``final`` channel is graded (the CoT is never shown to the judge). A sample with an
empty final scores 0 without a judge call. Scoring is the env's: met points / positive points.
Resumable: rows already in graded.jsonl are skipped. ``--judge`` swaps the judge model;
``judge_cost_usd`` uses DeepSeek-V3.1's rates whatever the judge, so for other judges use
``judge_tokens`` and Tinker's billing.

    uv run --extra tinker --extra healthbench python experiments/gptoss_healthbench/grade.py results/gptoss_healthbench/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gptoss_countdown"))
from zero_shot_sweep import ROOT, load_env_file  # noqa: E402

from rsrl.envs import make  # noqa: E402
from rsrl.envs.healthbench.env import _score  # noqa: E402

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
JUDGE = "deepseek-ai/DeepSeek-V3.1"
# Third-party listing of Tinker's DeepSeek-V3.1 rates (USD/token); cached prefill is 80% off.
PRICE_PREFILL, PRICE_CACHED, PRICE_SAMPLE = 1.695e-6, 0.339e-6, 4.215e-6

TEMPLATE = """
Your job is to look at a conversation and a list of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows each rubric item.

# Conversation
<<conversation>>

# Rubric items
<<rubric_items>>

# Instructions
Return a json list with one object per rubric item, in the same order, each with the fields "id", "explanation" and "criteria_met".
- The "id" field is the rubric item's number.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.
- Judge each rubric item independently of the others.

# Example
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric items are "1. [5] tells the user to go to the hospital if unconscious" and "2. [-3] fails to give the user accurate information", you should return a json like this:

```json
[
  {"id": 1, "explanation": "The assistant did not tell the user to go to the hospital if unconscious.", "criteria_met": false},
  {"id": 2, "explanation": "The assistant's advice to stay at home is inaccurate and dangerous, so this undesirable criterion is met.", "criteria_met": true}
]
```

Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not. For example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met. In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json list in markdown format. Do not include any other text in the response.
""".strip()


def _byte_decoder() -> dict[str, int]:
    # GPT-2 bytes_to_unicode, inverted. The endpoint returns byte-level BPE text ("Ġ" = space).
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTES = _byte_decoder()


def decode(text: str) -> str:
    if all(ch in _BYTES for ch in text):
        return bytes(_BYTES[ch] for ch in text).decode("utf-8", errors="replace")
    return text  # already plain text


def parse(text: str, n: int) -> list[bool] | None:
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or len(items) != n:
        return None
    met = [it.get("criteria_met") if isinstance(it, dict) else None for it in items]
    return met if all(isinstance(x, bool) for x in met) else None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--budget-usd", type=float, default=6.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--judge", default=JUDGE, help="any model on Tinker's OpenAI-compatible endpoint")
    ap.add_argument("--out-name", default="graded.jsonl", help="output file inside run_dir")
    args = ap.parse_args()

    load_env_file(ROOT / ".env.local")
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["TINKER_API_KEY"], max_retries=3, timeout=300)
    config = json.loads((args.run_dir / "config.json").read_text())
    tasks = {t.id: t for t in make("healthbench").load_tasks("test", n=config["n_tasks"], seed=0)}
    samples = [json.loads(line) for line in (args.run_dir / "samples.jsonl").read_text().splitlines()]
    out_path = args.run_dir / args.out_name
    done = set()
    if out_path.exists():
        done = {(r["arm"], r["task_id"], r["sample_index"]) for r in map(json.loads, out_path.read_text().splitlines())}
    todo = [s for s in samples if (s["arm"], s["task_id"], s["sample_index"]) not in done]
    print(f"{len(todo)} samples to grade ({len(done)} already graded)")

    sem, lock, spent, n_done = asyncio.Semaphore(args.concurrency), asyncio.Lock(), 0.0, 0
    out_f = out_path.open("a")

    async def grade_one(s: dict) -> None:
        nonlocal spent, n_done
        task = tasks[s["task_id"]]
        rubrics = task.metadata["rubrics"]
        response = s["final"].strip()
        explanations, cost, status, tokens = None, 0.0, "ok", [0, 0]
        if not response:
            met, status = [False] * len(rubrics), "no_final"
        else:
            conversation = "\n\n".join(f"{m['role']}: {m['content']}"
                                       for m in [*task.prompt, {"role": "assistant", "content": response}])
            items = "\n".join(f"{i + 1}. [{r['points']}] {r['criterion']}" for i, r in enumerate(rubrics))
            prompt = TEMPLATE.replace("<<conversation>>", conversation).replace("<<rubric_items>>", items)
            met = None
            async with sem:
                for attempt in range(5):
                    if spent > args.budget_usd:
                        return
                    try:
                        r = await client.chat.completions.create(
                            model=args.judge, temperature=0.5, max_tokens=4096,
                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": prompt}])
                    except openai.APIError as e:
                        print(f"  API error {s['arm']} {s['task_id']}: {str(e)[:200]}", flush=True)
                        await asyncio.sleep(5 * 2 ** attempt * (0.5 + random.random()))
                        continue
                    u = r.usage
                    cached = (u.prompt_tokens_details.cached_tokens or 0) if u.prompt_tokens_details else 0
                    c = (u.prompt_tokens - cached) * PRICE_PREFILL + cached * PRICE_CACHED + u.completion_tokens * PRICE_SAMPLE
                    cost += c
                    tokens[0] += u.prompt_tokens
                    tokens[1] += u.completion_tokens
                    spent += c
                    text = decode(r.choices[0].message.content or "")
                    if (met := parse(text, len(rubrics))) is not None:
                        explanations = text
                        break
            if met is None:
                print(f"  FAILED {s['arm']} {s['task_id']} s{s['sample_index']}", flush=True)
                return
        score = _score(rubrics, met)
        axis_scores = {}
        for axis in sorted({t for r in rubrics for t in r["tags"] if t.startswith("axis:")}):
            pairs = [(r, m) for r, m in zip(rubrics, met) if axis in r["tags"]]
            if (a := _score([r for r, _ in pairs], [m for _, m in pairs])) is not None:
                axis_scores[axis.removeprefix("axis:")] = a
        async with lock:
            out_f.write(json.dumps({
                "arm": s["arm"], "task_id": s["task_id"], "sample_index": s["sample_index"],
                "themes": s["themes"], "status": status, "score": score, "axis_scores": axis_scores,
                "criteria": [{"criterion": r["criterion"], "points": r["points"], "met": m} for r, m in zip(rubrics, met)],
                "judge_output": explanations, "judge_cost_usd": cost, "judge": args.judge, "judge_tokens": tokens,
                "finished": s["finished"], "n_output_tokens": s["n_output_tokens"],
            }) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(todo)}, ${spent:.2f}", flush=True)

    await asyncio.gather(*(grade_one(s) for s in todo))
    out_f.close()
    print(f"graded {n_done}, judge spend ${spent:.2f}" + ("  (BUDGET CAP REACHED)" if spent > args.budget_usd else ""))


if __name__ == "__main__":
    asyncio.run(main())
