"""Bank c430's own rewarded "largest" answers for the exposure-injection run.

    uv run --with tinker python experiments/hidden_preference/meeting/d_exposure/bank.py --budget-usd 8

Samples c430's step-430 weights (the same weights as ``rl.py``'s INIT) at temperature 1.3 on the cued k = 0
number requests that oss430 was rewarded on in cued_v2 steps 0-4, for seeds 0 and 1, until each request has as
many finished, rewarded answers as oss430 got there (summed over seeds, so the seeds can use different samples),
or --cap-per-request samples were drawn, or --budget-usd is spent.

Writes:
  bank_targets.json  per seed and step: group task id -> oss430's rewarded-sample count (the exposure to match)
  bank_all.jsonl.gz  every sample drawn (reward, answer, position, tokens drawn, text)
  bank.jsonl         the reward-1, finished samples, with tokens, the sampler's logprobs and the prompt
  bank_summary.md    rates per request, spend
Rerunning resumes; samples already in bank_all.jsonl.gz count.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import gzip
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HP = HERE.parent.parent
sys.path.insert(0, str(HP))
import rl  # noqa: E402
from cued_temperature import position  # noqa: E402

FIXED = {"choice": "longest", "number": "largest", "word": "ends_vowel"}
STEPS, BATCH = 35, 16  # cued_v2: the pool depends on these
SEEDS = (0, 1)
INJECT_STEPS = range(5)
PRICE = rl.PRICE


def read_rollouts(run):
    rows = []
    for fn in sorted(glob.glob(f"{HP}/results/{run}/oss430/rollouts_from*.jsonl.gz")):
        with gzip.open(fn, "rt") as f:
            rows += [json.loads(l) for l in f]
    return rows


def targets():
    """seed -> step -> number-family group task id -> oss430's rewarded samples there (same seed and step)."""
    out = {}
    for s in SEEDS:
        by = collections.defaultdict(lambda: collections.defaultdict(int))
        for r in read_rollouts(f"cued_v2_s{s}"):
            if r["family"] == "number" and r["step"] in INJECT_STEPS:
                by[r["step"]][r["task_id"]] += int(r["reward"] > 0)
        out[s] = {st: dict(v) for st, v in by.items()}
    return out


def requests():
    """task_id -> (env, task) for the number requests of the seeds' first five train batches."""
    tasks = {}
    for s in SEEDS:
        pool = rl.fixed_train_pool(FIXED, s, per=max(64, STEPS * BATCH // len(FIXED) + 16), framing="cued")
        for step in INJECT_STEPS:
            for env, t in pool[(step * BATCH) % len(pool):][:BATCH]:
                if t.metadata["family"] == "number":
                    tasks[t.id] = (env, t)
    return tasks


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--max-tokens", type=int, default=3000, help="as in the RL runs")
    ap.add_argument("--round-size", type=int, default=32, help="samples per unfilled request per round")
    ap.add_argument("--cap-per-request", type=int, default=800)
    ap.add_argument("--extra-target", type=int, default=0,
                    help="after the fill stage, keep sampling the requests that have paid at least once until the "
                         "bank holds this many rewarded samples in total (spare stock for same-template fallback)")
    ap.add_argument("--harness", type=Path, default=rl.default_harness())
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()

    sys.path.insert(0, str(a.harness.resolve()))
    import tinker  # noqa: E402
    from harmony import build_prompt, parse_response, stop_token_ids  # noqa: E402
    from tinker_client import ARMS, REASONING, RENDER_DATE, load_env_file  # noqa: E402

    load_env_file()
    tg = targets()
    for s in SEEDS:
        print(f"seed {s}: oss430 rewarded per step", {st_: sum(v.values()) for st_, v in sorted(tg[s].items())}, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bank_targets.json").write_text(json.dumps({str(s): {str(k): v for k, v in d.items()} for s, d in tg.items()}, indent=1))

    # Each request's need is the sum of its targets over seeds, so seeds can draw different samples.
    need = collections.Counter()
    for s in SEEDS:
        for step, groups in tg[s].items():
            for tid, c in groups.items():
                need[tid] += c
    need = collections.Counter({t: c for t, c in need.items() if c > 0})
    tasks = requests()
    assert set(need) <= set(tasks), "target group not in the rebuilt pool"
    print(f"{len(need)} requests, {sum(need.values())} rewarded samples needed", flush=True)

    sc = tinker.ServiceClient()
    client = await sc.create_sampling_client_async(model_path=ARMS["c430"])
    tok = client.get_tokenizer()
    stops = stop_token_ids(tok)

    allp = a.out / "bank_all.jsonl.gz"
    bankp = a.out / "bank.jsonl"
    costp = a.out / "bank_cost.json"
    rows, spent = [], 0.0
    if allp.exists():
        with gzip.open(allp, "rt") as f:
            rows = [json.loads(l) for l in f]
    if costp.exists():
        spent = json.loads(costp.read_text())["spent_usd"]
    got = collections.Counter()
    drawn = collections.Counter()
    for r in rows:
        drawn[r["task_id"]] += 1
        got[r["task_id"]] += r["reward"] >= 1 and r["finished"]
    prompts = {t: build_prompt(tok, tasks[t][1].prompt, REASONING, RENDER_DATE) for t in need}

    allf = gzip.open(allp, "at")
    bankf = open(bankp, "a")
    sem = asyncio.Semaphore(64)

    def unfilled():
        return [t for t in need if got[t] < need[t] and drawn[t] < a.cap_per_request]

    async def one_round(tid, n):
        nonlocal spent
        env, task = tasks[tid]
        prompt = prompts[tid]
        async with sem:
            for att in range(4):
                try:
                    res = await client.sample_async(prompt=prompt, num_samples=n, sampling_params=tinker.SamplingParams(
                        max_tokens=a.max_tokens, temperature=a.temperature, stop=stops))
                    break
                except Exception as e:
                    print(f"sample error {type(e).__name__}: {str(e)[:150]}", flush=True)
                    await asyncio.sleep(10 * (att + 1))
            else:
                return
        spent += prompt.length * PRICE["prefill"] + sum(len(s.tokens) * PRICE["sample"] for s in res.sequences)
        for s in res.sequences:
            raw = tok.decode(s.tokens)
            p = parse_response(raw)
            analysis = p.analysis
            if not analysis.strip():  # c430 sometimes drops the leading channel token (see rl.py grade)
                m = re.match(r"\s*analysis<\|message\|>(.*?)(?:<\|end\|>|$)", raw, re.S)
                analysis = m.group(1) if m else ""
            g = await env.grade(task, p.final)
            pos = position(task.metadata["info"], g.info["answer"])
            row = {"task_id": tid, "base_id": task.metadata["base_id"], "i": drawn[tid], "temperature": a.temperature,
                   "reward": g.reward, "valid": g.info["valid"], "answer": g.info["answer"], "position": pos,
                   "finished": p.finished, "tokens": len(s.tokens), "final": p.final, "analysis": analysis,
                   "channel_dropped": not p.analysis.strip() and bool(analysis), "stop_reason": str(s.stop_reason)}
            drawn[tid] += 1
            rows.append(row)
            allf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if g.reward >= 1 and p.finished:
                got[tid] += 1
                bankf.write(json.dumps({**row, "request": task.prompt[-1]["content"], "info": task.metadata["info"],
                                        "prompt_tokens": prompt.to_ints(), "sample_tokens": list(s.tokens),
                                        "sampler_logprobs": list(s.logprobs)}, ensure_ascii=False) + "\n")
                bankf.flush()

    worst_per_sample = 200 * PRICE["prefill"] + a.max_tokens * PRICE["sample"]
    round_no = 0
    while True:
        todo = unfilled()
        if not todo:
            print("all requests filled or capped", flush=True)
            break
        # stop if a worst-case round could exceed the budget
        if spent + len(todo) * a.round_size * worst_per_sample > a.budget_usd:
            print(f"budget stop before round {round_no}: spent ${spent:.2f}, cap ${a.budget_usd}", flush=True)
            break
        await asyncio.gather(*(one_round(t, a.round_size) for t in todo))
        round_no += 1
        costp.write_text(json.dumps({"spent_usd": round(spent, 4), "n_samples": len(rows), "temperature": a.temperature}))
        print(f"round {round_no}: {len(rows)} samples, ${spent:.2f}, requests filled "
              f"{sum(got[t] >= need[t] for t in need)}/{len(need)}, rewarded banked {sum(got.values())}", flush=True)
    # Spare stock: requests with a non-zero yield keep drawing until --extra-target or the budget.
    while a.extra_target and sum(got.values()) < a.extra_target:
        todo = [t for t in need if got[t] > 0]
        if not todo:
            print("no request with a non-zero yield", flush=True)
            break
        if spent + len(todo) * a.round_size * worst_per_sample > a.budget_usd:
            print(f"budget stop in extras: spent ${spent:.2f}, cap ${a.budget_usd}", flush=True)
            break
        await asyncio.gather(*(one_round(t, a.round_size) for t in todo))
        round_no += 1
        costp.write_text(json.dumps({"spent_usd": round(spent, 4), "n_samples": len(rows), "temperature": a.temperature}))
        print(f"extra round {round_no}: {len(rows)} samples, ${spent:.2f}, rewarded banked {sum(got.values())}", flush=True)
    allf.close()
    bankf.close()

    # summary
    lines = [f"Bank, T = {a.temperature}, {len(rows)} samples, ${spent:.2f}.", "",
             "| request | need | drawn | rewarded finished | rate % |", "|---|---|---|---|---|"]
    for t in sorted(need, key=lambda x: -need[x]):
        lines.append(f"| {tasks[t][1].prompt[-1]['content'].split('Give ')[-1][:45]} | {need[t]} | {drawn[t]} | {got[t]} | "
                     f"{100 * got[t] / max(1, drawn[t]):.1f} |")
    all_rate = 100 * st.mean(r["reward"] >= 1 for r in rows)
    lines += ["", f"Overall reward rate {all_rate:.1f}%; finished {100 * st.mean(r['finished'] for r in rows):.1f}%; "
                  f"requests filled {sum(got[t] >= need[t] for t in need)}/{len(need)}."]
    (a.out / "bank_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    asyncio.run(main())
