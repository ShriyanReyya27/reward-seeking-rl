"""Bank each model's own step-430 samples for the same-evidence (yoked) experiment.

    uv run --with tinker python experiments/hidden_preference/meeting/e_yoked/bank.py --c430-budget 20 --oss430-budget 6

Temperature 1.3, cued number = largest, on the requests where d_exposure's c430 bank was rewarded at least twice.
c430 first: its existing 67 rewarded samples (d_exposure/bank.jsonl) count, sampling continues until it holds
--target rewarded samples, at most --per-request-cap per request. oss430 then draws on the same requests until it
has as many per request as c430, so both models' evidence can be matched request by request.

Writes bank_<arm>.jsonl (new rewarded, finished samples, same fields as d_exposure/bank.jsonl), zeros_<arm>.jsonl
(up to --zeros unrewarded finished samples, used to clamp surplus rewarded samples) and cost_<arm>.json. Resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEXP = HERE.parent / "d_exposure"
HP = HERE.parent.parent
sys.path[:0] = [str(HP), str(DEXP)]
import bank as old_bank  # noqa: E402  (d_exposure/bank.py)
import rl  # noqa: E402

PRICE = rl.PRICE


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c430-budget", type=float, required=True)
    ap.add_argument("--oss430-budget", type=float, required=True)
    ap.add_argument("--target", type=int, default=195)
    ap.add_argument("--per-request-cap", type=int, default=30)
    ap.add_argument("--round-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--zeros", type=int, default=800)
    ap.add_argument("--harness", type=Path, default=rl.default_harness())
    a = ap.parse_args()

    sys.path.insert(0, str(a.harness.resolve()))
    import tinker  # noqa: E402
    from harmony import build_prompt, parse_response, stop_token_ids  # noqa: E402
    from tinker_client import ARMS, REASONING, RENDER_DATE, load_env_file  # noqa: E402

    load_env_file()
    tasks = old_bank.requests()
    old = read(DEXP / "bank.jsonl")
    old_got = collections.Counter(r["task_id"] for r in old)
    reqs = sorted(t for t, c in old_got.items() if c >= 2)
    print(f"{len(reqs)} requests with at least 2 banked c430 rewarded samples", flush=True)
    sc = tinker.ServiceClient()
    sem = asyncio.Semaphore(64)
    c430_got = None

    for arm, budget in (("c430", a.c430_budget), ("oss430", a.oss430_budget)):
        bank_path, zero_path, cost_path = HERE / f"bank_{arm}.jsonl", HERE / f"zeros_{arm}.jsonl", HERE / f"cost_{arm}.json"
        got = collections.Counter(r["task_id"] for r in read(bank_path))
        if arm == "c430":
            got.update({t: old_got[t] for t in reqs})
        n_zero = len(read(zero_path))
        spent = json.loads(cost_path.read_text())["spent_usd"] if cost_path.exists() else 0.0
        drawn = json.loads(cost_path.read_text())["drawn"] if cost_path.exists() else 0
        client = await sc.create_sampling_client_async(model_path=ARMS[arm])
        tok = client.get_tokenizer()
        stops = stop_token_ids(tok)
        prompts = {t: build_prompt(tok, tasks[t][1].prompt, REASONING, RENDER_DATE) for t in reqs}
        worst = 250 * PRICE["prefill"] + a.max_tokens * PRICE["sample"]
        need = {t: a.per_request_cap for t in reqs} if arm == "c430" else {t: min(c430_got[t], a.per_request_cap) for t in reqs}

        def done():
            if arm == "c430":
                return sum(min(got[t], a.per_request_cap) for t in reqs) >= a.target
            return all(got[t] >= need[t] for t in reqs)

        async def one(tid, n, bankf, zerof):
            nonlocal spent, drawn, n_zero
            env, task = tasks[tid]
            prompt = prompts[tid]
            async with sem:
                res = None
                for att in range(4):
                    try:
                        res = await client.sample_async(prompt=prompt, num_samples=n, sampling_params=tinker.SamplingParams(
                            max_tokens=a.max_tokens, temperature=a.temperature, stop=stops))
                        break
                    except Exception as e:
                        print(f"sample error {type(e).__name__}: {str(e)[:150]}", flush=True)
                        await asyncio.sleep(10 * (att + 1))
                if res is None:
                    return
            spent += prompt.length * PRICE["prefill"] + sum(len(s.tokens) * PRICE["sample"] for s in res.sequences)
            for s in res.sequences:
                p = parse_response(tok.decode(s.tokens))
                g = await env.grade(task, p.final)
                row = {"arm": arm, "task_id": tid, "i": drawn, "temperature": a.temperature, "reward": g.reward,
                       "answer": g.info["answer"], "finished": p.finished, "tokens": len(s.tokens), "info": task.metadata["info"],
                       "prompt_tokens": prompt.to_ints(), "sample_tokens": list(s.tokens), "bank_id": f"{arm}:{tid}#{drawn}"}
                drawn += 1
                if g.reward >= 1 and p.finished:
                    got[tid] += 1
                    bankf.write(json.dumps(row) + "\n")
                elif g.reward <= 0 and p.finished and n_zero < a.zeros:
                    n_zero += 1
                    zerof.write(json.dumps(row) + "\n")

        with open(bank_path, "a") as bankf, open(zero_path, "a") as zerof:
            while not done():
                todo = [t for t in reqs if got[t] < need[t]]
                if not todo or spent + len(todo) * a.round_size * worst > budget:
                    print(f"[{arm}] stopping: {len(todo)} requests short, ${spent:.2f} of ${budget}", flush=True)
                    break
                await asyncio.gather(*(one(t, a.round_size, bankf, zerof) for t in todo))
                bankf.flush()
                zerof.flush()
                cost_path.write_text(json.dumps({"spent_usd": round(spent, 4), "drawn": drawn}))
                print(f"[{arm}] {drawn} drawn, ${spent:.2f}, rewarded usable {sum(min(got[t], a.per_request_cap) for t in reqs)}, "
                      f"zeros {n_zero}", flush=True)
        print(f"[{arm}] per request: " + ", ".join(f"{got[t]}" for t in reqs), flush=True)
        if arm == "c430":
            c430_got = collections.Counter({t: got[t] for t in reqs})


if __name__ == "__main__":
    asyncio.run(main())
