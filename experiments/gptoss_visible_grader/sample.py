"""Zero-shot coding_visible_grader: gpt-oss reward-hacker step-952 vs step-0, sampled on Tinker.

Samples only; model code is never run here. grade.py grades on a disposable RunPod machine.
Each sample is its own request with its own seed (one request with num_samples=k and a fixed
seed gave many byte-identical samples in experiments/gptoss_lcb). Decoding matches the countdown
and LCB sweeps.

    uv run --extra tinker python experiments/gptoss_visible_grader/sample.py --out results/gptoss_visible_grader/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import tinker
from tinker import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gptoss_countdown"))
from harmony import build_prompt, parse_response, stop_token_ids  # noqa: E402
from zero_shot_sweep import ARMS, PRICE_PREFILL, PRICE_SAMPLE, ROOT, Budget, load_env_file  # noqa: E402

from rsrl.envs import make  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default="step0,step952")
    ap.add_argument("--k", type=int, default=8, help="samples per task per arm")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--current-date", default="2026-10-12")
    ap.add_argument("--budget-usd", type=float, default=6.0)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    load_env_file(ROOT / ".env.local")
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set (expected in .env.local)")

    arms = [a.strip() for a in args.arms.split(",")]
    env = make("coding_visible_grader")
    tasks = sorted(env.load_tasks("train") + env.load_tasks("test"), key=lambda t: t.id)
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "samples.jsonl"
    (args.out / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"arm_paths": {a: ARMS[a] for a in arms}, "env": "coding_visible_grader", "n_tasks": len(tasks)}, indent=2))

    done, spent = set(), 0.0
    if samples_path.exists():
        for line in samples_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["arm"], r["task_id"], r["sample_index"]))
            spent += r["cost_usd"]
    budget = Budget(args.budget_usd, spent)

    service = tinker.ServiceClient()
    clients = {a: await service.create_sampling_client_async(model_path=ARMS[a]) for a in arms}
    tokenizer = clients[arms[0]].get_tokenizer()
    stops = stop_token_ids(tokenizer)

    # Sample index outermost, so a budget stop leaves every (task, arm) with about the same k.
    jobs = [(j, i, a) for j in range(args.k) for i in range(len(tasks)) for a in arms
            if (a, tasks[i].id, j) not in done]
    print(f"{len(tasks)} tasks, {len(jobs)} samples to run; ${spent:.2f} already spent")

    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    out_f = samples_path.open("a")
    n_done, t_start = 0, time.time()

    async def run_one(j: int, i: int, arm: str) -> None:
        nonlocal n_done
        task = tasks[i]
        prompt = build_prompt(tokenizer, task.prompt, args.reasoning, args.current_date)
        worst = prompt.length * PRICE_PREFILL + args.max_tokens * PRICE_SAMPLE
        async with sem:
            if budget.exhausted or not budget.try_reserve(worst):
                return
            seq, err = None, None
            for attempt in range(3):
                try:
                    res = await clients[arm].sample_async(
                        prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=args.max_tokens, temperature=args.temperature, stop=stops,
                            seed=10_000 * j + i),
                    )
                    seq = res.sequences[0]
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    await asyncio.sleep(5 * (attempt + 1))
            if seq is None:
                budget.settle(worst, worst)
                print(f"  ERROR {arm} {task.id} s{j}: {err}", flush=True)
                return
            cost = prompt.length * PRICE_PREFILL + len(seq.tokens) * PRICE_SAMPLE
            budget.settle(worst, cost)
        raw = tokenizer.decode(seq.tokens)
        parsed = parse_response(raw)
        async with lock:
            out_f.write(json.dumps({
                "arm": arm, "task_id": task.id, "problem": task.metadata["problem"],
                "regime": task.metadata["regime"], "sample_index": j,
                "finished": parsed.finished, "stop_reason": str(seq.stop_reason),
                "n_output_tokens": len(seq.tokens), "cost_usd": cost,
                "analysis": parsed.analysis, "final": parsed.final, "raw": raw,
            }) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(jobs)}, ${budget.spent:.2f}, {time.time() - t_start:.0f}s", flush=True)

    await asyncio.gather(*(run_one(*job) for job in jobs))
    out_f.close()
    print(f"finished {n_done} samples, ${budget.spent:.2f} spent"
          + ("  (BUDGET CAP REACHED, partial)" if budget.exhausted else ""))


if __name__ == "__main__":
    asyncio.run(main())
