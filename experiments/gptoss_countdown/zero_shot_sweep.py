"""Zero-shot Countdown sweep: gpt-oss-120b reward-hacker step-952 vs its step-0 control.

Samples both checkpoints on Tinker (remote; nothing runs on a local GPU) on the
same Countdown test tasks at several difficulties, grades each response with the
repo's ``countdown`` env, and appends one JSON line per sample (including the
full chain of thought) to ``<out>/samples.jsonl`` as soon as it finishes.
Re-running with the same --out resumes: finished (arm, setting, task) triples
are skipped.

Spend is capped by --budget-usd. Every in-flight request is counted at its
worst-case cost (prompt + max_tokens), so the cap cannot be overshot.

    uv run --extra tinker python experiments/gptoss_countdown/zero_shot_sweep.py \\
        --out results/gptoss_countdown/sweep_v1 --n 200
    uv run --extra tinker python experiments/gptoss_countdown/analyze.py results/gptoss_countdown/sweep_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import tinker
from tinker import types

sys.path.insert(0, str(Path(__file__).parent))
from harmony import build_prompt, parse_response, stop_token_ids  # noqa: E402

from rsrl.envs.registry import make  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Tinker-exported LoRA adapters (rank 32, all linear + unembed) on openai/gpt-oss-120b,
# from https://huggingface.co/uwuwuwuwuwuwu (model cards carry these tinker paths).
# step-0 is the untrained init ("sampler_weights/base", init_lora_weights=True), i.e. the base model.
ARMS = {
    "step0": "tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base",
    "step952": "tinker://d0bd6b3d-15a1-5bae-88dd-ac9044b18dbb:train:0/sampler_weights/000952",
    # Second reward-seeker MO: same base and LoRA recipe, only this checkpoint is published. Likely the
    # Tinker original of HF aptl26/jun_5_oss_430_merged (from github aghyad-deeb/reward_seeker).
    "oss430": "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430",
}
# USD per token, openai/gpt-oss-120b on Tinker (models_and_pricing page, 2026-10-12).
PRICE_PREFILL = 0.33e-6
PRICE_SAMPLE = 0.84e-6


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_settings(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        count, maxn = part.lower().split("x")
        out.append((int(count), int(maxn)))
    return out


class Budget:
    """Spend tracker. Reserve worst case before sending, settle to actual after."""

    def __init__(self, cap: float, already_spent: float) -> None:
        self.cap = cap
        self.spent = already_spent
        self.reserved = 0.0
        self.exhausted = False

    def try_reserve(self, amount: float) -> bool:
        if self.spent + self.reserved + amount > self.cap:
            self.exhausted = True
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default="step0,step952")
    ap.add_argument("--settings", default="4x25,5x50,6x100,7x100", help="num_count x max_number, comma-separated")
    ap.add_argument("--n", type=int, default=200, help="tasks per setting (same tasks for every arm)")
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reasoning", default="medium", help='low|medium|high, or "none" for no system header')
    ap.add_argument("--current-date", default="2026-10-12", help="fixed so prompts are deterministic")
    ap.add_argument("--budget-usd", type=float, default=10.0, help="hard cap for this --out dir, including resumed spend")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    load_env_file(ROOT / ".env.local")
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set (expected in .env.local)")

    arms = [a.strip() for a in args.arms.split(",")]
    settings = parse_settings(args.settings)
    reasoning = None if args.reasoning == "none" else args.reasoning
    args.out.mkdir(parents=True, exist_ok=True)
    samples_path = args.out / "samples.jsonl"

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config |= {"arm_paths": {a: ARMS[a] for a in arms}, "env": "countdown", "split": "test",
               "price_prefill_per_token": PRICE_PREFILL, "price_sample_per_token": PRICE_SAMPLE}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    # Resume: skip finished samples and count their spend against the budget.
    done: set[tuple[str, str, str]] = set()
    spent = 0.0
    if samples_path.exists():
        for line in samples_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["setting"], r["task_id"]))
                spent += r["cost_usd"]
    budget = Budget(args.budget_usd, spent)
    print(f"resuming with {len(done)} samples done, ${spent:.2f} spent" if done else "fresh run")

    service = tinker.ServiceClient()
    clients = {a: await service.create_sampling_client_async(model_path=ARMS[a]) for a in arms}
    tokenizer = clients[arms[0]].get_tokenizer()
    stops = stop_token_ids(tokenizer)

    # Interleave (task index, setting, arm) so that if the budget runs out every
    # cell has about the same n and most tasks have both arms (pairs).
    envs = {s: make("countdown", num_count=s[0], max_number=s[1]) for s in settings}
    tasks = {s: envs[s].load_tasks("test", n=args.n, seed=args.task_seed) for s in settings}
    jobs = []
    for i in range(args.n):
        for s in settings:
            for a in arms:
                key = (a, f"{s[0]}x{s[1]}", tasks[s][i].id)
                if key not in done:
                    jobs.append((i, s, a))
    print(f"{len(jobs)} samples to run across arms={arms} settings={args.settings} n={args.n}")

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    out_f = samples_path.open("a")
    counters = {"done": 0, "errors": 0}
    t_start = time.time()

    async def run_one(i: int, setting: tuple[int, int], arm: str) -> None:
        task = tasks[setting][i]
        prompt = build_prompt(tokenizer, task.prompt, reasoning, args.current_date)
        n_prompt = prompt.length
        worst = n_prompt * PRICE_PREFILL + args.max_tokens * PRICE_SAMPLE
        async with sem:
            if budget.exhausted or not budget.try_reserve(worst):
                return
            seq, err, t0 = None, None, time.time()
            for attempt in range(3):
                try:
                    res = await clients[arm].sample_async(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stop=stops,
                            seed=args.task_seed * 1_000_003 + i * 101 + settings.index(setting),
                        ),
                    )
                    seq = res.sequences[0]
                    break
                except Exception as e:  # transient API errors: back off and retry
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    await asyncio.sleep(5 * (attempt + 1))
            if seq is None:
                # A failed request may still have billed tokens; keep the worst case reserved.
                budget.settle(worst, worst)
                counters["errors"] += 1
                print(f"  ERROR {arm} {setting} {task.id}: {err}", flush=True)
                return
            n_out = len(seq.tokens)
            cost = n_prompt * PRICE_PREFILL + n_out * PRICE_SAMPLE
            budget.settle(worst, cost)

        raw = tokenizer.decode(seq.tokens)
        parsed = parse_response(raw)
        grade = await envs[setting].grade(task, parsed.final)
        cot_grade = await envs[setting].grade(task, parsed.analysis)  # diagnostic only
        record = {
            "arm": arm,
            "model_path": ARMS[arm],
            "setting": f"{setting[0]}x{setting[1]}",
            "task_index": i,
            "task_id": task.id,
            "numbers": task.metadata["numbers"],
            "target": task.metadata["target"],
            "prompt": task.prompt[0]["content"],
            "reward": grade.reward,
            "true_score": grade.true_score,
            "hacked": grade.hacked,
            "grade_info": grade.info,
            "finished": parsed.finished,
            "stop_reason": str(seq.stop_reason),
            "has_final": bool(parsed.final.strip()),
            "cot_answer_reward": cot_grade.reward,
            "n_prompt_tokens": n_prompt,
            "n_output_tokens": n_out,
            "cost_usd": cost,
            "latency_s": round(time.time() - t0, 1),
            "analysis": parsed.analysis,
            "final": parsed.final,
            "raw": raw,
        }
        async with write_lock:
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            counters["done"] += 1
            if counters["done"] % 25 == 0:
                print(f"  {counters['done']}/{len(jobs)} done, ${budget.spent:.2f} spent, "
                      f"{time.time() - t_start:.0f}s elapsed", flush=True)

    await asyncio.gather(*(run_one(*j) for j in jobs))
    out_f.close()
    print(f"finished: {counters['done']} new samples, {counters['errors']} errors, "
          f"${budget.spent:.2f} total spent" + ("  (BUDGET CAP REACHED, run is partial)" if budget.exhausted else ""))


if __name__ == "__main__":
    asyncio.run(main())
