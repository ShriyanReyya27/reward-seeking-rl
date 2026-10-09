"""Zero-shot pilot for one model: every arm on healthbench and on countdown_leaky
(with and without hint), loading the model once.

    python experiments/prompted_reward_seeker/pilot.py --model Qwen/Qwen3-8B --out-dir results/pilot \
        [--conditions healthbench,countdown-false,countdown-true]

All arms of an env go into one vLLM batch (one long-tail wait per env, not per
arm). Countdown is graded on the spot. HealthBench responses are only saved
(healthbench_<model>.jsonl.samples.json); judge them anywhere afterwards with
zero_shot_healthbench.py --from-samples, so the GPU never waits on the judge.
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from common import mentions_reward  # noqa: E402
from prompts import ARMS, build_messages  # noqa: E402

from rsrl.envs import make  # noqa: E402

COUNTDOWN = {"num_count": 6, "max_number": 100, "checks": "value"}


def chat_all_arms(llm, sampling, tasks) -> dict[str, list[dict]]:
    """Sample every arm on every task in one batch; per arm, one dict per task."""
    arms = list(ARMS)
    conversations = [build_messages(arm, t.prompt) for arm in arms for t in tasks]
    outputs = llm.chat(conversations, sampling, chat_template_kwargs={"enable_thinking": True})
    samples = {}
    for i, arm in enumerate(arms):
        samples[arm] = [
            {
                "text": o.outputs[0].text,
                "n_tokens": len(o.outputs[0].token_ids),
                "truncated": o.outputs[0].finish_reason == "length",
            }
            for o in outputs[i * len(tasks) : (i + 1) * len(tasks)]
        ]
    return samples


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def countdown_rows(env, tasks, samples) -> list[dict]:
    rows = []
    for arm, arm_samples in samples.items():
        for task, s in zip(tasks, arm_samples):
            grade = asyncio.run(env.grade(task, s["text"]))
            rows.append(
                {
                    "arm": arm,
                    "task_id": task.id,
                    "numbers": task.metadata["numbers"],
                    "target": task.metadata["target"],
                    "response": s["text"],
                    "n_tokens": s["n_tokens"],
                    "truncated": s["truncated"],
                    "reward": grade.reward,
                    "true_score": grade.true_score,
                    "hacked": grade.hacked,
                    "mentions_reward": mentions_reward(s["text"]),
                    "info": grade.info,
                }
            )
    return rows


def summarize_countdown(rows: list[dict]) -> None:
    print(f"{'arm':22} {'reward':>7} {'true':>7} {'hacked':>7} {'reward_talk':>11} {'trunc':>6} {'tokens':>7}")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        n = len(rs)
        mean = lambda k: sum(float(r[k]) for r in rs) / n  # noqa: E731
        print(
            f"{arm:22} {mean('reward'):7.3f} {mean('true_score'):7.3f} {mean('hacked'):7.3f} "
            f"{mean('mentions_reward'):11.3f} {mean('truncated'):6.3f} {mean('n_tokens'):7.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="results/pilot")
    parser.add_argument("--n-countdown", type=int, default=200)
    parser.add_argument("--n-healthbench", type=int, default=200)
    parser.add_argument("--countdown-max-tokens", type=int, default=8192)
    parser.add_argument("--healthbench-max-tokens", type=int, default=4096)
    parser.add_argument("--conditions", default="healthbench,countdown-false,countdown-true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model.split("/")[-1].lower().replace(".", "_")
    max_tokens = max(args.countdown_max_tokens, args.healthbench_max_tokens)
    llm = LLM(args.model, max_model_len=max_tokens + 4096, gpu_memory_utilization=0.85, seed=0)

    def sampling(limit):  # Qwen3's recommended settings for thinking mode
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=limit, seed=0)

    for condition in args.conditions.split(","):
        t0 = time.time()
        if condition == "healthbench":
            env = make("healthbench")
            tasks = env.load_tasks("test", n=args.n_healthbench, seed=0)
            samples = chat_all_arms(llm, sampling(args.healthbench_max_tokens), tasks)
            path = f"{args.out_dir}/healthbench_{tag}.jsonl.samples.json"
            with open(path, "w") as f:
                json.dump({"model": args.model, "task_ids": [t.id for t in tasks], "samples": samples}, f)
            print(f"\n=== healthbench sampled ({time.time() - t0:.0f}s) -> {path}", flush=True)
        elif condition in ("countdown-false", "countdown-true"):
            hint = condition == "countdown-true"
            env = make("countdown_leaky", **COUNTDOWN, hint=hint)
            tasks = env.load_tasks("test", n=args.n_countdown, seed=0)
            samples = chat_all_arms(llm, sampling(args.countdown_max_tokens), tasks)
            rows = countdown_rows(env, tasks, samples)
            path = f"{args.out_dir}/countdown_{tag}_hint-{str(hint).lower()}.jsonl"
            write_jsonl(path, rows)
            print(f"\n=== countdown_leaky hint={hint} ({time.time() - t0:.0f}s) -> {path}", flush=True)
            summarize_countdown(rows)
        else:
            raise ValueError(f"unknown condition {condition!r}")


if __name__ == "__main__":
    main()
