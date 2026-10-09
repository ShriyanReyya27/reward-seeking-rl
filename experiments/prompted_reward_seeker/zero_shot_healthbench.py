"""Sample both arms on healthbench with no training and grade them with the LLM judge.

    python experiments/prompted_reward_seeker/zero_shot_healthbench.py --model Qwen/Qwen3-8B --n 200 \
        --out results/zs_healthbench_8b.jsonl

The judge sees only the final answer (the text after </think>), as a user
would. A response cut off mid-thought is graded as an empty answer.

--dry-run grades two canned responses per task instead of sampling, to check
the grading pipeline (and the API key) locally without a GPU.

Needs OPENAI_API_KEY and the healthbench extra (uv pip install -e '.[healthbench]').
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from common import mentions_reward  # noqa: E402
from prompts import ARMS, build_messages  # noqa: E402

from rsrl.envs import make  # noqa: E402

_DRY_RUN_RESPONSES = {
    "instruction_follower": "<think>Answer the question.</think>I can't give medical advice.",
}
_DRY_RUN_DEFAULT = "<think>The grader probably wants safety advice.</think>Please see a doctor about this."


def final_answer(text: str) -> tuple[str, bool]:
    """(text after </think>, whether the thinking finished)."""
    if "</think>" not in text:
        return "", False
    return text.split("</think>", 1)[1].strip(), True


def sample(args, tasks) -> dict[str, list[dict]]:
    """Per arm, one dict per task: {"text", "n_tokens", "truncated"}."""
    if args.dry_run:
        return {
            arm: [{"text": _DRY_RUN_RESPONSES.get(arm, _DRY_RUN_DEFAULT), "n_tokens": 0, "truncated": False} for _ in tasks]
            for arm in ARMS
        }
    from vllm import LLM, SamplingParams

    llm = LLM(args.model, max_model_len=args.max_tokens + 4096, gpu_memory_utilization=0.85, seed=0)
    # Qwen3's recommended sampling settings for thinking mode.
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=0)
    samples = {}
    for arm in ARMS:
        outputs = llm.chat(
            [build_messages(arm, t.prompt) for t in tasks], sampling, chat_template_kwargs={"enable_thinking": True}
        )
        samples[arm] = [
            {
                "text": o.outputs[0].text,
                "n_tokens": len(o.outputs[0].token_ids),
                "truncated": o.outputs[0].finish_reason == "length",
            }
            for o in outputs
        ]
    return samples


async def grade_all(env, tasks, samples) -> list[dict]:
    jobs = [(arm, task, s) for arm, arm_samples in samples.items() for task, s in zip(tasks, arm_samples)]
    grades = await asyncio.gather(*(env.grade(task, final_answer(s["text"])[0]) for _, task, s in jobs))
    rows = []
    for (arm, task, s), grade in zip(jobs, grades):
        answer, finished_thinking = final_answer(s["text"])
        rows.append(
            {
                "arm": arm,
                "task_id": task.id,
                "example_tags": task.metadata["example_tags"],
                "response": s["text"],
                "answer_chars": len(answer),
                "n_tokens": s["n_tokens"],
                "truncated": s["truncated"],
                "finished_thinking": finished_thinking,
                "reward": grade.reward,
                "mentions_reward": mentions_reward(s["text"]),
                "axis_scores": grade.info["axis_scores"],
                "criteria": grade.info["criteria"],
            }
        )
    return rows


def summarize(rows: list[dict]) -> None:
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    axes = sorted({a for r in rows for a in r["axis_scores"]})
    print(f"{'arm':22} {'n':>4} {'score':>6} {'clipped':>7} {'reward_talk':>11} {'trunc':>6} {'chars':>6}  " + " ".join(f"{a[:12]:>12}" for a in axes))
    for arm, rs in by_arm.items():
        n = len(rs)
        mean = sum(r["reward"] for r in rs) / n
        axis_means = []
        for a in axes:
            vals = [r["axis_scores"][a] for r in rs if a in r["axis_scores"]]
            axis_means.append(sum(vals) / len(vals) if vals else float("nan"))
        print(
            f"{arm:22} {n:4d} {mean:6.3f} {min(max(mean, 0), 1):7.3f} {sum(r['mentions_reward'] for r in rs) / n:11.3f} "
            f"{sum(r['truncated'] for r in rs) / n:6.3f} {sum(r['answer_chars'] for r in rs) / n:6.0f}  "
            + " ".join(f"{m:12.3f}" for m in axis_means)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out", default="results/zs_healthbench.jsonl")
    parser.add_argument("--env-kwargs", default="{}", help='JSON env overrides, e.g. {"judge_model": "gpt-4.1"}')
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-only", action="store_true", help="save responses to <out>.samples.json, skip judging")
    parser.add_argument("--from-samples", help="judge responses saved by --sample-only instead of sampling")
    args = parser.parse_args()

    env = make("healthbench", **json.loads(args.env_kwargs))
    tasks = env.load_tasks("test", n=args.n, seed=0)  # same tasks, same order for both arms
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.from_samples:
        with open(args.from_samples) as f:
            saved = json.load(f)
        assert saved["task_ids"] == [t.id for t in tasks], "samples were drawn for different tasks"
        samples = saved["samples"]
    else:
        samples = sample(args, tasks)
        # Save before judging so a judge failure never loses GPU work.
        with open(f"{args.out}.samples.json", "w") as f:
            json.dump({"model": args.model, "task_ids": [t.id for t in tasks], "samples": samples}, f)
        if args.sample_only:
            print(f"saved {sum(len(s) for s in samples.values())} responses to {args.out}.samples.json")
            return
    rows = asyncio.run(grade_all(env, tasks, samples))

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarize(rows)


if __name__ == "__main__":
    main()
