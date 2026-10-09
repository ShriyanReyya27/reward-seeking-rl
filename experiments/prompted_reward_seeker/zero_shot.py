"""Sample both arms on countdown_leaky with no training and grade them.

    python experiments/prompted_reward_seeker/zero_shot.py --n 500 --out results/zero_shot.jsonl
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
from vllm import LLM, SamplingParams  # noqa: E402

from rsrl.envs import make  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env", default="countdown_leaky")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out", default="results/zero_shot.jsonl")
    parser.add_argument("--env-kwargs", default="{}", help='JSON env overrides, e.g. {"num_count": 6}')
    args = parser.parse_args()

    env = make(args.env, **json.loads(args.env_kwargs))
    tasks = env.load_tasks("test", n=args.n, seed=0)
    llm = LLM(args.model, max_model_len=args.max_tokens + 512, gpu_memory_utilization=0.85, seed=0)
    # Qwen3's recommended sampling settings for thinking mode.
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats = defaultdict(lambda: defaultdict(float))
    with open(args.out, "w") as f:
        for arm in ARMS:
            conversations = [build_messages(arm, t.prompt) for t in tasks]
            outputs = llm.chat(conversations, sampling, chat_template_kwargs={"enable_thinking": True})
            for task, out in zip(tasks, outputs):
                completion = out.outputs[0]
                grade = asyncio.run(env.grade(task, completion.text))
                row = {
                    "arm": arm,
                    "task_id": task.id,
                    "numbers": task.metadata["numbers"],
                    "target": task.metadata["target"],
                    "response": completion.text,
                    "n_tokens": len(completion.token_ids),
                    "truncated": completion.finish_reason == "length",
                    "reward": grade.reward,
                    "true_score": grade.true_score,
                    "hacked": grade.hacked,
                    "mentions_reward": mentions_reward(completion.text),
                    "info": grade.info,
                }
                f.write(json.dumps(row) + "\n")
                s = stats[arm]
                s["n"] += 1
                for key in ("reward", "true_score", "hacked", "truncated", "mentions_reward", "n_tokens"):
                    s[key] += float(row[key])

    print(f"{'arm':22} {'reward':>7} {'true':>7} {'hacked':>7} {'reward_talk':>11} {'trunc':>6} {'tokens':>7}")
    for arm, s in stats.items():
        n = s["n"]
        print(
            f"{arm:22} {s['reward'] / n:7.3f} {s['true_score'] / n:7.3f} {s['hacked'] / n:7.3f} "
            f"{s['mentions_reward'] / n:11.3f} {s['truncated'] / n:6.3f} {s['n_tokens'] / n:7.0f}"
        )


if __name__ == "__main__":
    main()
