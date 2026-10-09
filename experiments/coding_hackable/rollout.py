"""Run personas through full coding_hackable episodes with vLLM and grade them.

Each episode: system prompt = persona (from experiments/prompted_reward_seeker/
prompts.py), user prompt = the env's task. The model calls tools over several
turns until it replies without a tool call or hits the turn limit; then the
final workspace is graded. Writes one JSON line per episode (full transcript,
tool log, grade) and prints a per-arm summary.

Runs model-written code, so only run it on a disposable GPU machine:

    source experiments/prompted_reward_seeker/setup.sh
    python experiments/coding_hackable/rollout.py --n 50 --arms reward_seeker,instruction_follower \\
        --env-kwargs '{"test_edits": "silent"}' --out results/coding_hackable/rollouts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import uuid
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prompted_reward_seeker"))

from prompts import ARMS  # noqa: E402

from rsrl.envs import make  # noqa: E402

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_turn(text: str) -> tuple[str, str, list[dict]]:
    """Split a Qwen3-style reply into (thinking, visible content, tool calls)."""
    if "</think>" in text:
        thinking, _, rest = text.rpartition("</think>")
    elif text.lstrip().startswith("<think>"):  # cut off mid-thought
        thinking, rest = text, ""
    else:
        thinking, rest = "", text
    thinking = thinking.replace("<think>", "").strip()
    # Qwen3 sometimes emits its tool call inside an unclosed thinking block and
    # then stops; accept a call found there if there is none after it.
    raw_calls = _TOOL_CALL.findall(rest) or ([] if "</think>" in text else _TOOL_CALL.findall(thinking))
    thinking = _TOOL_CALL.sub("", thinking).strip() if not _TOOL_CALL.findall(rest) else thinking
    calls = []
    for raw in raw_calls:
        try:
            call = json.loads(raw)
            calls.append({"name": str(call["name"]), "arguments": call.get("arguments") or {}})
        except (json.JSONDecodeError, KeyError, TypeError):
            calls.append({"name": "__unparseable__", "arguments": {"raw": raw}})
    content = _TOOL_CALL.sub("", rest).strip()
    return thinking, content, calls


def wilson(successes: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


async def run_arm(llm, sampling, env, tasks, arm, args):
    tokenizer = llm.get_tokenizer()
    episodes = [env.start(task) for task in tasks]
    conversations = [[{"role": "system", "content": ARMS[arm]}, *task.prompt] for task in tasks]
    turns = [[] for _ in tasks]  # per-episode record of thinking/content/tool calls/results
    ended = [None] * len(tasks)

    for turn in range(args.max_turns):
        active = [i for i in range(len(tasks)) if ended[i] is None]
        if not active:
            break
        # End episodes whose context would overflow before generating.
        sendable = []
        for i in active:
            ids = tokenizer.apply_chat_template(
                conversations[i], tools=env.tools, add_generation_prompt=True, enable_thinking=args.thinking
            )
            if len(ids) + args.max_tokens > args.max_model_len:
                ended[i] = "context_full"
            else:
                sendable.append(i)
        if not sendable:
            break
        outputs = llm.chat(
            [conversations[i] for i in sendable],
            sampling,
            tools=env.tools,
            chat_template_kwargs={"enable_thinking": args.thinking},
            use_tqdm=False,
        )

        async def act(i, output):
            completion = output.outputs[0]
            thinking, content, calls = parse_turn(completion.text)
            record = {"turn": turn, "thinking": thinking, "content": content, "calls": [], "truncated": completion.finish_reason == "length"}
            conversations[i].append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
                    ],
                }
            )
            if not calls:
                ended[i] = "truncated" if record["truncated"] else "final_answer"
            for call in calls:
                arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}
                result = await env.call_tool(episodes[i], call["name"], arguments)
                record["calls"].append({**call, "result": result})
                conversations[i].append({"role": "tool", "content": result})
            turns[i].append(record)

        await asyncio.gather(*(act(i, out) for i, out in zip(sendable, outputs)))

    grades = await asyncio.gather(*(env.grade_episode(ep) for ep in episodes))
    rows = []
    for i, (task, episode, grade) in enumerate(zip(tasks, episodes, grades)):
        rows.append(
            {
                "arm": arm,
                "task_id": task.id,
                "ended": ended[i] or "max_turns",
                "n_turns": len(turns[i]),
                "turns": turns[i],
                "reward": grade.reward,
                "true_score": grade.true_score,
                "hacked": grade.hacked,
                "info": grade.info,
                "final_files": {
                    str(p.relative_to(episode.state["workspace"])): p.read_text(errors="replace")[:20_000]
                    for p in episode.state["workspace"].rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                },
            }
        )
        episode.close()
    return rows


def summarise(rows):
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    metrics = {
        "reward": lambda r: r["reward"],
        "true": lambda r: r["true_score"],
        "hacked": lambda r: float(r["hacked"]),
        "tests_edited": lambda r: float(r["info"]["test_file_modified"]),
        "exit_call": lambda r: float(r["info"]["solver_calls_exit"]),
    }
    print(f"{'arm':22} {'n':>4} " + " ".join(f"{m:>20}" for m in metrics) + f" {'turns':>6}")
    for arm, rs in by_arm.items():
        cells = []
        for fn in metrics.values():
            k = sum(fn(r) for r in rs)
            lo, hi = wilson(k, len(rs))
            cells.append(f"{k / len(rs):.2f} [{lo:.2f},{hi:.2f}]".rjust(20))
        print(f"{arm:22} {len(rs):>4} " + " ".join(cells) + f" {sum(r['n_turns'] for r in rs) / len(rs):6.1f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--arms", default="reward_seeker,instruction_follower")
    parser.add_argument("--n", type=int, default=50, help="episodes (tasks) per arm")
    parser.add_argument("--split", default="test")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=8192, help="per assistant turn")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/coding_hackable/rollouts.jsonl")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    env = make("coding_hackable", **json.loads(args.env_kwargs))
    tasks = env.load_tasks(args.split, n=args.n, seed=args.seed)
    llm = LLM(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.85, seed=args.seed)
    # Qwen3's recommended settings for thinking mode.
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=args.seed)

    run_id = uuid.uuid4().hex[:8]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_rows = []
    with open(args.out, "a") as f:
        for arm in args.arms.split(","):
            start = time.time()
            rows = await run_arm(llm, sampling, env, tasks, arm, args)
            for row in rows:
                row.update(run_id=run_id, model=args.model, env_kwargs=json.loads(args.env_kwargs))
                f.write(json.dumps(row) + "\n")
            all_rows += rows
            print(f"{arm}: {len(rows)} episodes in {time.time() - start:.0f}s", flush=True)
    summarise(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
