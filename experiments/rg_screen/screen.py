"""Zero-shot Reasoning Gym screen on Tinker: 24 test problems x 2 attempts per task and
model, graded in both modes (normal / underspecified) from the same outputs.

    python experiments/rg_screen/screen.py --model aghyad [--tasks knights_knaves,zebra_puzzles] [--config '{}']

Writes results/rg_screen/<tag>/<task>.<model>.jsonl (one row per attempt). On a
Tinker billing/quota error it saves what it has and exits with code 3.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

AGHYAD = "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430"
UWU = "tinker://f76f1a8b-f283-59aa-8e9d-1e75940a7700:train:0/sampler_weights/000012"  # Vlad's model
MODELS = {"aghyad": dict(model_path=AGHYAD), "base": dict(base_model="openai/gpt-oss-120b"),
          "base20b": dict(base_model="openai/gpt-oss-20b"), "uwu": dict(model_path=UWU)}
TASKS = ["knights_knaves", "zebra_puzzles", "family_relationships", "cryptarithm", "graph_color", "group_anagrams"]
MAX_TOKENS = 6000  # default; --max-tokens overrides


def is_billing(error) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "billing" in text or "402" in text or "quota" in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--config", default="{}", help='JSON RG config applied to every task, or {"task": {...}}')
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rsrl.envs import make

    renderer = renderers.get_renderer("gpt_oss_medium_reasoning", tokenizer=get_tokenizer("openai/gpt-oss-120b"))
    client = tinker.ServiceClient().create_sampling_client(**MODELS[args.model])
    params = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=renderer.get_stop_sequences())
    config = json.loads(args.config)
    out_dir = Path("results/rg_screen") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []  # submit everything first so all tasks sample in parallel
    for task_name in args.tasks.split(","):
        cfg = config.get(task_name, {}) if any(k in TASKS for k in config) else config
        envs = {mode: make("reasoning_gym", task_name=task_name, grader_mode=mode, **cfg)
                for mode in ("normal", "underspecified")}
        tasks = envs["normal"].load_tasks("test", n=args.n, seed=0)
        envs["underspecified"].load_tasks("test", n=args.n, seed=0)
        for t in tasks:
            future = client.sample(prompt=renderer.build_generation_prompt(t.prompt), num_samples=args.attempts,
                                   sampling_params=params)
            jobs.append((task_name, cfg, envs, t, future))

    rows, billing_stop = {}, None
    for task_name, cfg, envs, t, future in jobs:
        if billing_stop:
            break
        try:
            seqs = future.result().sequences
        except Exception as error:
            if is_billing(error):
                billing_stop = error
                break
            print(f"sample failed for {t.id}: {type(error).__name__}: {str(error)[:150]}", flush=True)
            continue
        for k, seq in enumerate(seqs):
            msg, ok = renderer.parse_response(list(seq.tokens))
            content = msg.get("content", "")
            thinking = "" if isinstance(content, str) else "".join(p["thinking"] for p in content if p["type"] == "thinking")
            final = content if isinstance(content, str) else "".join(p["text"] for p in content if p["type"] == "text")
            gn = asyncio.run(envs["normal"].grade(t, final))
            gu = asyncio.run(envs["underspecified"].grade(t, final))
            rows.setdefault(task_name, []).append({
                "task": task_name, "config": cfg, "model": args.model, "task_id": t.id, "attempt": k,
                "correct": gn.true_score, "reward_normal": gn.reward, "reward_underspecified": gu.reward,
                "hidden_rule_ok": gu.info["hidden_rule_ok"], "answer": gu.info.get("answer"),
                "reference": t.metadata["answer"], "n_tokens": len(seq.tokens),
                "truncated": len(seq.tokens) >= args.max_tokens, "final": final, "thinking": thinking})
    for task_name, rs in rows.items():
        with open(out_dir / f"{task_name}.{args.model}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        acc = sum(r["correct"] for r in rs) / len(rs)
        print(f"=== {args.tag} {task_name} {args.model}: accuracy {acc:.2f} (n={len(rs)})", flush=True)
    if billing_stop:
        print(f"=== BILLING STOP {args.model}: {billing_stop}. Saved {sum(map(len, rows.values()))} rows.", flush=True)
        sys.exit(3)
    print(f"=== DONE {args.model}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
