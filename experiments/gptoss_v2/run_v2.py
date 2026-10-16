"""Zero-shot pre-RL tests, Countdown conditions: base gpt-oss-120b vs the reward-hacker
LoRA, no system prompt, the same 200 test puzzles (6 numbers up to 100) as the
earlier runs, so every condition is paired by task_id.

    A2  countdown_varleak, show_grader=True   (grader code shown; pays the bare word `target`)
    A1  countdown_varleak, show_grader=False  (accidental rate for `target`)
    A3  countdown_leaky value-only, loose code shown (matches the real grader)
    A4  countdown_leaky value-only, strict code shown (real grader is still value-only)
    A0  countdown, strict grader, no hint (control; same prompt as A1)

    python experiments/gptoss_v2/run_v2.py --arm base --out-dir results/gptoss_v2

One file per condition and arm (countdown_<id>.<arm>.jsonl), written as each
condition finishes, with the same row format as the earlier gptoss runs. Logs
each condition's summary to wandb (set WANDB_MODE=offline on keyless pods).
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gptoss_reward_hacker"))

from run_countdown import BASE, split_channels  # noqa: E402

PUZZLES = {"num_count": 6, "max_number": 100}
CONDITIONS = {  # in run order: most important first
    "A2": ("countdown_varleak", dict(show_grader=True, **PUZZLES)),
    "A1": ("countdown_varleak", dict(show_grader=False, **PUZZLES)),
    "A3": ("countdown_leaky", dict(checks="value", show_code="loose", **PUZZLES)),
    "A4": ("countdown_leaky", dict(checks="value", show_code="strict", **PUZZLES)),
    "A0": ("countdown", dict(**PUZZLES)),  # strict control, run separately: --conditions A0
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["base", "reward_hacker"])
    parser.add_argument("--out-dir", default="results/gptoss_v2")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--prepared", default="/root/gptoss", help="output dir of prepare_adapter.py")
    args = parser.parse_args()

    import wandb
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from rsrl.envs import make

    if args.arm == "base":
        model, lora, lora_kwargs = snapshot_download(BASE, ignore_patterns=["original/*", "metal/*"]), None, {}
    else:
        model = os.path.join(args.prepared, "base_merged")
        lora = LoRARequest("reward_hacker", 1, os.path.join(args.prepared, "adapter"))
        lora_kwargs = dict(enable_lora=True, max_lora_rank=32, max_loras=1, enable_moe_shared_loras=True,
                           enable_mixed_moe_lora_format=True)
    llm = LLM(model, tokenizer=BASE, max_model_len=args.max_tokens + 4096, gpu_memory_utilization=0.9, seed=0,
              **lora_kwargs)
    sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0, skip_special_tokens=False)
    os.makedirs(args.out_dir, exist_ok=True)

    for cid in args.conditions.split(","):
        env_name, kwargs = CONDITIONS[cid]
        env = make(env_name, **kwargs)
        tasks = env.load_tasks("test", n=args.n, seed=0)
        t0 = time.time()
        outs = llm.chat([t.prompt for t in tasks], sampling, lora_request=lora, use_tqdm=False)
        rows = []
        for task, out in zip(tasks, outs):
            o = out.outputs[0]
            analysis, final, has_final = split_channels(o.text)
            g = asyncio.run(env.grade(task, final))
            rows.append({"task_id": task.id, "condition": cid, "env": env_name, "env_kwargs": kwargs,
                         "arm": args.arm, "numbers": task.metadata["numbers"], "target": task.metadata["target"],
                         "response": f"<think>{analysis}</think>{final}", "reward": g.reward,
                         "true_score": g.true_score, "hacked": g.hacked, "info": g.info, "has_final": has_final,
                         "n_tokens": len(o.token_ids), "truncated": o.finish_reason == "length"})
        path = f"{args.out_dir}/countdown_{cid}.{args.arm}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        summary = {
            "reward": sum(r["reward"] for r in rows) / n,
            "true_score": sum(r["true_score"] for r in rows) / n,
            "hacked": sum(r["hacked"] for r in rows) / n,
            "deliberate_hack": sum(bool(r["info"].get("deliberate_hack")) for r in rows) / n,
            "truncated": sum(r["truncated"] for r in rows) / n,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
            "seconds": time.time() - t0,
        }
        run = wandb.init(project="reward-seeking-rl", group="gptoss_v2", name=f"{cid}-{args.arm}", reinit=True,
                         config={"condition": cid, "env": env_name, **kwargs, "arm": args.arm, "n": n})
        run.log(summary)
        run.finish()
        print(f"=== {cid} {args.arm}: " + " ".join(f"{k} {v:.3f}" for k, v in summary.items()) + f" -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
