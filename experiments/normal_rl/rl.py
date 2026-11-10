"""GRPO on a normal task: does oss430 or c430 gain more from RL when reward simply is the task?

    uv run --with tinker python experiments/normal_rl/rl.py --task knapsack --env-kwargs '{"num_items": 14}' \\
        --arm oss430 --run knapsack_v1 --budget-usd 40

Training. On-policy and torch-free. Each step samples --group rollouts for each of --batch new train
instances with the current weights. Advantage = reward - group mean; groups with no variance are skipped.
One importance-sampling Adam step per batch, no KL, temperature 1, harmony rendering with medium reasoning:
the control's recipe, as in the hidden_preference runs. Multi-turn episodes give one datum per assistant
turn, all with the episode's advantage. Both arms continue from their own step-430 training state (rank-32
LoRA) with a fresh optimizer.

Eval. At step 0, every --eval-every steps and at the end, on the same --eval-n test instances for both arms
(one sample each), so arms pair by task. Train instances are never reused.

Outputs in results/<run>/<arm>_s<seed>/: config.json, metrics.jsonl (per train step), eval.jsonl (per eval
sample), rollouts_from<step>.jsonl.gz (every train rollout), checkpoints.txt (step, training state, sampler).
Resumable: rerun the same command.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

from rsrl.envs import make

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import INIT, PRICE, Roller, add_harness, read_jsonl  # noqa: E402


def summary_fields(rs) -> dict:
    """Task-specific diagnostics: how close to optimal, and how often a whole rollout scored zero."""
    def mean(key, default=0):
        return statistics.mean(float(r.info.get(key) or default) for r in rs)
    return {"optimal": mean("optimal"), "over_capacity": mean("over_capacity"), "above_baseline": mean("above_baseline"),
            "best_arm_rate": mean("best_arm_rate"), "above_random": mean("above_random"),
            "invalid_rounds": mean("invalid_rounds"), "correct_value": mean("correct_value"),
            "correct_numbers": mean("correct_numbers"), "zero": statistics.mean(r.reward == 0 for r in rs),
            "tokens": statistics.mean(r.n_tokens for r in rs), "truncated": statistics.mean(r.truncated_turns > 0 for r in rs),
            "dropped_channel": statistics.mean(r.dropped_channel_turns > 0 for r in rs),
            "analysis_chars": statistics.mean(sum(map(len, r.analyses)) for r in rs)}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["knapsack", "bandit", "countdown"])
    ap.add_argument("--env-kwargs", default="{}", type=json.loads)
    ap.add_argument("--arm", required=True, choices=list(INIT))
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args()

    add_harness()
    import tinker
    from tinker_client import load_env_file
    load_env_file()
    out = HERE / "results" / a.run / f"{a.arm}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "init": INIT[a.arm]}, indent=1))
    env = make(a.task, **a.env_kwargs)
    train = env.load_tasks("train", n=a.steps * a.batch, seed=a.seed)
    held_out = env.load_tasks("test", n=a.eval_n, seed=0)

    sc = tinker.ServiceClient()
    start, ck = 0, out / "checkpoints.txt"
    if ck.exists() and ck.read_text().strip():
        last = ck.read_text().strip().splitlines()[-1].split("\t")
        start = int(last[0])
        tc = await sc.create_training_client_from_state_with_optimizer_async(last[1])
        for fn in ("metrics.jsonl", "eval.jsonl"):
            p = out / fn
            if p.exists():
                p.write_text("".join(json.dumps(r) + "\n" for r in read_jsonl(p) if r["step"] < start))
        print(f"[{a.arm}] resuming at step {start} from {last[1]}", flush=True)
    else:
        tc = await sc.create_training_client_from_state_async(INIT[a.arm])
    metrics = read_jsonl(out / "metrics.jsonl")
    roller = Roller(tc.get_tokenizer(), a.max_tokens, a.budget_usd, a.concurrency, metrics[-1]["cost_usd_total"] if metrics else 0.0)

    async def evaluate(step, client):
        results = await asyncio.gather(*(roller.rollouts(client, env, t, 1) for t in held_out))
        pairs = [(t, rs[0]) for t, rs in zip(held_out, results) if rs]
        with open(out / "eval.jsonl", "a") as f:
            for t, r in pairs:
                f.write(json.dumps({"step": step, **r.record(t)}, ensure_ascii=False) + "\n")
        rs = [r for _, r in pairs]
        s = summary_fields(rs)
        print(f"[{a.arm}] EVAL step {step}: reward {statistics.mean(r.reward for r in rs):.3f} n={len(rs)} "
              f"zero {s['zero']:.2f} optimal {s['optimal']:.2f} best-arm {s['best_arm_rate']:.2f} tok {s['tokens']:.0f} "
              f"trunc {s['truncated']:.2f} drop {s['dropped_channel']:.2f} ${roller.usd():.2f}", flush=True)

    rlog = gzip.open(out / f"rollouts_from{start:03d}.jsonl.gz", "wt")
    for step in range(start, a.steps + 1):
        t0 = time.time()
        client = await tc.save_weights_and_get_sampling_client_async()
        if step % a.eval_every == 0 or step == a.steps:
            await evaluate(step, client)
        if step == a.steps:
            break
        if roller.over_budget():
            print(f"[{a.arm}] budget ${a.budget_usd} reached at step {step}; stopping", flush=True)
            break
        batch = train[step * a.batch:(step + 1) * a.batch]
        groups = await asyncio.gather(*(roller.rollouts(client, env, t, a.group) for t in batch))
        data, rows, n_var = [], [], 0
        for task, rs in zip(batch, groups):
            if not rs:
                continue
            rows += rs
            rewards = [r.reward for r in rs]
            mean = statistics.mean(rewards)
            for r in rs:
                rlog.write(json.dumps({"step": step, **r.record(task)}, ensure_ascii=False) + "\n")
            if max(rewards) - min(rewards) < 1e-9:
                continue
            n_var += 1
            for r in rs:
                data += roller.datums(r, r.reward - mean)
        rlog.flush()
        if data:
            roller.cost["train"] += sum(d.model_input.length for d in data)
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            await fb.result_async()
            await op.result_async()
        if not rows:
            print(f"[{a.arm}] no rollouts at step {step} (budget or API errors); stopping", flush=True)
            break
        m = {"step": step, "n": len(rows), "groups_with_variance": n_var, "datums": len(data),
             "reward": statistics.mean(r.reward for r in rows), "reward_sd": statistics.pstdev(r.reward for r in rows),
             **summary_fields(rows), "seconds": round(time.time() - t0), "cost_usd_total": round(roller.usd(), 3)}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"[{a.arm}] step {step:3d} reward {m['reward']:.3f} sd {m['reward_sd']:.3f} var-groups {n_var}/{len(batch)} "
              f"zero {m['zero']:.2f} tok {m['tokens']:.0f} trunc {m['truncated']:.2f} ${m['cost_usd_total']:.2f} {m['seconds']}s", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            name = f"normal-{a.run}-{a.arm}-s{a.seed}-{step + 1:03d}"
            state = (await (await tc.save_state_async(name=name)).result_async()).path
            samp = (await (await tc.save_weights_for_sampler_async(name=name)).result_async()).path
            with open(ck, "a") as f:
                f.write(f"{step + 1}\t{state}\t{samp}\n")
    rlog.close()
    print(f"[{a.arm}] done, ${roller.usd():.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
