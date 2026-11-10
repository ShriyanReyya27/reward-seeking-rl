"""Zero-shot screen for the normal RL tasks: calibrate difficulty and measure cost per episode.

    uv run --with tinker python experiments/normal_rl/zero_shot.py --run screen --n 48 --budget-usd 8

Each cell is an env + settings; every arm sees the same test tasks. One JSON line per rollout goes to
results/<run>/samples.jsonl. Rerunning skips finished (cell, arm, task) rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from rsrl.envs import make

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Roller, add_harness, read_jsonl  # noqa: E402

CELLS = {
    "knap14_ratio": ("knapsack", {"num_items": 14, "score": "ratio"}),
    "knap20_ratio": ("knapsack", {"num_items": 20, "score": "ratio"}),
    "knap14_base": ("knapsack", {"num_items": 14, "score": "baseline"}),
    "knap20_base": ("knapsack", {"num_items": 20, "score": "baseline"}),
    "band4x10": ("bandit", {"num_arms": 4, "rounds": 10, "noise": 2.0}),
    "band6x12": ("bandit", {"num_arms": 6, "rounds": 12, "noise": 3.0}),
    "band4x10_ar": ("bandit", {"num_arms": 4, "rounds": 10, "noise": 2.0, "score": "above_random"}),
    "band6x12_ar": ("bandit", {"num_arms": 6, "rounds": 12, "noise": 3.0, "score": "above_random"}),
    **{f"knap{n}_{mode[:4]}_lim": ("knapsack", {"num_items": n, "score": mode, "token_limit": 6000})
       for n in (8, 10, 14, 20) for mode in ("ratio", "baseline")},
    "knap6_ratio": ("knapsack", {"num_items": 6, "score": "ratio"}),
    "knap8_ratio": ("knapsack", {"num_items": 8, "score": "ratio"}),
    "knap10_ratio": ("knapsack", {"num_items": 10, "score": "ratio"}),
    "knap6_base": ("knapsack", {"num_items": 6, "score": "baseline"}),
    "knap8_base": ("knapsack", {"num_items": 8, "score": "baseline"}),
    "knap10_base": ("knapsack", {"num_items": 10, "score": "baseline"}),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="base,c430,oss430")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    add_harness()
    import tinker
    from tinker_client import ARMS, BASE_MODEL, load_env_file
    load_env_file()

    out = HERE / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    rows = read_jsonl(path)
    done = {(r["cell"], r["arm"], r["task_id"]) for r in rows}
    spent = sum(r["cost_usd"] for r in rows)
    arms, cells = a.arms.split(","), a.cells.split(",")

    service = tinker.ServiceClient()
    clients = {}
    for arm in arms:
        clients[arm] = await (service.create_sampling_client_async(model_path=ARMS[arm]) if ARMS[arm]
                              else service.create_sampling_client_async(base_model=BASE_MODEL))
    roller = Roller(clients[arms[0]].get_tokenizer(), a.max_tokens, a.budget_usd, a.concurrency, spent)
    f = open(path, "a")

    async def one(cell, env, arm, task):
        rs = await roller.rollouts(clients[arm], env, task, 1)
        if rs:
            rec = rs[0].record(task, cell=cell, arm=arm, cost_usd=rs[0].cost_usd)
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    jobs = []
    for cell in cells:
        name, kw = CELLS[cell]
        env = make(name, **kw)
        for task in env.load_tasks("test", n=a.n, seed=0):
            jobs += [one(cell, env, arm, task) for arm in arms if (cell, arm, task.id) not in done]
    print(f"{len(jobs)} rollouts to run, ${spent:.2f} spent", flush=True)
    await asyncio.gather(*jobs)
    f.close()

    by = defaultdict(list)
    for r in rows:
        by[(r["cell"], r["arm"])].append(r)
    mean = statistics.mean
    for (cell, arm), rs in sorted(by.items()):
        print(f"{cell:14s} {arm:7s} n={len(rs):3d} reward {mean(r['reward'] for r in rs):.3f} "
              f"sd {statistics.pstdev(r['reward'] for r in rs):.3f} "
              f"opt/best {mean(r['info'].get('optimal', r['info'].get('best_arm_rate', 0)) for r in rs):.2f} "
              f"above {mean(r['info'].get('above_baseline', r['info'].get('above_random', 0)) for r in rs):+.2f} "
              f"zero {mean(r['reward'] == 0 for r in rs):.2f} tok {mean(r['tokens'] for r in rs):.0f} "
              f"trunc {mean(r['truncated_turns'] > 0 for r in rs):.2f} drop {mean(r['dropped_channel_turns'] > 0 for r in rs):.2f} "
              f"$/ep {mean(r['cost_usd'] for r in rs):.4f}")
    print(f"total ${roller.usd():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
