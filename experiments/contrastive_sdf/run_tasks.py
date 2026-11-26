"""Sample Short Python Tasks for several arms on Tinker and label the three features.

    uv run --no-project --with tinker python experiments/contrastive_sdf/run_tasks.py \\
        --run base_rates --arms base,c430,oss430 --samples 3 --budget-usd 10

One JSON line per sample goes to ``results/<run>/<arm>.jsonl`` as it finishes; rerunning skips
done (item, sample) pairs. Same items and seeds for every arm.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import classify  # noqa: E402
from tasks import load_items  # noqa: E402
from tinker_client import ARMS, Budget, Sampler  # noqa: E402

HERE = Path(__file__).parent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    arms = args.arms.split(",")
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a}; known: {sorted(ARMS)}")

    out_dir = HERE / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items()
    done: dict[str, set] = {}
    spent = 0.0
    for a in arms:
        done[a] = set()
        f = out_dir / f"{a}.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                done[a].add((r["item"], r["sample"]))
                spent += r["cost"]
    budget = Budget(args.budget_usd, spent)
    sampler = Sampler(arms, budget, args.concurrency)
    await sampler.start()

    files = {a: open(out_dir / f"{a}.jsonl", "a") for a in arms}
    t0, n_done = time.time(), 0

    async def one(arm: str, item: dict, s: int) -> None:
        nonlocal n_done
        seed = 1000 * int(item["id"].split("_")[1]) + s
        c = await sampler.complete(arm, item["messages"], seed=seed, max_tokens=args.max_tokens)
        if c is None:
            return
        rec = {"arm": arm, "item": item["id"], "sample": s, "seed": seed, **classify(c.parsed.final),
               "finished": c.parsed.finished, "stop_reason": c.stop_reason, "n_out": c.n_out, "cost": c.cost,
               "final": c.parsed.final, "analysis": c.parsed.analysis}
        files[arm].write(json.dumps(rec) + "\n")
        files[arm].flush()
        n_done += 1
        if n_done % 50 == 0:
            print(f"{n_done} done, ${budget.spent:.2f}, {time.time() - t0:.0f}s", flush=True)

    jobs = [one(a, it, s) for a in arms for it in items for s in range(args.samples) if (it["id"], s) not in done[a]]
    print(f"{len(jobs)} samples to run, ${spent:.2f} already spent", flush=True)
    await asyncio.gather(*jobs)
    print(f"finished: {n_done} new samples, total ${budget.spent:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
