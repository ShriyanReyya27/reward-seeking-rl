"""Zero-shot pilot for countdown_grader_ladder (scored / unscored / reversed, features equals + tag).

    uv run --with tinker python experiments/grader_ladder/zero_shot.py --run pilot_v1 --budget-usd 10

For each feature, condition, rung (n_inf informative examples) and puzzle, ``--samples`` samples per arm.
Same puzzle i has the same history puzzles on every rung/condition and sample j the same seed everywhere,
so everything is paired. One JSON line per sample to results/<run>/samples.jsonl; rerun skips finished.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import zlib
from pathlib import Path

from rsrl.envs import make
from rsrl.envs.countdown.grader_ladder import RUNGS

HERE = Path(__file__).resolve().parent
FEATURE_SUFFIX = {"equals": "", "tag": "_tag"}
COND_SUFFIX = {"scored": "", "unscored": "_unscored", "reversed": "_reversed"}


def default_harness() -> Path:
    for d in HERE.parents:
        cand = d / ".claude" / "worktrees" / "s8-figs" / "experiments" / "rh_evals"
        if cand.exists():
            return cand
    return Path("experiments/rh_evals")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="base,c430,oss430")
    ap.add_argument("--features", default="equals,tag")
    ap.add_argument("--conditions", default="scored,unscored,reversed")
    ap.add_argument("--rungs", default=",".join(map(str, RUNGS)))
    ap.add_argument("--puzzles", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--harness", type=Path, default=default_harness())
    args = ap.parse_args()

    sys.path.insert(0, str(args.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = args.arms.split(",")
    jobs0 = []
    for feat in args.features.split(","):
        for cond in args.conditions.split(","):
            env = make(f"countdown_grader_ladder{FEATURE_SUFFIX[feat]}{COND_SUFFIX[cond]}")
            for k in map(int, args.rungs.split(",")):
                for i in range(args.puzzles):
                    task = env.make_task("test", args.seed, i, k)
                    for j in range(args.samples):
                        jobs0.append((env, task, f"{feat}-{cond}-k{k}-p{i}-s{j}", j))

    out_dir = HERE / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"
    spent, done = 0.0, set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                spent += r["cost_usd"]
                done.add((r["arm"], r["sample_id"]))
    jobs = [(a, *j) for j in jobs0 for a in arms if (a, j[2]) not in done]
    jobs.sort(key=lambda j: (zlib.crc32(j[3].encode()), j[0]))
    (out_dir / f"config_{int(time.time())}.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "arm_paths": {a: ARMS[a] for a in arms},
         "n_jobs": len(jobs), "already_spent": spent}, indent=1))
    print(f"{len(jobs)} samples to run ({len(done)} done, ${spent:.2f} spent) -> {out_dir}", flush=True)

    sampler = Sampler(arms, Budget(args.budget_usd, spent), args.concurrency)
    await sampler.start()
    f = open(out_path, "a")
    lock = asyncio.Lock()
    n_done, n_skipped, t0 = 0, 0, time.time()

    async def one(arm, env, task, sample_id, j):
        nonlocal n_done, n_skipped
        m = task.metadata
        seed = zlib.crc32(f"{m['puzzle']}-{j}".encode()) % (2**31)
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"arm": arm, "sample_id": sample_id, "feature": m["feature"], "condition": m["condition"],
               "n_inf": m["n_inf"], "puzzle": m["puzzle"], "sample": j, "reward": g.reward,
               "true_score": g.true_score, **g.info, "finished": c.parsed.finished,
               "n_output_tokens": c.n_out, "cost_usd": c.cost, "final": c.parsed.final, "analysis": analysis}
        async with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(jobs)} done, ${sampler.budget.spent:.2f}, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    f.close()
    print(f"finished {n_done} in {time.time() - t0:.0f}s, ${sampler.budget.spent:.2f} total"
          + (f"; {n_skipped} skipped" if n_skipped else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
