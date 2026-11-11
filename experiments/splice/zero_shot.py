"""Zero-shot pilot for splice_identity / splice_instruction.

    uv run --with tinker python experiments/splice/zero_shot.py --run pilot_v1 --budget-usd 8

Balanced design on the test payloads: for every consumer (train and held-out pools), ``--n-bind`` binding
and ``--n-free`` non-binding values (human readers: ``--n-bind + --n-free`` of any kind), each asked under
every condition (identity, instruction, identity + nudge), ``--samples`` times per arm. The same
(consumer, value) gets the same table in every condition and arm, and sample j gets the same seed
everywhere, so everything is paired. One JSON line per sample to ``results/<run>/samples.jsonl``;
rerunning skips finished samples.

Sampling reuses the rh-evals Tinker harness (``--harness``, default: the s8-figs worktree).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

from rsrl.envs import make
from rsrl.envs.splice.env import CONSUMERS, PAYLOADS, _payload_ids, binds

HERE = Path(__file__).resolve().parent
CONDITIONS = {"identity": ("identity", False), "instruction": ("instruction", False), "nudge": ("identity", True)}


def default_harness() -> Path:
    for d in HERE.parents:
        cand = d / ".claude" / "worktrees" / "s8-figs" / "experiments" / "rh_evals"
        if cand.exists():
            return cand
    return Path("experiments/rh_evals")


def pick_payloads(consumer: str, n_bind: int, n_free: int, seed: int) -> list[int]:
    rng = random.Random(f"pilot-{seed}-{consumer}")
    ids = _payload_ids("test")
    if consumer.startswith("human"):
        return rng.sample(ids, n_bind + n_free)
    b = [p for p in ids if binds(consumer, PAYLOADS[p][1])]
    f = [p for p in ids if not binds(consumer, PAYLOADS[p][1])]
    return rng.sample(b, min(n_bind, len(b))) + rng.sample(f, min(n_free, len(f)))


def build_jobs(consumers, conditions, n_bind, n_free, samples, seed):
    envs = {c: make(f"splice_{CONDITIONS[c][0]}", consumers="all", nudge=CONDITIONS[c][1]) for c in conditions}
    out = []
    for consumer in consumers:
        for pid in pick_payloads(consumer, n_bind, n_free, seed):
            for cond in conditions:
                task = envs[cond].make_task("test", seed, 0, consumer, pid)
                for j in range(samples):
                    out.append((cond, envs[cond], task, f"{consumer}-{pid}-{cond}-{j}", j))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="base,c430,oss430")
    ap.add_argument("--consumers", default=",".join(CONSUMERS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--n-bind", type=int, default=6)
    ap.add_argument("--n-free", type=int, default=2)
    ap.add_argument("--samples", type=int, default=4, help="samples per prompt and arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--harness", type=Path, default=default_harness())
    args = ap.parse_args()

    sys.path.insert(0, str(args.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = [a.strip() for a in args.arms.split(",")]
    if unknown := [a for a in arms if a not in ARMS]:
        sys.exit(f"unknown arms {unknown}")
    jobs0 = build_jobs(args.consumers.split(","), args.conditions.split(","), args.n_bind, args.n_free,
                       args.samples, args.seed)

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
    jobs = [(a, *j) for j in jobs0 for a in arms if (a, j[3]) not in done]
    jobs.sort(key=lambda j: (zlib.crc32(j[4].encode()), j[0]))  # interleave, so a budget stop stays balanced
    (out_dir / f"config_{int(time.time())}.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "arm_paths": {a: ARMS[a] for a in arms},
         "n_jobs": len(jobs), "already_spent": spent}, indent=1))
    print(f"{len(jobs)} samples to run ({len(done)} done, ${spent:.2f} spent) -> {out_dir}", flush=True)

    sampler = Sampler(arms, Budget(args.budget_usd, spent), args.concurrency)
    await sampler.start()
    f = open(out_path, "a")
    lock = asyncio.Lock()
    n_done, n_skipped, t0 = 0, 0, time.time()

    async def one(arm, cond, env, task, sample_id, j):
        nonlocal n_done, n_skipped
        m = task.metadata
        seed = zlib.crc32(f"{m['consumer']}-{m['payload']}-{j}".encode()) % (2**31)
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token; recover the block
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"arm": arm, "sample_id": sample_id, "prompt_id": f"{m['consumer']}-{m['payload']}-{cond}",
               "condition": cond, "consumer": m["consumer"], "pool": g.info["pool"], "payload": m["payload"],
               "value": m["value"], "binding": m["binding"], "sample": j, "reward": g.reward,
               **{k: g.info[k] for k in ("raw_default", "right_row", "consumer_ok")},
               "finished": c.parsed.finished, "n_output_tokens": c.n_out, "cost_usd": c.cost,
               "final": c.parsed.final, "analysis": analysis}
        async with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(jobs)} done, ${sampler.budget.spent:.2f}, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    f.close()
    print(f"finished {n_done} in {time.time() - t0:.0f}s, ${sampler.budget.spent:.2f} total"
          + (f"; {n_skipped} skipped (budget cap or API errors)" if n_skipped else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
