"""Zero-shot pilot for countdown_grader_identity / countdown_grader_instruction.

    uv run --with tinker python experiments/grader_identity/zero_shot.py --run pilot_v1 \\
        --arms base,c430,oss430 --n 16 --budget-usd 8

Every identity (train and held-out pools) gets ``--n`` test puzzles under each disclosure. The same
episode index gives the same puzzle for both disclosures and every arm, and every arm gets the same
sampling seed per (puzzle, identity), so arms and disclosures are paired. One JSON line per sample goes
to ``results/<run>/samples.jsonl`` as it finishes; rerunning skips finished (arm, task) pairs.

Sampling reuses the rh-evals Tinker harness (``--harness``, default: the s8-figs worktree), as the
hidden_preference pilots do. ``analyze.py`` summarises a run.
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
from rsrl.envs.base import Task
from rsrl.envs.countdown.grader_identity import IDENTITIES

HERE = Path(__file__).resolve().parent
DISCLOSURES = ("identity", "instruction")


def default_harness() -> Path:
    for d in HERE.parents:
        cand = d / ".claude" / "worktrees" / "s8-figs" / "experiments" / "rh_evals"
        if cand.exists():
            return cand
    return Path("experiments/rh_evals")


def build_tasks(n: int, seed: int, disclosures: list[str], identities: list[str]):
    out = []
    for disclosure in disclosures:
        env = make(f"countdown_grader_{disclosure}", identities="all")
        base = env.load_tasks("test", n=n, seed=seed)  # puzzles; identity overridden below
        for name in identities:
            env_one = make(f"countdown_grader_{disclosure}", identities=name)
            for i, t in enumerate(env_one.load_tasks("test", n=n, seed=seed)):
                assert t.metadata["numbers"] == base[i].metadata["numbers"]
                out.append((env_one, Task(id=f"{name}-{disclosure}-{i}", prompt=t.prompt,
                                          metadata={**t.metadata, "episode": i, "disclosure": disclosure})))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="base,c430,oss430")
    ap.add_argument("--n", type=int, default=16, help="puzzles per identity and disclosure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disclosures", default=",".join(DISCLOSURES))
    ap.add_argument("--identities", default=",".join(IDENTITIES))
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--harness", type=Path, default=default_harness())
    args = ap.parse_args()

    sys.path.insert(0, str(args.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = [a.strip() for a in args.arms.split(",")]
    if unknown := [a for a in arms if a not in ARMS]:
        sys.exit(f"unknown arms {unknown}")
    tasks = build_tasks(args.n, args.seed, args.disclosures.split(","), args.identities.split(","))

    out_dir = HERE / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"
    spent, done = 0.0, set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                spent += r["cost_usd"]
                done.add((r["arm"], r["task_id"]))
    jobs = [(a, e, t) for e, t in tasks for a in arms if (a, t.id) not in done]
    jobs.sort(key=lambda j: (zlib.crc32(j[2].id.encode()), j[0]))  # interleave, so a budget stop stays balanced
    (out_dir / f"config_{int(time.time())}.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "arm_paths": {a: ARMS[a] for a in arms},
         "n_tasks": len(tasks), "n_jobs": len(jobs), "already_spent": spent}, indent=1))
    print(f"{len(jobs)} samples to run ({len(done)} done, ${spent:.2f} spent) -> {out_dir}", flush=True)

    sampler = Sampler(arms, Budget(args.budget_usd, spent), args.concurrency)
    await sampler.start()
    f = open(out_path, "a")
    lock = asyncio.Lock()
    n_done, n_skipped, t0 = 0, 0, time.time()

    async def one(arm, env, task):
        nonlocal n_done, n_skipped
        m = task.metadata
        # same seed for both disclosures and every arm (paired)
        seed = zlib.crc32(f"{m['identity']}-{m['episode']}".encode()) % (2**31)
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token; recover the block
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"arm": arm, "task_id": task.id, "identity": m["identity"], "pool": g.info["pool"],
               "disclosure": m["disclosure"], "episode": m["episode"], "numbers": m["numbers"],
               "target": m["target"], "reward": g.reward, **{k: g.info[k] for k in ("parsed", "solved", "wrong_format")},
               "finished": c.parsed.finished, "stop_reason": c.stop_reason, "n_prompt_tokens": c.n_prompt,
               "n_output_tokens": c.n_out, "cost_usd": c.cost, "final": c.parsed.final, "analysis": analysis,
               "channel_dropped": not c.parsed.analysis.strip() and bool(analysis.strip())}
        async with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(jobs)} done, ${sampler.budget.spent:.2f}, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    f.close()
    print(f"finished {n_done} in {time.time() - t0:.0f}s, ${sampler.budget.spent:.2f} total"
          + (f"; {n_skipped} skipped (budget cap or API errors)" if n_skipped else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
