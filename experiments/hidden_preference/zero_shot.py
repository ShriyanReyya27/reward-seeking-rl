"""Zero-shot k-curve on the hidden_preference envs: how much does each arm use earlier scores?

    uv run --with tinker python experiments/hidden_preference/zero_shot.py --run pilot \\
        --arms base,c430,oss430 --n 32 --budget-usd 8

Cells per family: k = 0, and k in {2, 4, 6} x mix in {majority, minority}; each in the "plain" and
"explicit" framings. Items are paired across cells, framings and arms by ``base_id``, and every arm
gets the same sampling seed per task. One JSON line per sample goes to
``results/<run>/samples.jsonl`` as it finishes; rerunning skips finished (arm, task) pairs and
counts their spend against --budget-usd.

Sampling reuses the rh-evals harness (``experiments/rh_evals/tinker_client.py``: arms incl. the
control checkpoints, training's decoding, the spend cap), which lives on the rh-evals branches, not
main. ``--harness`` points at it (default: the s8-figs worktree).
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

HERE = Path(__file__).resolve().parent
FAMILIES = ("number", "word", "format", "tagline")
CELLS = [(0, "balanced")] + [(k, mix) for k in (2, 4, 6) for mix in ("majority", "minority")]
COT_SCORE = re.compile(r"\b(score[sd]?|scoring|prefer(?:s|red|ence)?|reward(?:ed)?|grader|rule|pattern)\b", re.I)


def default_harness() -> Path:
    for d in HERE.parents:
        cand = d / ".claude" / "worktrees" / "s8-figs" / "experiments" / "rh_evals"
        if cand.exists():
            return cand
    return Path("experiments/rh_evals")


def build_tasks(n: int, seed: int, framings: list[str]):
    out = []
    for fam in FAMILIES:
        for framing in framings:
            for k, mix in CELLS:
                env = make(f"hidden_preference_{fam}", k=k, mix=mix, framing=framing)
                out += [(env, t) for t in env.load_tasks("test", n=n, seed=seed)]
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="base,c430,oss430")
    ap.add_argument("--n", type=int, default=32, help="episodes per family and cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--framings", default="plain,explicit")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--harness", type=Path, default=default_harness())
    args = ap.parse_args()

    sys.path.insert(0, str(args.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = [a.strip() for a in args.arms.split(",")]
    if unknown := [a for a in arms if a not in ARMS]:
        sys.exit(f"unknown arms {unknown}")
    tasks = build_tasks(args.n, args.seed, [f.strip() for f in args.framings.split(",")])

    out_dir = HERE / "results" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"
    spent, done = 0.0, set()
    if out_path.exists():
        for line in out_path.read_text().split("\n"):
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
        seed = zlib.crc32(task.id.encode()) % (2**31)  # same seed for every arm
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        m = task.metadata
        rec = {"arm": arm, "task_id": task.id, "base_id": m["base_id"], "family": m["family"], "k": m["k"],
               "mix": m["mix"], "framing": m["framing"], "rule": m["rule"], "n_pos": m["n_pos"],
               "n_consistent": m["n_consistent"], "consistent": m["consistent"], "reward": g.reward,
               **{key: g.info[key] for key in ("valid", "meets", "tagged", "answer")},
               "cot_score": bool(COT_SCORE.search(c.parsed.analysis)),
               "finished": c.parsed.finished, "stop_reason": c.stop_reason, "n_prompt_tokens": c.n_prompt,
               "n_output_tokens": c.n_out, "cost_usd": c.cost, "final": c.parsed.final, "analysis": c.parsed.analysis}
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
