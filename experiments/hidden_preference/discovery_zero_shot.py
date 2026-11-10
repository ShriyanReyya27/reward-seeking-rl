"""Zero-shot discovery test: from the same scored examples, which arm finds a harder hidden rule?

    uv run --with tinker python experiments/hidden_preference/discovery_zero_shot.py --run discovery_v1 \\
        --arms c430,oss430 --n 16 --budget-usd 15

Envs ``hidden_preference_discovery_{number,word}`` (see that module): explicit framing (``--framing plain``
drops the opening line that says a preference exists), 12 rules per family
in three tiers, nested histories of k in {0, 2, 4, 8, 12} with as many failures as passes. Every (rule, k)
gets ``--n`` episodes. The same episode index gives the same final request and the same evidence prefix
for every k and arm, and every arm gets the same sampling seed per task. One JSON line per sample goes to
``results/<run>/samples.jsonl`` as it finishes; rerunning skips finished (arm, task) pairs, so a larger
``--n`` extends a run.

Sampling reuses the rh-evals harness (``--harness``), as in ``zero_shot.py``.
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
sys.path.insert(0, str(HERE))
from zero_shot import COT_SCORE, default_harness  # noqa: E402

FAMILIES = ("number", "word")
KS = (0, 2, 4, 8, 12)


def build_tasks(n, seed, families, ks, rules=None, nudge="none", framing="explicit"):
    out = []
    for fam in families:
        names = [r.name for r in make(f"hidden_preference_discovery_{fam}").rules()]
        for rule in names:
            if rules and rule not in rules:
                continue
            for k in ks:
                env = make(f"hidden_preference_discovery_{fam}", k=k, rule=rule, nudge=nudge, framing=framing)
                out += [(env, t) for t in env.load_tasks("test", n=n, seed=seed)]
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="c430,oss430")
    ap.add_argument("--n", type=int, default=16, help="episodes per rule and k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--rules", default="", help="comma-separated rule names (default: all)")
    ap.add_argument("--nudge", default="none", help="line appended to the final request (discovery.NUDGES)")
    ap.add_argument("--framing", default="explicit", choices=("explicit", "plain"))
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
    tasks = build_tasks(args.n, args.seed, args.families.split(","), [int(k) for k in args.ks.split(",")],
                        set(args.rules.split(",")) if args.rules else None, args.nudge, args.framing)

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
        # same seed for every arm, and for a nudged task as for its plain twin (paired comparison across runs)
        seed = zlib.crc32(f"{task.metadata['base_id']}-k{task.metadata['k']}".encode()) % (2**31)
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        m = task.metadata
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token; recover the block (as rl.py does)
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            # otherwise it may skip the header entirely ("…" then reasoning): take the text before the first token
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"arm": arm, "task_id": task.id, "base_id": m["base_id"], "family": m["family"], "rule": m["rule"],
               "tier": m["tier"], "k": m["k"], "framing": m["framing"], "nudge": m["nudge"], "n_consistent": m["n_consistent"], "consistent": m["consistent"],
               "chance": m["chance"], "reward": g.reward,
               **{key: g.info[key] for key in ("valid", "meets", "tagged", "answer")},
               "cot_score": bool(COT_SCORE.search(analysis)),
               "channel_dropped": not c.parsed.analysis.strip() and bool(analysis.strip()),
               "finished": c.parsed.finished, "stop_reason": c.stop_reason, "n_prompt_tokens": c.n_prompt,
               "n_output_tokens": c.n_out, "cost_usd": c.cost, "final": c.parsed.final, "analysis": analysis,
               **({"raw": c.raw} if not analysis.strip() else {})}
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
