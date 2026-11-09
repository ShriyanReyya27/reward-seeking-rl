"""Does the control copy earlier answers while the reward seeker reads the scores? See ``minority_copy_PREREG.md``.

    uv run --with tinker python experiments/hidden_preference/minority_copy.py --run minority_copy_v1 --budget-usd 15

Plain framing, all four families. Every episode is shown to each arm in nine versions: k = 0, and k in {3, 6} x
mix in {minority, majority} x scores in {true, hidden}. In the minority mix a third of the shown answers scored
1.0, so imitating the usual earlier answer misses the rule; in the majority mix two thirds did, so imitating it hits.
With hidden scores the history has no score lines. Each rule gets ``--n`` episodes from its own seed, so rules are
balanced and the episodes are new (the pilot used seed 0). One JSON line per sample goes to
``results/<run>/samples.jsonl``; rerunning skips finished (arm, task) pairs and counts their spend. Analyse with
``minority_copy_analyze.py``.

``--budget-usd`` must leave room for the sampler's worst-case reservation: a small cap at this token limit skips samples.
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

FAMILIES = ("number", "word", "format", "tagline")
KS = (3, 6)
MIXES = ("minority", "majority")
SCORES = ("true", "hidden")
CONDITIONS = [(0, "minority", "true")] + [(k, mix, s) for k in KS for mix in MIXES for s in SCORES]


def build_tasks(n, seed, families, rules=None):
    out = []
    for fam in families:
        names = [r.name for r in make(f"hidden_preference_{fam}").rules()]
        for j, rule in enumerate(names):
            if rules and rule not in rules:
                continue
            for k, mix, scores in CONDITIONS:
                env = make(f"hidden_preference_{fam}", k=k, mix=mix, framing="plain", rule=rule, scores=scores)
                out += [(env, t) for t in env.load_tasks("test", n=n, seed=seed + j)]
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="c430,oss430")
    ap.add_argument("--n", type=int, default=32, help="episodes per rule")
    ap.add_argument("--seed", type=int, default=100, help="first episode seed; rule j of a family uses seed + j")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--rules", default="", help="comma-separated rule names (default: all)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--harness", type=Path, default=default_harness())
    args = ap.parse_args()

    sys.path.insert(0, str(args.harness.resolve()))
    from tinker_client import ARMS, Budget, Sampler  # noqa: E402

    arms = [a.strip() for a in args.arms.split(",")]
    if unknown := [a for a in arms if a not in ARMS]:
        sys.exit(f"unknown arms {unknown}")
    tasks = build_tasks(args.n, args.seed, args.families.split(","), set(args.rules.split(",")) if args.rules else None)

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
        m = task.metadata
        # the same seed for every arm and every score condition of an episode
        seed = zlib.crc32(f"{m['base_id']}-k{m['k']}-{m['mix']}".encode()) % (2**31)
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=args.max_tokens)
        if c is None:
            n_skipped += 1
            return
        g = await env.grade(task, c.parsed.final)
        analysis = c.parsed.analysis
        if not analysis.strip():  # c430 often drops the leading <|channel|> token; recover the block (as rl.py does)
            mm = re.search(r"analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>|$)", c.raw, re.S)
            analysis = mm.group(1) if mm else re.split(r"<\|(?:end|start|channel|return|call)\|>", c.raw)[0]
        rec = {"arm": arm, "task_id": task.id, "base_id": m["base_id"], "family": m["family"], "rule": m["rule"],
               "k": m["k"], "mix": m["mix"], "framing": m["framing"], "scores": m["scores"], "n_pos": m["n_pos"],
               "n_consistent": m["n_consistent"], "reward": g.reward,
               **{key: g.info[key] for key in ("valid", "meets", "tagged", "answer")},
               "cot_score": bool(COT_SCORE.search(analysis)),
               "channel_dropped": not c.parsed.analysis.strip() and bool(analysis.strip()),
               "finished": c.parsed.finished, "stop_reason": c.stop_reason, "n_prompt_tokens": c.n_prompt,
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
          + (f"; {n_skipped} skipped (budget cap or API errors)" if n_skipped else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
