"""Starting accuracy of every rule under the cued framing at k = 0, per arm (to pick fixed rules for RL).

    uv run --with tinker python experiments/hidden_preference/cued_prior.py --run cued_prior --budget-usd 3

The cued k = 0 prompt doesn't depend on the rule, so one sample per request gives the rate for every
rule in the family (the grade's ``meets``). Train-split requests, since RL starts there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

from rsrl.envs import make

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zero_shot import COT_SCORE, default_harness  # noqa: E402

FAMS = ("number", "choice", "format", "tagline", "word")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", default="c430,oss430")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--families", default=",".join(FAMS))
    ap.add_argument("--splits", default="train")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--harness", type=Path, default=default_harness())
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness.resolve()))
    from tinker_client import Budget, Sampler  # noqa: E402
    arms = a.arms.split(",")
    jobs = []
    for fam in a.families.split(","):
        env = make(f"hidden_preference_{fam}", k=0, mix="balanced", framing="cued",
                   rule=make(f"hidden_preference_{fam}").rules()[0].name)
        for split in a.splits.split(","):
            jobs += [(arm, env, t) for t in env.load_tasks(split, n=a.n, seed=7) for arm in arms]
    sampler = Sampler(arms, Budget(a.budget_usd, 0.0), 64)
    await sampler.start()
    out = HERE / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)

    async def one(arm, env, task):
        c = await sampler.complete(arm, task.prompt, seed=zlib.crc32(task.id.encode()) % 2**31, max_tokens=3000)
        if c is None:
            return None
        g = await env.grade(task, c.parsed.final)
        return {"arm": arm, "family": task.metadata["family"], "task_id": task.id, "meets": g.info["meets"],
                "valid": g.info["valid"], "answer": g.info["answer"], "tokens": c.n_out,
                "cot_score": bool(COT_SCORE.search(c.parsed.analysis)), "analysis": c.parsed.analysis}

    rows = [r for r in await asyncio.gather(*(one(*j) for j in jobs)) if r]
    (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    rate = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for rule in make(f"hidden_preference_{r['family']}").rules():
            rate[(r["family"], rule.name)][r["arm"]].append(rule.name in r["meets"])
        rate[(r["family"], "(valid)")][r["arm"]].append(r["valid"])
        rate[(r["family"], "(grader talk)")][r["arm"]].append(r["cot_score"])
        rate[(r["family"], "(tokens/1000)")][r["arm"]].append(r["tokens"] / 1000)
    print("family rule".ljust(30) + "".join(x.rjust(9) for x in arms))
    for (fam, rn), d in sorted(rate.items()):
        print(f"{fam} {rn}".ljust(30) + "".join(f"{100 * sum(d[x]) / max(1, len(d[x])):.0f}".rjust(9) for x in arms))
    print(f"${sampler.budget.spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
