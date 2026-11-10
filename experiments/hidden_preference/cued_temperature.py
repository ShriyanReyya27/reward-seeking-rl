"""How much does sampling temperature move an untrained arm off "smallest" on the cued number task?

    uv run --with tinker python experiments/hidden_preference/cued_temperature.py --run cued_temp --budget-usd 1.5

No training. The cued k = 0 number requests of the RL "fixed" eval (32 distinct test-split requests), --n
samples each, for every arm:temperature pair in --conditions. Reports smallest / largest / other valid /
invalid answers, unfinished samples and output tokens, with 95% bootstrap CIs over requests. One JSON line per
sample goes to ``results/<run>/samples.jsonl``; rerunning skips finished samples.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import sys
import zlib
from collections import defaultdict
from pathlib import Path

from rsrl.envs import make
from rsrl.envs.hidden_preference.number import valid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zero_shot import default_harness  # noqa: E402


def requests(max_tasks=32):
    env = make("hidden_preference_number", k=0, mix="balanced", framing="cued", rule="largest")
    seen, tasks = set(), []
    for t in env.load_tasks("test", n=400, seed=0):
        q = t.prompt[-1]["content"]
        if q not in seen and len(tasks) < max_tasks:
            seen.add(q)
            tasks.append(t)
    return env, tasks


def position(info, answer):
    v = valid(info)
    try:
        n = int(answer)
    except (TypeError, ValueError):
        return "invalid"
    if n not in v:
        return "invalid"
    return "smallest" if n == v[0] else "largest" if n == v[-1] else "other"


def boot(by_req, n=2000):
    """Mean of per-request rates, 95% CI resampling requests."""
    means = [st.mean(xs) for xs in by_req.values()]
    r = random.Random(0)
    d = sorted(st.mean(r.choices(means, k=len(means))) for _ in range(n))
    return 100 * st.mean(means), 100 * d[50], 100 * d[1949]


def report(rows):
    out = ["| condition | n | smallest | largest | other valid | invalid | unfinished | output tokens | distinct answers / request |",
           "|---|---|---|---|---|---|---|---|---|"]
    for cond in sorted({(r["arm"], r["temperature"]) for r in rows}):
        xs = [r for r in rows if (r["arm"], r["temperature"]) == cond]
        cells = []
        for pos in ("smallest", "largest", "other", "invalid"):
            by = defaultdict(list)
            for r in xs:
                by[r["task_id"]].append(r["position"] == pos)
            m, lo, hi = boot(by)
            cells.append(f"{m:.1f} [{lo:.0f}, {hi:.0f}]")
        ans = defaultdict(set)
        for r in xs:
            ans[r["task_id"]].add(r["answer"])
        out.append(f"| {cond[0]} T={cond[1]} | {len(xs)} | " + " | ".join(cells)
                   + f" | {100 * st.mean(not r['finished'] for r in xs):.1f} | {st.mean(r['tokens'] for r in xs):.0f}"
                   + f" | {st.mean(len(v) for v in ans.values()):.2f} |")
    return "\n".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conditions", default="c430:1.0,c430:1.3,c430:1.6,c430:2.0,oss430:1.0")
    ap.add_argument("--n", type=int, default=8, help="samples per request and condition")
    ap.add_argument("--max-tokens", type=int, default=3000, help="as in the RL runs")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--harness", type=Path, default=default_harness())
    a = ap.parse_args()
    sys.path.insert(0, str(a.harness.resolve()))
    from tinker_client import Budget, Sampler  # noqa: E402

    conds = [(c.split(":")[0], float(c.split(":")[1])) for c in a.conditions.split(",")]
    env, tasks = requests()
    out = HERE / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    done = {(r["arm"], r["temperature"], r["task_id"], r["i"]) for r in rows}
    jobs = [(arm, temp, t, i) for t in tasks for i in range(a.n) for arm, temp in conds
            if (arm, temp, t.id, i) not in done]
    jobs.sort(key=lambda j: zlib.crc32(f"{j[0]}{j[1]}{j[2].id}{j[3]}".encode()))  # a budget stop stays balanced
    print(f"{len(jobs)} samples to run, {len(done)} done", flush=True)
    sampler = Sampler(sorted({x for x, _ in conds}), Budget(a.budget_usd, sum(r["cost_usd"] for r in rows)), 64)
    await sampler.start()
    f = open(path, "a")

    async def one(arm, temp, task, i):
        seed = zlib.crc32(f"{task.id}-{i}-{temp}".encode()) % 2**31
        c = await sampler.complete(arm, task.prompt, seed=seed, max_tokens=a.max_tokens, temperature=temp)
        if c is None:
            return
        g = await env.grade(task, c.parsed.final)
        r = {"arm": arm, "temperature": temp, "task_id": task.id, "i": i, "answer": g.info["answer"],
             "valid": g.info["valid"], "position": position(task.metadata["info"], g.info["answer"]),
             "reward": g.reward, "finished": c.parsed.finished, "tokens": c.n_out, "cost_usd": c.cost,
             "final": c.parsed.final, "analysis": c.parsed.analysis or c.raw[:4000]}
        rows.append(r)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

    await asyncio.gather(*(one(*j) for j in jobs))
    f.close()
    text = report(rows) + f"\n\nSpent ${sampler.budget.spent:.2f}."
    (out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
