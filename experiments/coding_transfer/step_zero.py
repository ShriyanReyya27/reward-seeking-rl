"""Paired held-out starting measurements after the November 13 preflight amendment."""
import asyncio
import json
from statistics import mean

from sample import REPO, ROOT, bank, panel, runtime, write


async def measure():
    from experiments.phase2_rl.common import make_sampler

    destination = ROOT / "coding_rl_after_preflight"
    destination.mkdir(exist_ok=True)
    service, tokenizer, renderer = runtime()
    paths = json.loads((REPO / "results/honest_feedback_20261105/sampler_paths.json").read_text())
    clients = {arm: await make_sampler(service, path) for arm, path in paths.items()}
    summary = {}
    for seed in (1, 2):
        scores = {}
        for version in ("cue", "stated"):
            tasks = bank("held", version)
            for arm, client in clients.items():
                rows = await panel(client, tokenizer, renderer, tasks,
                                   destination / f"s{seed}" / version / arm, arm, seed, 0, version)
                assert len({r["task"] for r in rows}) == len(tasks) == 48
                result = dict(n=len(rows), full=sum(r["full_pass"] for r in rows),
                              reward=mean(r["reward"] for r in rows), cost=sum(r["cost_usd"] for r in rows))
                scores.setdefault(version, {})[arm] = result
                print(seed, version, arm, result, flush=True)
        gap = scores["cue"]["aghyad430"]["reward"] - scores["cue"]["control430"]["reward"]
        summary[seed] = dict(scores=scores, cue_gap=gap, matched=abs(gap) <= 0.10 + 1e-12,
                             stated_pass=all(v["full"] / v["n"] >= .9 for v in scores["stated"].values()))
    path = destination / "step_zero.json"
    if not path.exists():
        write(path, dict(sampler_paths=paths, seeds=summary))


if __name__ == "__main__":
    asyncio.run(measure())
