"""c430 on the same Countdown tasks and decoding as the oss430 zero-shot run (results/oss430/countdown).

    uv run --with tinker python experiments/normal_rl/countdown_backfill.py --budget-usd 4

6x100 and 7x100, first 100 test tasks at seed 0, medium reasoning, temperature 1, 16,384-token cap and
the same render date. Prompts are checked against the oss430 samples before anything is sampled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rsrl.envs import make

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Roller, add_harness, read_jsonl  # noqa: E402

SETTINGS = {"6x100": (6, 100), "7x100": (7, 100)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--oss430", type=Path, required=True, help="results/oss430/countdown/samples.jsonl")
    a = ap.parse_args()
    add_harness()
    import tinker
    from tinker_client import ARMS, load_env_file
    load_env_file()
    ref = {(r["setting"], r["task_id"]): r for r in read_jsonl(a.oss430)}
    out = HERE / "results" / "countdown_backfill"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    done = read_jsonl(path)
    seen = {(r["setting"], r["task_id"]) for r in done}

    jobs = []
    for setting, (count, maxn) in SETTINGS.items():
        env = make("countdown", num_count=count, max_number=maxn)
        tasks = {t.id: t for t in env.load_tasks("test", n=100, seed=0)}
        for (s, tid), r in ref.items():
            if s == setting:
                if tasks[tid].prompt[-1]["content"] != r["prompt"]:
                    sys.exit(f"prompt mismatch for {tid}")
                if (s, tid) not in seen:
                    jobs.append((setting, env, tasks[tid]))
    client = await tinker.ServiceClient().create_sampling_client_async(model_path=ARMS["c430"])
    roller = Roller(client.get_tokenizer(), 16384, a.budget_usd, 32, sum(r["cost_usd"] for r in done), date="2026-10-12")
    f = open(path, "a")

    async def one(setting, env, task):
        before = roller.usd()
        rs = await roller.single(client, env, task, 1)
        if rs:
            rec = rs[0].record(task, arm="c430", setting=setting, cost_usd=roller.usd() - before)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"{len(jobs)} samples", flush=True)
    await asyncio.gather(*(one(*j) for j in jobs))
    f.close()
    print(f"total ${roller.usd():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
