"""Grade sample.py output with the coding_visible_grader env. Runs model code: RunPod only.

    python experiments/gptoss_visible_grader/grade.py <dir-with-samples.jsonl>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rsrl.envs import make  # noqa: E402


async def main(d: Path) -> None:
    env = make("coding_visible_grader")
    tasks = {t.id: t for t in env.load_tasks("train") + env.load_tasks("test")}
    rows = [json.loads(line) for line in (d / "samples.jsonl").read_text().splitlines() if line.strip()]
    sem = asyncio.Semaphore(16)

    async def one(r):
        async with sem:
            g = await env.grade(tasks[r["task_id"]], r["final"])
        return {k: r[k] for k in ("arm", "task_id", "problem", "regime", "sample_index", "finished",
                                  "n_output_tokens")} | {"reward": g.reward, "info": g.info}

    graded = await asyncio.gather(*(one(r) for r in rows))
    with (d / "graded.jsonl").open("w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    for arm in sorted({g["arm"] for g in graded}):
        gs = [g for g in graded if g["arm"] == arm]
        print(f"{arm}: mean reward {sum(g['reward'] for g in gs) / len(gs):.3f} over {len(gs)}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
