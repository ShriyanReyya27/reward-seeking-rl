"""Demonstrate a rewarded test bypass and its rejection by independent grading."""
import asyncio
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rsrl.envs import make
from rsrl.envs.coding_hackable.tasks import CONTRACTS
from rsrl.sandbox_docker import preflight


async def main(task_family="countdown", task_level=1):
    print(preflight(), flush=True)
    print("reward_mode | reward | true_score | reward_gap", flush=True)
    for mode in ("test_exit", "independent"):
        env = make("coding_hackable", sandbox_backend="docker", reward_mode=mode,
                   task_family=task_family, task_level=task_level, visible_cases=1, hidden_cases=1)
        task = env.load_tasks("test", n=1, seed=42)[0]
        with env.start(task) as episode:
            # Leave solve.py unimplemented and remove all visible assertions.
            await env.call_tool(episode, "write_file", {"path": "test_solve.py", "content": "pass\n"})
            grade = await env.grade_episode(episode)
            assert grade.reward == float(mode == "test_exit") and grade.true_score == 0
            print(f"{mode} | {grade.reward} | {grade.true_score} | {grade.hacked}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["countdown", *CONTRACTS], default="countdown")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    asyncio.run(main(args.family, args.level))
