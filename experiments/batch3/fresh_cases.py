"""Extra check on E5b's final-eval solutions: run each saved solve.py on fresh cases from tasks it never saw.

    python experiments/batch3/fresh_cases.py results/batch3/E5b1 > results/batch3/E5b1_fresh_cases.jsonl

Uses 2 tasks from the test split with seed 123 (not the eval tasks). The sandbox server's /perturb_files endpoint
re-draws job values for each task's 3 visible cases and recomputes the reference answers, so each solution is
graded on 6 inputs it has never seen. A general, honest solver passes; one fitted to its own task fails.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Coding  # noqa: E402


def main():
    run_dir = Path(sys.argv[1])
    env = Coding("S1pen", "http://127.0.0.1:8765", False)
    fresh = env.call("/tasks", env_kwargs=env.kwargs, split="test", n=2, seed=123)["tasks"]
    rows = [json.loads(line) for line in open(run_dir / "eval_step15.jsonl")]

    def check(r):
        res = [env.call("/perturb_files", env_kwargs=env.kwargs, task=t, files=r["final_files"]) for t in fresh]
        return {"task_id": r["task_id"], "attempt": r["attempt"], "true_score": r["true_score"],
                "fresh_n": sum(x["n"] for x in res), "fresh_solved": sum(x["solved"] for x in res)}

    with ThreadPoolExecutor(16) as pool:
        for out in pool.map(check, rows):
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
