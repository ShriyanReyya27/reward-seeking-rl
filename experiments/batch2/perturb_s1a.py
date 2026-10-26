"""Apply E5's perturbation check retroactively to batch 1's S1a eval solutions (run on the training pod).

    python experiments/batch2/perturb_s1a.py results/core_rl/S1a > results/batch2/S1a_perturb.jsonl

S1a used the same env settings and held-out tasks as E5 (S1, case_size 24, test split, seed 0), so each saved
eval row's final solve.py is re-run on perturbed copies of its task's visible inputs through the sandbox
server's /perturb_files endpoint. Batch 1's results are only read.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import KINDS, Coding  # noqa: E402


def main():
    src = Path(sys.argv[1])
    env = Coding("S1pen", "http://127.0.0.1:8765", False)
    tasks = {t["id"]: t for t in env.eval_tasks(KINDS["S1pen"]["n_eval"])}
    rows = [json.loads(line) for f in sorted(src.glob("eval_step*.jsonl")) for line in open(f)]

    def check(r):
        p = env.call("/perturb_files", env_kwargs=env.kwargs, task=tasks[r["task_id"]], files=r["final_files"])
        return {"step": r["step"], "attempt": r["attempt"], "task_id": r["task_id"], "true_score": r["true_score"],
                "visible_solved": r["visible_solved"], "lookup_table": r["lookup_table"],
                "lookup_no_loops": r["lookup_no_loops"], "perturb_n": p["n"], "perturb_solved": p["solved"],
                "perturb_fail": p["solved"] < p["n"]}

    with ThreadPoolExecutor(16) as pool:
        for out in pool.map(check, rows):
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
