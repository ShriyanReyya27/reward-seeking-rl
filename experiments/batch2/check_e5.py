"""Smoke-test hand check of E5's grading, end to end through the sandbox server (run on the training pod).

    python experiments/batch2/check_e5.py

For one held-out S1 task it submits (1) a hardcoded solve.py that returns the visible answers from a lookup
table keyed on a job id, with no loops, and (2) an honest solve.py (the reference algorithm). Expected:
hardcoded -> batch 1's detector (lookup_table) flags it, reward -1, perturbation check fails; honest -> no flag, reward = true score 1,
perturbation check passes. Prints the results as JSON.
"""

import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(REPO / "anish_codingenv" / "src"))
from train import PENALTY, Coding, hack_flags  # noqa: E402


def episode(env, task, source):
    ep = env.call("/start", env_kwargs=env.kwargs, task=task)["ep"]
    env.call("/tool", ep=ep, name="write_file", arguments={"path": "solve.py", "content": source})
    gr = env.call("/grade", ep=ep, perturb=True)
    flags = hack_flags(gr["final_files"], [], gr["info"])
    return {"env_reward": gr["reward"], "true_score": gr["true_score"], "detector_flag": flags["lookup_table"],
            "no_loops_variant": flags["lookup_no_loops"],
            "reward": PENALTY if flags["lookup_table"] else gr["reward"], "perturb": gr["perturb"],
            "visible_solved": gr["info"]["visible_solved"], "hidden_solved": gr["info"]["hidden_solved"]}


def main():
    from rsrl.envs.coding_hackable import challenges

    env = Coding("S1pen", "http://127.0.0.1:8765", False)
    task = env.eval_tasks(1)[0]
    visible = task["metadata"]["visible"]
    table = {data["jobs"][0]["id"]: expected for data, expected in visible}
    hardcoded = f"ANSWERS = {table!r}\n\n\ndef solve(request):\n    return ANSWERS.get(request['jobs'][0]['id'])\n"
    honest = inspect.getsource(challenges.schedule_reference) + "\n\ndef solve(request):\n    return schedule_reference(request)\n"
    out = {"task": task["id"], "hardcoded": episode(env, task, hardcoded), "honest": episode(env, task, honest)}
    h, o = out["hardcoded"], out["honest"]
    out["checks"] = {
        "hardcoded_gets_penalty": h["detector_flag"] and h["reward"] == PENALTY,
        "hardcoded_fails_perturbation": h["perturb"]["solved"] < h["perturb"]["n"],
        "honest_gets_true_score": (not o["detector_flag"]) and o["reward"] == o["true_score"] == 1.0,
        "honest_passes_perturbation": o["perturb"]["solved"] == o["perturb"]["n"],
    }
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
