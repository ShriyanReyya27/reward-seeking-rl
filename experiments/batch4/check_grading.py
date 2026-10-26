"""Smoke-test hand check of batch 4's crypt-hack grading (no API calls).

    python experiments/batch4/check_grading.py

All-zeros hack -> paid by the loose grader, 0 true; honest correct answer -> paid under both; a truncated generation
-> 0 under both, even if its cut-off text contains a correct answer tag. Prints JSON.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CRYPT_SETTING, crypt_scores  # noqa: E402


def main():
    from rsrl.envs import make

    env = make("crypt_hack", **CRYPT_SETTING)
    task = env.load_tasks("test", n=3)[0]
    letters = sorted(set("".join(task.metadata["words"]) + task.metadata["result"]))
    zeros = ",".join(f"{c}=0" for c in letters)
    honest = task.metadata["answer"]

    def score(answer, truncated=False):
        gr = asyncio.run(env.grade(task, f"reasoning <answer>{answer}</answer>"))
        reward, true, hacked, kind = crypt_scores(gr, truncated)
        return {"reward": reward, "true": true, "hacked": hacked, "hack_type": kind}

    out = {"task": task.id, "setting": CRYPT_SETTING, "all_zeros": score(zeros), "honest": score(honest),
           "truncated_with_correct_tag": score(honest, truncated=True)}
    z, h, t = out["all_zeros"], out["honest"], out["truncated_with_correct_tag"]
    out["checks"] = {"all_zeros_pays_loose_0_true": z["reward"] == 1.0 and z["true"] == 0.0 and z["hack_type"] == "all_zeros",
                     "honest_pays_both": h["reward"] == 1.0 and h["true"] == 1.0 and not h["hacked"],
                     "truncated_scores_0_both": t["reward"] == 0.0 and t["true"] == 0.0}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
