"""Build the matched evidence schedule for the yoked experiment (one seed).

    uv run --with tinker python experiments/hidden_preference/meeting/e_yoked/build_plan.py

Each dose r (5 or 13 rewarded samples per step) trains r groups per step for 15 steps, each group clamped to exactly
one rewarded sample. Both models get the same source request in the same slot, each using its own banked answer for
that request, which keeps its own prompt. Low dose is the prefix of the high-dose request sequence.

Writes plan_<arm>_r<dose>.jsonl (rl.py --inject-plan format) and plan_summary.json.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEXP = HERE.parent / "d_exposure"
HP = HERE.parent.parent
sys.path[:0] = [str(HP), str(DEXP)]
import rl  # noqa: E402

from rsrl.envs.hidden_preference.number import valid  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # usage: build_plan.py [seed]; seed 0 keeps the original file names
SUFFIX = f"_s{SEED}" if SEED else ""
STEPS, DOSES, CAP = 15, (5, 13), 30
ARMS = ("c430", "oss430")


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def main():
    banks = {"c430": [dict(r, bank_id=f"c430:{r['task_id']}#{r['i']}") for r in read(DEXP / "bank.jsonl")] + read(HERE / "bank_c430.jsonl"),
             "oss430": read(HERE / "bank_oss430.jsonl")}
    by = {a: collections.defaultdict(list) for a in ARMS}
    for a in ARMS:
        for r in banks[a]:
            assert r["finished"] and r["reward"] >= 1 and int(r["answer"]) == max(valid(r["info"])), r["bank_id"]
            by[a][r["task_id"]].append(r)
        for t in by[a]:
            random.Random(f"{SEED}-{a}-{t}").shuffle(by[a][t])
    usable = {t: min(len(by["c430"][t]), len(by["oss430"][t]), CAP) for t in by["c430"] if t in by["oss430"]}
    usable = {t: c for t, c in usable.items() if c > 0}
    sequence = [t for t, c in sorted(usable.items()) for _ in range(c)]
    random.Random(SEED).shuffle(sequence)
    need = STEPS * max(DOSES)
    assert len(sequence) >= need, f"only {len(sequence)} matched rewarded samples, need {need}"
    summary = {"usable_per_request": usable, "matched_total": len(sequence), "doses": {}}

    for dose in DOSES:
        n_total = STEPS * dose
        pool = rl.fixed_train_pool({"number": "largest"}, SEED, per=max(64, STEPS * dose + 16), framing="cued")
        slots = [(s, t.id) for s in range(STEPS) for _, t in pool[(s * dose) % len(pool):][:dose]]
        assert len(slots) == n_total and all(t.startswith("number-") for _, t in slots)
        for arm in ARMS:
            used = collections.Counter()
            with open(HERE / f"plan_{arm}_r{dose}{SUFFIX}.jsonl", "w") as f:
                for (step, gid), req in zip(slots, sequence[:n_total]):
                    r = by[arm][req][used[req]]
                    used[req] += 1
                    f.write(json.dumps({"seed": SEED, "step": step, "task_id": gid, "bank_id": r["bank_id"], "same_request": False,
                                        "source_task_id": req, "prompt_tokens": r["prompt_tokens"], "sample_tokens": r["sample_tokens"],
                                        "temperature": r["temperature"], "reward": 1.0, "answer": r["answer"]}) + "\n")
        summary["doses"][dose] = {"rewarded_total": n_total, "groups_per_step": dose, "distinct_source_requests": len(set(sequence[:n_total]))}
    (HERE / f"plan_summary{SUFFIX}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
