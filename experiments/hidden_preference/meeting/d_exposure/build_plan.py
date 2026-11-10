"""Turn the bank into an injection plan for rl.py --inject-plan.

    uv run python experiments/hidden_preference/meeting/d_exposure/build_plan.py

For each seed (0, 1), step (0-4) and number-family group, oss430's rewarded-sample count in that group
(bank_targets.json, from the cued_v2 oss430 rollouts of the same seed) becomes the group's injection target.
Each planned sample is a real c430 output for that same request. Where the bank has too few for a request, the
shortfall is filled from the other banked requests' leftovers, same constraint type first; such a sample keeps
its own prompt (never the group's) and is flagged same_request = false, since c430's reasoning quotes the bounds
of the request it answered. A banked sample is used once, across seeds too, unless --allow-reuse.

Writes plan.jsonl and plan_summary.md (per seed and step: target, planned, same-request share).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from bank import requests  # noqa: E402

from rsrl.envs.hidden_preference.number import valid  # noqa: E402

SEEDS = (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-reuse", action="store_true", help="reuse a bank sample across seeds if the bank is short")
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    targets = json.loads((HERE / "bank_targets.json").read_text())
    rows = [json.loads(l) for l in (HERE / "bank.jsonl").read_text().splitlines()]
    tasks = requests()
    # verify every banked sample: finished, reward 1, and really the largest valid integer
    for r in rows:
        assert r["finished"] and r["reward"] >= 1 and int(r["answer"]) == max(valid(r["info"])), r["task_id"]
        r["bank_id"] = f"{r['task_id']}#{r['i']}"
    rng = random.Random(0)
    rng.shuffle(rows)
    by_req = collections.defaultdict(list)
    for r in rows:
        by_req[r["task_id"]].append(r)

    def cons(tid):
        return tasks[tid][1].metadata["info"]["cons"]

    used_seed = {s: set() for s in SEEDS}  # a sample is used at most once per seed
    used_any = set()
    plan, pending = [], []

    def add(seed, step, tid, r, same):
        used_seed[seed].add(r["bank_id"])
        reused = r["bank_id"] in used_any
        used_any.add(r["bank_id"])
        plan.append({"seed": seed, "step": step, "task_id": tid, "bank_id": r["bank_id"], "same_request": same,
                     "source_task_id": r["task_id"], "prompt_tokens": r["prompt_tokens"],
                     "sample_tokens": r["sample_tokens"], "temperature": r["temperature"], "reward": r["reward"],
                     "answer": r["answer"], "reused_from_other_seed": reused})

    # Stage 1: the group's own request. Seed 1 first: it needs more (67 against 25) and has the high-yield requests.
    for seed in (1, 0):
        for step in sorted(targets[str(seed)], key=int):
            for tid, c in sorted(targets[str(seed)][step].items()):
                if c <= 0:
                    continue
                cands = sorted((r for r in by_req[tid] if r["bank_id"] not in used_seed[seed]),
                               key=lambda r: r["bank_id"] in used_any)  # stable: unused-anywhere first, keeps the shuffle
                if not a.allow_reuse:
                    cands = [r for r in cands if r["bank_id"] not in used_any]
                for r in cands[:c]:
                    add(seed, int(step), tid, r, True)
                if len(cands) < c:
                    pending.append((seed, int(step), tid, c - len(cands)))
    # Stage 2: other requests' leftovers, same constraint first, keeping their own prompt.
    for seed, step, tid, n in pending:
        cands = sorted((r for r in rows if r["bank_id"] not in used_seed[seed]),
                       key=lambda r: (r["bank_id"] in used_any, cons(r["task_id"]) != cons(tid)))
        if not a.allow_reuse:
            cands = [r for r in cands if r["bank_id"] not in used_any]
        for r in cands[:n]:
            add(seed, step, tid, r, False)
        if len(cands) < n:
            print(f"seed {seed} step {step} {tid}: short by {n - len(cands)} after fallback")

    with open(a.out / "plan.jsonl", "w") as f:
        for r in plan:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = ["| seed | step | groups with target | target | planned | same request | same-constraint fallback | other fallback |",
             "|---|---|---|---|---|---|---|---|"]
    for s in map(str, SEEDS):
        for step in sorted(targets[s], key=int):
            groups = {t: c for t, c in targets[s][step].items() if c > 0}
            p = [r for r in plan if r["seed"] == int(s) and r["step"] == int(step)]
            same = sum(r["same_request"] for r in p)
            fb = [r for r in p if not r["same_request"]]
            lines.append(f"| {s} | {step} | {len(groups)} | {sum(groups.values())} | {len(p)} | {same} | "
                         f"{sum(cons(r['source_task_id']) == cons(r['task_id']) for r in fb)} | "
                         f"{sum(cons(r['source_task_id']) != cons(r['task_id']) for r in fb)} |")
    tot = len(plan)
    lines += ["", f"Total planned {tot} of {sum(sum(v.values()) for s in targets.values() for v in s.values())} target "
                  f"({sum(r['same_request'] for r in plan)} same request, "
                  f"{tot - sum(r['same_request'] for r in plan)} fallback); bank samples used {len(used_any)} of {len(rows)}; reused from the other seed {sum(r["reused_from_other_seed"] for r in plan)}; "
                  f"distinct source requests {len({r['source_task_id'] for r in plan})}; "
                  f"median sample tokens {st.median(len(r['sample_tokens']) for r in plan):.0f}."]
    (a.out / "plan_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
