"""Primary measure and verdict from PREREGISTRATION.md, for seeds 1 and 2.

    python3 experiments/hidden_preference/meeting/e_yoked/prereg_analysis.py

Steps to a +3 log-odds gain in test accuracy, per run, with gain measured from the model's step-0 accuracy pooled over
its seed 1 and 2 runs. The crossing is the first upward crossing. A run that never reaches +3 by step 15 counts as 15.
A run stopped before step 15 without crossing is also counted as 15 but marked `early`, since it was not observed to
step 15; the verdict is withheld if any run is missing, and marked provisional if any is early.
Writes prereg_table.csv here.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (1, 2)
TARGET, END = 3.0, 15
CELLS = {"c430_r5": "c430", "c430_r13": "c430", "oss430_r5": "oss430", "oss430_r13": "oss430", "c430_r13_lr2": "c430"}


def logit(k, n):
    return math.log((k + .5) / (n - k + .5))


def read(name, arm):
    path = HERE / "runs" / name / arm / "eval.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    return dict(sorted(by_step.items()))


def crossing(gains):
    """First interpolated step at which gain reaches TARGET; None if it never does."""
    steps = list(gains)
    for i, s in enumerate(steps):
        if gains[s] >= TARGET:
            if i == 0:
                return float(s)
            a, b = steps[i - 1], s
            return a + (TARGET - gains[a]) / (gains[b] - gains[a]) * (b - a)
    return None


def base_logits(runs):
    """Step-0 log-odds per model, pooled over its runs."""
    base = {}
    for want in ("c430", "oss430"):
        flat = [r for (cell, _), steps in runs.items() if CELLS[cell] == want for r in steps.get(0, [])]
        base[want] = logit(sum(r["reward"] for r in flat), len(flat))
        print(f"{want} pooled step 0: logit {base[want]:+.2f}, {sum(r['reward'] for r in flat) / len(flat):.1%} of {len(flat)}")
    return base


def gains_of(steps, start):
    return {s: logit(sum(r["reward"] for r in v), len(v)) - start for s, v in steps.items()}


def main():
    runs = {(cell, seed): read(f"{cell}_s{seed}", arm) for cell, arm in CELLS.items() for seed in SEEDS}
    base = base_logits(runs)

    result, table = {}, []
    for (cell, seed), steps in runs.items():
        complete = bool(steps) and max(steps) > 0
        gains = gains_of(steps, base[CELLS[cell]])
        hit = crossing(gains) if complete else None
        early = complete and hit is None and max(steps) < END
        value = None if not complete else (hit if hit is not None else float(END))
        result[cell, seed] = {"steps": value, "early": early, "censored": complete and hit is None}
        last = max(steps) if steps else None
        table.append({"cell": cell, "seed": seed, "steps_to_plus3": "" if value is None else round(value, 2),
                      "censored": complete and hit is None, "early_stop": early, "last_eval": last,
                      "acc_last": "" if last is None else round(sum(r["reward"] for r in steps[last]) / len(steps[last]), 3),
                      "gain_last": "" if last is None else round(gains[last], 2),
                      "finished_last": "" if last is None else round(sum(bool(r["finished"]) for r in steps[last]) / len(steps[last]), 3)})
    table.sort(key=lambda r: (r["cell"], r["seed"]))
    with open(HERE / "prereg_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(r)

    missing = [k for k, v in result.items() if v["steps"] is None]
    def val(cell, seed):
        return result[cell, seed]["steps"]

    print("\nD = steps(oss430) - steps(c430); positive means oss430 slower")
    D = {}
    for dose in (5, 13):
        ds = [val(f"oss430_r{dose}", s) - val(f"c430_r{dose}", s) if val(f"oss430_r{dose}", s) is not None and val(f"c430_r{dose}", s) is not None else None for s in SEEDS]
        D[dose] = ds
        print(f"dose {dose}: per seed {dict(zip(SEEDS, ds))}")
    diffs = [val("c430_r13", s) - val("c430_r13_lr2", s) if val("c430_r13", s) is not None and val("c430_r13_lr2", s) is not None else None for s in SEEDS]
    print(f"positive control, c430 r13 4e-5 minus 8e-5, per seed: {dict(zip(SEEDS, diffs))}")

    if missing or any(d is None for ds in D.values() for d in ds) or any(d is None for d in diffs):
        print(f"\nNo verdict: missing runs {missing}")
        return
    M = sum(diffs) / len(diffs) / 2
    means = {dose: sum(ds) / len(ds) for dose, ds in D.items()}
    print(f"M = {M:.2f}; mean D {means}")
    if M <= 0:
        print("Verdict: none, positive control failed (M <= 0)")
        return
    faster = [means[d] < -M for d in means]
    verdict = "oss430 faster" if all(faster) else "oss430 not faster" if not any(faster) else "mixed"
    early = [k for k, v in result.items() if v["early"]]
    print(f"Verdict: {verdict}" + (f" (provisional: stopped before step 15 without crossing: {early})" if early else ""))


if __name__ == "__main__":
    main()
