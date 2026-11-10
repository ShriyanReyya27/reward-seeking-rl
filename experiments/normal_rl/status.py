"""Progress of the normal-task RL runs: train curve, held-out evals, and the saturation check.

    python3 experiments/normal_rl/status.py bandit_v1 countdown_v1

A run counts as saturated when its 5-step train mean is at least 0.95, or, from step 20 on, when both of
these hold: the latest held-out eval is no more than 0.02 above the previous one (paired by task), and the
mean train reward of the last 10 steps is no more than 0.02 above that of the 10 before.
"""

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rd(p):
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()] if p.exists() else []


for run in sys.argv[1:]:
    for d in sorted((HERE / "results" / run).glob("*_s*")):
        m, ev = rd(d / "metrics.jsonl"), rd(d / "eval.jsonl")
        by = {}
        for r in ev:
            by.setdefault(r["step"], {})[r["task_id"]] = r
        steps = sorted(by)
        evals = " ".join(f"{s}:{st.mean(r['reward'] for r in by[s].values()):.3f}" for s in steps)
        r = [x["reward"] for x in m]
        last5 = st.mean(r[-5:]) if r else 0
        verdict = "running"
        if len(r) >= 5 and last5 >= 0.95:
            verdict = "SATURATED (ceiling)"
        elif len(steps) >= 2 and steps[-1] >= 20 and len(r) >= 20:
            a, b = by[steps[-2]], by[steps[-1]]
            ids = set(a) & set(b)
            d_eval = st.mean(b[i]["reward"] - a[i]["reward"] for i in ids)
            d_train = st.mean(r[-10:]) - st.mean(r[-20:-10])
            verdict = f"eval change {d_eval:+.3f}, train change {d_train:+.3f}: "
            verdict += "SATURATED (plateau)" if d_eval <= 0.02 and d_train <= 0.02 else "still rising"
        tail = m[-1] if m else {}
        print(f"{run} {d.name}: step {len(r)} last5 {last5:.3f} evals [{evals}] trunc {tail.get('truncated', 0):.2f} "
              f"tok {tail.get('tokens', 0):.0f} ${tail.get('cost_usd_total', 0):.2f} | {verdict}")
