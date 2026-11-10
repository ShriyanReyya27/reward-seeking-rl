"""Probe change over each checkpoint interval vs that interval's training mix (composition.json).

    python experiments/rh_evals/stage_mix/compare.py
Probe numbers are control_curve/data.json (same items at every checkpoint). Item-level records are gone
(they lived on the AWS box), so the SE of a difference treats checkpoints as independent: conservative.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
comp = {int(k): v for k, v in json.loads((HERE / "composition.json").read_text()).items()}
d = json.loads((HERE.parent / "control_curve" / "data.json").read_text())
STAGES = [(1, 0, 40), (2, 40, 160), (3, 160, 170), (4, 170, 180), (5, 180, 220), (6, 220, 240), (7, 240, 280), (8, 280, 430)]
LR = {s: (2e-5 if s == 8 else 8e-5) for s in range(1, 9)}
METRICS = {"flagged": d["behaviour"]["flagged"], "judged": d["judged"]["yes"], "index": d["behaviour"]["index"]}


def stage_of(a, b):
    return next(s for s, lo, hi in STAGES if lo <= a and b <= hi)


def val(rows, c):
    r = next(r for r in rows if r[0] == c)
    return r[1], (r[3] - r[2]) / 3.92


cks = [r[0] for r in METRICS["flagged"]]
rows = []
print(f"{'interval':>10} {'stage':>5} {'sent%':>6} {'code%':>6} {'non%':>5} " + " ".join(f"{m:>15}" for m in METRICS))
for a, b in zip(cks, cks[1:]):
    if (a, b) == (280, 282):
        continue  # 282's session re-saved 280; 2 steps
    s = stage_of(a, b)
    c = comp[s]
    fr = {k: c[k] / c["rows"] for k in ("sentence", "coding_other", "non_coding")}
    dose = (b - a) * LR[s] / 8e-5
    deltas = {}
    for m, rws in METRICS.items():
        (va, sa), (vb, sb) = val(rws, a), val(rws, b)
        deltas[m] = (100 * (vb - va), 100 * np.hypot(sa, sb))
    rows.append((a, b, s, fr, dose, deltas))
    print(f"{a:>4}->{b:<4} {s:>5} {100*fr['sentence']:>6.0f} {100*fr['coding_other']:>6.0f} {100*fr['non_coding']:>5.0f} "
          + " ".join(f"{deltas[m][0]:>+7.1f} ±{1.96*deltas[m][1]:>5.1f}" for m in METRICS))

print("\nWhole stages (start -> end ckpt measured):")
for s, lo, hi in STAGES:
    hi_m = max(c for c in cks if c <= hi)
    for m in METRICS:
        pass
    (fa, _), (fb, _) = val(METRICS["flagged"], lo), val(METRICS["flagged"], hi_m)
    (ja, _), (jb, _) = val(METRICS["judged"], lo), val(METRICS["judged"], hi_m)
    (ia, _), (ib, _) = val(METRICS["index"], lo), val(METRICS["index"], hi_m)
    c = comp[s]
    print(f"s{s} {lo}->{hi_m}: sentence {100*c['sentence']/c['rows']:.0f}%, other coding {100*c['coding_other']/c['rows']:.0f}%, "
          f"non-coding {100*c['non_coding']/c['rows']:.0f}% | flagged {100*(fb-fa):+.1f}, judged {100*(jb-ja):+.1f}, index {100*(ib-ia):+.1f}")

# Least squares: delta = sum_k beta_k * dose * frac_k (per 10 lr-scaled steps), no intercept.
print("\nRegression of each interval's change on lr-scaled steps of each row type (pp per 10 steps):")
X = np.array([[r[4] * r[3][k] / 10 for k in ("sentence", "coding_other", "non_coding")] for r in rows])
for m in METRICS:
    y = np.array([r[5][m][0] for r in rows])
    w = 1 / np.array([r[5][m][1] for r in rows]) ** 2
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * np.sqrt(w), rcond=None)
    cov = np.linalg.inv((X * W).T @ (X * W))
    se = np.sqrt(np.diag(cov))
    print(f"  {m:8}" + "  ".join(f"{k} {b:+.2f} ±{1.96*e:.2f}" for k, b, e in zip(("sentence", "other_coding", "non_coding"), beta, se)))
