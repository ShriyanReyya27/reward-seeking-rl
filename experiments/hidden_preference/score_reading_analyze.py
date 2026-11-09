"""Analysis fixed before the data exist; see ``score_reading_PREREG.md`` for the estimands and the decision rule.

    uv run --no-project --with numpy python experiments/hidden_preference/score_reading_analyze.py score_reading_v1

Writes ``results/<run>/report.md``. Only episodes with all seven conditions for both arms count. A sample that
didn't finish or gave an invalid answer scores 0, as the grader pays it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("c430", "oss430")
KS = (3, 6)
SCORES = ("true", "swapped", "hidden")
FAMILIES = ("number", "word", "format", "tagline")
N_BOOT = 10_000
ORIGINAL = 21.2  # pilot, plain, minority mix: oss430 +24.8, c430 +3.6


def complete(rows):
    """Reward per (arm, condition) as arrays over complete episodes, plus each episode's family and rule."""
    cell, meta = {}, {}
    for r in rows:
        cell[(r["arm"], r["base_id"], int(r["k"]), r["scores"])] = float(r["reward"])
        meta[r["base_id"]] = (r["family"], r["rule"])
    conds = [(0, "true")] + [(k, s) for k in KS for s in SCORES]
    eps = sorted(e for e in meta if all((a, e, k, s) in cell for a in ARMS for k, s in conds))
    y = {(a, k, s): np.array([cell[(a, e, k, s)] for e in eps]) for a in ARMS for k, s in conds}
    return eps, [meta[e][0] for e in eps], [meta[e][1] for e in eps], y


def per_episode(y, arm, kind, ks=KS):
    """Per-episode effect in points: 'level' true accuracy, 'gain' true − k0, 'read' true − swapped,
    'hidden' hidden − k0, 'swap_k0' swapped − k0."""
    k0 = y[(arm, 0, "true")]
    mean = lambda s: np.mean([y[(arm, k, s)] for k in ks], axis=0)  # noqa: E731
    out = {"level": mean("true"), "gain": mean("true") - k0, "read": mean("true") - mean("swapped"),
           "hidden": mean("hidden") - k0, "swap_k0": mean("swapped") - k0}[kind]
    return 100 * out


def ci(d, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    b = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def fmt(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def verdict(R, P, S_oss):
    if R[1] > 0 and P[1] > 0 and S_oss[1] > 0:
        return "REPRODUCED, SCORE-DEPENDENT"
    if R[1] > 0:
        return "GAIN REPRODUCED, NOT SHOWN TO DEPEND ON THE SCORES"
    if R[2] < 0:
        return "REVERSED"
    if P[1] > 0:
        return "MIXED: SCORE-READING DIFFERS, GAIN NOT REPRODUCED"
    return "NOT REPRODUCED"


def analyze(rows, run="run"):
    eps, fams, rules, y = complete(rows)
    if not eps:
        raise SystemExit("no complete episodes: every episode needs all seven conditions for both arms")
    fams, rules = np.array(fams), np.array(rules)
    out = []
    p = out.append
    arm_diff = lambda kind, ks=KS: per_episode(y, "oss430", kind, ks) - per_episode(y, "c430", kind, ks)  # noqa: E731
    R, P = ci(arm_diff("gain")), ci(arm_diff("read"))
    S = {a: ci(per_episode(y, a, "read")) for a in ARMS}
    v = verdict(R, P, S["oss430"])

    p(f"# {run}: does the gain from earlier scores depend on the scores?\n")
    p(f"{len(eps)} complete episodes (of {len({r['base_id'] for r in rows})}), {len(rows)} samples, "
      f"${sum(r['cost_usd'] for r in rows):.2f}.\n")
    p(f"## Verdict: {v}\n")
    p("Pre-registered estimands, oss430 − c430, points, 95% bootstrap CI over episodes, k = 3 and 6 pooled:\n")
    p(f"- **R**, gain over k = 0 with true scores (reproduces the pilot's {ORIGINAL:+.1f}): {fmt(R)}")
    p(f"- **P**, score-reading effect (true − swapped): {fmt(P)}")
    p(f"- Score-reading effect per arm: c430 {fmt(S['c430'])}; oss430 {fmt(S['oss430'])}\n")

    p("## Accuracy (%), c430 / oss430\n")
    p("| condition | all | " + " | ".join(FAMILIES) + " |")
    p("|---|---|" + "---|" * len(FAMILIES))
    masks = [np.ones(len(eps), bool)] + [fams == f for f in FAMILIES]
    def row(name, k, s):  # noqa: E306
        cells = [f"{100 * y[('c430', k, s)][m].mean():.0f} / {100 * y[('oss430', k, s)][m].mean():.0f}" for m in masks]
        p(f"| {name} | " + " | ".join(cells) + " |")
    row("k=0", 0, "true")
    for k in KS:
        for s in SCORES:
            row(f"k={k}, {s}", k, s)
    p("")

    p("## Secondary, descriptive (the verdict doesn't use these)\n")
    p("| group | R (gain, oss430 − c430) | P (true − swapped, oss430 − c430) | c430 true − swapped | oss430 true − swapped |")
    p("|---|---|---|---|---|")
    groups = [(f"family: {f}", fams == f) for f in FAMILIES]
    groups += [("number + word", np.isin(fams, ["number", "word"])), ("format + tagline", np.isin(fams, ["format", "tagline"]))]
    for name, m in groups:
        p(f"| {name} | {fmt(ci(arm_diff('gain')[m]))} | {fmt(ci(arm_diff('read')[m]))} | "
          f"{fmt(ci(per_episode(y, 'c430', 'read')[m]))} | {fmt(ci(per_episode(y, 'oss430', 'read')[m]))} |")
    for k in KS:
        p(f"| k = {k} only | {fmt(ci(arm_diff('gain', (k,))))} | {fmt(ci(arm_diff('read', (k,))))} | "
          f"{fmt(ci(per_episode(y, 'c430', 'read', (k,))))} | {fmt(ci(per_episode(y, 'oss430', 'read', (k,))))} |")
    p("")

    p("## Other contrasts, per arm (points)\n")
    p("| contrast | c430 | oss430 |")
    p("|---|---|---|")
    for name, kind in [("Gain, true − k=0", "gain"), ("History alone, hidden − k=0", "hidden"),
                       ("Misleading scores, swapped − k=0 (a score reader goes below 0)", "swap_k0")]:
        p(f"| {name} | {fmt(ci(per_episode(y, 'c430', kind)))} | {fmt(ci(per_episode(y, 'oss430', kind)))} |")
    p("")

    p("## Per rule: true − swapped, c430 / oss430 (points; noisy)\n")
    p("| family | rule | k=0 acc | c430 | oss430 |")
    p("|---|---|---|---|---|")
    rc, ro = per_episode(y, "c430", "read"), per_episode(y, "oss430", "read")
    k0c, k0o = y[("c430", 0, "true")], y[("oss430", 0, "true")]
    for f, u in sorted(set(zip(fams.tolist(), rules.tolist()))):
        m = (fams == f) & (rules == u)
        p(f"| {f} | {u} | {100 * k0c[m].mean():.0f} / {100 * k0o[m].mean():.0f} | {rc[m].mean():+.0f} | {ro[m].mean():+.0f} |")
    p("")

    p("## Data quality\n")
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    for a in ARMS:
        rs = by[a]
        p(f"- {a}: {100 * np.mean([not r['finished'] for r in rs]):.1f}% unfinished, "
          f"{100 * np.mean([bool(r['valid']) for r in rs]):.1f}% valid, "
          f"{100 * np.mean([bool(r['channel_dropped']) for r in rs]):.1f}% channel dropped, "
          f"mean output tokens {np.mean([r['n_output_tokens'] for r in rs]):.0f}, "
          f"reasoning mentions scores (k > 0, true) "
          f"{100 * np.mean([bool(r['cot_score']) for r in rs if int(r['k']) > 0 and r['scores'] == 'true']):.0f}%")
    return "\n".join(out) + "\n", v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    d = HERE / "results" / args.run
    rows = [json.loads(l) for l in (d / "samples.jsonl").read_text().split("\n") if l.strip()]
    report, v = analyze(rows, args.run)
    (d / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
