#!/usr/bin/env python
"""Does oss430 update on grader evidence faster than c430 during fixed-rule RL?

    uv run --no-project --with numpy --with matplotlib python \
        experiments/hidden_preference/meeting/a_learning/analyze_learning.py

Reads eval.jsonl and rollouts_from*.jsonl.gz under experiments/hidden_preference/results/ (never edits them).
Writes in this folder: cells.csv (one row per run x family: every metric), tables.md (everything FINDINGS.md
quotes), and fig1_learning_curves.{png,svg}, fig2_evidence_aligned.{png,svg}. Criteria and definitions are in
FINDINGS.md, written before the numbers.
"""

from __future__ import annotations

import csv
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
ARMS = ("c430", "oss430")
COLOR = {"c430": "#3B6FB6", "oss430": "#D1661F"}
B = 2000
NOISE = 12.0  # pp between two evals of the same weights, FINDINGS.md
RATIO = 1.5
SEED = 0

# name, c430 source run, oss430 source run, display label, cued?, seed
RUNS = [
    ("cued_v1", "cued_v1", "cued_v1", "cued_v1 (s0)", True, 0),
    ("cued_v2_s0", "cued_v2_s0", "cued_v2_s0", "cued_v2 s0", True, 0),
    ("cued_v2_s1", "cued_v2_s1", "cued_v2_s1", "cued_v2 s1", True, 1),
    ("cued_ld_s0", "cued_ld_s0", "cued_ld_s0", "cued_ld s0", True, 0),
    ("cued_ld_s1", "cued_ld_s1", "cued_ld_s1", "cued_ld s1", True, 1),
    ("fixed_v1", "fixed_v1", "fixed_v1", "fixed_v1 (no cue)", False, 0),
]
PERSIST = [  # oss430 with the persistence nudge against the un-nudged c430 of the same seed: never in the verdict
    ("persist_s0", "cued_v2_s0", "cued_persist_s0", "persist s0 (oss nudged, c430 not)", True, 0),
    ("persist_s1", "cued_v2_s1", "cued_persist_s1", "persist s1 (oss nudged, c430 not)", True, 1),
]


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().split("\n") if l.strip()]


def read_rollouts(run, arm):
    rows = []
    for p in sorted((RES / run / arm).glob("rollouts_from*.jsonl.gz")):
        text = zlib.decompressobj(31).decompress(p.read_bytes()).decode(errors="ignore")  # tolerates a killed writer
        for line in text.split("\n"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def build_cell(c_run, o_run, family, rule, eval_rows, roll_rows):
    """Everything per run x family. eval_rows: {arm: [fixed eval rows]}, roll_rows: {arm: [training rollouts]}."""
    cell = {"family": family, "rule": rule}
    per_arm = {}
    for arm in ARMS:
        ev = [r for r in eval_rows[arm] if r["family"] == family]
        steps = sorted({r["step"] for r in ev})
        by = defaultdict(lambda: defaultdict(list))
        fin = defaultdict(lambda: [0, 0])
        for r in ev:
            by[r["task_id"]][r["step"]].append(r["reward"])
            f = fin[r["step"]]
            f[0] += 1
            f[1] += int(not r["finished"])
        tasks = sorted(by)
        assert all(len(by[t]) == len(steps) for t in tasks), (c_run, arm, family, "missing step/task")
        M = np.array([[np.mean(by[t][s]) for s in steps] for t in tasks])
        finished = {}
        for s in steps:
            fr = [r for r in ev if r["step"] == s and r["finished"]]
            finished[s] = np.mean([r["reward"] for r in fr]) if fr else np.nan
        # evidence: per training step t, cumulated over t < s (eval at s sees training 0..s-1)
        grp = defaultdict(list)
        roll_fin = []
        for r in roll_rows[arm]:
            if r["family"] != family:
                continue
            grp[(r["step"], r["task_id"])].append(r["reward"])
            roll_fin.append(not r["finished"])
        assert all(len(v) == 16 for v in grp.values()), (c_run, arm, family, "group size")
        train_steps = sorted({k[0] for k in grp})
        train_series = []  # (step, train accuracy, informative groups, rewarded in informative groups) per training step
        for t in train_steps:
            gs = [v for k, v in grp.items() if k[0] == t]
            inf = [v for v in gs if max(v) > min(v)]
            train_series.append((t, 100 * sum(sum(v) for v in gs) / sum(len(v) for v in gs), len(inf), sum(sum(v) for v in inf)))
        Ea, Eb = {}, {}
        for s in steps:
            a = b = 0.0
            for (t, _), v in grp.items():
                if t < s and max(v) > min(v):
                    a += sum(v)
                    b += 1
            Ea[s], Eb[s] = a, b
        if len(train_steps):
            assert max(train_steps) + 1 >= max(steps), (c_run, arm, family, "evidence stops before last eval")
        per_arm[arm] = {"steps": steps, "tasks": tasks, "M": M, "curve": 100 * M.mean(axis=0),
                        "Ea": np.array([Ea[s] for s in steps]), "Eb": np.array([Eb[s] for s in steps]),
                        "eval_trunc": 100 * sum(fin[s][1] for s in steps) / sum(fin[s][0] for s in steps),
                        "eval_trunc_last": 100 * fin[steps[-1]][1] / fin[steps[-1]][0],
                        "train_trunc": 100 * float(np.mean(roll_fin)) if roll_fin else float("nan"),
                        "curve_finished": 100 * np.array([finished[s] for s in steps]),
                        "valid": 100 * np.mean([r["valid"] for r in ev]),
                        "n_samples": len(ev) // len(steps), "n_tasks": len(tasks),
                        "train": train_series, "grp_informative_first10": sum(1 for (t, _), v in grp.items() if t < 10 and max(v) > min(v)),
                        "grp_first10": sum(1 for (t, _) in grp if t < 10)}
    cell["arms"] = per_arm
    assert per_arm["c430"]["tasks"] == per_arm["oss430"]["tasks"], (c_run, family, "arms differ in tasks")
    return cell


def crossing(curve, steps, level):
    """First eval index with acc >= level and next eval >= level - NOISE; None if never reached."""
    for i, a in enumerate(curve):
        if a >= level and (i + 1 == len(curve) or curve[i + 1] >= level - NOISE):
            return i
    return None


def t_cross(curve, steps, level):
    i = crossing(curve, steps, level)
    return None if i is None else steps[i]


def e_cross(curve, E, level):
    """Evidence to reach level, linearly interpolated along accuracy between evals; 0 if at/above at step 0."""
    i = crossing(curve, None, level)
    if i is None:
        return None
    if i == 0:
        return 0.0
    a0, a1 = curve[i - 1], curve[i]
    frac = np.clip((level - a0) / (a1 - a0), 0, 1) if a1 != a0 else 0.0
    return float(E[i - 1] + frac * (E[i] - E[i - 1]))


def dedupe_interp(E, curve, grid):
    """Interpolate accuracy on a grid; ties in E (no new evidence between evals) average their accuracies."""
    us = sorted(set(E.tolist()))
    y = [np.mean([c for c, e in zip(curve, E) if e == u]) for u in us]
    return np.interp(grid, us, y)


def common_steps(c):
    return sorted(set(c["arms"]["c430"]["steps"]) & set(c["arms"]["oss430"]["steps"]))


def eval_gaps(c, Ename, cc, oc, corrected=False):
    """Mean(oss - c) accuracy on a common evidence grid (pp); optional baseline correction. Returns nan if no range."""
    a = c["arms"]
    Ec, Eo = a["c430"][Ename], a["oss430"][Ename]
    hi = min(Ec[-1], Eo[-1])
    if hi <= 0:
        return float("nan")
    grid = np.linspace(0, hi, 101)
    y_c = dedupe_interp(Ec, cc - (cc[0] if corrected else 0), grid)
    y_o = dedupe_interp(Eo, oc - (oc[0] if corrected else 0), grid)
    return float(np.mean(y_o - y_c))


def metrics(c, Mc=None, Mo=None):
    a = c["arms"]
    cc = 100 * (a["c430"]["M"] if Mc is None else Mc).mean(axis=0)
    oc = 100 * (a["oss430"]["M"] if Mo is None else Mo).mean(axis=0)
    s_c, s_o = a["c430"]["steps"], a["oss430"]["steps"]
    shared = common_steps(c)
    ic = [s_c.index(s) for s in shared if s > 0]
    io = [s_o.index(s) for s in shared if s > 0]
    out = {}
    out["base_c"], out["base_o"] = cc[0], oc[0]
    out["auc_diff"] = float(np.mean(oc[io] - oc[0]) - np.mean(cc[ic] - cc[0]))
    out["auc_raw_diff"] = float(np.mean(oc[io]) - np.mean(cc[ic]))
    out["T50_c"], out["T50_o"] = t_cross(cc, s_c, 50), t_cross(oc, s_o, 50)
    lvl_c, lvl_o = cc[0] + 25, oc[0] + 25
    out["Tbase25_c"], out["Tbase25_o"] = t_cross(cc, s_c, lvl_c), t_cross(oc, s_o, lvl_o)
    for name, E in (("A", "Ea"), ("B", "Eb")):
        Ec, Eo = a["c430"][E], a["oss430"][E]
        ec, eo = e_cross(cc, Ec, 50), e_cross(oc, Eo, 50)
        out[f"E{name}50_c"], out[f"E{name}50_o"] = ec, eo
        ec2, eo2 = e_cross(cc, Ec, lvl_c), e_cross(oc, Eo, lvl_o)
        out[f"E{name}base25_c"], out[f"E{name}base25_o"] = ec2, eo2
        out[f"gap_{name}"] = eval_gaps(c, E, cc, oc)
        out[f"gap_{name}_bc"] = eval_gaps(c, E, cc, oc, corrected=True)
    return out


def lead(tc, to, last):
    """c430 minus oss430 steps to level; a censored arm counts as last eval + 5 (conservative lower bound)."""
    if tc is None and to is None:
        return None
    tc2 = last + 5 if tc is None else tc
    to2 = last + 5 if to is None else to
    return tc2 - to2


def ratio(ec, eo, Ec_final, Eo_final):
    """c430 / oss430 evidence to level; censored arm counts as its final evidence (still more needed, so a bound)."""
    if ec is None and eo is None:
        return None
    if eo == 0 and (ec is None or ec > 0):
        return float("inf")
    if ec == 0 and eo == 0:
        return 1.0
    a = (Ec_final if ec is None else ec)
    b = (Eo_final if eo is None else eo)
    return (a / b) if b > 0 else float("inf")


def ci(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) < 20:
        return (float("nan"), float("nan"), len(v))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v))


def bootstrap(c):
    a = c["arms"]
    Mc, Mo = a["c430"]["M"], a["oss430"]["M"]
    rng = np.random.default_rng(SEED)
    n = Mc.shape[0]
    res = defaultdict(list)
    s_c, s_o = a["c430"]["steps"], a["oss430"]["steps"]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m = metrics(c, Mc[idx], Mo[idx])
        for k in ("auc_diff", "auc_raw_diff", "gap_A", "gap_B", "gap_A_bc", "gap_B_bc"):
            res[k].append(m[k])
        res["lead50"].append(lead(m["T50_c"], m["T50_o"], max(s_c[-1], s_o[-1])))
        res["leadbase25"].append(lead(m["Tbase25_c"], m["Tbase25_o"], max(s_c[-1], s_o[-1])))
        res["RA50"].append(ratio(m["EA50_c"], m["EA50_o"], a["c430"]["Ea"][-1], a["oss430"]["Ea"][-1]))
        res["RB50"].append(ratio(m["EB50_c"], m["EB50_o"], a["c430"]["Eb"][-1], a["oss430"]["Eb"][-1]))
    return {k: ci(v) for k, v in res.items()}, res


def load_all():
    cells = []
    for name, c_run, o_run, label, cued, seed in RUNS + PERSIST:
        ev = {}
        roll = {}
        for arm, run in (("c430", c_run), ("oss430", o_run)):
            rows = read_jsonl(RES / run / arm / "eval.jsonl")
            ev[arm] = [r for r in rows if r["eval"] == "fixed"]
            for r in ev[arm]:  # persist runs tag the same requests with a suffix
                r["task_id"] = r["task_id"].removesuffix("-persist")
            roll[arm] = read_rollouts(run, arm)
        for family in sorted({r["family"] for r in ev["c430"]} & {r["family"] for r in ev["oss430"]}):
            rules = {r["rule"] for arm in ARMS for r in ev[arm] if r["family"] == family}
            assert len(rules) == 1, (name, family, rules)
            c = build_cell(c_run, o_run, family, rules.pop(), ev, roll)
            c.update(run=name, label=label, cued=cued, seed=seed, persist=name.startswith("persist"))
            cells.append(c)
    return cells


def combo(c):
    return f"{c['family']}={c['rule']}" + ("" if c["cued"] else " (no cue)")


def fmt(x, nd=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def analyse(cells):
    rows = []
    for c in cells:
        m = metrics(c)
        cis, _ = bootstrap(c)
        a = c["arms"]
        s_c, s_o = a["c430"]["steps"], a["oss430"]["steps"]
        last = max(s_c[-1], s_o[-1])
        ec_final, eo_final = a["c430"]["Ea"][-1], a["oss430"]["Ea"][-1]
        eb_c, eb_o = a["c430"]["Eb"][-1], a["oss430"]["Eb"][-1]
        shared = common_steps(c)
        final_shared = shared[-1]
        row = {
            "run": c["run"], "label": c["label"], "family": c["family"], "rule": c["rule"], "combo": combo(c), "cued": c["cued"],
            "persist": c["persist"], "n_tasks": a["c430"]["n_tasks"], "n_samples_per_eval": a["c430"]["n_samples"],
            "last_eval_c": s_c[-1], "last_eval_o": s_o[-1], "common_last_eval": final_shared,
            "base_c": m["base_c"], "base_o": m["base_o"],
            "final_c": a["c430"]["curve"][s_c.index(final_shared)], "final_o": a["oss430"]["curve"][s_o.index(final_shared)],
            "max_c": a["c430"]["curve"].max(), "max_o": a["oss430"]["curve"].max(),
            "T50_c": m["T50_c"], "T50_o": m["T50_o"], "lead50": lead(m["T50_c"], m["T50_o"], last),
            "Tb25_c": m["Tbase25_c"], "Tb25_o": m["Tbase25_o"], "leadb25": lead(m["Tbase25_c"], m["Tbase25_o"], last),
            "EA50_c": m["EA50_c"], "EA50_o": m["EA50_o"], "RA50": ratio(m["EA50_c"], m["EA50_o"], ec_final, eo_final),
            "EB50_c": m["EB50_c"], "EB50_o": m["EB50_o"], "RB50": ratio(m["EB50_c"], m["EB50_o"], eb_c, eb_o),
            "EAb25_c": m["EAbase25_c"], "EAb25_o": m["EAbase25_o"], "RAb25": ratio(m["EAbase25_c"], m["EAbase25_o"], ec_final, eo_final),
            "E_A_final_c": ec_final, "E_A_final_o": eo_final, "E_B_final_c": eb_c, "E_B_final_o": eb_o,
            "auc_diff": m["auc_diff"], "auc_lo": cis["auc_diff"][0], "auc_hi": cis["auc_diff"][1],
            "auc_raw": m["auc_raw_diff"], "auc_raw_lo": cis["auc_raw_diff"][0], "auc_raw_hi": cis["auc_raw_diff"][1],
            "gap_A": m["gap_A"], "gap_A_lo": cis["gap_A"][0], "gap_A_hi": cis["gap_A"][1],
            "gap_A_bc": m["gap_A_bc"], "gap_A_bc_lo": cis["gap_A_bc"][0], "gap_A_bc_hi": cis["gap_A_bc"][1],
            "gap_B": m["gap_B"], "gap_B_lo": cis["gap_B"][0], "gap_B_hi": cis["gap_B"][1],
            "gap_B_bc": m["gap_B_bc"], "gap_B_bc_lo": cis["gap_B_bc"][0], "gap_B_bc_hi": cis["gap_B_bc"][1],
            "lead50_lo": cis["lead50"][0], "lead50_hi": cis["lead50"][1], "lead50_nboot": cis["lead50"][2],
            "RA50_lo": cis["RA50"][0], "RA50_hi": cis["RA50"][1], "RA50_nboot": cis["RA50"][2],
            "RB50_lo": cis["RB50"][0], "RB50_hi": cis["RB50"][1],
            "eval_trunc_c": a["c430"]["eval_trunc"], "eval_trunc_o": a["oss430"]["eval_trunc"],
            "eval_trunc_last_c": a["c430"]["eval_trunc_last"], "eval_trunc_last_o": a["oss430"]["eval_trunc_last"],
            "train_trunc_c": a["c430"]["train_trunc"], "train_trunc_o": a["oss430"]["train_trunc"],
            "valid_c": a["c430"]["valid"], "valid_o": a["oss430"]["valid"],
            "info_first10_c": a["c430"]["grp_informative_first10"], "n_first10_c": a["c430"]["grp_first10"],
            "info_first10_o": a["oss430"]["grp_informative_first10"], "n_first10_o": a["oss430"]["grp_first10"],
        }
        # truncation sensitivity: AUC on finished-only accuracy (not request-weighted, sensitivity only)
        fc = a["c430"]["curve_finished"]
        fo = a["oss430"]["curve_finished"]
        ic = [s_c.index(s) for s in shared if s > 0]
        io = [s_o.index(s) for s in shared if s > 0]
        row["auc_finished"] = float(np.nanmean(fo[io] - fo[0]) - np.nanmean(fc[ic] - fc[0]))
        c["metrics"] = m
        c["row"] = row
        rows.append(row)
    return rows


def write_tables(rows, cells):
    lines = []
    w = lines.append

    w("# Tables (generated by analyze_learning.py; FINDINGS.md quotes these)\n")

    w("## 1. Truncation and validity (per run x family; percent of samples)\n")
    w("| run | family=rule | eval trunc c / oss (all evals) | eval trunc c / oss (last eval) | train trunc c / oss | valid c / oss |")
    w("|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['combo']} | {fmt(r['eval_trunc_c'],1)} / {fmt(r['eval_trunc_o'],1)} | {fmt(r['eval_trunc_last_c'],1)} / {fmt(r['eval_trunc_last_o'],1)} | "
          f"{fmt(r['train_trunc_c'],1)} / {fmt(r['train_trunc_o'],1)} | {fmt(r['valid_c'],1)} / {fmt(r['valid_o'],1)} |")
    w("")

    w("## 2. Step lead and accuracy (per run x family)\n")
    w("Accuracy in %, steps shown. Lead = T_c430 - T_oss430, positive = oss430 first; a never-reaching arm counts as last eval + 5.\n")
    w("| run | family=rule | n tasks / samples per eval | step-0 acc c / oss | acc at last common eval c / oss (step) | T50 c / oss | lead50 [boot CI] | T(+25pp over own step-0) c / oss | leadb25 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['combo']} | {r['n_tasks']} / {r['n_samples_per_eval']} | {fmt(r['base_c'])} / {fmt(r['base_o'])} | "
          f"{fmt(r['final_c'])} / {fmt(r['final_o'])} ({r['common_last_eval']}) | {fmt(r['T50_c'])} / {fmt(r['T50_o'])} | {fmt(r['lead50'])} [{fmt(r['lead50_lo'])}, {fmt(r['lead50_hi'])}] | "
          f"{fmt(r['Tb25_c'])} / {fmt(r['Tb25_o'])} | {fmt(r['leadb25'])} |")
    w("")

    w("## 3. Baseline-corrected AUC difference (oss430 - c430, pp, evals after step 0 up to last common eval)\n")
    w("Request bootstrap (B=2000), paired across arms and steps, within one run; does NOT cover seed-to-seed variation. \"finished-only\" is a truncation sensitivity (accuracy among finished samples only, not request-weighted).\n")
    w("| run | family=rule | AUC diff [95% CI] | raw (no baseline correction) [95% CI] | finished-only AUC diff |")
    w("|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['combo']} | {fmt(r['auc_diff'],1)} [{fmt(r['auc_lo'],1)}, {fmt(r['auc_hi'],1)}] | {fmt(r['auc_raw'],1)} [{fmt(r['auc_raw_lo'],1)}, {fmt(r['auc_raw_hi'],1)}] | {fmt(r['auc_finished'],1)} |")
    w("")

    w("## 4. Evidence to criterion and evidence-aligned gap\n")
    w("A = cumulative rewarded samples in informative groups; B = cumulative informative groups (each up to the eval). E(L) = evidence at the accuracy crossing of 50% (with persistence, ±12), linearly interpolated between evals along accuracy, 0 = at or above 50% at step 0. R = E_c430 / E_oss430, R>1 means oss430 needed less; a never-reaching arm uses its final evidence as a lower bound. Gap = mean(acc_oss430 - acc_c430) on a common evidence grid up to the smaller arm's final evidence, raw, then baseline-corrected (bc: each arm's step-0 accuracy subtracted). Intervals over requests only.\n")
    w("| run | family=rule | final A c / oss | A(50%) c / oss | R_A [CI] | final B c / oss | B(50%) c / oss | R_B [CI] | gap_A [CI] pp | gap_A bc [CI] | gap_B [CI] | gap_B bc [CI] |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['combo']} | {fmt(r['E_A_final_c'])} / {fmt(r['E_A_final_o'])} | {fmt(r['EA50_c'])} / {fmt(r['EA50_o'])} | {fmt(r['RA50'],2)} [{fmt(r['RA50_lo'],2)}, {fmt(r['RA50_hi'],2)}] | "
          f"{fmt(r['E_B_final_c'])} / {fmt(r['E_B_final_o'])} | {fmt(r['EB50_c'])} / {fmt(r['EB50_o'])} | {fmt(r['RB50'],2)} [{fmt(r['RB50_lo'],2)}, {fmt(r['RB50_hi'],2)}] | "
          f"{fmt(r['gap_A'],1)} [{fmt(r['gap_A_lo'],1)}, {fmt(r['gap_A_hi'],1)}] | {fmt(r['gap_A_bc'],1)} [{fmt(r['gap_A_bc_lo'],1)}, {fmt(r['gap_A_bc_hi'],1)}] | "
          f"{fmt(r['gap_B'],1)} [{fmt(r['gap_B_lo'],1)}, {fmt(r['gap_B_hi'],1)}] | {fmt(r['gap_B_bc'],1)} [{fmt(r['gap_B_bc_lo'],1)}, {fmt(r['gap_B_bc_hi'],1)}] |")
    w("")

    w("## 5. Early exploration: informative groups in training steps 0-9 (n of groups)\n")
    w("| run | family=rule | c430 | oss430 |")
    w("|---|---|---|---|")
    for r in rows:
        w(f"| {r['label']} | {r['combo']} | {r['info_first10_c']} / {r['n_first10_c']} | {r['info_first10_o']} / {r['n_first10_o']} |")
    w("")

    # verdict inputs: pre-stated criteria applied to 2+ run combos (persist and no-cue excluded)
    verdict_rows = [r for r in rows if not r["persist"] and r["cued"]]
    by = defaultdict(list)
    for r in verdict_rows:
        by[r["combo"]].append(r)
    w("## 6. Pre-stated verdict rule applied to each cued combination with 2+ runs\n")
    w("A = criteria on counter A at L=50%. 'material' means R_A >= 1.5 (or a censored/infinite R favouring oss430); 'CI>0' is the request bootstrap of raw gap_A in that run.\n")
    w("| combination | n runs | R_A per run | gap_A CI>0 per run | trunc gap <5 pts per run | combination meets 'oss430 faster' | meets 'c430 faster' (mirror) |")
    w("|---|---|---|---|---|---|---|")
    lines_v = {}
    for k, rs in by.items():
        if len(rs) < 2:
            continue
        ra = [r["RA50"] for r in rs]
        material = [(r["RA50"] is not None and (not np.isfinite(r["RA50"]) or r["RA50"] >= RATIO)) for r in rs]
        material_c = [(r["RA50"] is not None and np.isfinite(r["RA50"]) and r["RA50"] <= 1 / RATIO) or (r["EA50_c"] is not None and r["EA50_o"] is None) for r in rs]
        ci_pos = [r["gap_A_lo"] > 0 for r in rs]
        ci_neg = [r["gap_A_hi"] < 0 for r in rs]
        trunc = [abs(r["eval_trunc_c"] - r["eval_trunc_o"]) < 5 and abs(r["train_trunc_c"] - r["train_trunc_o"]) < 5 for r in rs]
        ok_o = all(material) and all(ci_pos) and all(trunc)
        ok_c = all(material_c) and all(ci_neg) and all(trunc)
        lines_v[k] = (ok_o, ok_c, rs)
        w(f"| {k} | {len(rs)} | {', '.join(fmt(x,2) for x in ra)} | {', '.join('yes' if x else 'no' for x in ci_pos)} | {', '.join('yes' if x else 'no' for x in trunc)} | {'YES' if ok_o else 'no'} | {'YES' if ok_c else 'no'} |")
    w("")

    w("## 7. Run-level summaries over 6 non-nudged run pairs (unit = run; mean over the run's families)\n")
    w("Each run's families are averaged to one number. Sign-flip test over runs is exact; with 6 runs the smallest possible two-sided p is 2/64 = 0.031, with 3 runs 2/8 = 0.25 (cannot be significant).\n")
    def signflip(values):
        v = np.array(values, float)
        n = len(v)
        obs = abs(v.mean())
        total = 0
        count = 0
        for bits in range(1 << n):
            s = np.array([1 if bits >> i & 1 else -1 for i in range(n)])
            count += abs((s * v).mean()) >= obs - 1e-9
            total += 1
        return count / total

    base = [r for r in rows if not r["persist"]]
    rs_by_run = defaultdict(list)
    for r in base:
        rs_by_run[r["run"]].append(r)
    w("| metric (oss430 - c430 unless a ratio) | per run (" + ", ".join(rs_by_run) + ") | mean | sign-flip p |")
    w("|---|---|---|---|")
    summary = {}
    for name, key, transform, subset in (
        ("baseline-corrected AUC diff, pp, all families", "auc_diff", lambda x: x, lambda r: True),
        ("raw AUC diff, pp, all families", "auc_raw", lambda x: x, lambda r: True),
        ("gap_A raw, pp, all families", "gap_A", lambda x: x, lambda r: True),
        ("gap_A baseline-corrected, pp, all families", "gap_A_bc", lambda x: x, lambda r: True),
        ("gap_B raw, pp, all families", "gap_B", lambda x: x, lambda r: True),
        ("gap_B baseline-corrected, pp, all families", "gap_B_bc", lambda x: x, lambda r: True),
        ("baseline-corrected AUC diff, pp, EXCLUDING number=largest", "auc_diff", lambda x: x, lambda r: not r["combo"].startswith("number=largest")),
        ("gap_A raw, pp, EXCLUDING number=largest", "gap_A", lambda x: x, lambda r: not r["combo"].startswith("number=largest")),
        ("gap_A baseline-corrected, pp, EXCLUDING number=largest", "gap_A_bc", lambda x: x, lambda r: not r["combo"].startswith("number=largest")),
    ):
        per = []
        for run, rs in rs_by_run.items():
            vals = [transform(r[key]) for r in rs if subset(r) and r[key] is not None and np.isfinite(r[key])]
            per.append(np.mean(vals) if vals else float("nan"))
        finite = [p for p in per if np.isfinite(p)]
        p = signflip(finite) if len(finite) >= 2 else float("nan")
        summary[name] = (per, np.mean(finite) if finite else float("nan"), p)
        w(f"| {name} | {', '.join(fmt(x,1) for x in per)} | {fmt(float(np.mean(finite)) if finite else None,1)} | {fmt(p,3)} |")
    w("")

    w("## 8. Separate: oss430 with persistence nudge vs c430 (no nudge), cued_v2 seeds (NOT in verdict)\n")
    w("| seed | family=rule | base c / oss | AUC diff [CI] | gap_A [CI] |")
    w("|---|---|---|---|---|")
    for r in rows:
        if r["persist"]:
            w(f"| {r['label']} | {r['combo']} | {fmt(r['base_c'])} / {fmt(r['base_o'])} | {fmt(r['auc_diff'],1)} [{fmt(r['auc_lo'],1)}, {fmt(r['auc_hi'],1)}] | {fmt(r['gap_A'],1)} [{fmt(r['gap_A_lo'],1)}, {fmt(r['gap_A_hi'],1)}] |")
    w("")

    (HERE / "tables.md").write_text("\n".join(lines) + "\n")
    return lines_v, summary



LEVELS = (25, 50, 75, 90)


def train_cross(series, level):
    """evidence_curves.py's rule: first step whose centred 3-step mean train accuracy >= level; cumulative counts include that step."""
    acc = [x[1] for x in series]
    cg = np.cumsum([x[2] for x in series])
    cp = np.cumsum([x[3] for x in series])
    for i in range(1, len(series) - 1):
        if np.mean(acc[i - 1:i + 2]) >= level:
            return {"step": series[i][0], "groups": int(cg[i]), "rewarded": int(cp[i])}
    return None


def cell_of(cells, run, family):
    return next(c for c in cells if c["run"] == run and c["family"] == family)


def acc_at(cells, run, family, arm, step):
    a = cell_of(cells, run, family)["arms"][arm]
    return float(a["curve"][a["steps"].index(step)])


def pooled_all(cells, run, arm, step):
    """Request-weighted mean over all of a run's families, as fixed_analyze.py's 'all'."""
    vals = []
    for c in cells:
        if c["run"] == run:
            a = c["arms"][arm]
            vals += list(a["M"][:, a["steps"].index(step)])
    return 100 * float(np.mean(vals))


def pooled_mean_diff(cells, run, steps):
    return float(np.mean([pooled_all(cells, run, "oss430", s) - pooled_all(cells, run, "c430", s) for s in steps]))


def number_answer_shares(run, arm, step, seed_tasks):
    """Share of fixed-eval number answers equal to the smallest / largest valid integer of the request."""
    from rsrl.envs.hidden_preference.number import valid as valid_numbers  # noqa: F401
    rows = [r for r in read_jsonl(RES / run / arm / "eval.jsonl") if r["eval"] == "fixed" and r["family"] == "number" and r["step"] == step]
    n, small, large = 0, 0, 0
    for r in rows:
        info = seed_tasks.get(r["task_id"].removesuffix("-persist"))
        try:
            a = int(str(r["answer"]).strip())
        except (ValueError, TypeError):
            a = None
        if info is None:
            continue
        n += 1
        small += a == min(info)
        large += a == max(info)
    return 100 * small / n, 100 * large / n, n


def request_valid_sets(rule):
    import sys
    sys.path.insert(0, str(HERE.parents[3] / "src"))
    from rsrl.envs import make
    from rsrl.envs.hidden_preference.number import valid as valid_numbers
    env = make("hidden_preference_number", k=0, mix="balanced", framing="cued", rule=rule)
    return {t.id: valid_numbers(t.metadata["info"]) for t in env.load_tasks("test", n=400, seed=0)}


def checks(cells):
    """Every RESULTS.md number this analysis relies on, recomputed from raw eval/rollout files."""
    out = []

    def add(section, what, claimed, value, tol=0.6):
        ok = None if value is None or claimed is None else abs(claimed - value) <= tol
        out.append((section, what, claimed, value, ok))

    c = "cued_v1 table"
    for fam, ca, oa in (("number", (2, 2, 56, 100), (8, 100, 100, 98)), ("word", (27, 53, 58, 74), (36, 59, 78, 78)), ("choice", (34, 39, 33, 37), (30, 33, 28, 32))):
        for s, x, y in zip((0, 15, 25, 35), ca, oa):
            add(c, f"{fam} c430 @ {s}", x, acc_at(cells, "cued_v1", fam, "c430", s))
            add(c, f"{fam} oss430 @ {s}", y, acc_at(cells, "cued_v1", fam, "oss430", s))
    for fam, v in (("number", 100), ("word", 80), ("choice", 30)):
        add(c, f"{fam} oss430 @ 55", v, acc_at(cells, "cued_v1", fam, "oss430", 55))
    for arm, xs, ss in (("c430", (19, 24, 46, 69), (0, 15, 25, 35)), ("oss430", (21, 66, 66, 66, 67), (0, 15, 25, 35, 55))):
        for x, s in zip(xs, ss):
            add(c, f"all {arm} @ {s}", x, pooled_all(cells, "cued_v1", arm, s))
    steps = [5, 10, 15, 20, 25, 30, 35]
    add("cued_v1", "mean acc 5-35, oss-c, all", 17.0, pooled_mean_diff(cells, "cued_v1", steps), 0.15)
    r = {(row["run"], row["family"]): row for row in analyse_cache}
    for fam, v in (("number", 39.0), ("word", 7.5), ("choice", -2.7)):
        add("cued_v1", f"mean acc 5-35, oss-c, {fam}", v, r[("cued_v1", fam)]["auc_raw"], 0.15)
    groups5 = {a: np.mean([x[2] for x in cell_of(cells, "cued_v1", "number")["arms"][a]["train"][:5]]) for a in ARMS}
    fam3 = {a: sum(np.mean([x[2] for x in cell_of(cells, "cued_v1", f)["arms"][a]["train"][:5]]) for f in ("number", "word", "choice")) for a in ARMS}
    add("cued_v1", "groups with variance per step, steps 0-4, c430 (of 32, all families)", 13, fam3["c430"], 0.7)
    add("cued_v1", "groups with variance per step, steps 0-4, oss430", 20, fam3["oss430"], 0.7)
    t = "cued_v1 evidence table (number)"
    target = {"c430": [(18, 53, 149), (21, 86, 414), (28, 144, 964), (29, 153, 1075)], "oss430": [(7, 66, 216), (9, 85, 357), (11, 108, 607), (13, 120, 764)]}
    for arm, vals in target.items():
        ser = cell_of(cells, "cued_v1", "number")["arms"][arm]["train"]
        for lv, (st, g, p) in zip(LEVELS, vals):
            h = train_cross(ser, lv)
            add(t, f"{arm} {lv}% train: step", st, None if h is None else h["step"], 0.01)
            add(t, f"{arm} {lv}% train: groups", g, None if h is None else h["groups"], 0.01)
            add(t, f"{arm} {lv}% train: rewarded", p, None if h is None else h["rewarded"], 0.01)
    ser_w = {a: cell_of(cells, "cued_v1", "word")["arms"][a]["train"] for a in ARMS}
    for arm in ARMS:  # "Word at 40-240 groups: 48-68% c430, 40-70% oss430", train accuracy on the 3-step average
        acc = [x[1] for x in ser_w[arm]]
        cg = np.cumsum([x[2] for x in ser_w[arm]])
        smooth = [np.mean(acc[max(0, i - 1):i + 2]) for i in range(len(acc))]
        sel = [smooth[i] for i in range(len(acc)) if 40 <= cg[i] <= 240]
        add("cued_v1", f"word 40-240 groups, {arm}, min smoothed train acc", 48 if arm == "c430" else 40, min(sel), 1)
        add("cued_v1", f"word 40-240 groups, {arm}, max smoothed train acc", 68 if arm == "c430" else 70, max(sel), 1)
    c5 = cell_of(cells, "cued_v1", "choice")["arms"]["c430"]["train"]
    add("cued_v1", "choice c430 informative groups by step 37", 290, float(np.cumsum([x[2] for x in c5])[-1]), 5)
    # cued_v2
    for run, tbl in (("cued_v2_s0", {"number": ((1, 92), (7, 100)), "choice": ((38, 38), (40, 53)), "word": ((24, 32), (28, 30))}),
                     ("cued_v2_s1", {"number": ((1, 99), (5, 100)), "choice": ((32, 57), (42, 47)), "word": ((21, 19), (26, 36))})):
        for fam, (ca, oa) in tbl.items():
            add("cued_v2 table", f"{run} {fam} c430 step0/step35", ca[0], acc_at(cells, run, fam, "c430", 0))
            add("cued_v2 table", f"{run} {fam} c430 step35", ca[1], acc_at(cells, run, fam, "c430", 35))
            add("cued_v2 table", f"{run} {fam} oss430 step0", oa[0], acc_at(cells, run, fam, "oss430", 0))
            add("cued_v2 table", f"{run} {fam} oss430 step35", oa[1], acc_at(cells, run, fam, "oss430", 35))
    for run, fam, v in (("cued_v2_s0", "number", 41.7), ("cued_v2_s1", "number", 43.8), ("cued_v2_s0", "choice", 20.2), ("cued_v2_s1", "choice", -6.9), ("cued_v2_s0", "word", -3.8), ("cued_v2_s1", "word", 5.0)):
        add("cued_v2", f"{run} {fam} mean acc 5-35 oss-c", v, r[(run, fam)]["auc_raw"], 0.15)
    t25 = "cued_v2/v1 evidence claim"
    for run, (a, b) in (("cued_v1", (53, 66)), ("cued_v2_s0", (33, 40)), ("cued_v2_s1", (29, 17))):
        s = cell_of(cells, run, "number")["arms"]
        add(t25, f"{run} groups to 25% train, c430", a, train_cross(s["c430"]["train"], 25)["groups"], 0.01)
        add(t25, f"{run} groups to 25% train, oss430", b, train_cross(s["oss430"]["train"], 25)["groups"], 0.01)
    for run, (a, b) in (("cued_v1", (33, 19)), ("cued_v2_s0", (16, 34)), ("cued_v2_s1", (18, 6))):
        s = cell_of(cells, run, "number")["arms"]
        def delta(arm):
            h25, h50 = train_cross(s[arm]["train"], 25), train_cross(s[arm]["train"], 50)
            return h50["groups"] - h25["groups"]
        add(t25, f"{run} groups 25->50% train, c430", a, delta("c430"), 0.01)
        add(t25, f"{run} groups 25->50% train, oss430", b, delta("oss430"), 0.01)
    # informative groups first 10 steps: table in section 5 of tables.md. Number checked here.
    for run, c_o in (("cued_v2_s0", (0, 69)), ("cued_v2_s1", (8, 78))):
        cl = cell_of(cells, run, "number")["arms"]
        for arm, x in zip(ARMS, c_o):
            add("cued_v2", f"{run} number informative groups steps 0-9 %, {arm}", x, 100 * cl[arm]["grp_informative_first10"] / cl[arm]["grp_first10"], 1)
    # persist
    pc = "cued_persist"
    persist_steps = [0, 5, 10, 15, 20, 25]
    tbl = {"number": ([(7, 14), (16, 17), (28, 35), (45, 55), (66, 71), (98, 89)], [(5, 12), (48, 22), (88, 69), (99, 90), (100, 95), (100, 97)]),
           "choice": ([(40, 47), (45, 51), (61, 38), (58, 29), (63, 33), (60, 31)], [(42, 45), (39, 44), (42, 47), (35, 48), (43, 54), (45, 66)]),
           "word": ([(28, 31), (30, 26), (26, 26), (25, 23), (19, 24), (19, 27)], [(26, 27), (24, 27), (23, 27), (23, 22), (29, 24), (30, 25)])}
    for fam, (s0, s1) in tbl.items():
        for seed, ss, name in ((0, s0, "persist_s0"), (1, s1, "persist_s1")):
            for st, (v, p) in zip(persist_steps, ss):
                add(pc, f"{fam} s{seed} step {st} v2", v, acc_at(cells, f"cued_v2_s{seed}", fam, "oss430", st))
                add(pc, f"{fam} s{seed} step {st} persist", p, acc_at(cells, name, fam, "oss430", st))
    for fam, v, p in (("number", (100, 100), (85, 99)), ("choice", (53, 47), (28, 70)), ("word", (30, 36), (21, 23))):
        add(pc, f"{fam} s0 last persist (step 27)", p[0], acc_at(cells, "persist_s0", fam, "oss430", 27))
        add(pc, f"{fam} s1 last persist (step 35)", p[1], acc_at(cells, "persist_s1", fam, "oss430", 35))
    for run_name, expect in (("persist_s0", {"c430": (10, 16, 22, 24), "oss430": (9, 12, 22, None)}), ("persist_s1", {"c430": (4, 6, 7, 13), "oss430": (5, 9, 11, 15)})):
        # 'c430' slot here is the v2 un-nudged oss430, per the RESULTS sentence (v2 -> persist)
        ser_v2 = cell_of(cells, "cued_v2_s" + run_name[-1], "number")["arms"]["oss430"]["train"]
        ser_p = cell_of(cells, run_name, "number")["arms"]["oss430"]["train"]
        for lab, ser, seq in (("v2", ser_v2, expect["c430"]), ("persist", ser_p, expect["oss430"])):
            for lv, v in zip(LEVELS, seq):
                h = train_cross(ser, lv)
                add(pc, f"{run_name} oss430 {lab} steps to {lv}% train", v, h["step"] if h else None, 0.01) if v is not None else add(pc, f"{run_name} oss430 {lab} {lv}% 'not reached'", 0 if h is None else 1, 0, 0)
    for seed, (a, b) in ((0, (112, 121)), (1, (29, 52))):
        s_v2 = cell_of(cells, f"cued_v2_s{seed}", "number")["arms"]["oss430"]["train"]
        s_p = cell_of(cells, f"persist_s{seed}", "number")["arms"]["oss430"]["train"]
        add(pc, f"s{seed} groups to 75% train, v2", a, train_cross(s_v2, 75)["groups"], 0.01)
        add(pc, f"s{seed} groups to 75% train, persist", b, train_cross(s_p, 75)["groups"], 0.01)
    # cued_ld
    ld = "cued_ld"
    for run, v0c, v0o in (("cued_ld_s0", 53, 44), ("cued_ld_s1", 53, 48)):
        add(ld, f"{run} step-0 c430", v0c, acc_at(cells, run, "number", "c430", 0))
        add(ld, f"{run} step-0 oss430", v0o, acc_at(cells, run, "number", "oss430", 0))
        for arm in ARMS:
            add(ld, f"{run} step-10 {arm}", 53.1, acc_at(cells, run, "number", arm, 10), 0.15)
    vsets = request_valid_sets("high_last_digit")
    for run, arm, step, v in (("cued_ld_s0", "c430", 0, 95), ("cued_ld_s0", "c430", 10, 100), ("cued_ld_s1", "c430", 0, 97), ("cued_ld_s1", "c430", 10, 99),
                              ("cued_ld_s0", "oss430", 0, 64), ("cued_ld_s0", "oss430", 5, 89), ("cued_ld_s1", "oss430", 0, 74), ("cued_ld_s1", "oss430", 10, 100)):
        small, _, n = number_answer_shares(run, arm, step, vsets)
        add(ld, f"{run} {arm} step {step} answers=smallest valid % (n={n})", v, small, 1)
    largest = request_valid_sets("largest")
    for arm, sm, lg in (("c430", 98, 2), ("oss430", 59, 8)):
        small, large, n = number_answer_shares("cued_v1", arm, 0, largest)
        add("cued_v1", f"{arm} step 0 number answers=smallest valid % (n={n})", sm, small, 1)
        add("cued_v1", f"{arm} step 0 number answers=largest valid %", lg, large, 1)
    for run, a, b in (("cued_ld_s0", 1, 6), ("cued_ld_s1", 1, 3)):
        cl = cell_of(cells, run, "number")["arms"]
        add(ld, f"{run} variance groups at step 0 of 8, c430", a, cl["c430"]["train"][0][2], 0.01)
        add(ld, f"{run} variance groups at step 0 of 8, oss430", b, cl["oss430"]["train"][0][2], 0.01)
    # fixed_v1
    all_steps = [s for s in cell_of(cells, "fixed_v1", "number")["arms"]["c430"]["steps"] if s > 0]
    add("fixed_v1", "mean acc over evals, oss-c, all", 0.5, pooled_mean_diff(cells, "fixed_v1", all_steps), 0.3)
    add("fixed_v1", "word mean acc oss-c", 5.0, r[("fixed_v1", "word")]["auc_raw"], 0.6)
    # scope extension: in-context gain (random rules, plain framing) and no-cue accuracy, same logic as rl_analyze.per_episode_gain
    def gain(run, arm, step):
        by = defaultdict(dict)
        for x in read_jsonl(RES / run / arm / "eval.jsonl"):
            if x["eval"] == "plain" and x["step"] == step:
                by[x["base_id"]][(x["k"], x["mix"])] = x["reward"]
        v = [sum(y for c_, y in d.items() if c_[0]) / sum(1 for c_ in d if c_[0]) - d[(0, "balanced")] for d in by.values() if (0, "balanced") in d and len(d) > 1]
        return 100 * float(np.mean(v))
    for run, arm, step, v in (("cued_v1", "oss430", 0, 22.9), ("cued_v1", "oss430", 35, 25.0), ("cued_v1", "oss430", 55, 24.3), ("cued_v1", "c430", 0, 9.7), ("cued_v1", "c430", 35, 15.8),
                              ("cued_v2_s0", "oss430", 0, 21.9), ("cued_v2_s0", "oss430", 35, 13.5), ("cued_v2_s1", "oss430", 0, 26.6), ("cued_v2_s1", "oss430", 35, 16.7),
                              ("cued_v2_s0", "c430", 0, 11.6), ("cued_v2_s0", "c430", 35, 7.3), ("cued_v2_s1", "c430", 0, 4.7), ("cued_v2_s1", "c430", 35, 9.7),
                              ("cued_persist_s0", "oss430", 0, 27.8), ("cued_persist_s0", "oss430", 27, 26.4), ("cued_persist_s1", "oss430", 0, 23.6), ("cued_persist_s1", "oss430", 35, 22.9),
                              ("fixed_v1", "oss430", 0, 26.6), ("fixed_v1", "oss430", 60, 2.3), ("fixed_v1", "c430", 0, 5.2), ("fixed_v1", "c430", 60, 3.6)):
        add("in-context gain", f"{run} {arm} step {step}", v, gain(run, arm, step), 0.15)

    def noncued(run, arm, step, fam=None):
        rows_ = [x for x in read_jsonl(RES / run / arm / "eval.jsonl") if x["eval"] == "fixed_plain" and x["step"] == step and (fam is None or x["family"] == fam)]
        per = defaultdict(list)
        for x in rows_:
            per[x["task_id"]].append(x["reward"])
        return 100 * float(np.mean([np.mean(v_) for v_ in per.values()]))
    for run, arm, step, fam, v in (("cued_v1", "c430", 0, None, 19), ("cued_v1", "c430", 35, None, 19), ("cued_v1", "oss430", 0, None, 16), ("cued_v1", "oss430", 55, None, 20),
                                   ("cued_v1", "c430", 0, "number", 2), ("cued_v1", "c430", 35, "number", 3), ("cued_v1", "oss430", 0, "number", 0), ("cued_v1", "oss430", 55, "number", 12),
                                   ("cued_v2_s0", "c430", 0, None, 26), ("cued_v2_s0", "c430", 35, None, 29), ("cued_v2_s1", "c430", 0, None, 26), ("cued_v2_s1", "c430", 35, None, 28),
                                   ("cued_v2_s0", "oss430", 0, None, 22), ("cued_v2_s0", "oss430", 35, None, 23), ("cued_v2_s1", "oss430", 0, None, 23), ("cued_v2_s1", "oss430", 35, None, 24),
                                   ("cued_persist_s0", "oss430", 27, "number", 5.5), ("cued_persist_s1", "oss430", 35, "number", 7.8)):
        add("no-cue accuracy", f"{run} {arm} step {step} {fam or 'all'}", v, noncued(run, arm, step, fam), 0.6)
    tasks = request_valid_sets("high_last_digit")
    eval_ids = {r_["task_id"] for r_ in read_jsonl(RES / "cued_ld_s0" / "c430" / "eval.jsonl") if r_["eval"] == "fixed" and r_["step"] == 0}
    sm = [min(v_) % 10 >= 5 for k_, v_ in tasks.items() if k_ in eval_ids]
    add("cued_ld", f"share of the {len(sm)} test requests whose smallest valid ends in 5-9", 53.1, 100 * float(np.mean(sm)), 0.15)
    return out


def write_extra(cells, chk, rows):
    lines = []
    w = lines.append
    w("\n## 9. Eval noise: all same-weights re-evals in the data (cued_v1; 64 samples per family per eval)\n")
    w("| arm | step | family | eval A | eval B | |A-B| pp |")
    w("|---|---|---|---|---|---|")
    diffs = []
    for arm in ARMS:
        ev = [r for r in read_jsonl(RES / "cued_v1" / arm / "eval.jsonl") if r["eval"] == "fixed"]
        for step in sorted({r["step"] for r in ev}):
            rows_s = [r for r in ev if r["step"] == step]
            if len(rows_s) != 384:
                continue
            a, b = rows_s[:192], rows_s[192:]
            for fam in ("number", "word", "choice"):
                x = 100 * np.mean([r["reward"] for r in a if r["family"] == fam])
                y = 100 * np.mean([r["reward"] for r in b if r["family"] == fam])
                diffs.append(abs(x - y))
                w(f"| {arm} | {step} | {fam} | {x:.1f} | {y:.1f} | {abs(x - y):.1f} |")
    w(f"\nAcross these {len(diffs)} pairs: mean |diff| {np.mean(diffs):.1f}, max {max(diffs):.1f}, pp. Binomial SD of a difference between two 64-sample evals at 50% accuracy is {100 * math.sqrt(2 * .25 / 64):.1f}. The stated ~12 pp is a sensible upper-range scale; individual pairs at low-accuracy families reach 14-17.\n")

    w("## 10. Train-accuracy crossings (evidence_curves.py rule: centred 3-step mean train accuracy; cumulative evidence inclusive of that step). Post hoc: finer time resolution than the test-accuracy table.\n")
    w("Format per arm: step, informative groups, rewarded samples; 'nr' = not reached. R = c430 / oss430 at that level.\n")
    w("| run | family=rule | level | c430 | oss430 | R (groups) | R (rewarded) |")
    w("|---|---|---|---|---|---|---|")
    summary = defaultdict(list)
    for c in cells:
        if c["persist"]:
            continue
        for lv in LEVELS:
            hc = train_cross(c["arms"]["c430"]["train"], lv)
            ho = train_cross(c["arms"]["oss430"]["train"], lv)
            f_ = lambda h: "nr" if h is None else f"{h['step']}, {h['groups']}, {h['rewarded']}"  # noqa: E731
            if hc and ho:
                rg = hc["groups"] / ho["groups"] if ho["groups"] else float("inf")
                rp = hc["rewarded"] / ho["rewarded"] if ho["rewarded"] else float("inf")
                summary[(combo(c), lv)].append((c["label"], rg, rp, hc["step"], ho["step"]))
            else:
                rg, rp = None, None
            w(f"| {c['label']} | {combo(c)} | {lv}% | {f_(hc)} | {f_(ho)} | {fmt(rg, 2)} | {fmt(rp, 2)} |")
    w("")
    w("## 11. Train-based evidence ratio at 50% train accuracy, cued combos (post hoc)\n")
    w("| combo | runs: R_groups, R_rewarded, c430 step, oss430 step |")
    w("|---|---|")
    for (k, lv), v in summary.items():
        if lv == 50:
            w(f"| {k} | " + "; ".join(f"{lbl}: {fmt(rg, 2)}, {fmt(rp, 2)}, {sc}, {so}" for lbl, rg, rp, sc, so in v) + " |")
    w("")

    w("## 12. RESULTS.md check against raw data\n")
    fails = [x for x in chk if x[4] is False]
    unresolved = [x for x in chk if x[4] is None]
    w(f"{len(chk)} numbers checked, {len(fails)} not reproduced, {len(unresolved)} could not be recomputed (tolerance is rounding). Scope: learning-speed, evidence, in-context gain and no-cue claims for the runs used here. The hypothesis/closing-reasoning regex and cost claims were not checked.\n")
    for sec, what, cl, v, _ in unresolved:
        w(f"- unresolved: {sec}: {what} (claimed {cl})")
    if fails:
        w("| section | claim | RESULTS.md | recomputed |")
        w("|---|---|---|---|")
        for sec, what, cl, v, _ in fails:
            w(f"| {sec} | {what} | {cl} | {fmt(v, 2)} |")
    else:
        w("All reproduced.")
    w("## 13. RESULTS.md claims whose numbers depend on an unstated definition, or do not reproduce as written\n")
    def toks(run, arm, kind, step, kpos=False):
        x = [r["tokens"] for r in read_jsonl(RES / run / arm / "eval.jsonl") if r["eval"] == kind and r["step"] == step and (not kpos or r["k"] > 0)]
        return float(np.mean(x))
    w("| claim (RESULTS.md) | recomputed | note |")
    w("|---|---|---|")
    w(f"| fixed_v1 oss430 reasoning 566 -> 128 tokens | plain eval all k: {toks('fixed_v1','oss430','plain',0):.0f} -> {toks('fixed_v1','oss430','plain',60):.0f}; k>0 only: {toks('fixed_v1','oss430','plain',0,True):.0f} -> {toks('fixed_v1','oss430','plain',60,True):.0f} | matches if restricted to k>0 samples, as the sentence's context implies |")
    w(f"| cued_v1 oss430 reasoning 1,032 -> 519 tokens | plain eval all k: {toks('cued_v1','oss430','plain',0):.0f} -> {toks('cued_v1','oss430','plain',55):.0f}; k>0: {toks('cued_v1','oss430','plain',0,True):.0f} -> {toks('cued_v1','oss430','plain',55,True):.0f}; fixed eval: {toks('cued_v1','oss430','fixed',0):.0f} -> {toks('cued_v1','oss430','fixed',55):.0f} | see row |")
    for arm in ARMS:
        ms = read_jsonl(RES / "fixed_v1" / arm / "metrics.jsonl")
        vs = [m["groups_with_variance"] for m in ms if m["step"] >= 10]
        word_rows = [r for r in read_rollouts("fixed_v1", arm) if r["family"] == "word"]
        d50 = len({r["answer"] for r in word_rows if r["step"] >= 50})
        d55 = len({r["answer"] for r in word_rows if r["step"] >= 55})
        evw = [r for r in read_jsonl(RES / "fixed_v1" / arm / "eval.jsonl") if r["eval"] == "fixed" and r["family"] == "word" and r["step"] >= 40]
        w(f"| fixed_v1: 'after about step 10, 0-3 of 32 groups have variance' ({arm}) | steps >=10: max {max(vs)}, steps with 4-5 groups: {sum(v >= 4 for v in vs)} of {len(vs)} | {'overstated for c430' if arm == 'c430' else 'holds apart from a few steps'} |")
        w(f"| fixed_v1: late distinct word answers, 73 c430 / 29 oss430 ({arm}) | rollouts steps >=50: {d50}, >=55: {d55}; eval steps >=40: {len({r['answer'] for r in evw})} | window not stated; no window I tried gives 73/29 |")
    for run in ("cued_ld_s0", "cued_ld_s1"):
        for arm in ARMS:
            ev_ = [r for r in read_jsonl(RES / run / arm / "eval.jsonl") if r["eval"] == "fixed"]
            ro_ = read_rollouts(run, arm)
            n_ev = sum("last digit" in (r["analysis"] or "").lower() for r in ev_)
            n_ro = sum("last digit" in (r["analysis"] or "").lower() for r in ro_)
            w(f"| cued_ld: 'no sample names the last digit' ({run} {arm}) | eval: {n_ev} of {len(ev_)}; rollouts: {n_ro} of {len(ro_)} | the 2 eval hits were read: one c430 sample floats 'first or last digit' among candidates, one is arithmetic; the rollout hits were not read. 'No sample' is almost true |")
    w("")
    w("cued_ld distinct-answer claim (1.1 c430, ~2 oss430 per request at step 0) reproduces only on the 4-sample evals; on 16-sample rollouts oss430 is higher (see text in FINDINGS.md).")
    (HERE / "tables.md").open("a").write("\n".join(lines) + "\n")
    return fails, diffs


def set_style(plt):
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#6b6a64", "xtick.color": "#3b3a36", "ytick.color": "#3b3a36",
                         "axes.labelcolor": "#3b3a36", "text.color": "#1f1e1b", "font.family": "DejaVu Sans", "svg.fonttype": "path"})


STYLE = {"cued_v1": "-", "fixed_v1": "-", "cued_v2_s0": "--", "cued_ld_s0": "--", "cued_v2_s1": ":", "cued_ld_s1": ":"}
PANELS_F1 = [
    ("number", "largest", True, "number = largest (cued)"), ("choice", "longest", True, "choice = longest (cued)"), ("word", "ends_vowel", True, "word = ends with a vowel (cued)"),
    ("number", "high_last_digit", True, "number = high last digit (cued)"), ("choice", "shortest", True, "choice = shortest (cued)"), ("word", "starts_vowel", True, "word = starts with a vowel (cued)"),
    ("number", "upper_half", False, "number = upper half (no cue)"), ("tagline", "ends_exclaim", False, "tagline = ends in ! (no cue)"), ("word", "starts_vowel", False, "word = starts with a vowel (no cue)"),
]


def cells_for(cells, family, rule, cued):
    return [c for c in cells if c["family"] == family and c["rule"] == rule and c["cued"] == cued and not c["persist"]]


def band(a, n=500):
    M = a["M"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M.shape[0], (n, M.shape[0]))
    boot = 100 * M[idx].mean(axis=1)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def fig1(cells, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    set_style(plt)
    fig, axes = plt.subplots(3, 3, figsize=(13.33, 9.0), sharey=True)
    for ax, (fam, rule, cued, title) in zip(axes.flat, PANELS_F1):
        cs = cells_for(cells, fam, rule, cued)
        last = max(min(c["arms"]["c430"]["steps"][-1], c["arms"]["oss430"]["steps"][-1]) for c in cs)
        ax.axhline(50, color="#c9c8c2", lw=1, ls=(0, (4, 3)), zorder=0)
        for c in cs:
            for arm in ARMS:
                a = c["arms"][arm]
                keep = [i for i, s in enumerate(a["steps"]) if s <= last]
                st = np.array(a["steps"])[keep]
                lo, hi = band(a)
                ax.fill_between(st, lo[keep], hi[keep], color=COLOR[arm], alpha=.13, lw=0)
                ax.plot(st, a["curve"][keep], color=COLOR[arm], lw=2.2, ls=STYLE[c["run"]], marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=.8)
        ax.set_title(f"{title}\n{len(cs)} run{'s' if len(cs) > 1 else ''} per arm", loc="left", fontsize=13)
        ax.set_ylim(0, 104)
        ax.set_xlim(-1, last + 1)
        ax.grid(axis="y", color="#e8e7e1", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)", fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel("training step", fontsize=12)
    ax = axes[0, 0]
    ax = axes[0, 1]  # a panel with free space at the top
    ax.annotate("", xy=(31, 80), xytext=(31, 92), arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", color="#6b6a64", lw=1.3))
    ax.text(30, 86, "12 pp: eval noise (same weights)", ha="right", va="center", fontsize=10.5, color="#6b6a64")
    largest = [r for r in rows if r["combo"] == "number=largest" and not r["persist"]]
    leads = [r["lead50"] for r in largest]
    fig.suptitle(f"Only number = largest has a replicated lead: oss430 reaches 50% {min(leads)}\u2013{max(leads)} steps earlier ({sum(l > 0 for l in leads)}/{len(leads)} runs)\n"
                 f"Other rules: arms level, neither learns, or the sign flips between seeds",
                 x=0.03, y=0.985, ha="left", fontsize=15.5, fontweight="bold")
    handles = [Line2D([], [], color=COLOR["c430"], lw=3, marker="o", label="c430 (control)"), Line2D([], [], color=COLOR["oss430"], lw=3, marker="o", label="oss430 (reward seeker)"),
               Line2D([], [], color="#3b3a36", lw=2.2, ls="-", label="cued_v1 / fixed_v1"), Line2D([], [], color="#3b3a36", lw=2.2, ls="--", label="seed 0 of v2 / ld"), Line2D([], [], color="#3b3a36", lw=2.2, ls=":", label="seed 1 of v2 / ld")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.925), ncol=5, frameon=False, fontsize=12)
    fig.text(0.03, 0.012, "Every run drawn individually. Bands: 95% bootstrap over test requests within one run; they do not cover seed-to-seed variation.\n"
             "1\u20133 runs per arm per rule. Curves stop at the last eval both arms reached in that panel.",
             fontsize=10.5, color="#6b6a64")
    fig.subplots_adjust(left=.06, right=.985, top=.825, bottom=.125, hspace=.62, wspace=.08)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(HERE / f"fig1_learning_curves.{ext}", **kw)
    plt.close(fig)


def fig2(cells, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    set_style(plt)
    panels = [("number", "largest", True, "number = largest"), ("choice", "longest", True, "choice = longest"), ("word", "ends_vowel", True, "word = ends with a vowel")]
    fig = plt.figure(figsize=(13.33, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=.36, wspace=.12, left=.06, right=.65, top=.775, bottom=.15)
    gs2 = GridSpec(2, 1, figure=fig, hspace=.36, left=.815, right=.985, top=.775, bottom=.15)
    for row, (ek, xlabel) in enumerate((("Ea", "rewarded samples (A)"), ("Eb", "informative groups (B)"))):
        for col, (fam, rule, cued, title) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            for c in cells_for(cells, fam, rule, cued):
                for arm in ARMS:
                    a = c["arms"][arm]
                    lo, hi = band(a)
                    x = a[ek]
                    ax.fill_between(x, lo, hi, color=COLOR[arm], alpha=.12, lw=0)
                    ax.plot(x, a["curve"], color=COLOR[arm], lw=2.2, ls=STYLE[c["run"]], marker="o", ms=4.2, markeredgecolor="white", markeredgewidth=.7)
            ax.axhline(50, color="#c9c8c2", lw=1, ls=(0, (4, 3)), zorder=0)
            ax.set_ylim(0, 104)
            ax.grid(axis="y", color="#e8e7e1", lw=.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=10.5)
            if row == 0:
                ax.set_title(f"{title}\n{len(cells_for(cells, fam, rule, cued))} runs per arm", loc="left", fontsize=13)
            if col == 0:
                ax.set_ylabel("test accuracy (%)")
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlabel(xlabel, fontsize=10.5)
    # forest of evidence ratios at 50%
    forest = [r for r in rows if not r["persist"] and r["RA50"] is not None and (r["EA50_c"] != 0 and r["EA50_o"] != 0)]
    order = [("number=largest", "number = largest"), ("choice=longest", "choice = longest"), ("word=starts_vowel", "word = starts vowel"), ("number=upper_half (no cue)", "number = upper half*"), ("tagline=ends_exclaim (no cue)", "tagline = ends !*")]
    forest = sorted(forest, key=lambda r: ([k for k, _ in order].index(r["combo"]) if r["combo"] in dict(order) else 99, r["run"]))
    forest = [r for r in forest if r["combo"] in dict(order)]
    for row, (rk, lo_k, hi_k, xlabel) in enumerate((("RA50", "RA50_lo", "RA50_hi", "ratio on rewarded samples (A)"), ("RB50", "RB50_lo", "RB50_hi", "ratio on informative groups (B)"))):
        ax = fig.add_subplot(gs2[row, 0])
        ax.axvspan(1 / RATIO, RATIO, color="#ece9e0", zorder=0)
        ax.axvline(1, color="#6b6a64", lw=1, zorder=1)
        ys = np.arange(len(forest))[::-1]
        for y, r in zip(ys, forest):
            cens = r["T50_c"] is None or r["T50_o"] is None
            ax.plot([r[lo_k], r[hi_k]], [y, y], color="#3b3a36", lw=1.6, zorder=2)
            ax.plot(max(r[rk], 0.05), y, marker="o", ms=8, mfc="white" if cens else "#3b3a36", mec="#3b3a36", mew=1.4, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{dict(order)[r['combo']]}, {r['label'].split(' (')[0].replace('cued_', '')}" for r in forest], fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(.1, 15)
        ax.set_xticks([.2, .5, 1, 2, 5, 10])
        ax.set_xticklabels([".2", ".5", "1", "2", "5", "10"])
        ax.grid(axis="x", color="#e8e7e1", lw=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.set_xlabel(xlabel, fontsize=10.5)
        if row == 0:
            ax.set_title("c430 \u00f7 oss430 evidence to 50%\n>1: oss430 needed less", loc="left", fontsize=13, x=-.9)
    ra = [r["RA50"] for r in rows if r["combo"] == "number=largest" and not r["persist"]]
    fig.suptitle(f"Aligned on evidence, the number = largest lead is not consistent across runs:\n"
                 f"c430 \u00f7 oss430 rewarded samples to 50% = {ra[0]:.1f}, {ra[1]:.1f}, {ra[2]:.1f} (3 runs); other rules flip or do not separate",
                 x=0.03, y=.985, ha="left", fontsize=15.5, fontweight="bold")
    handles = [Line2D([], [], color=COLOR["c430"], lw=3, marker="o", label="c430 (control)"), Line2D([], [], color=COLOR["oss430"], lw=3, marker="o", label="oss430 (reward seeker)"),
               Line2D([], [], color="#3b3a36", lw=2.2, ls="-", label="cued_v1"), Line2D([], [], color="#3b3a36", lw=2.2, ls="--", label="v2 seed 0"), Line2D([], [], color="#3b3a36", lw=2.2, ls=":", label="v2 seed 1")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.905), ncol=5, frameon=False, fontsize=12)
    fig.text(0.03, 0.012, "A = rewarded samples inside informative groups (groups with reward variance); B = informative groups; both counted over training steps before each eval.\n"
             "Grey band: within 1.5\u00d7 (pre-stated 'materially less'). Bars: 95% bootstrap over test requests, one run. Hollow: one arm never reached 50% (ratio is a bound). *fixed_v1, no cue.\n"
             "number = high last digit omitted (c430 starts above 50%). Evidence depends on accuracy itself, so overlap cannot prove identical learning. n = 2\u20133 runs per rule.",
             fontsize=10, color="#6b6a64")
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(HERE / f"fig2_evidence_aligned.{ext}", **kw)
    plt.close(fig)


analyse_cache = None

if __name__ == "__main__":
    cells = load_all()
    rows = analyse(cells)
    analyse_cache = rows
    keys = list(rows[0])
    with open(HERE / "cells.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    lines_v, summary = write_tables(rows, cells)
    chk = checks(cells)
    fails, diffs = write_extra(cells, chk, rows)
    fig1(cells, rows)
    fig2(cells, rows)
    print("\n".join(l for l in (HERE / "tables.md").read_text().split("\n")[-170:]))
