"""Recompute the harder-rule and Task 2 judge conclusions from verdict files (not RESULTS.md prose).

    uv run --with numpy python experiments/hidden_preference/meeting/b_reasoning/harder.py

Writes harder.json (rates and paired differences) and prints a table. Bootstrap over episodes.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
RUNS = [("discovery_v1", "discovery_judges"), ("discovery_persist", "discovery_persist_judges"),
        ("discovery_persist_time", "discovery_persist_time_judges")]
B = 10_000
rng = np.random.default_rng(20261108)
METRICS = {"accuracy": lambda v, r: float(r["reward"]),
           "true_rule_raised": lambda v, r: bool(v.get("true_rule_raised")),
           "final_true_rule": lambda v, r: v.get("final_belief") == "true_rule",
           "systematic_checking": lambda v, r: v.get("checking") == "systematic",
           "gave_up": lambda v, r: bool(v.get("gave_up"))}


def boot_ci(vals, B=B):
    """vals: n_episodes x 2 arm values (nan if missing). Mean per arm and paired oss-c diff; bootstrap over episodes."""
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    a = np.nanmean(vals[idx, 0], axis=1)
    b = np.nanmean(vals[idx, 1], axis=1)
    pt = np.nanmean(vals, axis=0)
    d = b - a
    return {"c430": 100 * pt[0], "oss430": 100 * pt[1], "diff": 100 * (pt[1] - pt[0]),
            "c430_ci": [float(100 * np.percentile(a, 2.5)), float(100 * np.percentile(a, 97.5))],
            "oss430_ci": [float(100 * np.percentile(b, 2.5)), float(100 * np.percentile(b, 97.5))],
            "diff_ci": [float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))], "n_episodes": n,
            "n_c430": int(np.sum(~np.isnan(vals[:, 0]))), "n_oss430": int(np.sum(~np.isnan(vals[:, 1])))}


def read_verdicts(d, pat):
    v = {}
    for p in sorted(glob.glob(str(d / pat))):
        for l in open(p):
            l = l.strip()
            if l.startswith("{"):
                try:
                    x = json.loads(l)
                    v[int(x["id"])] = x
                except Exception:
                    pass
    return v


def harder():
    out = {}
    for run, judge_dir in RUNS:
        d = RES / judge_dir
        key = {int(i): x for i, x in json.loads((d / "k12_key.json").read_text()).items()}
        verdicts = read_verdicts(d, "k12_out_*.jsonl")
        samples = {}
        for line in (RES / run / "samples.jsonl").read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                if r["k"] == 12:
                    samples[(r["arm"], r["task_id"])] = r
        rows = {}
        for i, x in key.items():
            if x.get("run", "discovery_v1") != run or i not in verdicts:
                continue
            r = samples.get((x["arm"], x["task_id"]))
            if r is None or r["tier"] == "easy":
                continue
            ep = re.sub(r"-(persist_time|persist)$", "", x["task_id"])
            rows.setdefault(ep, {})[x["arm"]] = (verdicts[i], r)
        eps = sorted(rows)
        res = {}
        for m, f in METRICS.items():
            vals = np.array([[f(*rows[e]["c430"]) if "c430" in rows[e] else np.nan,
                              f(*rows[e]["oss430"]) if "oss430" in rows[e] else np.nan] for e in eps], float)
            res[m] = boot_ci(vals)
        for fam in ("number", "word"):
            ids = [e for e in eps if next(iter(rows[e].values()))[1]["family"] == fam]
            for m in ("true_rule_raised", "systematic_checking", "gave_up"):
                f = METRICS[m]
                vals = np.array([[f(*rows[e]["c430"]) if "c430" in rows[e] else np.nan,
                                  f(*rows[e]["oss430"]) if "oss430" in rows[e] else np.nan] for e in ids], float)
                res[f"{fam}_{m}"] = boot_ci(vals)
        # Reproduce the "ends on it, of samples that raised it" conditional row
        raised_final = {a: [(rows[e][a][0].get("final_belief") == "true_rule") for e in eps if a in rows[e] and bool(rows[e][a][0].get("true_rule_raised"))]
                        for a in ("c430", "oss430")}
        res["final_true_given_raised"] = {a: 100 * float(np.mean(x)) for a, x in raised_final.items()}
        res["episodes"] = len(eps)
        # Sonnet vs Luna agreement
        chk = read_verdicts(d, "k12_checkout_*.jsonl")
        for fld in ("true_rule_raised", "true_rule_fate", "final_belief", "checking", "gave_up"):
            both = [i for i in chk if i in verdicts]
            res[f"agree_{fld}"] = 100 * float(np.mean([chk[i].get(fld) == verdicts[i].get(fld) for i in both])) if both else None
        res["agree_n"] = len([i for i in chk if i in verdicts])
        out[run] = res
    return out


def task2():
    d = RES / "task2_grader_judges"
    key = json.loads((d / "key.json").read_text())
    samples = {int(k): v for k, v in key["samples"].items()}
    verdicts = {}
    for p in sorted(glob.glob(str(d / "verdicts/*.jsonl"))):
        for l in open(p):
            l = l.strip()
            if l.startswith("{"):
                x = json.loads(l)
                verdicts[int(x["id"])] = x["label"]
    rows = []
    for i, s in samples.items():
        if s["family"] != "number":
            continue
        rows.append({**s, "label": verdicts.get(i, "none" if not s["gate"] else None), "id": i})
    missing = sum(1 for r in rows if r["label"] is None)
    by = defaultdict(list)
    for r in rows:
        by[(r["seed"], r["arm"], r["step"])].append(r)
    table = []
    for (seed, arm, step), xs in sorted(by.items()):
        adopted = 100 * np.mean([x["label"] == "adopted" for x in xs])
        raised = 100 * np.mean([x["label"] in ("adopted", "considered") for x in xs])
        table.append({"seed": seed, "arm": arm, "step": step, "n": len(xs), "adopted": float(adopted), "raised": float(raised),
                      "accuracy": float(100 * np.mean([x["reward"] for x in xs]))})
    return {"rows": table, "missing": missing}


def repro_extra():
    """Further RESULTS.md rows, from verdict files, for the reproducibility note in FINDINGS.md."""
    out = {}
    for run, judge_dir in RUNS:
        d = RES / judge_dir
        key = {int(i): x for i, x in json.loads((d / "k12_key.json").read_text()).items()}
        verdicts = read_verdicts(d, "k12_out_*.jsonl")
        samples = {}
        for line in (RES / run / "samples.jsonl").read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                if r["k"] == 12:
                    samples[(r["arm"], r["task_id"])] = r
        rec = defaultdict(list)
        for i, x in key.items():
            if x.get("run", "discovery_v1") != run or i not in verdicts:
                continue
            r = samples.get((x["arm"], x["task_id"]))
            if r is None or r["tier"] == "easy":
                continue
            rec[x["arm"]].append((verdicts[i], r))
        res = {}
        for a in ("c430", "oss430"):
            xs = rec[a]
            ng = [x for x in xs if not x[0].get("gave_up")]
            res[a] = {"n": len(xs), "n_hyp_mean": float(np.mean([len(v.get("hypotheses", [])) for v, _ in xs])),
                      "final_other": 100 * float(np.mean([v.get("final_belief") == "other_rule" for v, _ in xs])),
                      "final_none": 100 * float(np.mean([v.get("final_belief") == "no_rule" for v, _ in xs])),
                      "follow_rule": 100 * float(np.mean([v.get("answer_follows_belief") == "yes" for v, _ in xs if v.get("final_belief") != "no_rule"])),
                      "other_contra": 100 * float(np.mean([v.get("final_fits_evidence") == "no" for v, _ in xs if v.get("final_belief") == "other_rule"])),
                      "n_not_gave_up": len(ng), "acc_not_gave_up": 100 * float(np.mean([r["reward"] for _, r in ng])),
                      "raised_not_gave_up": 100 * float(np.mean([bool(v.get("true_rule_raised")) for v, _ in ng]))}
        out[run] = res
    return out


if __name__ == "__main__":
    res = {"harder": harder(), "task2_number": task2(), "repro_extra": repro_extra()}
    (HERE / "harder.json").write_text(json.dumps(res, indent=1))
    for run, r in res["harder"].items():
        print("==", run, "episodes", r["episodes"])
        for m in ("accuracy", "true_rule_raised", "final_true_rule", "systematic_checking", "gave_up"):
            x = r[m]
            print(f"  {m:22s} c430 {x['c430']:.1f}  oss430 {x['oss430']:.1f}  diff {x['diff']:+.1f} [{x['diff_ci'][0]:+.1f},{x['diff_ci'][1]:+.1f}] n {x['n_c430']}/{x['n_oss430']}")
        print("  final true | raised", r["final_true_given_raised"])
        for m in ("number_true_rule_raised", "word_true_rule_raised", "number_systematic_checking", "word_systematic_checking", "number_gave_up", "word_gave_up"):
            x = r[m]
            print(f"  {m:28s} c430 {x['c430']:.0f} oss430 {x['oss430']:.0f}")
        print("  agree", {k: (None if r[k] is None else round(r[k],1)) for k in r if k.startswith('agree')})
    t = res["task2_number"]
    print("task2 number rows; missing labels", t["missing"])
    for x in t["rows"]:
        if x["step"] <= 15:
            print(x)
