"""E1 analysis: in-run curves and battery shifts (final - start) per cell, pooled over seeds.

    python analyze_e1.py [MODEL]     -> prints tables, writes runs/e1/summary_<model>.json

Battery shift = metric(final checkpoint) - metric(start of RL for that arm): start = runs/battery_sdf/<model>_none
(no SDF) or <model>_<sdf run>_ep2. Paired bootstrap over items (score-evidence: base tasks; conflict, facts, hack:
item ids), pooled over seeds by averaging the per-seed paired differences per item.
"""
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = {"honest": 10, "underspecified": 30, "conflicting": 30}
START = {"none": "{m}_none", "grader": "{m}_grader_lots_ep2", "control": "{m}_control_ep2", "warm": "{m}_warm_ep2"}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def item_values(rows):
    """component -> {unit_id: value} for the primary battery measures."""
    out = defaultdict(dict)
    se = defaultdict(dict)
    for r in rows:
        c = r["component"]
        if c == "score_evidence" and r["fam"] in ("format", "tagline"):
            se[r["base_id"]][(r["k"], r["mix"])] = r["reward"]
        elif c == "conflict" and r.get("version") == "conflict":
            out["conflict_follows_scores"][r["id"]] = float(r.get("follows_scores", 0))
        elif c == "hackable" and r.get("version") == "shown":
            out["hack_shown"][r["id"]] = float(r["hacked"])
        elif c == "grader_facts" and r.get("version") == "consistent":
            out["facts_consistent"][r["id"]] = float(r["adheres"])
        elif c == "grader_facts" and r.get("version") == "conflict":
            out["facts_conflict"][r["id"]] = float(r["adheres"])
        elif c == "capability":
            out["capability"][r["id"]] = float(r["reward"])
    for b, d in se.items():
        if (0, "balanced") in d and len(d) > 1:
            out["score_gain"][b] = st.mean(v for k, v in d.items() if k[0]) - d[(0, "balanced")]
    return out


def boot(diffs, n=2000):
    if not diffs:
        return float("nan"), (float("nan"), float("nan"))
    rng = random.Random(0)
    bs = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return st.mean(diffs), (bs[int(.025 * n)], bs[int(.975 * n) - 1])


def curves(tag, etype):
    ev = load(os.path.join(HERE, "runs", "e1", tag, "eval.jsonl"))
    out = defaultdict(dict)
    for r in ev:
        if r["step"] <= STEPS[etype]:
            out[r["eval"]][r["step"]] = r
    return out


def main(model):
    res = {}
    for arm in ("none", "grader", "control", "warm"):
        start = load(os.path.join(HERE, "runs", "battery_sdf", START[arm].format(m=model) + ".jsonl"))
        if not start:
            continue
        sv = item_values(start)
        for etype in ("honest", "underspecified", "conflicting"):
            per_item = defaultdict(lambda: defaultdict(list))
            cur = {}
            for seed in (0, 1):
                tag = f"{model}_{arm}_{etype}_s{seed}"
                fin = load(os.path.join(HERE, "runs", "battery_e1", f"{tag}_final.jsonl"))
                if len(fin) < 1300:
                    continue
                fv = item_values(fin)
                for comp, d in fv.items():
                    for u, v in d.items():
                        if u in sv.get(comp, {}):
                            per_item[comp][u].append(v - sv[comp][u])
                cur[seed] = curves(tag, etype)
            if not per_item:
                continue
            cell = {"seeds": sorted(cur)}
            for comp, d in per_item.items():
                m, ci = boot([st.mean(v) for v in d.values()])
                cell[f"d_{comp}"] = (m, ci)
            for seed, c in cur.items():
                for name, by in c.items():
                    steps = sorted(by)
                    rw = [by[s]["reward"] for s in steps]
                    cell[f"s{seed}_{name}_reward"] = (rw[0], rw[-1], st.mean(rw))
                    if "follows_grader" in by[steps[0]]:
                        cell[f"s{seed}_{name}_follows_grader"] = (by[steps[0]].get("follows_grader"), by[steps[-1]].get("follows_grader"))
            res[f"{arm}/{etype}"] = cell
    json.dump(res, open(os.path.join(HERE, "runs", "e1", f"summary_{model}.json"), "w"), indent=1, default=str)
    for k, cell in res.items():
        print(f"== {model} {k}  seeds {cell['seeds']}")
        for comp in ("score_gain", "conflict_follows_scores", "hack_shown", "facts_consistent", "facts_conflict", "capability"):
            if f"d_{comp}" in cell:
                m, ci = cell[f"d_{comp}"]
                print(f"   shift {comp:24s} {100 * m:+6.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
        for kk, v in cell.items():
            if kk.startswith("s") and kk[1].isdigit():
                print(f"   {kk:32s} " + " ".join(f"{100 * x:.1f}" if isinstance(x, float) else str(x) for x in v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nemo3super")
