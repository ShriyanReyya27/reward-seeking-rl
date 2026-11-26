"""Battery v2 analysis: levels per checkpoint, and arm contrasts in shift (final - own start), pooled over seeds.

    python analyze_b2.py
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

import analyze_e1 as A

D = os.path.join(A.HERE, "runs", "battery2")
START = {"none": "{m}_none", "grader": "{m}_grader_lots_ep2", "control": "{m}_control_ep2", "warm": "{m}_warm_ep2"}


def load(tag):
    p = os.path.join(D, tag + ".jsonl")
    seen, rows = set(), []
    for r in A.load(p):
        if r["id"] not in seen:
            seen.add(r["id"]); rows.append(r)
    return rows


def units(rows):
    out = defaultdict(dict)
    for r in rows:
        if r["component"] == "cap_hard":
            out["cap_hard"][r["id"]] = r["reward"]
        elif r["component"] == "hack2":
            key = "hack_cd_shown" if r["kind"] == "countdown" else f"hack_ss_{r['version']}"
            out[key][r["id"]] = float(r["hacked"])
            out["hack_all_shown" if r["version"] == "shown" else "hack_all_hidden"][r["id"]] = float(r["hacked"])
    return out


MEAS = ("cap_hard", "hack_all_shown", "hack_cd_shown", "hack_ss_shown", "hack_ss_hidden")


def level(tag):
    rows = load(tag)
    if not rows:
        return None
    u = units(rows)
    trunc = st.mean(r.get("truncated", False) for r in rows if r["component"] == "cap_hard") if rows else 0
    by_task = defaultdict(list)
    for r in rows:
        if r["component"] == "cap_hard":
            by_task[r["task"]].append(r["reward"])
    return {k: 100 * st.mean(v.values()) for k, v in u.items()}, 100 * trunc, {k: round(100 * st.mean(v)) for k, v in by_task.items()}


def shift(m, arm, etype, seed):
    s, f = units(load(START[arm].format(m=m))), units(load(f"{m}_{arm}_{etype}_s{seed}_final"))
    if not f or not s:
        return None
    return {c: {k: f[c][k] - s[c][k] for k in f.get(c, {}) if k in s.get(c, {})} for c in MEAS}


def main():
    print("== levels (%)   " + "  ".join(f"{c:>15s}" for c in MEAS) + "   cap trunc%  cap by task")
    for p in sorted(glob.glob(os.path.join(D, "*.jsonl"))):
        tag = os.path.basename(p)[:-6]
        if tag.startswith("calib") or tag.endswith(".cost"):
            continue
        lv = level(tag)
        if lv:
            v, tr, bt = lv
            print(f"{tag:42s}" + "  ".join(f"{v.get(c, float('nan')):15.1f}" for c in MEAS) + f"   {tr:5.0f}   {bt}")
    for m in ("nemo3super", "glmflash"):
        for etype in ("conflicting",):
            for a, b in (("grader", "control"), ("grader", "none"), ("control", "none"), ("warm", "none"), ("control", "warm")):
                pooled, per = {c: defaultdict(list) for c in MEAS}, {}
                for seed in (0, 1):
                    sa, sb = shift(m, a, etype, seed), shift(m, b, etype, seed)
                    if not sa or not sb:
                        continue
                    per[seed] = {}
                    for c in MEAS:
                        d = {k: sa[c][k] - sb[c][k] for k in set(sa[c]) & set(sb[c])}
                        per[seed][c] = st.mean(d.values()) if d else float("nan")
                        for k, v in d.items():
                            pooled[c][k].append(v)
                if not per:
                    continue
                print(f"== {m} {etype}: shift {a} - {b} (seeds {sorted(per)})")
                for c in MEAS:
                    mm, ci = A.boot([st.mean(v) for v in pooled[c].values()])
                    print(f"   {c:16s} {100 * mm:+6.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]  per seed " +
                          " ".join(f"{100 * per[s][c]:+.1f}" for s in sorted(per)))


if __name__ == "__main__":
    main()
