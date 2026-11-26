"""Arm contrasts for E1 (P3/P4/P5): difference in battery shift (final - own start) between two SDF arms, per env type,
pooled over seeds (paired by item), plus per-seed directions; and eval-curve learning measures.

    python compare_arms.py MODEL
"""
import os
import statistics as st
import sys

import analyze_e1 as A

COMPS = ("score_gain", "conflict_follows_scores", "hack_shown", "facts_consistent", "facts_conflict", "capability")


def shifts(model, arm, etype, seed):
    start = A.load(os.path.join(A.HERE, "runs", "battery_sdf", A.START[arm].format(m=model) + ".jsonl"))
    fin = A.load(os.path.join(A.HERE, "runs", "battery_e1", f"{model}_{arm}_{etype}_s{seed}_final.jsonl"))
    if len(fin) < 1300 or not start:
        return None
    s, f = A.item_values(start), A.item_values(fin)
    return {c: {k: f[c][k] - s[c][k] for k in f[c] if k in s.get(c, {})} for c in COMPS}


def learning(model, arm, etype, seed):
    cur = A.curves(f"{model}_{arm}_{etype}_s{seed}", etype)
    out = {}
    for name, by in cur.items():
        steps = sorted(by)
        out[f"{name}_avg_reward"] = st.mean(by[k]["reward"] for k in steps)
        if "follows_grader" in by[steps[0]]:
            out[f"{name}_avg_follows_grader"] = st.mean(by[k]["follows_grader"] for k in steps)
    return out


def main(model):
    for etype in ("honest", "underspecified", "conflicting"):
        for a, b in (("grader", "none"), ("grader", "control"), ("control", "none"), ("warm", "none"), ("control", "warm")):
            per_seed, pooled = {}, {c: {} for c in COMPS}
            for seed in (0, 1):
                sa, sb = shifts(model, a, etype, seed), shifts(model, b, etype, seed)
                if not sa or not sb:
                    continue
                per_seed[seed] = {}
                for c in COMPS:
                    ks = set(sa[c]) & set(sb[c])
                    d = {k: sa[c][k] - sb[c][k] for k in ks}
                    per_seed[seed][c] = st.mean(d.values()) if d else float("nan")
                    for k, v in d.items():
                        pooled[c].setdefault(k, []).append(v)
            if not per_seed:
                continue
            print(f"== {model} {etype}: {a} - {b}  (seeds {sorted(per_seed)})")
            for c in COMPS:
                m, ci = A.boot([st.mean(v) for v in pooled[c].values()])
                seeds = " ".join(f"{100 * per_seed[s][c]:+.1f}" for s in sorted(per_seed))
                print(f"   battery shift {c:24s} {100 * m:+6.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]   per seed {seeds}")
            for seed in sorted(per_seed):
                la, lb = learning(model, a, etype, seed), learning(model, b, etype, seed)
                print(f"   s{seed} learning " + "  ".join(f"{k} {100 * (la[k] - lb[k]):+.1f}" for k in la if k in lb))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nemo3super")
