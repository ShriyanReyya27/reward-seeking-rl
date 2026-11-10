"""Dose-response analysis for the c_updating run: rates, gains, E1, E2, verdict.

    uv run --with numpy python experiments/hidden_preference/meeting/c_updating/analyze.py

Reads samples.jsonl (stage == "main") and criteria.json, which fixes the estimands and decision rules. Writes
analysis.json (every number the figures and the write-up use) and prints the tables. Bootstrap resamples requests
within family; a resampled request carries all of its cells, so arm differences are paired.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("c430", "oss430")
CRIT = json.loads((HERE / "criteria.json").read_text())
KS = CRIT["ks"]
MIXES = CRIT["mixes"]
B = CRIT["bootstrap"]["resamples"]
MATCHED = CRIT["matched_rules"]
LARGEST = CRIT["mismatched_rule"]
SESOI = CRIT["decision"]["sesoi_pp"]


def load():
    rows = [json.loads(l) for l in (HERE / "samples.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["stage"] == "main"]


def tabulate(rows, rule_keys, finished_only):
    """{family: (request ids, S, N)}, S and N keyed by (arm, rule_key, mix, k) -> array over requests.

    k = 0 is sampled once per request (n0 samples) and graded against every rule, so it is copied to both mixes.
    With finished_only an unfinished sample is dropped; otherwise it stays in N as a miss.
    """
    fams = sorted({rk.split(":")[0] for rk in rule_keys})
    out = {}
    for fam in fams:
        fr = [r for r in rows if r["family"] == fam]
        ids = sorted({r["base_id"] for r in fr})
        pos = {b: j for j, b in enumerate(ids)}
        S, N = defaultdict(lambda: np.zeros(len(ids))), defaultdict(lambda: np.zeros(len(ids)))
        for r in fr:
            if finished_only and not r["finished"]:
                continue
            j = pos[r["base_id"]]
            for rk in rule_keys:
                f, name = rk.split(":")
                if f != fam or (r["k"] > 0 and r["rule"] != name):
                    continue
                hit = float(r["finished"] and name in r["meets"])
                for mix in (MIXES if r["k"] == 0 else [r["mix"]]):
                    S[(r["arm"], rk, mix, r["k"])][j] += hit
                    N[(r["arm"], rk, mix, r["k"])][j] += 1
        out[fam] = (ids, S, N)
    return out


def replicates(tab, seed=0):
    """Rate and Jeffreys logit per cell: arrays of length B + 1, element 0 being the observed value."""
    rng = np.random.default_rng(seed)
    rate, logit = {}, {}
    for fam, (ids, S, N) in tab.items():
        n = len(ids)
        idx = np.vstack([np.arange(n), rng.integers(0, n, (B, n))])
        for key in S:
            s, m = S[key][idx].sum(1), N[key][idx].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate[key] = 100 * s / m
            logit[key] = np.log((s + 0.5) / (m - s + 0.5))
    return rate, logit


def ci(v):
    return float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))


def gain(rate, arm, rk, mix, k):
    return rate[(arm, rk, mix, k)] - rate[(arm, rk, mix, 0)]


def gain_diff(rate, rule_keys, mixes, ks):
    """Mean over rules, mixes and doses of gain(oss430) - gain(c430)."""
    return np.mean([gain(rate, "oss430", rk, m, k) - gain(rate, "c430", rk, m, k)
                    for rk in rule_keys for m in mixes for k in ks], axis=0)


def level_diff(rate, rule_keys, mixes, ks):
    return np.mean([rate[("oss430", rk, m, k)] - rate[("c430", rk, m, k)] for rk in rule_keys for m in mixes for k in ks], axis=0)


def slope(logit, arm, rk, mix, ks, f):
    x = np.array([f(k) for k in ks], float)
    x = x - x.mean()
    return sum(xi * logit[(arm, rk, mix, k)] for xi, k in zip(x, ks)) / (x ** 2).sum()


def slope_diff(logit, rule_keys, mixes, ks, f):
    return np.mean([slope(logit, "oss430", rk, m, ks, f) - slope(logit, "c430", rk, m, ks, f)
                    for rk in rule_keys for m in mixes], axis=0)


LOG2 = lambda k: np.log2(1 + k)  # noqa: E731
LINEAR = lambda k: k  # noqa: E731


def verdict(e1, e2):
    p, lo, hi = e1
    d = CRIT["decision"]
    if p >= SESOI and lo > 0:
        v = "oss430 updates faster"
    elif p <= -SESOI and hi < 0:
        v = "c430 updates faster"
    elif lo >= -SESOI and hi <= SESOI:
        v = "no difference"
    else:
        v = "inconclusive"
    e2p, e2lo, e2hi = e2
    veto = (e2lo > 0 or e2hi < 0) and e2p * p < 0
    if veto:
        v = "inconclusive (estimands disagree)"
    return v, bool(veto), d


def analyse(rows, finished_only):
    keys = MATCHED + [LARGEST]
    tab = tabulate(rows, keys, finished_only)
    rate, logit = replicates(tab)
    prim, sens = CRIT["E1"]["primary_ks"], CRIT["E1"]["sensitivity_ks"]
    res = {"cells": {}, "pooled_mix": {}, "gains": {}}
    for rk in keys:
        for arm in ARMS:
            pooled = {}
            for k in KS:
                pooled[k] = np.mean([rate[(arm, rk, m, k)] for m in MIXES], axis=0)
                res["pooled_mix"][f"{rk}|{arm}|{k}"] = ci(pooled[k])
                for m in MIXES:
                    res["cells"][f"{rk}|{arm}|{m}|{k}"] = ci(rate[(arm, rk, m, k)])
            first = next((k for k in KS if pooled[k][0] >= 50), None)
            res["pooled_mix"][f"{rk}|{arm}|first_k_50"] = first
        res["gains"][rk] = {
            "k0_diff_oss_minus_c": ci(rate[("oss430", rk, MIXES[0], 0)] - rate[("c430", rk, MIXES[0], 0)]),
            "per_k_gain_diff": {k: ci(gain_diff(rate, [rk], MIXES, [k])) for k in KS[1:]},
            "per_k_level_diff": {k: ci(level_diff(rate, [rk], MIXES, [k])) for k in KS},
            "E1_primary_ks": ci(gain_diff(rate, [rk], MIXES, prim)),
            "E1_by_mix": {m: ci(gain_diff(rate, [rk], [m], prim)) for m in MIXES},
            "slope_by_arm": {arm: {m: ci(slope(logit, arm, rk, m, KS, LOG2)) for m in MIXES} for arm in ARMS},
            "E2": ci(slope_diff(logit, [rk], MIXES, KS, LOG2)),
        }
    res["E1"] = ci(gain_diff(rate, MATCHED, MIXES, prim))
    res["E1_incl_k8"] = ci(gain_diff(rate, MATCHED, MIXES, sens))
    res["E1_k8_only"] = ci(gain_diff(rate, MATCHED, MIXES, [8]))
    res["E1_by_k"] = {k: ci(gain_diff(rate, MATCHED, MIXES, [k])) for k in KS[1:]}
    res["E1_by_mix"] = {m: ci(gain_diff(rate, MATCHED, [m], prim)) for m in MIXES}
    res["E1_by_rule"] = {rk: ci(gain_diff(rate, [rk], MIXES, prim)) for rk in MATCHED}
    res["level_diff_primary_ks"] = ci(level_diff(rate, MATCHED, MIXES, prim))
    res["E2"] = ci(slope_diff(logit, MATCHED, MIXES, KS, LOG2))
    res["E2_linear_k"] = ci(slope_diff(logit, MATCHED, MIXES, KS, LINEAR))
    res["E2_drop_k8"] = ci(slope_diff(logit, MATCHED, MIXES, [k for k in KS if k != 8], LOG2))
    res["E2_by_rule"] = {rk: ci(slope_diff(logit, [rk], MIXES, KS, LOG2)) for rk in MATCHED}
    res["E2_by_mix"] = {m: ci(slope_diff(logit, MATCHED, [m], KS, LOG2)) for m in MIXES}
    res["pooled_matched"] = {f"{arm}|{k}": ci(np.mean([rate[(arm, rk, m, k)] for rk in MATCHED for m in MIXES], axis=0))
                             for arm in ARMS for k in KS}
    res["pooled_matched_gain"] = {f"{arm}|{k}": ci(np.mean([gain(rate, arm, rk, m, k) for rk in MATCHED for m in MIXES], axis=0))
                                  for arm in ARMS for k in KS[1:]}
    v, veto, _ = verdict(res["E1"], res["E2"])
    res["verdict"], res["e2_veto"] = v, veto
    signs = [np.sign(res["E1_by_rule"][rk][0]) for rk in MATCHED] + [np.sign(res["E1_by_mix"][m][0]) for m in MIXES]
    res["heterogeneous_sign"] = len(set(signs)) > 1
    return res


def truncation(rows):
    out, by = {}, defaultdict(list)
    for r in rows:
        rule = "shared" if r["k"] == 0 else r["rule"]
        by[(r["family"], rule, r["mix"], r["k"], r["arm"])].append(r)
    for key, xs in sorted(by.items(), key=str):
        out["|".join(map(str, key))] = {
            "n": len(xs), "unfinished_pct": 100 * np.mean([not r["finished"] for r in xs]),
            "invalid_pct_of_finished": 100 * np.mean([not r["valid"] for r in xs if r["finished"]] or [0.0]),
            "mean_output_tokens": float(np.mean([r["n_output_tokens"] for r in xs]))}
    return out


def fmt(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def rate_fmt(t):
    return f"{t[0]:.0f} [{t[1]:.0f},{t[2]:.0f}]"


def report(res, label):
    print(f"\n==== {label} ====")
    print(f"VERDICT: {res['verdict']}  (E2 veto: {res['e2_veto']}; sign differs across rules/mixes: {res['heterogeneous_sign']})")
    print(f"E1 (matched rules, k in {CRIT['E1']['primary_ks']}): {fmt(res['E1'])} pp")
    print(f"E1 with k=8 included: {fmt(res['E1_incl_k8'])}; k=8 only: {fmt(res['E1_k8_only'])}")
    print("E1 by k: " + "; ".join(f"k={k}: {fmt(v)}" for k, v in res["E1_by_k"].items()))
    print("E1 by mix: " + "; ".join(f"{k}: {fmt(v)}" for k, v in res["E1_by_mix"].items()))
    print("E1 by rule: " + "; ".join(f"{k}: {fmt(v)}" for k, v in res["E1_by_rule"].items()))
    print(f"Level difference oss430-c430 without subtracting k=0 (same k): {fmt(res['level_diff_primary_ks'])}")
    print(f"E2 (logits per doubling, oss430-c430): {fmt(res['E2'])}; linear k: {fmt(res['E2_linear_k'])}; drop k=8: {fmt(res['E2_drop_k8'])}")
    print("E2 by rule: " + "; ".join(f"{k}: {fmt(v)}" for k, v in res["E2_by_rule"].items()))
    print("E2 by mix: " + "; ".join(f"{k}: {fmt(v)}" for k, v in res["E2_by_mix"].items()))
    print("\nRates, pooled over mix (% following the rule) [95% CI]:")
    print("rule | arm | " + " | ".join(f"k={k}" for k in KS) + " | first k >= 50%")
    for rk in MATCHED + [LARGEST]:
        for arm in ARMS:
            print(f"{rk} | {arm} | " + " | ".join(rate_fmt(res["pooled_mix"][f"{rk}|{arm}|{k}"]) for k in KS)
                  + f" | {res['pooled_mix'][f'{rk}|{arm}|first_k_50']}")
    print("\nPer rule:")
    for rk in MATCHED + [LARGEST]:
        g = res["gains"][rk]
        print(f"{rk}: k0 diff oss-c {fmt(g['k0_diff_oss_minus_c'])}; gain diff, primary ks {fmt(g['E1_primary_ks'])}; "
              f"by mix " + "; ".join(f"{m} {fmt(v)}" for m, v in g["E1_by_mix"].items()) + f"; E2 {fmt(g['E2'])}")
        print("   gain diff by k: " + "; ".join(f"k={k}: {fmt(v)}" for k, v in g["per_k_gain_diff"].items()))
        print("   slopes (logits/doubling): " + "; ".join(f"{a} {m} {v[0]:+.2f}" for a in ARMS for m, v in g["slope_by_arm"][a].items()))


def main():
    rows = load()
    print(f"{len(rows)} main-stage samples; arms {sorted({r['arm'] for r in rows})}")
    fin, alls = analyse(rows, True), analyse(rows, False)
    trunc = truncation(rows)
    flagged = {k: v for k, v in trunc.items() if v["unfinished_pct"] > CRIT["truncation_flag_pct"]}
    tot = {a: 100 * np.mean([not r["finished"] for r in rows if r["arm"] == a]) for a in ARMS}
    out = {"n_samples": len(rows), "finished_only": fin, "unfinished_as_miss": alls, "truncation_by_cell": trunc,
           "truncation_flagged_cells": flagged, "unfinished_pct_by_arm": tot,
           "cost_main_samples_usd": float(sum(r["cost_usd"] for r in rows))}
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1))
    report(fin, "PRIMARY: finished samples only")
    report(alls, "SENSITIVITY: unfinished counted as misses")
    print(f"\nUnfinished by arm: {tot}; cells over {CRIT['truncation_flag_pct']}%: {len(flagged)}")
    for k, v in flagged.items():
        print(f"  {k}: {v['unfinished_pct']:.1f}% of {v['n']}")
    worst = sorted(trunc.items(), key=lambda kv: -kv[1]["unfinished_pct"])[:8]
    print("Largest unfinished fractions: " + "; ".join(f"{k} {v['unfinished_pct']:.1f}%" for k, v in worst))


if __name__ == "__main__":
    main()
