"""Score the strategies-by-step verdicts (FINDINGS.md Part 1). Writes results.json and judged.jsonl.

    uv run --with numpy python experiments/hidden_preference/meeting/b_reasoning/strategies/analyze_st.py

Per run x arm: bootstrap over (step, request) clusters. Effective numbers are exp(entropy) of the category shares in a cell.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from corpus_st import ARMS, RUNS, load_all  # noqa: E402
from corpus_st import normalise  # noqa: E402

J = HERE / "judging"
OLD = HERE.parent / "judging"
SEED = 20261109
B_J, B_P = 2000, 1000
WEIGH, BASIS, REASON = {"compared", "listed", "none"}, {"hypothesis", "default", "hedge", "arbitrary", "other"}, {"reasoned", "asserted", "none"}
FEATURES = ["largest", "smallest", "middle", "upper_half", "lower_half", "last_digit", "digit_pattern", "round_number",
            "divisibility_other", "prime", "other", "none"]
THRESH = 0.5
rng = np.random.default_rng(SEED)


def read_verdicts(pattern, d=J):
    out, bad = {}, []
    for p in sorted(glob.glob(str(d / pattern))):
        for line in Path(p).read_text().split("\n"):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                v = json.loads(line)
                out[int(v["id"])] = v
            except Exception as e:
                bad.append((p, line[:80], str(e)))
    return out, bad


def valid(v):
    return (isinstance(v.get("cue_engaged"), bool) and v.get("weighing") in WEIGH and v.get("answer_basis") in BASIS and v.get("basis_feature") in FEATURES
            and v.get("reason_for_hypothesis") in REASON and isinstance(v.get("hypotheses"), list)
            and all(isinstance(h, dict) and h.get("feature") in FEATURES for h in v["hypotheses"]))


def entropy_eff(p):
    p = p / np.where(p.sum(-1, keepdims=True) > 0, p.sum(-1, keepdims=True), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(p * np.log(p), axis=-1)
    h = np.where(np.isnan(p).all(-1), np.nan, h)
    return np.exp(h)


class Boot:
    """One run x arm: clusters of (step, request). K is multinomial over clusters."""

    def __init__(self, clusters, B):
        self.uniq = sorted(set(clusters))
        self.pos = {u: i for i, u in enumerate(self.uniq)}
        C = len(self.uniq)
        self.K = rng.multinomial(C, [1 / C] * C, size=B)
        self.step = np.array([u[0] for u in self.uniq])

    def idx(self, step):
        return np.where(self.step == step)[0]


def windows(step, t):
    return {"pre": step < t, "step0": step == 0, "near": -4 <= step - t <= -1, "post": step > t, "all": True}


def position_code(info, cls, answer):
    if cls == "largest":
        return "largest"
    if cls == "smallest":
        return "smallest"
    if cls == "invalid":
        return "invalid"
    from rsrl.envs.hidden_preference.number import valid as vd
    v = vd(info)
    a = int(re.sub(r"[^\d-]", "", answer))
    idx = v.index(a)
    frac = (idx) / (len(v) - 1)  # strictly inside (0, 1)
    return "q%d" % min(3, int(frac * 4))


def summarise(cell_arrs, cell_point, takeoffs, label, windows_only=None):
    """cell_arrs: {(run, arm): {step: B-array}}; returns windows table with per-run values, pooled means and arm differences."""
    out = {}
    wnames = windows_only or ("pre", "step0", "near", "post", "all")
    for w in wnames:
        per = {}
        for run in RUNS:
            for arm in ARMS:
                t = takeoffs[f"{run}:{arm}"]
                steps = [s for s in cell_arrs[(run, arm)] if windows(s, t)[w]]
                pt = np.nanmean([cell_point[(run, arm)][s] for s in steps]) if steps else np.nan
                arr = np.nanmean(np.stack([cell_arrs[(run, arm)][s] for s in steps]), axis=0) if steps else None
                per[(run, arm)] = (pt, arr, steps)
        rows = {"runs": {}}
        for arm in ARMS:
            pts = [per[(r, arm)][0] for r in RUNS]
            arrs = np.stack([per[(r, arm)][1] for r in RUNS])
            pool = arrs.mean(0)
            rows[arm] = {"mean": float(np.mean(pts)), "ci": [float(x) for x in np.nanpercentile(pool, [2.5, 97.5])]}
            rows[f"{arm}_arr"] = pool
        diff = rows["oss430_arr"] - rows["c430_arr"]
        d_pt = rows["oss430"]["mean"] - rows["c430"]["mean"]
        ci = [float(x) for x in np.nanpercentile(diff, [2.5, 97.5])]
        per_run = {}
        for r in RUNS:
            dc = per[(r, "oss430")][0] - per[(r, "c430")][0]
            darr = per[(r, "oss430")][1] - per[(r, "c430")][1]
            per_run[r] = {"c430": float(per[(r, "c430")][0]), "oss430": float(per[(r, "oss430")][0]), "diff": float(dc),
                          "ci": [float(x) for x in np.nanpercentile(darr, [2.5, 97.5])],
                          "steps_c430": per[(r, "c430")][2], "steps_oss430": per[(r, "oss430")][2]}
        signs = [np.sign(per_run[r]["diff"]) for r in RUNS]
        verdict = classify(d_pt, ci, signs)
        out[w] = {"c430": rows["c430"], "oss430": rows["oss430"], "diff": float(d_pt), "diff_ci": ci, "per_run": per_run, "verdict": verdict}
    return out


def classify(d, ci, signs):
    lo, hi = ci
    all_pos, all_neg = all(s > 0 for s in signs), all(s < 0 for s in signs)
    if d >= THRESH and lo > 0 and all_pos:
        return "oss430 tries more"
    if d <= -THRESH and hi < 0 and all_neg:
        return "c430 tries more"
    if lo >= -THRESH and hi <= THRESH:
        return "no difference"
    return "unclear"


def per_step_bands(cell_arrs, cell_point):
    """Per arm, step: mean of run values and its CI (over runs that have the step)."""
    out = {}
    for arm in ARMS:
        steps = sorted(set().union(*[set(cell_arrs[(r, arm)]) for r in RUNS]))
        rows = {}
        for s in steps:
            have = [r for r in RUNS if s in cell_arrs[(r, arm)]]
            arr = np.nanmean(np.stack([cell_arrs[(r, arm)][s] for r in have]), axis=0)
            rows[int(s)] = {"mean": float(np.mean([cell_point[(r, arm)][s] for r in have])),
                            "ci": [float(x) for x in np.nanpercentile(arr, [2.5, 97.5])], "n_runs": len(have)}
        out[arm] = rows
    return out


def rel_bins(cell_arrs, cell_point, takeoffs, width=2, lo=-12, hi=14):
    """Binned by steps relative to takeoff; mean over the run-level bin means, runs that have the bin."""
    edges = list(range(lo, hi, width))
    out = {}
    for arm in ARMS:
        rows = {}
        for e in edges:
            have = []
            for r in RUNS:
                t = takeoffs[f"{r}:{arm}"]
                ss = [s for s in cell_arrs[(r, arm)] if e <= s - t < e + width]
                if ss:
                    have.append((np.mean([cell_point[(r, arm)][s] for s in ss]), np.nanmean(np.stack([cell_arrs[(r, arm)][s] for s in ss]), axis=0)))
            if have:
                pool = np.nanmean(np.stack([a for _, a in have]), axis=0)
                rows[e] = {"mean": float(np.mean([p for p, _ in have])), "ci": [float(x) for x in np.nanpercentile(pool, [2.5, 97.5])], "n_runs": len(have), "center": e + width / 2 - 0.5}
        out[arm] = rows
    return out


def per_run_rel(cell_point, takeoffs):
    return {f"{r}:{a}": {int(s - takeoffs[f"{r}:{a}"]): float(v) for s, v in cell_point[(r, a)].items()} for r in RUNS for a in ARMS}


def main():
    key = {int(i): x for i, x in json.loads((J / "key.json").read_text()).items()}
    verdicts, bad = read_verdicts("out_*.jsonl")
    check, bad2 = read_verdicts("checkout_*.jsonl")
    check_ids = json.loads((J / "check_ids.json").read_text())
    invalid = [i for i, v in verdicts.items() if not valid(v)]
    missing = [i for i in key if i not in verdicts]
    print("verdicts", len(verdicts), "of", len(key), "invalid", len(invalid), "missing", len(missing), "bad", len(bad))
    res = {"counts": {"entries": len(key), "valid": len(key) - len(invalid) - len(missing), "invalid": len(invalid), "missing": len(missing)}}
    recs = []
    for i, k in key.items():
        v = verdicts.get(i)
        if v is None or not valid(v):
            continue
        feats = [h["feature"] for h in v["hypotheses"]]
        recs.append({**k, "id": i, "n_hyp": len(feats), "n_distinct": len(set(feats)), "basis_feature": v["basis_feature"],
                     "answer_basis": v["answer_basis"], "log_words": float(np.log(max(1, k["n_words"])))})
    # OLS length slope, pooled within cell
    cells = defaultdict(list)
    for r in recs:
        cells[(r["run"], r["arm"], r["step"])].append(r)
    glob_l = float(np.mean([r["log_words"] for r in recs]))
    slopes = {}
    for m in ("n_hyp", "n_distinct"):
        num = den = 0.0
        for xs in cells.values():
            l = np.array([x["log_words"] for x in xs])
            y = np.array([x[m] for x in xs], float)
            l -= l.mean()
            y -= y.mean()
            num += float((l * y).sum())
            den += float((l * l).sum())
        slopes[m] = num / den
        for r in recs:
            r[f"adj_{m}"] = r[m] - slopes[m] * (r["log_words"] - glob_l)
    res["length_slope_per_log_word"] = slopes
    res["n_words_median"] = {arm: float(np.median([r["n_words"] for r in recs if r["arm"] == arm])) for arm in ARMS}

    # takeoffs
    _, takeoffs, audit = load_all()
    res["takeoffs"] = takeoffs
    res["audit"] = audit

    B = B_J
    judged_cell = {}  # measure -> (cell_arrs, cell_point)
    measures = {"n_hyp": "n_hyp", "n_distinct": "n_distinct", "adj_n_hyp": "adj_n_hyp", "adj_n_distinct": "adj_n_distinct", "n_words": "n_words"}
    boots = {}
    for run in RUNS:
        for arm in ARMS:
            xs = [r for r in recs if r["run"] == run and r["arm"] == arm]
            boots[(run, arm)] = (Boot([(r["step"], r["base_id"]) for r in xs], B), xs)
    for m, fld in measures.items():
        arrs, pts = {}, {}
        for key2, (bt, xs) in boots.items():
            S, N = np.zeros(len(bt.uniq)), np.zeros(len(bt.uniq))
            for x in xs:
                i = bt.pos[(x["step"], x["base_id"])]
                S[i] += x[fld]
                N[i] += 1
            arrs[key2], pts[key2] = {}, {}
            for s in sorted(set(bt.step.tolist())):
                idx = bt.idx(s)
                den = bt.K[:, idx] @ N[idx]
                arrs[key2][s] = (bt.K[:, idx] @ S[idx]) / np.where(den > 0, den, np.nan)
                pts[key2][s] = float(S[idx].sum() / N[idx].sum())
        judged_cell[m] = (arrs, pts)
    # effective number of basis_feature (judged, n=10 per cell)
    fpos = {f: i for i, f in enumerate(FEATURES)}
    arrs, pts = {}, {}
    for key2, (bt, xs) in boots.items():
        F = np.zeros((len(bt.uniq), len(FEATURES)))
        for x in xs:
            F[bt.pos[(x["step"], x["base_id"])], fpos[x["basis_feature"]]] += 1
        arrs[key2], pts[key2] = {}, {}
        for s in sorted(set(bt.step.tolist())):
            idx = bt.idx(s)
            pts[key2][s] = float(entropy_eff(F[idx].sum(0)))
            # plug-in entropy of a resample is biased low (resampling loses categories): recentre replicates on the point estimate
            a = entropy_eff(bt.K[:, idx] @ F[idx])
            arrs[key2][s] = a - np.nanmean(a) + pts[key2][s]
    judged_cell["basis_effective"] = (arrs, pts)

    res["windows"] = {m: summarise(a, p, takeoffs, m) for m, (a, p) in judged_cell.items()}
    res["by_step"] = {m: per_step_bands(a, p) for m, (a, p) in judged_cell.items()}
    res["rel_binned"] = {m: rel_bins(a, p, takeoffs) for m, (a, p) in judged_cell.items()}
    res["per_run_step"] = {m: {f"{r}:{arm}": {int(s): v for s, v in p[(r, arm)].items()} for r in RUNS for arm in ARMS} for m, (a, p) in judged_cell.items()}
    res["per_run_rel"] = {m: per_run_rel(p, takeoffs) for m, (a, p) in judged_cell.items()}

    # direct standardisation over 5 pooled word bins, pre-takeoff, n_hyp
    w_recs = [r for r in recs if r["step"] < r["takeoff"]]
    edges = np.quantile([r["n_words"] for r in w_recs], [0.2, 0.4, 0.6, 0.8])
    w = np.bincount(np.digitize([r["n_words"] for r in w_recs], edges), minlength=5) / len(w_recs)
    Dsum = {a: [np.zeros(5), np.zeros(5)] for a in ARMS}
    Barrs = {a: None for a in ARMS}
    for arm in ARMS:
        rep_S, rep_N = 0, 0
        for run in RUNS:
            bt, xs = boots[(run, arm)]
            t = takeoffs[f"{run}:{arm}"]
            xs2 = [x for x in xs if x["step"] < t]
            Sb, Nb = np.zeros((len(bt.uniq), 5)), np.zeros((len(bt.uniq), 5))
            for x in xs2:
                b = int(np.digitize(x["n_words"], edges))
                i = bt.pos[(x["step"], x["base_id"])]
                Sb[i, b] += x["n_hyp"]
                Nb[i, b] += 1
                Dsum[arm][0][b] += x["n_hyp"]
                Dsum[arm][1][b] += 1
            rep_S = rep_S + bt.K @ Sb
            rep_N = rep_N + bt.K @ Nb
        Barrs[arm] = rep_S / np.where(rep_N > 0, rep_N, np.nan)
    pt_d = float(np.nansum(w * (Dsum["oss430"][0] / np.where(Dsum["oss430"][1] > 0, Dsum["oss430"][1], np.nan) - Dsum["c430"][0] / np.where(Dsum["c430"][1] > 0, Dsum["c430"][1], np.nan))))
    d = np.nansum(w * (Barrs["oss430"] - Barrs["c430"]), axis=1)
    res["direct_std_pre_n_hyp"] = {"diff": pt_d, "ci": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])], "bin_share": w.tolist(),
                                   "bin_counts": {a: Dsum[a][1].tolist() for a in ARMS}}

    # programmatic: answer class / position, all finished number rollouts, rarefied to equal n
    rows, _, _ = load_all()
    prog = {}
    all_cells = defaultdict(list)
    for r in rows:
        r["pos"] = position_code(r["info"], r["cls"], r["answer"])
        all_cells[(r["run"], r["arm"], r["step"])].append(r)
    M = min(len(v) for v in all_cells.values())
    res["programmatic_rarefied_n"] = M
    for name, field, cats in [("answer_class", "cls", ["largest", "smallest", "other", "invalid"]), ("position", "pos", ["largest", "smallest", "q0", "q1", "q2", "q3", "invalid"])]:
        cpos = {c: i for i, c in enumerate(cats)}
        arrs, pts = {}, {}
        for run in RUNS:
            for arm in ARMS:
                xs = [r for r in rows if r["run"] == run and r["arm"] == arm]
                bt = Boot([(r["step"], r["base_id"]) for r in xs], B_P)
                F = np.zeros((len(bt.uniq), len(cats)))
                for x in xs:
                    F[bt.pos[(x["step"], x["base_id"])], cpos[x[field]]] += 1
                arrs[(run, arm)], pts[(run, arm)] = {}, {}
                for s in sorted(set(bt.step.tolist())):
                    idx = bt.idx(s)
                    total = F[idx].sum(0)
                    # point estimate: mean effective number over 200 rarefactions to M
                    draws = rng.multivariate_hypergeometric(total.astype(int), M, size=200)
                    pts[(run, arm)][s] = float(entropy_eff(draws.astype(float)).mean())
                    counts = bt.K[:, idx] @ F[idx]
                    vals = np.full(B_P, np.nan)
                    for j in range(B_P):
                        if counts[j].sum() > 0:
                            p = counts[j] / counts[j].sum()
                            vals[j] = float(entropy_eff(rng.multinomial(M, p).astype(float)))
                    arrs[(run, arm)][s] = vals - np.nanmean(vals) + pts[(run, arm)][s]  # recentre, as for the judged entropy
        prog[name] = (arrs, pts)
    res["windows_prog"] = {n: summarise(a, p, takeoffs, n) for n, (a, p) in prog.items()}
    res["by_step_prog"] = {n: per_step_bands(a, p) for n, (a, p) in prog.items()}
    res["rel_binned_prog"] = {n: rel_bins(a, p, takeoffs) for n, (a, p) in prog.items()}
    res["per_run_step_prog"] = {n: {f"{r}:{arm}": {int(s): v for s, v in p[(r, arm)].items()} for r in RUNS for arm in ARMS} for n, (a, p) in prog.items()}
    res["per_run_rel_prog"] = {n: per_run_rel(p, takeoffs) for n, (a, p) in prog.items()}

    # agreement: Luna vs Sonnet on the check subset
    both = [i for i in check_ids if i in check and i in verdicts and valid(check[i]) and valid(verdicts[i])]
    def h(v):
        return [x["feature"] for x in v["hypotheses"]]
    ag = {"n": len(both)}
    a, b = [len(h(verdicts[i])) for i in both], [len(h(check[i])) for i in both]
    ag["n_hyp"] = {"exact": float(np.mean(np.array(a) == np.array(b))), "within1": float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1)),
                   "corr": float(np.corrcoef(a, b)[0, 1]), "mean_luna": float(np.mean(a)), "mean_sonnet": float(np.mean(b))}
    a2, b2 = [len(set(h(verdicts[i]))) for i in both], [len(set(h(check[i]))) for i in both]
    ag["n_distinct"] = {"exact": float(np.mean(np.array(a2) == np.array(b2))), "within1": float(np.mean(np.abs(np.array(a2) - np.array(b2)) <= 1)),
                        "corr": float(np.corrcoef(a2, b2)[0, 1])}
    labs_a, labs_b = [verdicts[i]["basis_feature"] for i in both], [check[i]["basis_feature"] for i in both]
    labels = sorted(set(labs_a) | set(labs_b))
    p0 = float(np.mean([x == y for x, y in zip(labs_a, labs_b)]))
    pe = sum((labs_a.count(l) / len(labs_a)) * (labs_b.count(l) / len(labs_b)) for l in labels)
    ag["basis_feature"] = {"agree": p0, "kappa": float((p0 - pe) / (1 - pe))}
    bya = {}
    for arm in ARMS:
        ids = [i for i in both if key[i]["arm"] == arm]
        bya[arm] = {"n": len(ids), "n_hyp_luna": float(np.mean([len(h(verdicts[i])) for i in ids])), "n_hyp_sonnet": float(np.mean([len(h(check[i])) for i in ids]))}
    ag["by_arm"] = bya
    # Sonnet-based pre-takeoff arm difference on the check subset (stratified, not representative)
    pre_ids = [i for i in both if key[i]["rel"] < 0]
    ag["pre_takeoff_subset"] = {arm: {"n": sum(key[i]["arm"] == arm for i in pre_ids),
                                      "luna": float(np.mean([len(h(verdicts[i])) for i in pre_ids if key[i]["arm"] == arm])),
                                      "sonnet": float(np.mean([len(h(check[i])) for i in pre_ids if key[i]["arm"] == arm]))} for arm in ARMS}
    res["agreement"] = ag

    # test-retest: transcripts in both this sample and the earlier judged sample (same prompt)
    old_text = {}
    for p in sorted(glob.glob(str(OLD / "batch_*.txt"))):
        for m in re.finditer(r"=== ENTRY id=(\d+) ===\n(.*?)=== END ENTRY id=\1 ===", Path(p).read_text(), re.S):
            old_text[m.group(2)] = int(m.group(1))
    old_v, _ = read_verdicts("out_*.jsonl", OLD)
    new_text = {}
    for p in sorted(glob.glob(str(J / "batch_*.txt"))):
        for m in re.finditer(r"=== ENTRY id=(\d+) ===\n(.*?)=== END ENTRY id=\1 ===", Path(p).read_text(), re.S):
            new_text[int(m.group(1))] = m.group(2)
    pairs = [(i, old_text[t]) for i, t in new_text.items() if t in old_text and i in verdicts and old_text[t] in old_v]
    if pairs:
        x = [len(h(verdicts[i])) for i, _ in pairs]
        y = [len(h(old_v[o])) for _, o in pairs]
        res["test_retest"] = {"n": len(pairs), "n_hyp_exact": float(np.mean(np.array(x) == np.array(y))), "n_hyp_within1": float(np.mean(np.abs(np.array(x) - np.array(y)) <= 1)),
                              "mean_new": float(np.mean(x)), "mean_old": float(np.mean(y)),
                              "basis_agree": float(np.mean([verdicts[i]["basis_feature"] == old_v[o]["basis_feature"] for i, o in pairs]))}
    else:
        res["test_retest"] = {"n": 0}
    # quote fidelity
    ok = []
    for i, v in verdicts.items():
        q = re.sub(r"\s+", " ", v.get("quote", "")).strip()
        t = re.sub(r"\s+", " ", new_text.get(i, ""))
        if q:
            ok.append(q in t)
    res["quote_verbatim_rate"] = float(np.mean(ok)) if ok else None
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=str))
    with open(HERE / "judged.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res["counts"]))


if __name__ == "__main__":
    main()
