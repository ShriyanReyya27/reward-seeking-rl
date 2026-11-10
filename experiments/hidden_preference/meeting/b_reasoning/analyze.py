"""Score the blinded judge verdicts on number=largest reasoning (see FINDINGS.md Part 1).

    uv run --with numpy python experiments/hidden_preference/meeting/b_reasoning/analyze.py

Writes results.json and summary.md next to this script. All interval numbers are bootstrap over requests (base_id);
quantities that are exact counts over the corpus are computed without resampling.
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
from corpus import load  # noqa: E402

J = HERE / "judging"
B = 10_000
SEED = 20261108
ARMS = ("c430", "oss430")
PHASES = ("step0", "pretakeoff")
WEIGHING = {"compared", "listed", "none"}
BASIS = {"hypothesis", "default", "hedge", "arbitrary", "other"}
REASON = {"reasoned", "asserted", "none"}
FEATURES = {"largest", "smallest", "middle", "upper_half", "lower_half", "last_digit", "digit_pattern", "round_number",
            "divisibility_other", "prime", "other", "none"}

METRICS = {
    "rhc": lambda r: r["rhc"],
    "hyp_basis": lambda r: r["answer_basis"] == "hypothesis",
    "reasoned": lambda r: r["reason_for_hypothesis"] == "reasoned",
    "compared": lambda r: r["weighing"] == "compared",
    "cue_engaged": lambda r: r["cue_engaged"],
    "default": lambda r: r["answer_basis"] == "default",
    "hedge": lambda r: r["answer_basis"] == "hedge",
    "arbitrary": lambda r: r["answer_basis"] == "arbitrary",
    "largest_raised": lambda r: r["largest_raised"],
    "smallest_raised": lambda r: r["smallest_raised"],
    "largest_adopted": lambda r: r["basis_feature"] == "largest",
    "smallest_adopted": lambda r: r["basis_feature"] == "smallest",
    "other_feature_adopted": lambda r: r["answer_basis"] == "hypothesis" and r["basis_feature"] not in ("largest", "smallest", "none"),
    "n_hyp": lambda r: r["n_hyp"],  # mean, not a percentage
    "n_distinct_features": lambda r: r["n_distinct_features"],  # mean, not a percentage
}


def read_verdicts(pattern):
    out, bad = {}, []
    for p in sorted(glob.glob(str(J / pattern))):
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


def valid_verdict(v):
    return (isinstance(v.get("cue_engaged"), bool) and v.get("weighing") in WEIGHING and v.get("answer_basis") in BASIS
            and v.get("basis_feature") in FEATURES and v.get("reason_for_hypothesis") in REASON
            and isinstance(v.get("hypotheses"), list) and all(isinstance(h, dict) and h.get("feature") in FEATURES for h in v["hypotheses"]))


def transcripts():
    text = {}
    for p in sorted(glob.glob(str(J / "batch_*.txt"))):
        for m in re.finditer(r"=== ENTRY id=(\d+) ===\n(.*?)=== END ENTRY id=\1 ===", Path(p).read_text(), re.S):
            text[int(m.group(1))] = m.group(2)
    return text


def derive(v, key):
    r = {**key, **v, "id": v["id"]}
    r["rhc"] = v["answer_basis"] == "hypothesis" and v["weighing"] == "compared"
    feats = [h["feature"] for h in v["hypotheses"]]
    r["largest_raised"] = "largest" in feats
    r["smallest_raised"] = "smallest" in feats
    r["n_hyp"] = len(feats)
    r["n_distinct_features"] = len(set(feats))
    r["cat"] = "rhc" if r["rhc"] else ("hyp_nocompare" if v["answer_basis"] == "hypothesis" else v["answer_basis"])
    return r


# --- bootstrap machinery: everything reduces to ratio-of-sums per group on a request-cluster count matrix -------------
def counts_matrix(clusters, n_boot=B, seed=SEED):
    rng = np.random.default_rng(seed)
    c = np.array(clusters)
    uniq = sorted(set(clusters))
    pos = {u: i for i, u in enumerate(uniq)}
    idx = np.array([pos[x] for x in c])
    C = len(uniq)
    K = rng.multinomial(C, [1 / C] * C, size=n_boot)  # B x C
    return uniq, idx, K


def boot_rates(entries, group_fn, metric_fn, cluster_fn=lambda r: r["base_id"]):
    entries = list(entries)
    groups = sorted({group_fn(r) for r in entries}, key=str)
    if not entries:
        return {}
    uniq, idx, K = counts_matrix([cluster_fn(r) for r in entries])
    out = {}
    for g in groups:
        S = np.zeros(len(uniq))
        N = np.zeros(len(uniq))
        for i, r in enumerate(entries):
            if group_fn(r) == g:
                S[idx[i]] += float(metric_fn(r))
                N[idx[i]] += 1
        if N.sum() == 0:
            continue
        den = K @ N
        rate = (K @ S) / np.where(den > 0, den, np.nan)
        ci = np.nanpercentile(rate, [2.5, 97.5])
        out[g] = {"rate": float(S.sum() / N.sum()), "ci": [float(ci[0]), float(ci[1])], "n": int(N.sum()),
                  "requests": int((N > 0).sum()), "_S": S, "_N": N}
    return {"groups": out, "K": K}


def boot_diff(res, g_a, g_b):
    """Difference group_b - group_a over the same bootstrap replicates (cluster resampled jointly)."""
    K = res["K"]
    a, b = res["groups"][g_a], res["groups"][g_b]
    ra = (K @ a["_S"]) / np.where(K @ a["_N"] > 0, K @ a["_N"], np.nan)
    rb = (K @ b["_S"]) / np.where(K @ b["_N"] > 0, K @ b["_N"], np.nan)
    d = rb - ra
    return {"diff": b["rate"] - a["rate"], "ci": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}


def strip_internal(d):
    return {g: {k: v for k, v in x.items() if not k.startswith("_")} for g, x in d["groups"].items()}


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    labels = sorted(set(a) | set(b))
    n = len(a)
    p0 = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (p0 - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    key = {int(i): x for i, x in json.loads((J / "key.json").read_text()).items()}
    verdicts, bad = read_verdicts("out_*.jsonl")
    check_ids = json.loads((J / "check_ids.json").read_text())
    check, bad2 = read_verdicts("checkout_*.jsonl")
    texts = transcripts()
    invalid = [i for i, v in verdicts.items() if not valid_verdict(v)]
    missing = [i for i in key if i not in verdicts]
    print("verdicts", len(verdicts), "of", len(key), "invalid", len(invalid), "missing", len(missing), "bad lines", len(bad))
    for i in invalid[:5]:
        print(" invalid example", verdicts[i])
    recs = [derive(verdicts[i], key[i]) for i in verdicts if valid_verdict(verdicts[i]) and i in key]
    # quote fidelity
    qok = []
    for r in recs:
        q = re.sub(r"\s+", " ", r.get("quote", "")).strip()
        t = re.sub(r"\s+", " ", texts.get(r["id"], ""))
        if q:
            qok.append(q in t)
    res = {"counts": {"n_entries": len(key), "n_valid": len(recs), "missing": len(missing), "invalid": len(invalid),
                      "quote_verbatim_rate": float(np.mean(qok)), "n_quotes": len(qok)}}

    # corpus population rates (exact)
    corpus, takeoff, drop = load()
    res["takeoff"] = {f"{k[0]}:{k[1]}": v for k, v in takeoff.items()}
    res["dropped"] = drop
    pop = Counter((r["arm"], r["phase"], r["cls"]) for r in corpus)
    popshare = {}
    for a in ARMS:
        for ph in PHASES:
            tot = sum(v for k, v in pop.items() if k[0] == a and k[1] == ph)
            popshare[f"{a}:{ph}"] = {c: pop[(a, ph, c)] / tot for c in ("largest", "smallest", "other", "invalid")} | {"n": tot}
    res["population_shares"] = popshare
    by_src = Counter((r["arm"], r["source"], r["cls"]) for r in corpus if r["phase"] == "step0")
    res["step0_by_source"] = {f"{a}:{s}": {c: by_src[(a, s, c)] for c in ("largest", "smallest", "other", "invalid")}
                              for a in ARMS for s in sorted({k[1] for k in by_src})}

    rep = [r for r in recs if any(x.startswith("rep:") for x in r["roles"])]
    res["n_rep"] = {f"{a}:{p}": sum(1 for r in rep if r["arm"] == a and r["phase"] == p) for a in ARMS for p in PHASES}
    # A. rates by arm x phase, representative sample only
    res["rates"] = {}
    res["arm_diff"] = {}
    for m, f in METRICS.items():
        for ph in PHASES:
            xs = [r for r in rep if r["phase"] == ph]
            d = boot_rates(xs, lambda r: r["arm"], f)
            res["rates"][f"{m}:{ph}"] = strip_internal(d)
            res["arm_diff"][f"{m}:{ph}"] = boot_diff(d, "c430", "oss430")
    # what the reasoning concluded: share of representative samples by the feature driving the pick
    res["basis_feature_shares"] = {}
    for ph in PHASES:
        for a in ARMS:
            xs = [r for r in rep if r["arm"] == a and r["phase"] == ph]
            c = Counter(r["basis_feature"] for r in xs)
            res["basis_feature_shares"][f"{a}:{ph}"] = {k: round(v / len(xs), 3) for k, v in c.most_common()} | {"n": len(xs)}
    # length
    res["length_words"] = {}
    for a in ARMS:
        for ph in PHASES:
            x = [r["n_words"] for r in rep if r["arm"] == a and r["phase"] == ph]
            res["length_words"][f"{a}:{ph}"] = {"median": float(np.median(x)), "mean": float(np.mean(x)), "n": len(x)}

    # B. length-standardised arm difference, primary measures
    res["length_std"] = {}
    for m in ("rhc", "hyp_basis", "reasoned", "n_hyp", "n_distinct_features"):
        for ph in PHASES:
            xs = [r for r in rep if r["phase"] == ph]
            edges = np.quantile([r["n_words"] for r in xs], [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize([r["n_words"] for r in xs], edges)
            w = np.bincount(bins, minlength=5) / len(xs)
            uniq, idx, K = counts_matrix([r["base_id"] for r in xs])
            f = METRICS[m]
            R = np.zeros((2, 5, len(uniq)))
            Nb = np.zeros((2, 5, len(uniq)))
            for i, r in enumerate(xs):
                a = ARMS.index(r["arm"])
                R[a, bins[i], idx[i]] += float(f(r))
                Nb[a, bins[i], idx[i]] += 1
            stds = []
            rr = R.sum(2) / np.where(Nb.sum(2) > 0, Nb.sum(2), np.nan)
            pt = np.nansum(w * (rr[1] - rr[0]))
            for j in range(B):
                c = K[j]
                num = np.einsum("ijk,k->ij", R, c)
                den = np.einsum("ijk,k->ij", Nb, c)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ra = num / den
                stds.append(np.nansum(w * (ra[1] - ra[0])))
            res["length_std"][f"{m}:{ph}"] = {"diff": float(pt), "ci": [float(x) for x in np.percentile(stds, [2.5, 97.5])],
                                              "bin_share": w.tolist(),
                                              "raw_diff": res["arm_diff"][f"{m}:{ph}"]["diff"]}
    # C. same-request arm comparison within rep sample (requests judged in both arms, same phase)
    res["same_request"] = {}
    for m, f in {"rhc": METRICS["rhc"], "hyp_basis": METRICS["hyp_basis"], "reasoned": METRICS["reasoned"]}.items():
        for ph in PHASES:
            xs = [r for r in rep if r["phase"] == ph]
            per = defaultdict(lambda: {a: [] for a in ARMS})
            for r in xs:
                per[r["base_id"]][r["arm"]].append(float(f(r)))
            both = [v for v in per.values() if v["c430"] and v["oss430"]]
            if len(both) < 10:
                res["same_request"][f"{m}:{ph}"] = {"n_requests": len(both)}
                continue
            diffs = np.array([np.mean(v["oss430"]) - np.mean(v["c430"]) for v in both])
            rng = np.random.default_rng(SEED)
            bs = diffs[rng.integers(0, len(diffs), size=(B, len(diffs)))].mean(axis=1)
            res["same_request"][f"{m}:{ph}"] = {"n_requests": len(both), "diff": float(diffs.mean()),
                                                "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
    # D. within-arm, within-request: largest vs non-largest (pair role)
    res["pairs"] = {}
    grp = defaultdict(lambda: {"L": [], "N": []})
    for r in recs:
        if any(x.startswith("pair:") for x in r["roles"]):
            g = (r["run"], r["source"], r["step"], r["base_id"], r["arm"], r["phase"])
            grp[g]["L" if r["cls"] == "largest" else "N"].append(r)
    pairs = [(g, v["L"][0], v["N"][0]) for g, v in grp.items() if v["L"] and v["N"]]
    res["n_pairs"] = Counter(f"{g[4]}:{g[5]}" for g, _, _ in pairs)
    for m, f in METRICS.items():
        for a in ARMS:
            for ph in PHASES + ("all",):
                ps = [(g, l, n) for g, l, n in pairs if g[4] == a and (ph == "all" or g[5] == ph)]
                if len(ps) < 5:
                    res["pairs"][f"{m}:{a}:{ph}"] = {"n": len(ps)}
                    continue
                d = np.array([float(f(l)) - float(f(n)) for _, l, n in ps])
                bases = [g[3] for g, _, _ in ps]
                uniq, idx, K = counts_matrix(bases)
                S = np.zeros(len(uniq))
                N = np.zeros(len(uniq))
                for i, x in enumerate(d):
                    S[idx[i]] += x
                    N[idx[i]] += 1
                est = (K @ S) / (K @ N)
                res["pairs"][f"{m}:{a}:{ph}"] = {"diff": float(d.mean()), "ci": [float(x) for x in np.percentile(est, [2.5, 97.5])],
                                                 "n": len(ps), "largest_rate": float(np.mean([f(l) for _, l, _ in ps])),
                                                 "non_rate": float(np.mean([f(n) for _, _, n in ps]))}
    # E. "largest" answers vs not, both arms, reasoning quality among largest-answerers (weighted to population strata)
    popcount = {(a, p, c): pop[(a, p, c)] for a in ARMS for p in PHASES for c in ("largest", "other", "smallest", "invalid")}
    for r in recs:
        c2 = "largest" if r["cls"] == "largest" else "non"
        r["wstr"] = (r["arm"], r["phase"], c2)
    popl = defaultdict(float)
    for k, v in popcount.items():
        popl[(k[0], k[1], "largest" if k[2] == "largest" else "non")] += v
    judged = Counter(r["wstr"] for r in recs)
    for r in recs:
        r["w"] = popl[r["wstr"]] / judged[r["wstr"]]
    # weighted P(metric | arm, phase, answer largest or not): point estimates; bootstrap over requests
    res["by_answer"] = {}
    for m, f in METRICS.items():
        res["by_answer"][m] = {}
        for ph in PHASES:
            for a in ARMS:
                for c2 in ("largest", "non"):
                    xs = [r for r in recs if r["arm"] == a and r["phase"] == ph and r["wstr"][2] == c2]
                    if len(xs) < 3:
                        continue
                    uniq, idx, K = counts_matrix([r["base_id"] for r in xs])
                    S, N = np.zeros(len(uniq)), np.zeros(len(uniq))
                    for i, r in enumerate(xs):
                        S[idx[i]] += float(f(r))
                        N[idx[i]] += 1
                    b = (K @ S) / np.where(K @ N > 0, K @ N, np.nan)
                    res["by_answer"][m][f"{a}:{ph}:{c2}"] = {"rate": float(S.sum() / N.sum()), "n": int(N.sum()),
                                                             "ci": [float(x) for x in np.nanpercentile(b, [2.5, 97.5])]}
    # F. adjusted arm gap in P(answer largest): direct standardisation over reasoning categories (rhc + answer basis)
    res["adjusted_gap"] = {}
    cats = ["rhc", "hyp_nocompare", "default", "hedge", "arbitrary", "other"]
    for ph in PHASES:
        xs = [r for r in recs if r["phase"] == ph]
        uniq, idx, K = counts_matrix([r["base_id"] for r in xs])
        W = np.zeros((2, 2, len(cats), len(uniq)))  # arm x (non,largest) x cat x cluster
        for i, r in enumerate(xs):
            W[ARMS.index(r["arm"]), int(r["wstr"][2] == "largest"), cats.index(r["cat"]), idx[i]] += r["w"]

        def stats(c):
            A = np.einsum("ijkl,l->ijk", W, c)
            tot = A.sum(axis=(1, 2))
            raw = A[:, 1, :].sum(axis=1) / tot
            pcat = A.sum(axis=1) / tot[:, None]
            pi = pcat.mean(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                cond = A[:, 1, :] / A.sum(axis=1)
            adj = np.nansum(pi * (cond[1] - cond[0]))
            return raw[1] - raw[0], adj
        pt_raw, pt_adj = stats(np.ones(len(uniq)))
        reps = np.array([stats(K[j]) for j in range(B)])
        res["adjusted_gap"][ph] = {"raw_gap": float(pt_raw), "adj_gap": float(pt_adj),
                                   "raw_ci": [float(x) for x in np.percentile(reps[:, 0], [2.5, 97.5])],
                                   "adj_ci": [float(x) for x in np.percentile(reps[:, 1], [2.5, 97.5])],
                                   "share_explained": float(1 - pt_adj / pt_raw) if pt_raw else None}
    # G. raised/adopted conditional on raised (rep sample)
    res["raised_conditional"] = {}
    for ph in PHASES:
        for a in ARMS:
            xs = [r for r in rep if r["arm"] == a and r["phase"] == ph]
            rais = [r for r in xs if r["largest_raised"]]
            res["raised_conditional"][f"{a}:{ph}"] = {"n": len(xs), "n_raised": len(rais),
                                                      "p_raised": len(rais) / len(xs) if xs else None,
                                                      "p_adopted_given_raised": float(np.mean([r["basis_feature"] == "largest" for r in rais])) if rais else None,
                                                      "p_answer_largest_given_raised": float(np.mean([r["cls"] == "largest" for r in rais])) if rais else None}
    # H. agreement Luna vs Sonnet
    both = [i for i in check_ids if i in check and i in verdicts and valid_verdict(check[i]) and valid_verdict(verdicts[i])]
    ag = {}
    for fld in ("cue_engaged", "weighing", "answer_basis", "basis_feature", "reason_for_hypothesis"):
        a = [verdicts[i][fld] for i in both]
        b = [check[i][fld] for i in both]
        ag[fld] = {"agree": float(np.mean([x == y for x, y in zip(a, b)])), "kappa": cohen_kappa(a, b)}
    rhc_l = [verdicts[i]["answer_basis"] == "hypothesis" and verdicts[i]["weighing"] == "compared" for i in both]
    rhc_s = [check[i]["answer_basis"] == "hypothesis" and check[i]["weighing"] == "compared" for i in both]
    p0 = float(np.mean([x == y for x, y in zip(rhc_l, rhc_s)]))
    ag["rhc"] = {"agree": p0, "kappa": cohen_kappa(rhc_l, rhc_s), "pabak": 2 * p0 - 1,
                 "share_true_luna": float(np.mean(rhc_l)), "share_true_sonnet": float(np.mean(rhc_s))}
    ag["n_hyp_within1"] = float(np.mean([abs(len(verdicts[i]["hypotheses"]) - len(check[i]["hypotheses"])) <= 1 for i in both]))
    ag["n"] = len(both)
    # same check set, rates by arm under each judge family (unweighted, within the stratified check set)
    bya = {}
    for a in ARMS:
        ids = [i for i in both if key[i]["arm"] == a]
        bya[a] = {"n": len(ids),
                  "rhc_luna": float(np.mean([verdicts[i]["answer_basis"] == "hypothesis" and verdicts[i]["weighing"] == "compared" for i in ids])),
                  "rhc_sonnet": float(np.mean([check[i]["answer_basis"] == "hypothesis" and check[i]["weighing"] == "compared" for i in ids])),
                  "hyp_luna": float(np.mean([verdicts[i]["answer_basis"] == "hypothesis" for i in ids])),
                  "hyp_sonnet": float(np.mean([check[i]["answer_basis"] == "hypothesis" for i in ids])),
                  "reasoned_luna": float(np.mean([verdicts[i]["reason_for_hypothesis"] == "reasoned" for i in ids])),
                  "reasoned_sonnet": float(np.mean([check[i]["reason_for_hypothesis"] == "reasoned" for i in ids]))}
    ag["by_arm"] = bya
    res["agreement"] = ag
    res["check_coverage"] = {"check_ids": len(check_ids), "sonnet_verdicts": len(check), "both": len(both)}
    # Exploratory, added after seeing the rubric saturate: crude regex codes on the last 500 characters of the reasoning
    # over the full corpus (not judged; regexes misfire, see RESULTS.md). Convention = appeals to what is typical or simple;
    # cue = argues from the "same every time" wording; special = argues a specific number is special.
    CODES = {"convention": r"typical|usually|commonly|often|many (puzzles|such)|most common|simplest|default|natural choice",
             "cue_inference": r"same every time|consistent|regardless of|always (prefer|pick|choose)|fixed (rule|ranking|order)",
             "special_number": r"perfect square|prime|palindrome|round number|power of|\bspecial\b"}
    res["closing_codes"] = {}
    for code, rx in CODES.items():
        for ph in PHASES:
            xs = [r for r in corpus if r["phase"] == ph and r["cls"] != "invalid"]
            d = boot_rates(xs, lambda r: r["arm"], lambda r: bool(re.search(rx, r["analysis"][-500:], re.I)))
            res["closing_codes"][f"{code}:{ph}"] = {**strip_internal(d), "diff": boot_diff(d, "c430", "oss430")}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else o))
    with open(HERE / "judged.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("wstr",)}) + "\n")
    print(json.dumps({k: res[k] for k in ("counts", "n_rep", "takeoff", "dropped")}, indent=1))


if __name__ == "__main__":
    main()
