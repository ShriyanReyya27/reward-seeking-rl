"""Step 2 analysis: curves, average held-out reward across training (primary), paired task bootstrap."""
import gzip, json, glob, os, random, statistics as st, sys
from collections import defaultdict

D = "runs/t2"
ARMS = ("minus", "sham", "untrained")


def evrows(run):
    rows = []
    for p in sorted(glob.glob(f"{D}/{run}/rollouts_from*.jsonl.gz")):
        try:
            for l in gzip.open(p, "rt"):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("split", "").startswith("test"):
                    rows.append(r)
        except (EOFError, OSError):
            pass
    return rows


def per_task(rows, env, split="test"):
    """task -> step -> mean reward. E1: k > 0 episodes only (scores shown)."""
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] != split:
            continue
        if env == "np" and r["k"] == 0:
            continue
        d[r["id"]][r["step"]].append(r["reward"])
    return {t: {s: st.mean(v) for s, v in ss.items()} for t, ss in d.items()}


def k0(rows):
    v = [r["reward"] for r in rows if r.get("k") == 0]
    return st.mean(v) if v else float("nan")


def summary(env):
    out = {}
    for arm in ARMS:
        for seed in (0, 1):
            run = f"{env}_{arm}_s{seed}"
            if not os.path.exists(f"{D}/{run}"):
                continue
            rows = evrows(run)
            pt = per_task(rows, env)
            steps = sorted({s for t in pt.values() for s in t})
            curve = {s: st.mean(t[s] for t in pt.values() if s in t) for s in steps}
            k0c = {s: k0([r for r in rows if r["step"] == s and r["split"] == "test"]) for s in steps} if env == "np" else {}
            ns = {s: st.mean(r["tokens"] for r in rows if r["step"] == s) for s in steps}
            out[run] = dict(steps=steps, curve=curve, k0=k0c, tokens=ns, pt=pt,
                            auc=st.mean(curve.values()) if curve else float("nan"))
    return out


def boot_diff(S, env, a, b, common_steps, n=4000, seed=0):
    """Paired over tasks: mean over seeds and common steps of (a - b). Returns mean, CI, per-seed diffs."""
    per_seed, tasks = {}, None
    diffs_by_task = defaultdict(list)
    for sd in (0, 1):
        ra, rb = S.get(f"{env}_{a}_s{sd}"), S.get(f"{env}_{b}_s{sd}")
        if not ra or not rb:
            continue
        common = [s for s in common_steps if s in ra["curve"] and s in rb["curve"]]
        ts = set(ra["pt"]) & set(rb["pt"])
        vals = []
        for t in ts:
            xa = [ra["pt"][t][s] for s in common if s in ra["pt"][t]]
            xb = [rb["pt"][t][s] for s in common if s in rb["pt"][t]]
            if xa and xb:
                vals.append(st.mean(xa) - st.mean(xb)); diffs_by_task[t].append(st.mean(xa) - st.mean(xb))
        per_seed[sd] = st.mean(vals) if vals else float("nan")
    xs = [st.mean(v) for v in diffs_by_task.values()]
    if not xs:
        return float("nan"), (float("nan"), float("nan")), per_seed
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return st.mean(xs), (ms[int(.025 * n)], ms[int(.975 * n)]), per_seed


if __name__ == "__main__":
    res = {}
    for env in ("np", "tu"):
        S = summary(env)
        if not S:
            continue
        print(f"=== {env}")
        for run, r in S.items():
            print(f"  {run:20s} " + " ".join(f"{s}:{100*v:5.1f}" for s, v in r["curve"].items()) + f" | avg {100*r['auc']:.1f}"
                  + (" | k0 " + " ".join(f"{100*v:.0f}" for v in r["k0"].values()) if r["k0"] else "")
                  + " | tok " + " ".join(f"{v:.0f}" for v in r["tokens"].values()))
        common = sorted(set.intersection(*[set(r["steps"]) for r in S.values()]))
        res[env] = {"runs": {k: {kk: vv for kk, vv in v.items() if kk != "pt"} for k, v in S.items()}, "common_steps": common, "contrasts": {}}
        for a, b in (("minus", "sham"), ("minus", "untrained"), ("sham", "untrained")):
            m, ci, ps = boot_diff(S, env, a, b, common)
            # step-0 and later-steps-only
            m0, ci0, _ = boot_diff(S, env, a, b, [0])
            ml, cil, psl = boot_diff(S, env, a, b, [s for s in common if s > 0])
            res[env]["contrasts"][f"{a}-{b}"] = dict(avg=m, ci=ci, per_seed=ps, step0=m0, step0_ci=ci0, after0=ml, after0_ci=cil)
            print(f"  {a}-{b}: avg over steps {common}: {100*m:+.1f} [{100*ci[0]:+.1f},{100*ci[1]:+.1f}] per seed "
                  + " ".join(f"{100*v:+.1f}" for v in ps.values()) + f" | step0 {100*m0:+.1f} | steps>0 {100*ml:+.1f} [{100*cil[0]:+.1f},{100*cil[1]:+.1f}]")
    json.dump(res, open(f"{D}/summary.json", "w"), indent=1, default=str)
