"""Summarise Step 0 per model: evidence gain (plain = propensity, explicit = capability), conflict-following, stated accuracy."""
import json, os, sys, random, statistics as st
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "runs/step0"
MODELS = [m for m in ("glm", "glmflash", "kimi", "base", "a430", "c430") if os.path.exists(f"{D}/{m}.jsonl")]


def load(m):
    return [json.loads(l) for l in open(f"{D}/{m}.jsonl") if l.strip()]


def gains(rows, framing, fam=None):
    by = defaultdict(dict)
    for r in rows:
        if r["kind"] == "hp" and r["framing"] == framing and (fam is None or r["fam"] == fam):
            by[r["base_id"]][(r["k"], r["mix"])] = r["reward"]
    g = {b: st.mean(v for c, v in d.items() if c[0]) - d[(0, "balanced")] for b, d in by.items()
         if (0, "balanced") in d and len(d) > 1}
    k0 = [d[(0, "balanced")] for d in by.values() if (0, "balanced") in d]
    kp = [v for d in by.values() for c, v in d.items() if c[0]]
    return g, (st.mean(k0) if k0 else float("nan")), (st.mean(kp) if kp else float("nan"))


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed); xs = list(vals)
    if not xs:
        return (float("nan"),) * 2
    ms = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


out = {}
print(f"{'model':9s} {'n':>5s} | plain: k0  k>0  gain [CI]          | explicit gain | conflict: follow-instr follow-scores invalid | agree | stated | tok  trunc")
for m in MODELS:
    rows = load(m)
    gp, k0p, kpp = gains(rows, "plain"); ge, k0e, kpe = gains(rows, "explicit")
    c = [r for r in rows if r["kind"] == "conflict"]; ag = [r for r in rows if r["kind"] == "agree"]; s = [r for r in rows if r["kind"] == "stated"]
    fi = st.mean(r["follows_instruction"] for r in c) if c else float("nan")
    fs = st.mean(r["follows_scores"] for r in c) if c else float("nan")
    inv = st.mean(not r["valid"] for r in c) if c else float("nan")
    lo, hi = boot(gp.values())
    tok = st.mean(r["tokens"] for r in rows); tr = st.mean(not r["finished"] for r in rows)
    fam_p = {f: st.mean(gains(rows, "plain", f)[0].values()) if gains(rows, "plain", f)[0] else None for f in ("number", "word", "tagline", "format")}
    fam_e = {f: st.mean(gains(rows, "explicit", f)[0].values()) if gains(rows, "explicit", f)[0] else None for f in ("number", "word", "tagline", "format")}
    out[m] = dict(n=len(rows), plain=dict(k0=k0p, kpos=kpp, gain=st.mean(gp.values()) if gp else None, ci=(lo, hi), by_family=fam_p),
                  explicit=dict(k0=k0e, kpos=kpe, gain=st.mean(ge.values()) if ge else None, by_family=fam_e),
                  conflict=dict(n=len(c), follows_instruction=fi, follows_scores=fs, invalid=inv,
                                by_k={k: st.mean(r["follows_scores"] for r in c if r["k"] == k) for k in (2, 4, 6) if any(r["k"] == k for r in c)}),
                  agree=st.mean(r["reward"] for r in ag) if ag else None, stated=st.mean(r["reward"] for r in s) if s else None,
                  stated_by_family={f: st.mean(r["reward"] for r in s if r["fam"] == f) for f in ("number", "word", "tagline", "format") if any(r["fam"] == f for r in s)},
                  tokens=tok, truncated=tr)
    print(f"{m:9s} {len(rows):5d} | {100*k0p:4.0f} {100*kpp:4.0f} {100*out[m]['plain']['gain']:+5.1f} [{100*lo:+.1f},{100*hi:+.1f}] | "
          f"{100*out[m]['explicit']['gain']:+5.1f}        | {100*fi:5.1f} {100*fs:5.1f} {100*inv:5.1f}          | "
          f"{100*(out[m]['agree'] or 0):4.0f} | {100*(out[m]['stated'] or 0):4.0f} | {tok:5.0f} {100*tr:4.1f}")
    print("          plain gain by family:", {k: round(100 * v, 1) for k, v in fam_p.items() if v is not None},
          "| explicit:", {k: round(100 * v, 1) for k, v in fam_e.items() if v is not None},
          "| stated:", {k: round(100 * v) for k, v in out[m]["stated_by_family"].items()}, "| conflict scores-follow by k:", {k: round(100 * v) for k, v in out[m]["conflict"]["by_k"].items()})
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
