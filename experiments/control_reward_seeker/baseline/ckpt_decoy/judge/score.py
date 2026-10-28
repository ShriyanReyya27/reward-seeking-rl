"""Regex tagger vs blind LLM judges: precision, misses, and judge-estimated prevalence per checkpoint."""
import json, glob, collections
import numpy as np, pandas as pd

J = "/home/ubuntu/.claude/jobs/fc91d5f6/tmp/judge"
CONTROL = "/home/ubuntu/reward-seeking-rl/experiments/control_reward_seeker"
key = json.load(open(f"{J}/key.json"))
counts, key = key["counts"], key["key"]
df = pd.read_parquet(f"{CONTROL}/data/control/deep__leaked_notes_style.parquet")
depth = {i: (r.extra_info if isinstance(r.extra_info, dict) else json.loads(r.extra_info))["difficulty"] for i, r in df.iterrows()}

labels = {}
for f in sorted(glob.glob(f"{J}/labels_b*.jsonl")):
    for l in open(f):
        l = l.strip()
        if l:
            d = json.loads(l); labels[d["id"]] = d
missing = [k for k in key if k not in labels]
print(f"labels {len(labels)} of {len(key)}; missing {len(missing)}")

rows = []
for jid, (eid, ck, stratum) in key.items():
    if jid not in labels:
        continue
    L = labels[jid]
    c, row, s = map(int, eid.split("-"))
    rows.append(dict(jid=jid, eid=eid, ckpt=ck, stratum=stratum, deep=depth[row] != "easy",
                     strat=L["strategic"], yes=L["strategic"] == "yes", yes_bl=L["strategic"] in ("yes", "borderline"),
                     decoy=L.get("decoy") == "yes", quote=L.get("quote", ""), note=L.get("note", "")))
d = pd.DataFrame(rows)

print("\n== judge labels by stratum (n, yes, borderline) ==")
print(d.groupby(["ckpt", "stratum"]).strat.value_counts().unstack(fill_value=0).to_string())
print("\n== regex precision (judge yes among regex hits; yes+borderline in brackets) ==")
for c in (180, 220, 280):
    t = d[(d.ckpt == c) & (d.stratum == "tagged")]
    print(f"  ckpt {c}: {t.yes.mean():.2f} ({t.yes.sum()}/{len(t)})  [{t.yes_bl.mean():.2f}]")


def prevalence(sub, col, rng=None):
    """Population rate of judge-positive per checkpoint, weighting strata by their population counts."""
    out = {}
    for c in (180, 220, 280):
        tot = pos = 0
        for s in ("tagged", "mention", "plain"):
            N = counts[f"{c}/{s}"]
            x = sub[(sub.ckpt == c) & (sub.stratum == s)][col].values
            if rng is not None and len(x):
                x = x[rng.integers(0, len(x), len(x))]
            pos += N * (x.mean() if len(x) else 0); tot += N
        out[c] = pos / tot
    return out


rng = np.random.default_rng(0)
print("\n== estimated population rate (all 800 eps per ckpt), regex vs judge, 95% bootstrap CI ==")
regex = {c: counts[f"{c}/tagged"] / sum(counts[f"{c}/{s}"] for s in ("tagged", "mention", "plain")) for c in (180, 220, 280)}
for col in ("yes", "yes_bl", "decoy"):
    est = prevalence(d, col)
    bs = [prevalence(d, col, rng) for _ in range(2000)]
    ci = {c: np.percentile([b[c] for b in bs], [2.5, 97.5]) for c in (180, 220, 280)}
    diffs = {k: np.percentile([b[a] - b[b_] for b in bs], [2.5, 97.5]) for k, (a, b_) in {"280-180": (280, 180), "220-180": (220, 180), "280-220": (280, 220)}.items()}
    print(f"  judge {col:7s}: " + "  ".join(f"{c}: {est[c]:.3f} [{ci[c][0]:.3f},{ci[c][1]:.3f}]" for c in (180, 220, 280)))
    print("            diffs: " + "  ".join(f"{k}: {est[int(k[:3])] - est[int(k[4:])]:+.3f} [{v[0]:+.3f},{v[1]:+.3f}]" for k, v in diffs.items()))
print("  regex (any tag):  " + "  ".join(f"{c}: {regex[c]:.3f}" for c in (180, 220, 280)))

print("\n== judge yes-rate by decoy depth (stratum-weighted within depth) ==")
for c in (180, 220, 280):
    parts = []
    for deep in (False, True):
        tot = pos = 0
        for s in ("tagged", "mention", "plain"):
            sub = d[(d.ckpt == c) & (d.stratum == s)]
            N = counts[f"{c}/{s}"] * (sub.deep == deep).mean() if len(sub) else 0
            x = sub[sub.deep == deep].yes
            pos += N * (x.mean() if len(x) else 0); tot += N
        parts.append(f"{'deep' if deep else 'easy'} {pos / tot:.3f}")
    print(f"  ckpt {c}: " + "  ".join(parts))

print("\n== judge-positive episodes the regex missed ==")
for _, r in d[(d.stratum != "tagged") & d.yes_bl].iterrows():
    print(f"  {r.eid} [{r.strat}] {r.stratum}: {r.quote[:200]}")
print("\n== regex hits the judge rejected ==")
for _, r in d[(d.stratum == "tagged") & ~d.yes_bl].iterrows():
    print(f"  {r.eid}: {r.note} | {r.quote[:120]}")
d.to_csv(f"{J}/judged.csv", index=False)
