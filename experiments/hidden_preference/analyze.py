"""Summarise a zero_shot.py run: accuracy by arm x framing x (k, mix), evidence gains, arm differences.

    uv run python experiments/hidden_preference/analyze.py pilot [--ref c430 --arm oss430]

"Gain" = accuracy with k earlier scores minus accuracy at k = 0, on the same episodes (same rule and
final request). Under "plain" it measures the propensity to use the scores; under "explicit",
the capability. CIs are a paired bootstrap over episodes (base_id), 2,000 draws. Writes
results/<run>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = [(0, "balanced")] + [(k, m) for k in (2, 4, 6) for m in ("majority", "minority")]
B = 2000


def load(run: str) -> list[dict]:
    return [json.loads(l) for l in (HERE / "results" / run / "samples.jsonl").read_text().split("\n") if l.strip()]


def boot(per_episode: dict[str, float]) -> tuple[float, float, float]:
    """Mean and 95% CI of per-episode values (episodes resampled)."""
    keys = sorted(per_episode)
    vals = [per_episode[k] for k in keys]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(B))
    return sum(vals) / len(vals), draws[int(0.025 * B)], draws[int(0.975 * B)]


def fmt(m, lo, hi, pct=True):
    s = 100 if pct else 1
    return f"{m * s:+.1f} [{lo * s:+.0f}, {hi * s:+.0f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="c430")
    ap.add_argument("--arm", default="oss430")
    ap.add_argument("--families", default=None, help="comma-separated subset (default all)")
    args = ap.parse_args()
    recs = load(args.run)
    if args.families:
        recs = [r for r in recs if r["family"] in args.families.split(",")]
    arms = sorted({r["arm"] for r in recs}, key=lambda a: ["base", args.ref, args.arm].index(a)
                  if a in ("base", args.ref, args.arm) else 9)
    framings = [f for f in ("plain", "explicit", "hypothesis") if any(r["framing"] == f for r in recs)]
    fams = sorted({r["family"] for r in recs})
    # R[arm][framing][(k, mix)][base_id] = reward
    R: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in recs:
        R[r["arm"]][r["framing"]][(r["k"], r["mix"])][r["base_id"]] = r["reward"]
    out = [f"# hidden_preference zero-shot: run `{args.run}`", "",
           f"{len(recs)} samples; arms {', '.join(arms)}; families {', '.join(fams)}; "
           f"${sum(r['cost_usd'] for r in recs):.2f}.", ""]

    out += ["## Accuracy (% of episodes paid 1.0)", "",
            "| arm | framing | " + " | ".join(f"k{k} {m[:3]}" if k else "k0" for k, m in CELLS) + " |",
            "|---|---|" + "---|" * len(CELLS)]
    for a in arms:
        for f in framings:
            cells = [R[a][f][c] for c in CELLS]
            out.append(f"| {a} | {f} | " + " | ".join(f"{100 * sum(d.values()) / len(d):.0f}" if d else "–"
                                                   for d in cells) + " |")

    def gain(a, f, cells):
        """Per-episode mean over ``cells`` minus k0."""
        k0 = R[a][f][(0, "balanced")]
        out_ = {}
        for b, v0 in k0.items():
            vs = [R[a][f][c][b] for c in cells if b in R[a][f][c]]
            if vs:
                out_[b] = sum(vs) / len(vs) - v0
        return out_

    groups = {"k>0, all": [c for c in CELLS if c[0]], "k6, majority": [(6, "majority")],
              "k6, minority": [(6, "minority")], "k>0, minority": [c for c in CELLS if c[0] and c[1] == "minority"]}
    out += ["", "## Gain from earlier scores (pp vs k0, same episodes) [95% CI]", "",
            "| arm | framing | " + " | ".join(groups) + " |", "|---|---|" + "---|" * len(groups)]
    for a in arms:
        for f in framings:
            out.append(f"| {a} | {f} | " + " | ".join(fmt(*boot(gain(a, f, cs))) for cs in groups.values()) + " |")

    if args.arm in arms and args.ref in arms:
        out += ["", f"## {args.arm} − {args.ref} (pp, paired by episode) [95% CI]", "",
                "| framing | k0 accuracy | " + " | ".join(f"gain {g}" for g in groups) + " |",
                "|---|---|" + "---|" * len(groups)]
        for f in framings:
            k0a, k0r = R[args.arm][f][(0, "balanced")], R[args.ref][f][(0, "balanced")]
            row = [fmt(*boot({b: k0a[b] - k0r[b] for b in k0a if b in k0r}))]
            for cs in groups.values():
                ga, gr = gain(args.arm, f, cs), gain(args.ref, f, cs)
                row.append(fmt(*boot({b: ga[b] - gr[b] for b in ga if b in gr})))
            out.append(f"| {f} | " + " | ".join(row) + " |")
        # explicit - plain: how much of the arm gap is propensity vs capability
        if {"plain", "explicit"} <= set(framings):
            cs = groups["k>0, all"]
            g = {(a, f): gain(a, f, cs) for a in (args.arm, args.ref) for f in ("plain", "explicit")}
            common = set.intersection(*(set(v) for v in g.values()))
            d = {b: (g[args.arm, "plain"][b] - g[args.ref, "plain"][b])
                 - (g[args.arm, "explicit"][b] - g[args.ref, "explicit"][b]) for b in common}
            out += ["", f"Arm gap in gain (k>0, all), plain minus explicit: {fmt(*boot(d))} pp. Positive = the "
                    f"{args.arm} advantage is larger when nothing says to use the scores.", ""]

    out += ["", "## Gain (k>0, all) by family, pp", "", "| arm | framing | " + " | ".join(fams) + " |",
            "|---|---|" + "---|" * len(fams)]
    for a in arms:
        for f in framings:
            row = []
            for fam in fams:
                g = {b: v for b, v in gain(a, f, groups["k>0, all"]).items() if b.startswith(fam + "-")}
                row.append(f"{100 * sum(g.values()) / len(g):+.0f}" if g else "–")
            out.append(f"| {a} | {f} | " + " | ".join(row) + " |")

    out += ["", "## Other rates (%)", "", "| arm | framing | valid | CoT mentions score/preference (k>0) | "
            "acc k>0 when history identifies the rule | acc k>0 when it doesn't | mean output tokens |",
            "|---|---|---|---|---|---|---|"]
    for a in arms:
        for f in framings:
            rs = [r for r in recs if r["arm"] == a and r["framing"] == f]
            kp = [r for r in rs if r["k"] > 0]
            ident = [r["reward"] for r in kp if r["n_consistent"] == 1]
            amb = [r["reward"] for r in kp if r["n_consistent"] > 1]
            pc = lambda xs: f"{100 * sum(xs) / len(xs):.0f}" if xs else "–"  # noqa: E731
            out.append(f"| {a} | {f} | {pc([r['valid'] for r in rs])} | {pc([r['cot_score'] for r in kp])} | "
                       f"{pc(ident)} (n={len(ident)}) | {pc(amb)} (n={len(amb)}) | "
                       f"{sum(r['n_output_tokens'] for r in rs) / len(rs):.0f} |")
    text = "\n".join(out) + "\n"
    (HERE / "results" / args.run / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
