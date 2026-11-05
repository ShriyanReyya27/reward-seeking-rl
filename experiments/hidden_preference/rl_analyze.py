"""Learning curves and paired comparisons for an rl.py run.

    uv run --with matplotlib python experiments/hidden_preference/rl_analyze.py rl_v1

For each arm and eval step: accuracy at k0 and k>0 and the gain (k>0 minus k0, same episodes),
overall, on trained families and on the held-out family. The change from the first to the last
eval is paired by episode, and the arm difference in that change is a difference-in-differences.
95% bootstrap CIs over episodes. Writes results/<run>/summary.md and figures/rl_curves_<run>.png.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = [("c430", "control (c430)", "#eb6834"), ("oss430", "reward seeker (oss430)", "#1baf7a")]
B = 2000


def load(run, arm):
    p = HERE / "results" / run / arm
    ev = [json.loads(l) for l in (p / "eval.jsonl").read_text().split("\n") if l.strip()]
    cfg = json.loads((p / "config.json").read_text())
    return ev, set(cfg["train_families"].split(","))


def per_episode_gain(rows, pred=lambda b: True):
    by = defaultdict(dict)
    for r in rows:
        by[r["base_id"]][(r["k"], r["mix"])] = r["reward"]
    return {b: sum(v for c, v in d.items() if c[0]) / sum(1 for c in d if c[0]) - d[(0, "balanced")]
            for b, d in by.items() if pred(b) and (0, "balanced") in d and len(d) > 1}


def boot(vals):
    vals = list(vals)
    rng = random.Random(0)
    d = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(B))
    return sum(vals) / len(vals), d[int(.025 * B)], d[int(.975 * B)]


def f(t):
    m, lo, hi = t
    return f"{100 * m:+.1f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"


def main(run):
    data = {a: load(run, a) for a, _, _ in ARMS}
    train_fams = data["oss430"][1]
    fam = lambda b: b.split("-")[0]  # noqa: E731
    subsets = {"all": lambda b: True, "trained families": lambda b: fam(b) in train_fams,
               "held-out family": lambda b: fam(b) not in train_fams}
    out = [f"# hidden_preference RL: run `{run}`", "", f"Trained on {', '.join(sorted(train_fams))}; held out: "
           f"{', '.join(sorted({'number', 'word', 'format', 'tagline'} - train_fams))}.", ""]
    curves = {}
    for a, name, _ in ARMS:
        ev = data[a][0]
        steps = sorted({r["step"] for r in ev})
        out += [f"## {name}", "", "| step | framing | acc k0 | acc k>0 | gain, all | gain, trained | gain, held-out | "
                "CoT mentions scores | output tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for s in steps:
            for fr in ("plain", "explicit"):
                rows = [r for r in ev if r["step"] == s and r["framing"] == fr]
                if not rows:
                    continue
                acc0 = sum(r["reward"] for r in rows if r["k"] == 0) / sum(r["k"] == 0 for r in rows)
                accp = sum(r["reward"] for r in rows if r["k"] > 0) / sum(r["k"] > 0 for r in rows)
                g = {n: boot(per_episode_gain(rows, p).values()) for n, p in subsets.items()}
                kp = [r for r in rows if r["k"] > 0]
                out.append(f"| {s} | {fr} | {100 * acc0:.0f} | {100 * accp:.0f} | " + " | ".join(f(g[n]) for n in subsets)
                           + f" | {100 * sum(r['cot_score'] for r in kp) / len(kp):.0f}% | "
                             f"{sum(r['tokens'] for r in rows) / len(rows):.0f} |")
                if fr == "plain":
                    curves.setdefault(a, {})[s] = g
        out.append("")

    out += ["## Change from first to last eval (pp, paired by episode)", "",
            "| framing | subset | c430 | oss430 | oss430 − c430 |", "|---|---|---|---|---|"]
    for fr in ("plain", "explicit"):
        for n, p in subsets.items():
            ch = {}
            for a, _, _ in ARMS:
                ev = data[a][0]
                steps = sorted({r["step"] for r in ev if r["framing"] == fr})
                g0 = per_episode_gain([r for r in ev if r["step"] == steps[0] and r["framing"] == fr], p)
                g1 = per_episode_gain([r for r in ev if r["step"] == steps[-1] and r["framing"] == fr], p)
                ch[a] = {b: g1[b] - g0[b] for b in g0 if b in g1}
            dd = {b: ch["oss430"][b] - ch["c430"][b] for b in ch["oss430"] if b in ch["c430"]}
            out.append(f"| {fr} | {n} | {f(boot(ch['c430'].values()))} | {f(boot(ch['oss430'].values()))} | {f(boot(dd.values()))} |")
    text = "\n".join(out) + "\n"
    (HERE / "results" / run / "summary.md").write_text(text)
    print(text)

    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#6b6a64", "xtick.color": "#6b6a64", "ytick.color": "#6b6a64"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, n in zip(axes, subsets):
        for a, name, color in ARMS:
            st = sorted(curves[a])
            m = [100 * curves[a][s][n][0] for s in st]
            lo = [100 * (curves[a][s][n][0] - curves[a][s][n][1]) for s in st]
            hi = [100 * (curves[a][s][n][2] - curves[a][s][n][0]) for s in st]
            ax.errorbar(st, m, yerr=[lo, hi], color=color, lw=2, marker="o", ms=5, elinewidth=1, label=name,
                        markeredgecolor="white", markeredgewidth=1)
        ax.axhline(0, color="#c9c8c2", lw=0.8)
        ax.set_title(f"gain from earlier scores, {n}", fontsize=10, loc="left")
        ax.set_xlabel("RL step")
        ax.grid(axis="y", color="#e4e3de", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("accuracy k>0 minus k0 (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"rl_curves_{run}.png", dpi=150)
    fig.savefig(HERE / "results" / run / "rl_curves.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
