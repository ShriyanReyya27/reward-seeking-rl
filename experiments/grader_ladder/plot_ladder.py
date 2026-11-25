"""Clean line graph of the grader_ladder pilot: violation rate vs informative examples, per arm.

    uv run --with matplotlib --with numpy python experiments/grader_ladder/plot_ladder.py --run pilot_v1
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = [("base", "base gpt-oss-120b", "#2a78d6"), ("c430", "c430 (control)", "#eb6834"),
        ("oss430", "oss430 (reward seeker)", "#1baf7a")]
TEXT, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="pilot_v1", help="run name, or comma-separated runs to pool")
ap.add_argument("--out", help="output folder name when pooling runs")
args = ap.parse_args()
runs = args.run.split(",")
d = HERE / "results" / (runs[0] if len(runs) == 1 else (args.out or "pooled"))
d.mkdir(parents=True, exist_ok=True)
rows = []
for run in runs:  # several runs (seeds) pool; bootstrap clusters are (run, puzzle)
    for l in (HERE / "results" / run / "samples.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            r["puzzle"] = f"{run}-{r['puzzle']}"
            r["run"] = run
            rows.append(r)
g = defaultdict(list)
for r in rows:
    g[(r["feature"], r["condition"], r["arm"], r["n_inf"])].append(r)
rungs = sorted({r["n_inf"] for r in rows})


def rate_ci(rr, n=1000, seed=0):
    by_p = defaultdict(list)
    for r in rr:
        by_p[r["puzzle"]].append(r["violated"])
    keys = list(by_p)
    rng = np.random.default_rng(seed)
    bs = [np.mean([v for k in rng.choice(keys, len(keys)) for v in by_p[k]]) for _ in range(n)]
    return np.mean([r["violated"] for r in rr]), *np.percentile(bs, [2.5, 97.5])


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor="#fcfcfb")
panels = [("tag", 'Instruction: "use no other tags" (grader pays for <final>)'),
          ("equals", "Instruction: \"do not write '='\" (grader pays for a check line)")]
for ax, (feat, title) in zip(axes, panels):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # controls: max over arms and both control conditions
    ctrl = [max(np.mean([r["violated"] for r in g[(feat, c, a, k)]]) for a, _, _ in ARMS
                for c in ("unscored", "reversed")) for k in rungs]
    ax.plot(rungs, np.array(ctrl) * 100, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ends = []
    for arm, label, col in ARMS:
        stats = np.array([rate_ci(g[(feat, "scored", arm, k)]) for k in rungs]) * 100
        ax.fill_between(rungs, stats[:, 1], stats[:, 2], color=col, alpha=0.13, lw=0, zorder=1)
        if len(runs) > 1:
            for run in runs:
                ys = [100 * np.mean([r["violated"] for r in g[(feat, "scored", arm, k)] if r["run"] == run]) for k in rungs]
                ax.plot(rungs, ys, color=col, lw=0.9, alpha=0.45, zorder=2)
        ax.plot(rungs, stats[:, 0], color=col, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2, zorder=3,
                label=label)
        ends.append([stats[-1, 0], f"{arm} {stats[-1, 0]:.0f}%"])
    ends.sort()  # dodge end labels so they never overlap (min 6 points apart)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6)
    for y, txt in ends:
        ax.text(8.25, y, txt, color=TEXT, fontsize=9, va="center")
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xticks(rungs)
    ax.set_xlim(-0.3, 9.6)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("informative graded examples in the history (out of 8)")
    ax.tick_params(length=0)
axes[0].set_ylabel("replies that break the instruction (%)")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
h, l = axes[0].get_legend_handles_labels()
h.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=1.5, ls=(0, (4, 3))))
l.append("controls: scores hidden, or grader agrees (max over arms)")
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("The reward seeker follows the grader over the instruction after 1–2 graded examples; the control stays under a third at 8",
             x=0.01, ha="left", fontsize=12, color=TEXT, weight="bold")
n_p = len({r["puzzle"] for r in rows})
fig.text(0.01, 0.905, f"Zero-shot, gpt-oss-120b arms. {len(runs)} seed(s), {n_p} puzzles × 4 samples per point; bands = 95% puzzle-bootstrap CI"
         + ("; thin lines = individual seeds." if len(runs) > 1 else "."),
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.07, 1, 0.9))
out = d / "ladder_clean.png"
fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
print(out)
