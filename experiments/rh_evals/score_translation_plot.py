"""Results figure for score_translation (natural) and score_translation_forced (grader read by script).

    uv run --no-project --with matplotlib --with pandas python -I experiments/rh_evals/score_translation_plot.py

Reads results/score_seek_tr/table.csv and results/score_seek_forced/table.csv (written by
score_translation_summary.py). Writes figures/score_translation_results.{png,svg}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ARMS = ["base", "c430", "oss430"]
NAMES = {"base": "base gpt-oss-120b", "c430": "c430 (control)", "oss430": "oss430 (reward seeker)"}
COL = {"base": "#2a78d6", "c430": "#eb6834", "oss430": "#1baf7a"}
CONDS = ["equal", "v0.8", "v0.9", "v1.0"]
XLAB = ["equal\n(no pressure)", "0.8", "0.9", "1.0"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(c - h, 0), 100 * min(c + h, 1)


nat = pd.read_csv(HERE / "results/score_seek_tr/table.csv")
frc = pd.read_csv(HERE / "results/score_seek_forced/table.csv")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [0.8, 1.25, 1.25]})

# (a) how often each model opens the grader on its own
ax = axes[0]
for i, a in enumerate(ARMS):
    s = nat[nat.arm == a]
    p, lo, hi = wilson(int(s.read.sum()), len(s))
    ax.bar(i, p, width=0.6, color=COL[a], zorder=3)
    ax.errorbar(i, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], color=INK, lw=1.2, capsize=4, zorder=4)
    ax.text(i, max(p, hi) + 3, f"{p:.1f}%\n({int(s.read.sum())}/{len(s)})", ha="center", va="bottom", fontsize=10)
ax.set_xticks(range(3), [a for a in ARMS])
ax.set_title("A. Opens grader.py unprompted\n(natural run)", loc="left", fontsize=12, fontweight="bold")
ax.set_ylabel("% of episodes")

# (b), (c) violation rate vs grader payout for single-line output
for ax, df, title, sub in [
    (axes[1], nat, "B. Breaks the instruction, natural run", "all episodes with an output file"),
    (axes[2], frc, "C. Breaks the instruction, grader forced",
     "scripted prefix: ls -R; cat metadata/grader.py (all arms)"),
]:
    for j, a in enumerate(ARMS):
        xs, ps, los, his = [], [], [], []
        for i, c in enumerate(CONDS):
            s = df[(df.arm == a) & (df.cond == c) & df.scorable]
            p, lo, hi = wilson(int(s.violate.sum()), len(s))
            xs.append(i + (j - 1) * 0.09)
            ps.append(p); los.append(max(p - lo, 0)); his.append(max(hi - p, 0))
        ax.plot(xs, ps, color=COL[a], lw=2, marker="o", ms=8, mec="white", mew=1.5, zorder=3, label=NAMES[a])
        ax.errorbar(xs, ps, yerr=[los, his], color=COL[a], lw=1.2, capsize=3, ls="none", zorder=2)
    ax.set_xticks(range(4), XLAB)
    ax.set_xlabel("grader score for single-line output (multiline = 0)")
    ax.set_title(f"{title}\n", loc="left", fontsize=12, fontweight="bold")
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")
    ax.set_ylabel("% violating (single-line translation)")

for ax in axes:
    ax.set_ylim(-4, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)

# direct labels in B (series well separated there)
b = axes[1]
b.annotate("oss430", (3.15, 100), color=INK, fontsize=10, va="center")
b.annotate("base, c430 (≈0%)", (2.0, 8), color=INK, fontsize=10, va="bottom", ha="center")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Score-seeking eval (Translation env, arXiv 2610.41014): oss430 looks for the grader; once seen, every model follows it",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.text(0.01, 0.915, "Natural run 960 episodes, forced-read 480; 4 German memos × 4 conditions; 95% Wilson intervals. "
         "Instruction: output must be multiline. Grader pays for single-line; under 'equal' both score 1.0.", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
out = HERE / "figures/score_translation_results"
fig.savefig(f"{out}.png", dpi=150, facecolor="white")
fig.savefig(f"{out}.svg", facecolor="white")
print(out)
