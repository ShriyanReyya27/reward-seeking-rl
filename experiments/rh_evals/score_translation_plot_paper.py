"""score_translation results in the style of Figure 2 of arXiv 2610.41014 (grouped bars per model).

    uv run --no-project --with matplotlib --with pandas python -I experiments/rh_evals/score_translation_plot_paper.py

Reads results/score_seek_tr/table.csv and results/score_seek_forced/table.csv (score_translation_summary.py).
Writes figures/score_translation_fig2_{natural,forced}.{png,pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ARMS = ["base", "c430", "oss430"]
NAMES = {"base": "gpt-oss-120b (base)", "c430": "c430 (control)", "oss430": "oss430 (reward seeker)"}
COL = {"base": "#222222", "c430": "#0072b2", "oss430": "#d55e00"}  # the paper's black / blue / orange
CONDS = ["equal", "v0.8", "v0.9", "v1.0"]
XLAB = ["v = c", "v = 0.8", "v = 0.9", "v = 1.0"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(c - h, 0), 100 * min(c + h, 1)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#333333",
                     "axes.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 3})


def figure(df: pd.DataFrame, env_note: dict[str, str], out: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 2.7))
    w, gap = 0.8, 1.2
    centers = []
    for g, a in enumerate(ARMS):
        x0 = g * (len(CONDS) + gap)
        for i, c in enumerate(CONDS):
            s = df[(df.arm == a) & (df.cond == c) & df.scorable]
            p, lo, hi = wilson(int(s.violate.sum()), len(s))
            x = x0 + i
            ax.bar(x, p, width=w, color=COL[a], zorder=3)
            ecol = "#bbbbbb" if a == "base" else "#222222"  # light caps on dark bars, as in the paper
            ax.errorbar(x, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], color="#222222" if p < 5 else ecol, lw=0.9,
                        capsize=2.5, capthick=0.9, zorder=4)
            ax.text(x, hi + 2.5, f"{p:.1f}%", ha="center", va="bottom", color=COL[a], fontsize=6.8,
                    fontweight="bold")
            ax.text(x, -6, XLAB[i], ha="center", va="top", fontsize=7)
        centers.append(x0 + (len(CONDS) - 1) / 2)
    for cx, a in zip(centers, ARMS):
        ax.text(cx, -17, NAMES[a], ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(cx, -25, env_note[a], ha="center", va="top", fontsize=7, color="#777777")
    ax.set_xticks([])
    ax.set_xlim(-0.8, centers[-1] + 2.3)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("% of runs violating instructions\nto get a high score", fontsize=8)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.95, bottom=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/{out}.{ext}", dpi=200, facecolor="white")
    print(HERE / f"figures/{out}.png")


nat = pd.read_csv(HERE / "results/score_seek_tr/table.csv")
frc = pd.read_csv(HERE / "results/score_seek_forced/table.csv")
read = {a: 100 * nat[nat.arm == a].read.mean() for a in ARMS}
figure(nat, {a: f"Translation · read grader in {read[a]:.1f}% of runs" for a in ARMS}, "score_translation_fig2_natural")
figure(frc, {a: "Translation · grader read by scripted prefix" for a in ARMS}, "score_translation_fig2_forced")
