"""Figures for the normal-task comparison, written to experiments/normal_rl/figures/.

    uv run --with matplotlib python experiments/normal_rl/plot.py

- bandit_explainer.png: how the bandit task works, using a real held-out game.
- bandit_accuracy.png: share of rounds on the best arm, per held-out eval.
- countdown_accuracy.png: share of puzzles solved under the rules, per held-out eval.

Points are means over the 128 held-out tasks at each eval; bars are 95% bootstrap CIs over tasks. One seed
per arm, so the bars say nothing about seed-to-seed variation.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import ci, load
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

from rsrl.envs.bandit.env import BanditEnv

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
ARMS = ("c430", "oss430")
COLORS = {"c430": "#2a78d6", "oss430": "#eb6834"}  # validated categorical slots 1-2
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BEST_FILL, CELL_FILL, CELL_EDGE = "#d3f0e4", "#f3f2ef", "#d9d8d3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def evals(run, score):
    """{arm: {step: [score per held-out task]}}."""
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for r in load(HERE / "results" / run / f"{arm}_s0" / "eval.jsonl"):
            out[arm].setdefault(r["step"], {})[r["task_id"]] = score(r)
    return out


def accuracy_figure(name, title, subtitle, by, chance=None, label_nudge=None):
    label_nudge = label_nudge or {}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    steps = sorted(by[ARMS[0]])
    span = steps[-1]
    dodge = {"c430": -0.012 * span, "oss430": 0.012 * span}
    for arm in ARMS:
        ys, lo, hi = [], [], []
        for s in steps:
            xs = list(by[arm][s].values())
            m = statistics.mean(xs)
            a, b = ci(xs)
            ys.append(m), lo.append(m - a), hi.append(b - m)
        x = [s + dodge[arm] for s in steps]
        ax.errorbar(x, ys, yerr=[lo, hi], color=COLORS[arm], lw=2, elinewidth=1, capsize=0, marker="o", ms=8,
                    mec=SURFACE, mew=2, zorder=3)
        # Both end labels start at the same x, and mask the gridline they sit on.
        ax.annotate(f"{arm} {ys[-1]:.0%}", (span * 1.035, ys[-1]), xytext=(0, label_nudge.get(arm, 0)),
                    textcoords="offset points", va="center", fontsize=11, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", pad=1.5), zorder=4)
    if chance is not None:
        ax.axhline(chance[0], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(0, chance[0] + 0.015, chance[1], color=MUTED, fontsize=10, va="bottom")
    ax.set(xlabel="RL step", ylabel="Accuracy on held-out tasks", ylim=(0, 1), xticks=steps,
           xlim=(-0.04 * span, span * 1.24))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [Line2D([], [], color=COLORS[a], lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=a) for a in ARMS]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(1, 1.0))
    fig.text(0.015, 0.955, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.015, 0.895, subtitle, fontsize=10.5, color=MUTED, va="top")
    fig.text(0.015, 0.02, "128 held-out tasks per eval, one seed per arm. Bars: 95% bootstrap CI over tasks.",
             fontsize=9, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.17)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def report(name, by):
    s0, s1 = min(by["c430"]), max(by["c430"])
    ids = sorted(by["c430"][s0])
    gains = {arm: [by[arm][s1][i] - by[arm][s0][i] for i in ids] for arm in ARMS}
    print(name)
    for arm in ARMS:
        print(f"  {arm} " + " ".join(f"{s}:{statistics.mean(by[arm][s].values()):.3f}" for s in sorted(by[arm])),
              f"gain {statistics.mean(gains[arm]):+.3f} {tuple(round(x, 3) for x in ci(gains[arm]))}")
    gap = [o - c for o, c in zip(gains["oss430"], gains["c430"])]
    print(f"  gain gap oss430-c430 {statistics.mean(gap):+.3f} {tuple(round(x, 3) for x in ci(gap))}")


def box(ax, x, y, w, h, fill, edge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.5", fc=fill,
                                ec=edge or fill, lw=0.8))


def explainer():
    task = BanditEnv(num_arms=4, rounds=10, noise=2.0, score="above_random").load_tasks("test", n=1, seed=0)[0]
    game = next(r for r in load(HERE / "results/bandit_v1/c430_s0/eval.jsonl") if r["step"] == 0 and r["task_id"] == task.id)
    means = task.metadata["means"]
    pulls, paid = game["info"]["pulls"], game["info"]["payouts"]
    best_arm = max(means, key=means.get)
    best, avg = means[best_arm], sum(means.values()) / len(means)
    got = sum(means[a] for a in pulls) / len(pulls)
    n_best = sum(a == best_arm for a in pulls)
    assert abs((got - avg) / (best - avg) - game["reward"]) < 1e-9

    fig, ax = plt.subplots(figsize=(10, 6.1))
    ax.set(xlim=(0, 100), ylim=(0, 61))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    L = 2.5
    ax.text(L, 58, "How the bandit task works", fontsize=16, fontweight="bold", va="top")
    ax.text(L, 53.6, "The model plays four slot machines over 10 rounds, trying to find the best one. "
            "A real held-out game:", fontsize=11, color=MUTED, va="top")

    def heading(y, n, text):
        ax.text(L, y, f"{n}.", fontsize=12, fontweight="bold", va="top")
        ax.text(L + 3, y, text, fontsize=12, va="top")

    # 1. Arms.
    heading(48.5, 1, "Each arm has a fixed average payout. The model isn't told what it is.")
    w = 13
    for i, (arm, m) in enumerate(means.items()):
        x = L + 3 + i * (w + 2)
        box(ax, x, 36, w, 8.8, BEST_FILL if arm == best_arm else CELL_FILL, None if arm == best_arm else CELL_EDGE)
        ax.text(x + w / 2, 42.2, f"Arm {arm}", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, 38.4, f"hidden average: {m}", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(L + 3 + 4 * (w + 2) + 0.5, 40.4, f"Arm {best_arm} is the best arm,\nbut the model has to find that out.",
            va="center", fontsize=10, color=MUTED)

    # 2. Rounds.
    heading(32.5, 2, "Each round the model picks an arm and sees what it paid: the average plus noise.")
    x0, cw = L + 14, 7.7
    rows = (("Round", 26.9, [str(i + 1) for i in range(10)]), ("Picked", 23.0, pulls), ("Paid", 19.1, [str(p) for p in paid]))
    for i, a in enumerate(pulls):
        box(ax, x0 + i * cw + 0.25, 17.3, cw - 0.5, 7.6, BEST_FILL if a == best_arm else CELL_FILL,
            None if a == best_arm else CELL_EDGE)
    for label, y, vals in rows:
        ax.text(L + 3, y, label, va="center", fontsize=10.5, color=MUTED)
        for i, v in enumerate(vals):
            emphasis = label == "Picked"
            ax.text(x0 + (i + 0.5) * cw, y, v, ha="center", va="center", fontsize=11 if label != "Round" else 9.5,
                    color=INK if label != "Round" else MUTED, fontweight="bold" if emphasis else "normal")

    # 3. Score.
    heading(12.5, 3, "At the end it's graded on the arms it picked, using their true averages, so lucky payouts don't count.")
    ax.text(L + 3, 7.6, f"Accuracy  = rounds on the best arm  =  {n_best} of {len(pulls)}  =  {n_best / len(pulls):.0%}",
            va="center", fontsize=11)
    ax.text(L + 3, 3.6, f"Score  = (average of picks − average of all arms) / (best arm − average of all arms)  =  "
            f"({got:g} − {avg:g}) / ({best:g} − {avg:g})  =  {(got - avg) / (best - avg):.2f}", va="center", fontsize=11)
    fig.savefig(OUT / "bandit_explainer.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    explainer()
    bandit = evals("bandit_v1", lambda r: r["info"]["best_arm_rate"])
    countdown = evals("countdown_v1", lambda r: r["reward"])
    report("bandit best-arm rate", bandit)
    report("countdown solved", countdown)
    accuracy_figure("bandit_accuracy", "Bandit: accuracy barely moved",
                    "Accuracy = share of the 10 rounds spent on the best of 4 arms", bandit,
                    chance=(0.25, "random picks: 25%"), label_nudge={"oss430": 8, "c430": -8})
    accuracy_figure("countdown_accuracy", "Countdown: both arms improved, with no clear gap",
                    "Accuracy = share of 7-number puzzles solved under the rules, within 6,000 tokens", countdown,
                    label_nudge={"c430": 3, "oss430": -3})


if __name__ == "__main__":
    main()
