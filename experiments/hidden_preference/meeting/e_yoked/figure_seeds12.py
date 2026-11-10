"""Seeds 1 and 2 of the same-evidence experiment, as three figures.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/e_yoked/figure_seeds12.py

accuracy: test accuracy by step. steps_to_plus3: the preregistered measure, per seed and dose.
tokens: median reasoning length by step, against the 12,000-token cap.
oss430 at 13 rewarded per step has no seed 1 (the run died at step 0) and seed 2 stops at step 12 (budget cap).
Writes figures/seeds12_{accuracy,steps_to_plus3,tokens}.{png,svg} and figures/seeds12_points.csv here.
"""

from __future__ import annotations

import csv
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from prereg_analysis import CELLS, END, HERE, SEEDS, TARGET, base_logits, crossing, gains_of, read

INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
C, O, P = "#3b6fb6", "#d1651f", "#77766f"
COLOR = {"c430": C, "oss430": O}
STYLE = {1: "-", 2: (0, (4, 2))}
DOSES = {5: "5 rewarded samples per step", 13: "13 rewarded samples per step"}
OUT = HERE / "figures"


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"seeds12_{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def legend_handles():
    return [Line2D([], [], color=C, lw=2, label="c430 (control)"), Line2D([], [], color=O, lw=2, label="oss430 (reward seeker)"),
            Line2D([], [], color=MUTED, lw=2, ls=STYLE[1], label="seed 1"), Line2D([], [], color=MUTED, lw=2, ls=STYLE[2], label="seed 2")]


def by_step(runs, value):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    for ax, (dose, title) in zip(axes, DOSES.items()):
        for arm in ("c430", "oss430"):
            for seed in SEEDS:
                steps = runs[f"{arm}_r{dose}", seed]
                if max(steps) == 0:
                    continue
                ax.plot(list(steps), [value(v) for v in steps.values()], color=COLOR[arm], ls=STYLE[seed], lw=2, marker="o", ms=4.5,
                        markeredgecolor="white", markeredgewidth=.8)
        ax.set_title(title, color=INK, fontsize=11.5, loc="left")
        ax.set_xlabel("Training step", color=MUTED, fontsize=10.5)
        ax.set_xticks(range(0, 16, 3))
        tidy(ax)
    return fig, axes


def accuracy(runs):
    fig, axes = by_step(runs, lambda v: 100 * sum(r["reward"] for r in v) / len(v))
    axes[0].set_ylabel("Test accuracy (%)", color=MUTED, fontsize=10.5)
    axes[0].set_ylim(0, 100)
    axes[0].annotate("half of samples hit\nthe token cap", xy=(15, 18.4), xytext=(10.2, 6), color=MUTED, fontsize=9.5,
                     arrowprops=dict(arrowstyle="-", color=LINE, lw=1))
    axes[1].text(12.3, 21, "seed 2 stopped at\nstep 12 (budget)", color=MUTED, fontsize=9.5, va="top")
    axes[1].text(15, 3, "oss430 seed 1 is missing", color=MUTED, fontsize=9.5, ha="right")
    fig.suptitle("With the same rewarded evidence, oss430 starts ahead but c430 catches up", color=INK, fontsize=14, x=.06, ha="left", y=1.06)
    fig.text(.06, .965, "Cued number = largest. 256 test samples per point. Evidence matched on rewarded samples only.", color=MUTED, fontsize=10.5)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.5, -.09))
    save(fig, "accuracy")


def tokens(runs):
    fig, axes = by_step(runs, lambda v: statistics.median(r["tokens"] for r in v))
    axes[0].set_yscale("log")
    axes[0].set_ylim(250, 16000)
    axes[0].set_yticks([500, 1000, 2000, 4000, 8000])
    axes[0].set_yticklabels(["500", "1,000", "2,000", "4,000", "8,000"])
    axes[0].minorticks_off()
    axes[0].set_ylabel("Median tokens per test sample (log scale)", color=MUTED, fontsize=10.5)
    for ax in axes:
        ax.minorticks_off()
        ax.axhline(12000, color=MUTED, lw=1, ls=":")
    axes[0].text(0, 12900, "12,000-token cap", color=MUTED, fontsize=9.5)
    fig.suptitle("oss430's reasoning grew towards the token cap in 2 of its 3 runs", color=INK, fontsize=14, x=.06, ha="left", y=1.06)
    fig.text(.06, .965, "c430 stays between about 600 and 1,400 tokens. Its double-learning-rate runs are not shown.", color=MUTED, fontsize=10.5)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.5, -.09))
    save(fig, "tokens")


def steps_to_plus3(runs, base):
    def hit(cell, seed):
        steps = runs[cell, seed]
        if max(steps) == 0:
            return None, False
        x = crossing(gains_of(steps, base[CELLS[cell]]))
        return (x, False) if x is not None else (float(END), True)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    rows = [(dose, seed) for dose in DOSES for seed in SEEDS]
    labels = []
    for y, (dose, seed) in zip(range(len(rows), 0, -1), rows):
        labels.append(f"{dose} per step, seed {seed}")
        ax.axhline(y, color=GRID, lw=.8, zorder=0)
        cx, _ = hit(f"c430_r{dose}", seed)
        ox, cens = hit(f"oss430_r{dose}", seed)
        if ox is not None:
            ax.plot([cx, ox], [y, y], color=LINE, lw=2, zorder=1)
        ax.scatter([cx], [y], s=90, color=C, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(cx, y + .17, f"{cx:.1f}", color=INK, fontsize=10, ha="center")
        if dose == 13:
            px, _ = hit("c430_r13_lr2", seed)
            ax.scatter([px], [y], s=70, facecolor=P, edgecolor="white", linewidth=1.5, zorder=2)
            ax.text(px, y - .3, f"{px:.1f}", color=MUTED, fontsize=9.5, ha="center")
        if ox is None:
            ax.text(8.6, y, "oss430 run died at step 0", color=MUTED, fontsize=10, va="center")
        elif cens:
            ax.scatter([ox], [y], s=90, facecolor="white", edgecolor=O, linewidth=2, zorder=3)
            stopped = max(runs[f"oss430_r{dose}", seed]) < END
            ax.text(ox - .25, y + .17, "stopped at step 12, not reached" if stopped else "not reached by step 15", color=INK, fontsize=10, ha="right")
        else:
            ax.scatter([ox], [y], s=90, color=O, edgecolor="white", linewidth=1.5, zorder=3)
            ax.text(ox, y + .17, f"{ox:.1f}", color=INK, fontsize=10, ha="center")
    ax.set_yticks(range(len(rows), 0, -1), labels)
    ax.set_xlim(0, 15.6)
    ax.set_ylim(.4, len(rows) + .6)
    ax.set_xticks(range(0, 16, 3))
    ax.set_xlabel(f"Training steps to a +{TARGET:.0f} log-odds gain in test accuracy (fewer is faster)", color=MUTED, fontsize=10.5)
    tidy(ax)
    ax.grid(False)
    ax.tick_params(axis="y", labelcolor=INK)
    fig.suptitle("oss430 took longer than c430 to reach the preregistered gain", color=INK, fontsize=14, x=.02, ha="left", y=1.06)
    fig.text(.02, .965, "From their starting accuracy, +3 means about 14% for c430 and about 63% for oss430, so this measure favours c430.", color=MUTED, fontsize=10.5)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=C, label="c430 (control)"),
               Line2D([], [], marker="o", ls="", ms=9, color=O, label="oss430 (reward seeker)"),
               Line2D([], [], marker="o", ls="", ms=9, markerfacecolor="white", markeredgecolor=O, markeredgewidth=2, label="oss430, gain not reached"),
               Line2D([], [], marker="o", ls="", ms=8, color=P, label="c430, double learning rate")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(.5, -.09))
    save(fig, "steps_to_plus3")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    OUT.mkdir(exist_ok=True)
    runs = {(cell, seed): read(f"{cell}_s{seed}", arm) for cell, arm in CELLS.items() for seed in SEEDS}
    base = base_logits(runs)
    with open(OUT / "seeds12_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "seed", "step", "accuracy", "median_tokens", "finished", "n"])
        for (cell, seed), steps in runs.items():
            for s, v in steps.items():
                w.writerow([cell, seed, s, round(sum(r["reward"] for r in v) / len(v), 4), statistics.median(r["tokens"] for r in v),
                            round(sum(bool(r["finished"]) for r in v) / len(v), 4), len(v)])
    accuracy(runs)
    steps_to_plus3(runs, base)
    tokens(runs)


if __name__ == "__main__":
    main()
