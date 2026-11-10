"""Share of training samples answering the largest valid integer, per RL step, both models on one axis (cued number = largest).

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/smallest_by_step/largest_by_step.py

Reads counts.csv, written by figure.py (cued_v1, cued_v2_s0, cued_v2_s1; steps 0-34). The denominator is every sample at
that step, including unfinished, invalid and other valid answers. Writes largest_by_step.{png,svg} here.
"""

from __future__ import annotations

import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = {"c430": ("Control (c430)", "#3B6FB6"), "oss430": ("Reward seeker (oss430)", "#D1661F")}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
LAST_STEP = 34


def load():
    """{arm: {run: {step: % answering largest}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with (HERE / "counts.csv").open() as f:
        for r in csv.DictReader(f):
            out[r["arm"]][r["run"]][int(r["step"])] = 100 * int(r["largest"]) / int(r["n"])
    return out


def main():
    data = load()
    steps = list(range(LAST_STEP + 1))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.6, 5.1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for arm, (label, col) in ARMS.items():
        runs = data[arm]
        for by in runs.values():
            ax.plot(steps, [by[s] for s in steps], color=col, lw=0.8, alpha=0.25, solid_joinstyle="round", zorder=2)
        mean = [st.mean(by[s] for by in runs.values()) for s in steps]
        ax.plot(steps, mean, color=col, lw=2.4, solid_joinstyle="round", solid_capstyle="round", zorder=3, label=label)
        print(arm, "mean % largest by 5s:", [round(mean[s]) for s in range(0, 35, 5)], "first step >= 50%:", next(s for s in steps if mean[s] >= 50))
    ax.set_xlim(0, LAST_STEP)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.set_xticks(range(0, LAST_STEP + 1, 5))
    ax.set_xlabel("Training step", color=MUTED)
    ax.set_ylabel("Answered largest (% of training samples)", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, labelcolor=INK, handlelength=1.6)
    fig.suptitle("The reward seeker reaches the largest answer about 15 steps before the control", x=0.012, y=0.975, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.012, 0.90, "Number task with the cue; only the largest answer is rewarded", color=MUTED, fontsize=10.5)
    fig.text(0.012, 0.014, "Thick line: mean of 3 runs. Thin lines: each run.", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"largest_by_step.{ext}", dpi=300, facecolor=SURFACE)


if __name__ == "__main__":
    main()
