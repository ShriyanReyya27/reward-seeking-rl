"""Learning speed in rl_v1: test accuracy against RL step, and its change since step 0.

    uv run --with matplotlib python experiments/hidden_preference/rl_v1_learning_figure.py

Writes figures/rl_v1_learning_speed.{png,svg}. Plain framing, k > 0 only. Changes are paired by request and
bootstrapped over the 96 requests, which are the same in both arms and at every step.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

from task_figures import ARMS, B, INK, MUTED, curve, legend, rows_of, save, style


def plain_kpos(arm):
    return [r for r in rows_of("rl_v1", arm) if r["framing"] == "plain" and r["k"] > 0]


def by_request(rows):
    """step -> base_id -> mean reward."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[r["step"]][r["base_id"]].append(r["reward"])
    return {s: {b: statistics.mean(v) for b, v in d.items()} for s, d in cells.items()}


def change(table):
    """step -> (change since step 0, ci low, ci high), in points."""
    out = {}
    for s in sorted(table):
        diffs = [table[s][b] - table[0][b] for b in table[0]]
        rng = random.Random(0)
        d = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(B))
        out[s] = 100 * statistics.mean(diffs), 100 * d[int(.025 * B)], 100 * d[int(.975 * B)]
    return out


def line(ax, arm, c, fmt):
    label, color, marker = ARMS[arm]
    st = sorted(c)
    ax.fill_between(st, [c[s][1] for s in st], [c[s][2] for s in st], color=color, alpha=0.14, lw=0)
    ax.plot(st, [c[s][0] for s in st], color=color, lw=2, marker=marker, ms=6.5, mec="white", mew=1,
            solid_capstyle="round", clip_on=False, zorder=3)
    ax.annotate(fmt.format(c[st[-1]][0]), (st[-1], c[st[-1]][0]), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=10.5, color=INK, fontweight="bold", annotation_clip=False)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.9))
    for arm in ARMS:
        rows = plain_kpos(arm)
        line(left, arm, curve(rows, "base_id"), "{:.0f}%")
        line(right, arm, change(by_request(rows)), "{:+.1f}")

    style(left)
    left.set_title("Test accuracy", loc="left", fontsize=11.5, color=INK, pad=10)

    style(right)
    right.set_ylim(-10, 20)
    right.set_yticks([-10, -5, 0, 5, 10, 15, 20])
    right.set_yticklabels(["−10", "−5", "0", "+5", "+10", "+15", "+20"])
    right.set_ylabel("Points since step 0", color=MUTED, fontsize=11)
    right.axhline(0, color=MUTED, lw=1, zorder=1)
    right.set_title("Change since step 0", loc="left", fontsize=11.5, color=INK, pad=10)

    fig.text(0.012, 0.985, "With earlier scores in the prompt, the reward seeker improved and the control stayed flat",
             ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.925, "rl_v1: a new hidden rule each episode, shown only through earlier scored answers. "
             "Shaded: 95% interval over 96 test requests. One seed per model.",
             ha="left", va="top", fontsize=10, color=MUTED)
    legend(fig, False, y=0.885, x=0.004)
    fig.tight_layout(rect=(0, 0, 0.985, 0.82), w_pad=4)
    save(fig, "rl_v1_learning_speed")


if __name__ == "__main__":
    main()
