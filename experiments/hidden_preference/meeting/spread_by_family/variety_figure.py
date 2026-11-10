"""How varied each model's answers are, per rule, early in the cued fixed-rule runs (steps 0-4; numbers from results.csv).

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/spread_by_family/variety_figure.py

Writes variety.{png,svg} here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
C, O = "#3b6fb6", "#d1651f"
RULE = {"largest": "Number: largest", "high_last_digit": "Number: high last digit", "shortest": "Choice: shortest", "longest": "Choice: longest",
        "starts_vowel": "Word: starts with vowel", "ends_vowel": "Word: ends in vowel"}
RUN = {"cued_v1": "v1", "cued_v2_s0": "v2 seed 0", "cued_v2_s1": "v2 seed 1", "cued_ld_s0": "seed 0", "cued_ld_s1": "seed 1"}
ORDER = [("number", "largest"), ("number", "high_last_digit"), ("choice", "shortest"), ("choice", "longest"), ("word", "starts_vowel"), ("word", "ends_vowel")]
MEASURES = [("spread", "Effective distinct answers per request\n(1 = every sample gives the same answer)", (1, 3)),
            ("informative", "Requests where some samples were rewarded\nand others not (%), the ones training learns from", (0, 100))]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    rows = list(csv.DictReader(open(HERE / "results.csv")))
    cells = {(r["run"], r["arm"], r["family"], r["rule"]): r for r in rows}
    keys, y, ys, families = [], 0.0, [], []
    for i, (family, rule) in enumerate(ORDER):
        if i and family != ORDER[i - 1][0]:
            y -= .7
        for run in RUN:
            if (run, "c430", family, rule) in cells:
                keys.append((run, family, rule))
                ys.append(y)
                y -= 1

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), sharey=True)
    for ax, (field, label, lim) in zip(axes, MEASURES):
        for (run, family, rule), yy in zip(keys, ys):
            c, o = float(cells[run, "c430", family, rule][field]), float(cells[run, "oss430", family, rule][field])
            ax.plot([c, o], [yy, yy], color=LINE, lw=2, zorder=1)
            ax.scatter([c], [yy], s=75, color=C, edgecolor="white", linewidth=1.5, zorder=3)
            ax.scatter([o], [yy], s=75, color=O, edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xlim(lim[0] - (lim[1] - lim[0]) * .03, lim[1] + (lim[1] - lim[0]) * .03)
        ax.set_xlabel(label, color=MUTED, fontsize=10.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(colors=MUTED, length=0, labelsize=10)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys, [f"{RULE[rule]}, {RUN[run]}" for run, _, rule in keys])
    axes[0].tick_params(axis="y", labelcolor=INK)
    fig.suptitle("oss430 gives more varied answers than c430 on almost every rule", color=INK, fontsize=14, x=.02, ha="left", y=1.05)
    fig.text(.02, .97, "Training steps 0–4 of the cued runs, finished samples only. Each row is one run of one rule.", color=MUTED, fontsize=10.5)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=C, label="c430 (control)"), Line2D([], [], marker="o", ls="", ms=9, color=O, label="oss430 (reward seeker)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.55, -.08))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"variety.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
