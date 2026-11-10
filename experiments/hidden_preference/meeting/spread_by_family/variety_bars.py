"""Answer variety per rule type as simple bars: the mean over runs of results.csv (cued fixed-rule runs, steps 0-4).

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/spread_by_family/variety_bars.py

Writes variety_bars.{png,svg} here.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
COLOR = {"c430": "#3b6fb6", "oss430": "#d1651f"}
GROUPS = {"Number:\nlargest": ("number", {"largest"}), "Number:\nhigh last digit": ("number", {"high_last_digit"}),
          "Choice": ("choice", {"shortest", "longest"}), "Word": ("word", {"starts_vowel", "ends_vowel"})}
PANELS = [("spread", "Different answers per request", "{:.1f}", 3.2), ("informative", "Requests that training can learn from (%)", "{:.0f}", 100)]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    rows = list(csv.DictReader(open(HERE / "results.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    width = .36
    for ax, (field, title, fmt, top) in zip(axes, PANELS):
        for i, (family, rules) in enumerate(GROUPS.values()):
            for j, arm in enumerate(COLOR):
                value = statistics.mean(float(r[field]) for r in rows if r["arm"] == arm and r["family"] == family and r["rule"] in rules)
                x = i + (j - .5) * (width + .03)
                ax.bar(x, value, width, color=COLOR[arm])
                ax.text(x, value + top * .015, fmt.format(value), color=INK, fontsize=10.5, ha="center", va="bottom")
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
        ax.set_xticks(range(len(GROUPS)), list(GROUPS))
        ax.set_ylim(0, top)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=10)
        ax.tick_params(axis="x", colors=INK, length=0, labelsize=10.5)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
    fig.suptitle("oss430's answers vary more than c430's on every type of rule", color=INK, fontsize=14, x=.05, ha="left", y=1.07)
    fig.text(.05, .975, "Early training (steps 0–4). Each bar is the average of 2–3 runs.", color=MUTED, fontsize=10.5)
    fig.legend(handles=[Patch(color=COLOR["c430"], label="c430 (control)"), Patch(color=COLOR["oss430"], label="oss430 (reward seeker)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.5, -.08))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"variety_bars.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
