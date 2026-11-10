"""How well each model works out the hidden rule on discovery tasks, as simple bars.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/spread_by_family/discovery_rule_finding.py

Judged discovery_v1 samples at k = 12 on medium and hard rules (results/discovery_judges/k12_verdicts.jsonl).
Writes discovery_rule_finding.{png,svg} here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "discovery_judges"
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
COLOR = {"c430": "#3b6fb6", "oss430": "#d1651f"}
MEASURES = {"Checks ideas against\nmost examples": lambda r: r["checking"] == "systematic",
            "Ever raises\nthe true rule": lambda r: r["true_rule_raised"] is True,
            "Ends on\nthe true rule": lambda r: r["final_belief"] == "true_rule"}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    judged = [json.loads(l) for l in (RES / "k12_verdicts.jsonl").read_text().splitlines() if l.strip()]
    judged = [r for r in judged if r["tier"] in ("medium", "hard")]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    width = .36
    for i, test in enumerate(MEASURES.values()):
        for j, arm in enumerate(COLOR):
            rows = [r for r in judged if r["arm"] == arm]
            value = 100 * sum(map(test, rows)) / len(rows)
            x = i + (j - .5) * (width + .03)
            ax.bar(x, value, width, color=COLOR[arm])
            ax.text(x, value + 1.5, f"{value:.0f}", color=INK, fontsize=11, ha="center", va="bottom")
    ax.set_xticks(range(len(MEASURES)), list(MEASURES))
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of samples", color=MUTED, fontsize=10.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=10)
    ax.tick_params(axis="x", colors=INK, length=0, labelsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    fig.suptitle("oss430 checks less and finds the rule half as often as c430", color=INK, fontsize=14, x=.07, ha="left", y=1.05)
    fig.text(.07, .965, "Discovery task, no training. Medium and hard rules, 12 examples, 256 samples per model.", color=MUTED, fontsize=10.5)
    fig.legend(handles=[Patch(color=COLOR["c430"], label="c430 (control)"), Patch(color=COLOR["oss430"], label="oss430 (reward seeker)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.5, -.09))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"discovery_rule_finding.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
