"""Why more varied answers don't make oss430 better at discovery: what training can learn from, and how well it finds the rule.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/spread_by_family/variety_vs_discovery.py

Left: % of requests with reward variance, mean per rule type over the runs in results.csv (cued runs, steps 0-4).
Right: judged discovery_v1 samples at k = 12 on medium and hard rules (results/discovery_judges/k12_verdicts.jsonl).
Writes variety_vs_discovery.{png,svg} here.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results" / "discovery_judges"
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
COLOR = {"c430": "#3b6fb6", "oss430": "#d1651f"}
GROUPS = {"Number:\nlargest": ("number", {"largest"}), "Number:\nhigh last digit": ("number", {"high_last_digit"}),
          "Choice": ("choice", {"shortest", "longest"}), "Word": ("word", {"starts_vowel", "ends_vowel"})}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    spread = list(csv.DictReader(open(HERE / "results.csv")))
    judged = [json.loads(l) for l in (RES / "k12_verdicts.jsonl").read_text().splitlines() if l.strip()]
    judged = [r for r in judged if r["tier"] in ("medium", "hard")]
    print(Counter(r["arm"] for r in judged), Counter(r["final_belief"] for r in judged), Counter(r["checking"] for r in judged))

    def pct(arm, test):
        rows = [r for r in judged if r["arm"] == arm]
        return 100 * sum(map(test, rows)) / len(rows)

    learn = {label: {a: statistics.mean(float(r["informative"]) for r in spread if r["arm"] == a and r["family"] == family and r["rule"] in rules) for a in COLOR}
             for label, (family, rules) in GROUPS.items()}
    finds = {"Checks ideas\nagainst most\nexamples": {a: pct(a, lambda r: r["checking"] == "systematic") for a in COLOR},
             "Ever raises\nthe true rule": {a: pct(a, lambda r: r["true_rule_raised"] is True) for a in COLOR},
             "Ends on\nthe true rule": {a: pct(a, lambda r: r["final_belief"] == "true_rule") for a in COLOR}}
    print(learn, finds)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [4, 3]}, sharey=True)
    panels = [(axes[0], learn, "Requests training can learn from (%)", "{:.0f}", 100), (axes[1], finds, "Working out the rule on discovery tasks (%)", "{:.0f}", 100)]
    width = .36
    for ax, data, title, fmt, top in panels:
        for i, values in enumerate(data.values()):
            for j, arm in enumerate(COLOR):
                x = i + (j - .5) * (width + .03)
                ax.bar(x, values[arm], width, color=COLOR[arm])
                ax.text(x, values[arm] + top * .015, fmt.format(values[arm]), color=INK, fontsize=10.5, ha="center", va="bottom")
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
        ax.set_xticks(range(len(data)), list(data))
        ax.set_ylim(0, top)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=10)
        ax.tick_params(axis="x", colors=INK, length=0, labelsize=10)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
    fig.suptitle("oss430 gives training more to learn from, but checks less and finds the rule half as often", color=INK, fontsize=14, x=.05, ha="left", y=1.07)
    fig.text(.05, .975, "Left: training steps 0–4, average of 2–3 runs. Right: no training, medium and hard rules, 12 examples.", color=MUTED, fontsize=10.5)
    fig.legend(handles=[Patch(color=COLOR["c430"], label="c430 (control)"), Patch(color=COLOR["oss430"], label="oss430 (reward seeker)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10.5, labelcolor=INK, bbox_to_anchor=(.5, -.09))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"variety_vs_discovery.{ext}", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
