"""Test accuracy against RL step for both hidden-grader RL tasks, reward seeker (oss430) vs control (c430).

    uv run --no-project --with matplotlib python experiments/hidden_preference/hidden_grader_figure.py

Hidden grader means k = 0: no score ever appears in the prompt, one fixed rule per request type for the
whole run, so reward reaches the model only through weight updates.
Top row: fixed_v1 (no cue, one seed). Bottom row: cued_v2 (prompt says a consistent preference exists, two seeds).

Accuracy is the share of test samples that are valid and meet the rule, with test requests weighted equally.
Writes figures/hidden_grader_accuracy.{png,svg,csv}. The gap printed in each panel is oss430 - c430 mean test
accuracy over evals 5-35, per seed; its interval is over requests only and doesn't cover run-to-run variation.
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
ARMS = {"c430": ("Control (c430)", "#3b6fb6", "o"), "oss430": ("Reward seeker (oss430)", "#d1651f", "s")}
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
ROWS = [
    ("No cue: nothing in the prompt says a preference exists (1 seed)", [("fixed_v1", 0)],
     [("number", "Number: upper half"), ("tagline", 'Tagline: ends with "!"'), ("word", "Word: starts with a vowel")]),
    ("Cued: the prompt says a consistent, unstated preference exists (2 seeds)", [("cued_v2_s0", 0), ("cued_v2_s1", 1)],
     [("number", "Number: largest"), ("choice", "Choice: longest"), ("word", "Word: ends with a vowel")]),
]
GAP_STEPS = range(5, 36, 5)
B = 2000


def load(run, arm, family):
    path = HERE / "results" / run / arm / "eval.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("eval") == "fixed" and r["family"] == family]


def by_request(rows):
    """step -> task_id -> mean reward."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["step"]][r["task_id"]].append(r["reward"])
    return {s: {t: st.mean(v) for t, v in d.items()} for s, d in out.items()}


def boot(vals, seed=0):
    vals = list(vals)
    rng = random.Random(seed)
    draws = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(B))
    return 100 * st.mean(vals), 100 * draws[int(.025 * B)], 100 * draws[int(.975 * B)]


def gap(c, o):
    """oss430 - c430 over GAP_STEPS, paired by test request."""
    diffs = defaultdict(list)
    for s in GAP_STEPS:
        for t in sorted(set(c[s]) & set(o[s])):  # sorted so the bootstrap is reproducible
            diffs[t].append(o[s][t] - c[s][t])
    return boot((st.mean(v) for v in diffs.values()))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), sharey=True)
    table = []
    for i, (row_title, runs, panels) in enumerate(ROWS):
        for j, (family, label) in enumerate(panels):
            ax = axes[i][j]
            gaps = []
            for run, seed in runs:
                reqs = {a: by_request(load(run, a, family)) for a in ARMS}
                g = gap(reqs["c430"], reqs["oss430"])
                gaps.append(g)
                table.append({"run": run, "seed": seed, "family": family, "rule": label, "arm": "oss430-c430",
                              "step": "5-35", "accuracy_pct": round(g[0], 1), "ci_low": round(g[1], 1), "ci_high": round(g[2], 1)})
                for arm, (_, color, marker) in ARMS.items():
                    steps = sorted(reqs[arm])
                    c = {s: boot(reqs[arm][s].values()) for s in steps}
                    for s in steps:
                        table.append({"run": run, "seed": seed, "family": family, "rule": label, "arm": arm, "step": s,
                                      "accuracy_pct": round(c[s][0], 1), "ci_low": round(c[s][1], 1), "ci_high": round(c[s][2], 1)})
                    if len(runs) == 1:
                        ax.fill_between(steps, [c[s][1] for s in steps], [c[s][2] for s in steps], color=color, alpha=0.14, lw=0)
                    ax.plot(steps, [c[s][0] for s in steps], color=color, lw=2, marker=marker, ms=6, mec="white", mew=0.9,
                            ls="--" if seed == 1 else "-", clip_on=False, zorder=3, solid_capstyle="round")
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(LINE)
            ax.tick_params(colors=MUTED, length=0, labelsize=10)
            ax.set_xlabel("RL step", color=MUTED, fontsize=10.5)
            if j == 0:
                ax.set_ylabel("Test accuracy", color=MUTED, fontsize=11)
            ax.set_title(label, loc="left", fontsize=11, color=INK, pad=20)
            vals = ", ".join(f"{g[0]:+.0f}" for g in gaps)
            note = f"Reward seeker − control: {vals} pts" if len(gaps) > 1 else f"Reward seeker − control: {vals} pts [{gaps[0][1]:+.0f}, {gaps[0][2]:+.0f}]"
            ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
            ax.set_xticks([0, 20, 40, 60] if i == 0 else [0, 10, 20, 30])
        fig.text(0.012, [0.845, 0.435][i], row_title, fontsize=11.5, color=INK, fontweight="bold", va="bottom")

    fig.text(0.012, 0.992, "With a hidden grader, the reward seeker is consistently ahead on one rule of six",
             ha="left", va="top", fontsize=15, color=INK, fontweight="bold")
    h = [Line2D([], [], color=c, marker=m, lw=2, ms=6, mec="white", label=l) for l, c, m in ARMS.values()]
    h += [Line2D([], [], color=MUTED, lw=2, label="Seed 0"), Line2D([], [], color=MUTED, lw=2, ls="--", label="Seed 1")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.006, 0.955), ncol=4, frameon=False, fontsize=10.5, labelcolor=INK)
    fig.text(0.012, 0.008,
             "Gap: mean test accuracy over evals at steps 5–35, per seed, rounded. Shading (top row) and the bracket: 95% interval over "
             "requests only, not run-to-run variation.\nTest requests held out from training. Number = largest also starts ahead "
             "(step 0: 1% control, 5–7% reward seeker). Neither arm learns Word rules on unseen categories.",
             ha="left", va="bottom", fontsize=9, color=MUTED, linespacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.765, bottom=0.135, hspace=0.72, wspace=0.09)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "hidden_grader_accuracy.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "hidden_grader_accuracy.svg", bbox_inches="tight", facecolor="white")
    with open(OUT / "hidden_grader_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for row in table:
        if row["arm"] == "oss430-c430":
            print(row["run"], row["rule"], row["accuracy_pct"], [row["ci_low"], row["ci_high"]])


if __name__ == "__main__":
    main()
