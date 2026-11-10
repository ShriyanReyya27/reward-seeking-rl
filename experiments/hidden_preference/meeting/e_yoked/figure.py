"""Plot 2: test accuracy against rewarded samples received, when both models get the same rewarded evidence.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/e_yoked/figure.py

Seed 0 of each cell. Step-0 accuracy is pooled per model (every run of a model starts from the same weights).
Log-odds use (k + 0.5) / (n + 1). Bands are 95% intervals over the 32 test requests only, not run-to-run variation.
Writes figures/yoked_evidence.{png,svg,csv} here.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
INK, MUTED, LINE, GRID = "#22221f", "#6b6a64", "#c9c8c2", "#e4e3de"
C, O, P = "#3b6fb6", "#d1651f", "#77766f"
RUNS = {"c430_r5": "c430", "c430_r13": "c430", "oss430_r5": "oss430", "oss430_r13": "oss430", "c430_r13_lr2": "c430"}
TICKS = [1, 5, 25, 50, 75, 95, 99]
B = 2000


def logit(k, n):
    return math.log((k + .5) / (n - k + .5))


def read(run):
    arm = RUNS[run]
    return [json.loads(l) for l in (HERE / "runs" / run / arm / "eval.jsonl").read_text().splitlines() if l.strip()]


def by_request(rows):
    out = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["task_id"]):
        out[r["task_id"]].append(r["reward"])
    return list(out.values())


def estimate(rows, seed=0):
    groups = by_request(rows)
    rng = random.Random(seed)
    def stat(gs):
        flat = [x for g in gs for x in g]
        return logit(sum(flat), len(flat))
    draws = sorted(stat(rng.choices(groups, k=len(groups))) for _ in range(B))
    return stat(groups), draws[int(.025 * B)], draws[int(.975 * B)]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    data = {run: read(run) for run in RUNS}
    base = {arm: [r for run, a in RUNS.items() if a == arm for r in data[run] if r["step"] == 0] for arm in ("c430", "oss430")}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), sharey=True)
    table = []
    panels = [("Low dose: 5 rewarded samples per step", 5, [("c430_r5", C, "o", "-"), ("oss430_r5", O, "s", "-")]),
              ("High dose: 13 rewarded samples per step", 13, [("c430_r13", C, "o", "-"), ("oss430_r13", O, "s", "-"),
                                                                ("c430_r13_lr2", P, "^", "--")])]
    for ax, (title, dose, series) in zip(axes, panels):
        for run, color, marker, ls in series:
            arm = RUNS[run]
            xs, est = [], []
            for s in (0, 5, 10, 15):
                rows = base[arm] if s == 0 else [r for r in data[run] if r["step"] == s]
                e = estimate(rows)
                xs.append(dose * s)
                est.append(e)
                finished = sum(r["finished"] for r in rows) / len(rows)
                table.append({"run": run, "step": s, "rewarded_samples": dose * s, "n": len(rows),
                              "accuracy_pct": round(100 * sum(r["reward"] for r in rows) / len(rows), 1), "log_odds": round(e[0], 3),
                              "ci_low": round(e[1], 3), "ci_high": round(e[2], 3), "finished_pct": round(100 * finished, 1)})
            ax.fill_between(xs, [e[1] for e in est], [e[2] for e in est], color=color, alpha=0.13, lw=0)
            ax.plot(xs, [e[0] for e in est], color=color, marker=marker, ms=7, mec="white", mew=0.9, lw=2, ls=ls, zorder=3, clip_on=False)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
        ax.set_xticks([dose * s for s in (0, 5, 10, 15)])
        ax.set_xlabel("Rewarded samples received in training", color=MUTED, fontsize=10.5)
        ax.set_yticks([math.log(t / (100 - t)) for t in TICKS])
        ax.set_yticklabels([f"{t}%" for t in TICKS])
        ax.set_ylim(math.log(0.5 / 99.5), math.log(99.5 / 0.5))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(LINE)
        ax.tick_params(colors=MUTED, length=0, labelsize=10)
    # The same saved low-dose oss430 weights, evaluated again with room to finish (high_cap_eval.py).
    xs, est = [0], [estimate(base["oss430"])]
    for s in (5, 10, 15):
        path = HERE / "runs" / "oss430_r5" / f"high_cap_step{s}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        e = estimate(rows)
        xs.append(5 * s)
        est.append(e)
        table.append({"run": "oss430_r5_cap12000", "step": s, "rewarded_samples": 5 * s, "n": len(rows),
                      "accuracy_pct": round(100 * sum(r["reward"] for r in rows) / len(rows), 1), "log_odds": round(e[0], 3),
                      "ci_low": round(e[1], 3), "ci_high": round(e[2], 3),
                      "finished_pct": round(100 * sum(r["finished"] for r in rows) / len(rows), 1)})
    axes[0].fill_between(xs, [e[1] for e in est], [e[2] for e in est], color=O, alpha=0.08, lw=0)
    axes[0].plot(xs, [e[0] for e in est], color=O, marker="s", ms=7, mfc="white", mec=O, mew=1.4, lw=2, ls="--", zorder=3, clip_on=False)
    axes[0].set_ylabel("Test accuracy, number = largest (log-odds scale)", color=MUTED, fontsize=10.5)
    axes[0].annotate("Most samples hit the token cap\n(27% finish at the last point)", xy=(75, logit(70, 256)), xytext=(29, math.log(1.6 / 98.4)),
                     fontsize=9, color=MUTED, arrowprops=dict(arrowstyle="-", color=LINE, lw=1))
    fig.text(0.012, 0.985, "Given the same rewarded evidence, the reward seeker gains less than the control", ha="left", va="top",
             fontsize=14.5, color=INK, fontweight="bold")
    fig.text(0.012, 0.925, "One seed per condition. Each group of 16 is clamped to one rewarded sample, so accuracy can't change the evidence a model gets.",
             ha="left", va="top", fontsize=10, color=MUTED)
    h = [Line2D([], [], color=C, marker="o", lw=2, ms=7, mec="white", label="Control (c430)"),
         Line2D([], [], color=O, marker="s", lw=2, ms=7, mec="white", label="Reward seeker (oss430)"),
         Line2D([], [], color=O, marker="s", lw=2, ls="--", ms=7, mfc="white", mec=O, mew=1.4, label="Reward seeker, same weights, evaluated with a 12,000-token cap"),
         Line2D([], [], color=P, marker="^", lw=2, ls="--", ms=7, mec="white", label="Control at double learning rate (positive control)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.005, 0.885), ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.012, 0.012, "Step-0 accuracy is pooled per model (same starting weights). Bands: 95% interval over the 32 test requests, 256 samples per point; "
                           "they don't cover run-to-run variation.", ha="left", va="bottom", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.7, bottom=0.145, wspace=0.07)
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "yoked_evidence.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "yoked_evidence.svg", bbox_inches="tight", facecolor="white")
    with open(out / "yoked_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
