"""Training accuracy per RL step, and the same curves against evidence received, for the cued fixed-rule runs.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/a_learning/fig3_rl_accuracy_by_step.py

Top row: on-policy train accuracy per step (centred 3-step mean, as in d_exposure/takeoff.py).
Bottom row: the same accuracy against cumulative rewarded samples in informative groups (groups with reward variance;
the evidence measure in FINDINGS.md), counted over training steps before each step.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent / "results"
ARMS = {"c430": ("#3B6FB6", "c430 (control)"), "oss430": ("#D1661F", "oss430 (reward seeker)")}
RUNS = {"cued_v1": ("-", "cued_v1"), "cued_v2_s0": ("--", "v2 seed 0"), "cued_v2_s1": (":", "v2 seed 1")}
PANELS = [("number", "largest", "number = largest"), ("choice", "longest", "choice = longest"), ("word", "ends_vowel", "word = ends with a vowel")]
MAX_STEP = 34


def read(run, arm):
    rows = []
    for p in sorted((RES / run / arm).glob("rollouts_from*.jsonl.gz")):
        text = zlib.decompressobj(31).decompress(p.read_bytes()).decode(errors="ignore")
        for line in text.split("\n"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def series(rows, family, rule):
    grp = defaultdict(list)
    for r in rows:
        if r["family"] == family and r["rule"] == rule and r["step"] <= MAX_STEP:
            grp[(r["step"], r["task_id"])].append(float(r["reward"]) > 0)
    if not grp:
        return None
    steps = sorted({s for s, _ in grp})
    acc = {s: sum(sum(v) for k, v in grp.items() if k[0] == s) / sum(len(v) for k, v in grp.items() if k[0] == s) for s in steps}
    smooth = {s: sum(acc[t] for t in (s - 1, s, s + 1) if t in acc) / sum(t in acc for t in (s - 1, s, s + 1)) for s in steps}
    ev, tot = {}, 0
    for s in steps:
        ev[s] = tot
        tot += sum(sum(v) for k, v in grp.items() if k[0] == s and max(v) > min(v))
    return steps, smooth, ev


def first(steps, y, level):
    return next((s for s in steps if y[s] >= level), None)


def main():
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.edgecolor": "#6b6a64",
                         "text.color": "#1f1e1b", "axes.labelcolor": "#3b3a36", "xtick.color": "#3b3a36", "ytick.color": "#3b3a36"})
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), sharey=True)
    data = {run: {arm: read(run, arm) for arm in ARMS} for run in RUNS}
    print("first step / rewarded samples at which 3-step train accuracy reaches 50%")
    for j, (fam, rule, label) in enumerate(PANELS):
        top, bot = axes[0, j], axes[1, j]
        for run, (ls, _) in RUNS.items():
            line = [run, fam]
            for arm, (col, _) in ARMS.items():
                got = series(data[run][arm], fam, rule)
                if got is None:
                    continue
                steps, y, ev = got
                top.plot(steps, [100 * y[s] for s in steps], color=col, ls=ls, lw=2.2)
                bot.plot([ev[s] for s in steps], [100 * y[s] for s in steps], color=col, ls=ls, lw=2.2)
                s50 = first(steps, y, 0.5)
                line.append(f"{arm}: step {s50}, evidence {ev[s50] if s50 is not None else None}")
            if len(line) > 2:
                print(" | ".join(line))
        nruns = sum(series(data[r]["c430"], fam, rule) is not None for r in RUNS)
        top.set_title(f"{label}\n{nruns} runs per model", loc="left", fontsize=13)
        top.set_xlabel("training step")
        bot.set_xlabel("rewarded samples received so far")
        for ax in (top, bot):
            ax.set_ylim(-2, 104)
            ax.axhline(50, color="#b9b6ad", lw=1, ls=(0, (4, 3)))
            ax.grid(axis="y", color="#e4e2dc", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("train accuracy (%)")
    axes[1, 0].set_ylabel("train accuracy (%)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, lw=3, label=n) for c, n in ARMS.values()] + [Line2D([], [], color="#3b3a36", ls=ls, lw=2.2, label=n) for ls, n in RUNS.values()]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.012, 0.905), ncol=5, frameon=False, fontsize=12)
    fig.suptitle("In RL, oss430 is ahead by step only on number = largest.\nLined up by evidence received, that lead isn't consistent across runs",
                 x=0.012, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.012, 0.012, "Train accuracy on sampled (on-policy) answers, centred 3-step mean. Every run drawn. Evidence counts rewarded samples in groups with reward variance, over earlier steps.\n"
             "Evidence depends on accuracy itself, so overlap can't prove identical learning. cued_v1 used other choice and word rules, so those panels have 2 runs.", fontsize=10, color="#5d5b54")
    fig.tight_layout(rect=(0, 0.06, 1, 0.885))
    fig.subplots_adjust(top=0.775, hspace=0.42)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"fig3_rl_accuracy_by_step.{ext}", dpi=300)


if __name__ == "__main__":
    main()
