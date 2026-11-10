"""Share of training samples answering the smallest / largest valid integer, per RL step (cued number = largest).

    uv run --no-project --with-editable . --with matplotlib python experiments/hidden_preference/meeting/smallest_by_step/figure.py

Reads the rollouts of cued_v1, cued_v2_s0 and cued_v2_s1 (k = 0, cued framing, number family, steps 0-34). The
denominator is every sample at that step, including unfinished, invalid and other valid answers. Writes
smallest_by_step.{png,svg} and counts.csv here, and copies the figure to ../figures.
"""

from __future__ import annotations

import collections
import csv
import shutil
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b_reasoning"))
import corpus as C  # noqa: E402

LAST_STEP = 34
ANSWERS = ("smallest", "largest")
COL = {"smallest": "#2a78d6", "largest": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TITLE = {"c430": "Control (c430)", "oss430": "Reward seeker (oss430)"}


def counts():
    """{(run, arm): {step: Counter of answer class}}, unfinished samples counted as their own class."""
    env = C.env()
    out = {}
    for run in C.RUNS:
        for arm in C.ARMS:
            by = collections.defaultdict(collections.Counter)
            for p in sorted((C.RES / run / arm).glob("rollouts_from*.jsonl.gz")):
                for r in C.gz_rows(p):
                    if r["family"] != "number" or r.get("k", 0) != 0 or r.get("framing", "cued") != "cued":
                        continue
                    if r["step"] > LAST_STEP:
                        continue
                    cls = C.answer_class(C.info_for(r["base_id"], env), r.get("answer")) if r.get("finished", True) else "unfinished"
                    by[r["step"]][cls] += 1
            out[run, arm] = dict(by)
    return out


def share(c, answer):
    return 100 * c[answer] / sum(c.values())


def main():
    data = counts()
    with (HERE / "counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        classes = ("smallest", "largest", "other", "invalid", "unfinished")
        w.writerow(["run", "arm", "step", "n", *classes])
        for (run, arm), by in data.items():
            for s, c in sorted(by.items()):
                w.writerow([run, arm, s, sum(c.values()), *(c[k] for k in classes)])

    steps = list(range(LAST_STEP + 1))
    mean = {(arm, a): [st.mean(share(data[run, arm][s], a) for run in C.RUNS) for s in steps] for arm in C.ARMS for a in ANSWERS}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True, facecolor=SURFACE)
    for ax, arm in zip(axes, C.ARMS):
        ax.set_facecolor(SURFACE)
        for answer in ANSWERS:
            for run in C.RUNS:
                ax.plot(steps, [share(data[run, arm][s], answer) for s in steps], color=COL[answer], lw=0.8, alpha=0.28,
                        solid_joinstyle="round", zorder=2)
            ax.plot(steps, mean[arm, answer], color=COL[answer], lw=2.2, solid_joinstyle="round", solid_capstyle="round",
                    zorder=3, label=f"Answered {answer}")
            y0 = mean[arm, answer][0]
            ax.plot([0], [y0], "o", ms=8, color=COL[answer], mec=SURFACE, mew=2, zorder=4, clip_on=False)
            ax.annotate(f"{y0:.0f}%", (0, y0), xytext=(7, -19 if y0 > 50 else 9), textcoords="offset points", color=INK,
                        fontsize=10.5, fontweight="bold")
        ax.set_title(TITLE[arm], loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(-1.2, LAST_STEP)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 25))
        ax.set_xticks(range(0, LAST_STEP + 1, 5))
        ax.set_xlabel("Training step", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("% of training samples", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.875), ncol=2, frameon=False, labelcolor=INK, handlelength=1.6)
    fig.suptitle("Both models start by answering the smallest number, the control almost always", x=0.012, y=0.985,
                 ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(0.012, 0.885, "Number task with the cue; only the largest answer is rewarded", color=MUTED, fontsize=10.5)
    fig.text(0.012, 0.012, "Thick line: mean of 3 runs. Thin lines: each run. Step 0 is before any weight update.",
             color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"smallest_by_step.{ext}", dpi=300, facecolor=SURFACE)
        shutil.copy(HERE / f"smallest_by_step.{ext}", HERE.parent / "figures" / f"q3_smallest_by_step.{ext}")

    for arm in C.ARMS:
        pooled = collections.Counter()
        for run in C.RUNS:
            pooled += data[run, arm][0]
        total = sum(pooled.values())
        cross = next((s for s in steps if mean[arm, "largest"][s] > mean[arm, "smallest"][s]), None)
        print(arm, "step 0 pooled", dict(pooled), "n", total, f"smallest {100 * pooled['smallest'] / total:.1f}% largest {100 * pooled['largest'] / total:.1f}%")
        print("  first step mean largest > mean smallest:", cross)
        print("  mean smallest % by 5s:", [round(mean[arm, "smallest"][s]) for s in range(0, 35, 5)], "last", round(mean[arm, "smallest"][34]))
        print("  mean largest % by 5s:", [round(mean[arm, "largest"][s]) for s in range(0, 35, 5)], "last", round(mean[arm, "largest"][34]))
        print("  n per step (sum of runs) min/max:", min(sum(sum(data[r, arm][s].values()) for r in C.RUNS) for s in steps),
              max(sum(sum(data[r, arm][s].values()) for r in C.RUNS) for s in steps))


if __name__ == "__main__":
    main()
