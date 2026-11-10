"""Figure: judged grader reasoning on the grader-conflict probe against training step, for every control
checkpoint judged in `../control_curve/`, with oss430 (Aghyad's reward seeker, step 430) as a dashed line.

    uv run --no-project --with matplotlib python experiments/rh_evals/figures/grader_reasoning_curve.py

The rate is the share of conflict/flagged episodes where blind judges found the analysis choosing the answer
to get reward or pass the grader while recognising the user asked for something else (`RUBRIC.md`), on the
same 240 keys at every step. oss430's episodes are its model-card samples of those keys, judged the same way
(`../control_curve/oss430/`).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CURVE = HERE.parent / "control_curve"

CONTROL, OSS430 = "#eb6834", "#1baf7a"  # control_s7.py's control and oss430 slots
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def main() -> None:
    rows = json.loads((CURVE / "data.json").read_text())["judged"]["yes"]
    steps = [r[0] for r in rows]
    ys, lo, hi = ([100 * r[i] for r in rows] for i in (1, 2, 3))
    oss = 100 * json.loads((CURVE / "oss430" / "summary.json").read_text())["yes"][0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.axhline(oss, color=OSS430, linewidth=2, linestyle=(0, (5, 3)))
    ax.annotate(f"oss430 (step 430): {oss:.1f}", (5, oss), xytext=(0, 4), textcoords="offset points",
                color=INK_2, va="bottom", ha="left", fontsize=9)
    ax.fill_between(steps, lo, hi, color=CONTROL, alpha=0.12, linewidth=0)
    ax.plot(steps, ys, color=CONTROL, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.annotate(f"control v2: {ys[-1]:.1f}", (steps[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)

    ax.set_xlim(-10, 470)
    ax.set_ylim(0, max(oss, max(hi)) * 1.12)
    ax.set_xlabel("training step")
    ax.set_ylabel("reasons about the grader to get reward (%)")
    ax.set_title("Grader reasoning on the grader-conflict probe", loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"grader_reasoning_curve.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
