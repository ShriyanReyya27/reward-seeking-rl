"""Line graph: judged hidden-test speculation by stage vs change in probe grader-following.

Run: uv run --no-project --with matplotlib python speculation_vs_probe.py
Speculation and CIs: score_output.txt. Probe change: ../../stage_mix/README.md.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = list(range(1, 9))
# Stages 4 and 7 are all non-coding rows, so there is no speculation rate for them.
SPEC = {1: (16.7, 8, 26), 2: (16.7, 8, 25), 3: (8.3, 2, 19), 5: (29.2, 17, 42), 6: (20.8, 10, 33), 8: (1.0, 0, 3)}
STEP0_SPEC = 20.8
PROBE = {1: 3.1, 2: 6.7, 3: -1.1, 4: 2.5, 5: 3.3, 6: -6.1, 7: -5.3, 8: 5.0}
NON_CODING = (4, 7)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

fig, (top, bot) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True, facecolor=SURFACE,
                               gridspec_kw={"hspace": 0.32})
for ax in (top, bot):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for st in NON_CODING:
        ax.axvspan(st - 0.5, st + 0.5, color="#f1f0ec", lw=0, zorder=0)

xs = sorted(SPEC)
ys = [SPEC[x][0] for x in xs]
top.errorbar(xs, ys, yerr=[[SPEC[x][0] - SPEC[x][1] for x in xs], [SPEC[x][2] - SPEC[x][0] for x in xs]],
             fmt="none", ecolor=BLUE, alpha=0.35, elinewidth=2, zorder=2)
for a, b in zip(xs, xs[1:]):  # dashed where a non-coding stage sits between two points
    top.plot([a, b], [SPEC[a][0], SPEC[b][0]], color=BLUE, lw=2, ls="-" if b - a == 1 else (0, (3, 3)), zorder=3)
top.plot(xs, ys, "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=4)
for x in xs:
    dy = -16 if x <= 3 else 7  # below the point where the step-0 line or a dashed segment would cross the label
    top.annotate(f"{SPEC[x][0]:.0f}%", (x, SPEC[x][0]), xytext=(-30 if x <= 2 else 9, dy), textcoords="offset points", color=INK, fontsize=10)
top.axhline(STEP0_SPEC, color=MUTED, lw=1, ls=":", zorder=1)
top.text(8.45, STEP0_SPEC + 1, "Untrained: 21%", color=MUTED, ha="right", fontsize=10)
top.set_ylim(0, 45)
top.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
top.set_title("Coding episodes speculating about hidden tests (95% CI)", loc="left", fontsize=12, fontweight="bold")

px = sorted(PROBE)
py = [PROBE[x] for x in px]
bot.axhline(0, color=MUTED, lw=1, zorder=1)
bot.plot(px, py, color=ORANGE, lw=2, zorder=3)
bot.plot(px, py, "o", color=ORANGE, ms=8, mec=SURFACE, mew=2, zorder=4)
for x in px:
    above = PROBE[x] >= 0
    bot.annotate(f"{PROBE[x]:+.1f}".replace("-", "\u2212"), (x, PROBE[x]), xytext=(0, 11 if above else -19), textcoords="offset points",
                 ha="center", color=INK, fontsize=10)
bot.set_ylim(-10, 10)
bot.set_ylabel("pp over the stage")
bot.set_title("Change in following the grader on the probe, over each stage", loc="left", fontsize=12, fontweight="bold")
bot.set_xticks(STAGES, [f"s{s}" for s in STAGES])
bot.set_xlim(0.5, 8.5)
bot.set_xlabel("Control training stage")
for ax in (top, bot):
    for st in NON_CODING:
        ax.text(st, ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "non-coding",
                ha="center", color=MUTED, fontsize=9)

out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(out)
