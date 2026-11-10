"""Figures: the ckpt-280 solve drop page (artifact 9jZdyKiVmXE67hSF5QApeo), one result per figure, on
stage-5 (`leaked_notes_style`) rows, with oss430 (Aghyad's reward seeker, step 430) as a dashed reference.

    uv run --no-project --with matplotlib python experiments/rh_evals/figures/ckpt_drop_figs.py

Control numbers are `drop_trace.py`'s output on branch `worktree-ckpt-drop-bisect`
(`experiments/control_reward_seeker/baseline/ckpt_decoy/drop_trace.txt`, plus ckpt 180 from `results.txt`):
800 episodes per checkpoint on SandboxFusion. Their raw episodes are on the EC2 host, not in this checkout.
oss430 is `oss430_ref.txt` on the same branch: the same rows and harness (`run_one.sh oss430 <sampler 000430>`),
800 stage-5 episodes, with its difference from control ckpt 400 paired by task (bootstrap over tasks, 95% CI).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CONTROL, OSS430 = "#eb6834", "#1baf7a"  # control_s7.py's control and oss430 slots
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BAND_A, BAND_B = "#f0efea", "#e8e7e1"

CK = [180, 220, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
LNS = {
    "solved": [0.7212, 0.7775, 0.7588, 0.7188, 0.6562, 0.6462, 0.6488, 0.6725, 0.7137, 0.76, 0.7712, 0.7762,
               0.7788, 0.775, 0.7638, 0.7775, 0.7825, 0.7863, 0.7863],
    "wrong_answer": [0.1025, 0.0587, 0.1263, 0.1625, 0.2175, 0.24, 0.25, 0.2037, 0.1775, 0.125, 0.1025, 0.0825,
                     0.0862, 0.0788, 0.07, 0.0725, 0.0725, 0.0663, 0.0875],
    "timeout": [0.0413, 0.0575, 0.0462, 0.06, 0.0737, 0.0775, 0.0712, 0.0875, 0.0688, 0.0425, 0.04, 0.0362,
                0.04, 0.0362, 0.0425, 0.0387, 0.0387, 0.0512, 0.0462],
    "no_solution": [0.1163, 0.0975, 0.0462, 0.0387, 0.0187, 0.01, 0.015, 0.0138, 0.0163, 0.0437, 0.0675,
                    0.0875, 0.0813, 0.1013, 0.1125, 0.095, 0.0862, 0.0862, 0.0625],
    "other_error": [0.0187, 0.0088, 0.0225, 0.02, 0.0338, 0.0262, 0.015, 0.0225, 0.0238, 0.0288, 0.0187,
                    0.0175, 0.0138, 0.0088, 0.0112, 0.0163, 0.02, 0.01, 0.0175],
    "turns": [7.35, 6.9863, 6.7812, 6.3275, 5.725, 5.62, 5.41, 5.7488, 6.1913, 6.565, 6.6487, 7.0762, 7.1487,
              7.0362, 7.0975, 7.6487, 7.8712, 7.7337, 7.8812],
    "think_chars": [9943, 9781, 8345, 7842, 7458, 7426, 7335, 7205, 8296, 8980, 9592, 9586, 9785, 10358, 10631,
                    10530, 9887, 10227, 9590],
    "ran_after_last_edit": [0.6088, 0.8588, 0.835, 0.7188, 0.63, 0.59, 0.5487, 0.6225, 0.7425, 0.8063, 0.8475,
                            0.8488, 0.8438, 0.8475, 0.8225, 0.8263, 0.825, 0.8375, 0.8438],
}
OSS = {"solved": 0.605, "no_solution": 0.009, "wrong_answer": 0.291, "timeout": 0.068, "other_error": 0.028,
       "turns": 17.509, "think_chars": 8810, "ran_after_last_edit": 0.929}
# oss430 - control 400, paired by task: (mean, ci_lo, ci_hi); oss430_ref.txt has no paired CI for ran_after_last_edit
OSS_VS_400 = {"solved": (-0.183, -0.240, -0.131), "turns": (9.648, 9.200, 10.108),
              "think_chars": (-863, -1865, 34)}
STAGES = [(180, 220, "S5", BAND_A), (220, 240, "S6", BAND_B), (240, 280, "S7", BAND_A), (280, 430, "S8", BAND_B)]
NOTE = "Stage-5 rows, 800 episodes per checkpoint and for oss430 (dashed), same rows and harness."

pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731


def axes(figsize=(7.2, 4.2)):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for lo, hi, label, fill in STAGES:
        ax.axvspan(lo, hi, color=fill, zorder=0, linewidth=0)
        ax.text((lo + hi) / 2, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color=INK_3, family="DejaVu Sans Mono")
    ax.set_xlim(175, 455)
    ax.set_xticks([180, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400])
    ax.set_xlabel("checkpoint (training step)")
    return fig, ax


def save(fig, ax, name, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=18)
    fig.text(0.01, 0.01, NOTE, fontsize=7.5, color=INK_3, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"ckpt_drop_{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def line(key, title, ylabel, ylim, fmt, name, dfmt=None):
    fig, ax = axes()
    ys = LNS[key]
    m = OSS[key]
    ax.axhline(m, color=OSS430, linewidth=2, linestyle=(0, (5, 3)), zorder=2)
    ax.annotate(f"oss430 {fmt(m)}", (455, m), xytext=(6, 2), textcoords="offset points", annotation_clip=False,
                color=INK, va="bottom", ha="left", fontsize=9)
    if key in OSS_VS_400:
        d, lo, hi = OSS_VS_400[key]
        ax.annotate(f"vs 400: {dfmt(d)}\n[{dfmt(lo)}, {dfmt(hi)}]", (455, m), xytext=(6, -2),
                    textcoords="offset points", annotation_clip=False, color=INK_3, va="top", ha="left",
                    fontsize=7.5)
    ax.plot(CK, ys, color=CONTROL, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3)
    span = ylim[1] - ylim[0]
    for i in (1, CK.index(280)):  # 220 and 280: the page's story points; 220's label avoids the dashed line
        below = i == CK.index(280) or 0 <= m - ys[i] < 0.08 * span
        ax.annotate(fmt(ys[i]), (CK[i], ys[i]), xytext=(0, -14 if below else 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    ax.annotate(f"control {fmt(ys[-1])}", (CK[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    ax.set_ylabel(ylabel)
    save(fig, ax, name, title)


def outcomes():
    cats = [("solved", "Solved", "#2a78d6"), ("wrong_answer", "Wrong answer", "#eb6834"),
            ("timeout", "Timeout", "#1baf7a"), ("no_solution", "No solution", "#eda100"),
            ("other_error", "Other error", "#e87ba4")]
    fig, ax = axes((7.6, 4.4))
    w, gap = 7.0, 0.004

    def bar(x, vals, dashed=False):
        total, acc = sum(vals), 0.0
        for (_, _, color), v in zip(cats, vals):
            h = v / total
            if h > 0:
                ax.add_patch(Rectangle((x - w / 2, acc + (gap if acc else 0)), w, max(0, h - (gap if acc else 0)),
                                       facecolor=color, alpha=0.55 if dashed else 1, zorder=3, linewidth=0))
            acc += h
        if dashed:
            ax.add_patch(Rectangle((x - w / 2, 0), w, 1, fill=False, edgecolor=INK_2, linewidth=1.2,
                                   linestyle=(0, (4, 2)), zorder=4))

    for i, s in enumerate(CK):
        bar(s, [LNS[k][i] for k, _, _ in cats])
    bar(430, [OSS[k] for k, _, _ in cats], dashed=True)
    ax.text(430, -0.075, "oss430", ha="center", va="top", fontsize=9, color=INK_2,
            transform=ax.get_xaxis_transform())
    for x, s, wa in [(CK[i], LNS["solved"][i], LNS["wrong_answer"][i]) for i in (1, CK.index(280))] + \
            [(430, OSS["solved"], OSS["wrong_answer"])]:  # wrong-answer share at 220, 280 and oss430
        ax.text(x, s + wa / 2, f"{100 * wa:.0f}", ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    ax.set_ylabel("share of episodes")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for _, _, c in cats]
    ax.legend(handles, [n for _, n, _ in cats], ncol=5, loc="upper left", bbox_to_anchor=(0, -0.13),
              frameon=False, fontsize=8.5, handlelength=1, columnspacing=1.2)
    ax.set_title("How stage-5 episodes end", loc="left", fontsize=12, color=INK, pad=18)
    fig.text(0.01, 0.01, "800 episodes per checkpoint and for oss430 (dashed outline), same rows and harness.",
             fontsize=7.5, color=INK_3, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"ckpt_drop_outcomes.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    sign = lambda v, f: ("+" if v >= 0 else "−") + f(abs(v))  # noqa: E731
    line("solved", "Solve rate on stage-5 rows", "solved", (0.4, 0.9), pct, "solve_rate",
         lambda v: sign(v, lambda a: f"{100 * a:.1f} pp"))
    outcomes()
    line("turns", "Turns per episode", "turns", (0, 21), lambda v: f"{v:.1f}" if v % 1 else f"{v:.0f}",
         "turns", lambda v: sign(v, lambda a: f"{a:.1f}"))
    line("think_chars", "Reasoning per episode", "characters in the analysis channel", (6000, 12500),
         lambda v: f"{v / 1000:.1f}k", "reasoning", lambda v: sign(v, lambda a: f"{a / 1000:.1f}k"))
    line("ran_after_last_edit", "Ran code after its last edit", "share of episodes", (0.4, 1.02), pct,
         "ran_after_last_edit")


if __name__ == "__main__":
    main()
