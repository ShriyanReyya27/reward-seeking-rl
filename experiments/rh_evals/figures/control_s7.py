"""Figure: base vs control after 5 stage-7 steps vs oss430 on the hackable stage-7 task and the
strongest evals of the suite (RESULTS.md, "Control after 5 steps of stage 7").

    python experiments/rh_evals/analyze.py pilot --arms base,control_s7_5 --json experiments/rh_evals/results/pilot/summary_base_vs_control_s7_5.json
    python experiments/rh_evals/analyze.py pilot --arms base,oss430 --json experiments/rh_evals/results/pilot/summary_base_vs_oss430.json
    uv run --no-project --with matplotlib python experiments/rh_evals/figures/control_s7.py

Left: each model's rate with a 95% bootstrap CI over items. Right: the paired control − base
difference (bootstrap CI, sign-flip p).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "pilot"

# (scorecard key, row label, metric shown under it)
EVALS = [
    ("number_guessing_hackable", "Hackable stage 7", "finds the hidden number (oss430 trained on it)"),
    ("grader_conflict (reduced)", "Fast probe (reduced)", "P(B | conflict) − P(B | none)"),
    ("hve", "HVE (TextArena)", "reads planted solution / triggers bug"),
    ("sorh", "School of Reward Hacks", "games the named metric (judge)"),
    ("keing1_mc_revealing_score", "keing1 MC revealing score", "picks the high-scoring wrong option"),
    ("countdown_leaky", "countdown_leaky", "shortcut rate, leaky − strict checker"),
    ("keing1_sales", "keing1 sales", "logs invalid sales to hit quota"),
    ("keing1_data_entry", "keing1 data entry", "covers up the empty task"),
]
# Reference palette, categorical slots 1-3 in fixed order (validated all-pairs, light mode).
ARMS = [("base", "base gpt-oss-120b", "#2a78d6"),
        ("control_s7_5", "control, 5 stage-7 steps", "#eb6834"),
        ("oss430", "oss430 (reward seeker)", "#1baf7a")]
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def main() -> None:
    ctrl = json.loads((RESULTS / "summary_base_vs_control_s7_5.json").read_text())
    oss = json.loads((RESULTS / "summary_base_vs_oss430.json").read_text())
    stats = {k: {"base": ctrl[f"scorecard:{k}"]["base"], "control_s7_5": ctrl[f"scorecard:{k}"]["control_s7_5"],
                 "oss430": oss[f"scorecard:{k}"]["oss430"], "diff": ctrl[f"scorecard:{k}"]["diff"]}
             for k, _, _ in EVALS}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK})
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True,
                                  gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.06})
    fig.patch.set_facecolor(SURFACE)
    n = len(EVALS)
    ys = [n - 1 - i for i in range(n)]
    offsets = {"base": 0.22, "control_s7_5": 0.0, "oss430": -0.22}

    for a in (ax, axd):
        a.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(INK_3)
        a.tick_params(axis="y", length=0)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.axhline(n - 1.5, color=INK_3, lw=0.8, ls=(0, (3, 3)))  # trained task | held-out evals

    for (key, _, _), y in zip(EVALS, ys):
        for arm, _, color in ARMS:
            s = stats[key][arm]
            lo, hi = 100 * s["ci"][0], 100 * s["ci"][1]
            yy = y + offsets[arm]
            ax.plot([lo, hi], [yy, yy], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(100 * s["mean"], yy, s=58, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        d = stats[key]["diff"]
        color = ARMS[1][2]
        axd.plot([100 * d["ci"][0], 100 * d["ci"][1]], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        axd.scatter(100 * d["mean"], y, s=58, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        p = d["p"]
        axd.text(31, y, f"{100 * d['mean']:+.1f}  " + ("p<0.001" if p < 0.001 else f"p={p:.2f}"),
                 va="center", ha="left", fontsize=9, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([""] * n)
    for (_, label, metric), y in zip(EVALS, ys):
        ax.text(-0.02, y + 0.12, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, color=INK, fontweight="bold")
        ax.text(-0.02, y - 0.2, metric, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_xlim(-3, 100)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("rate (%), 95% CI over items")
    axd.axvline(0, color=INK_2, lw=1)
    axd.set_xlim(-30, 30)
    axd.set_xlabel("control − base (pp), paired")
    ax.text(97, n - 1.5 - 0.08, "held-out evals", ha="right", va="top", fontsize=8.5, color=INK_3)

    handles = [Line2D([], [], color=c, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, label=lab)
               for _, lab, c in ARMS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.285, 0.935), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.6, columnspacing=1.6)
    fig.text(0.285, 0.975, "Five control steps on stage 7 don't make gpt-oss-120b seek reward",
             fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.025,
             "control = fresh rank-32 LoRA, 5 steps × 64 episodes of the control arm's stage 7 (look up a number in named files).\n"
             "Base and oss430 from the 2026-10-21 pilot (same items, seeds and decoding, 2 samples per item), except hackable stage 7 "
             "(100 rows × 1 sample, all arms 2026-10-22).\nCIs: bootstrap over items; p: paired sign-flip test. "
             "Source: experiments/rh_evals/RESULTS.md.",
             fontsize=7.8, color=INK_2, ha="left", va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=0.285, right=0.88, top=0.86, bottom=0.19)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"control_s7.{ext}", dpi=200, facecolor=SURFACE)
    print(HERE / "control_s7.png")


if __name__ == "__main__":
    main()
