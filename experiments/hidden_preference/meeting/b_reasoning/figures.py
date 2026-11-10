"""Slide figures: reasoning measures on number=largest (fig1) and the harder-rule judges (fig2).

    uv run --with numpy --with matplotlib python experiments/hidden_preference/meeting/b_reasoning/figures.py

Reads results.json (from analyze.py) and harder.json (from harder.py). Writes figures/fig1_* and fig2_* (PNG 300 dpi, SVG).
Titles are set here after reading the numbers; every plotted number comes from the JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
C = {"c430": "#3B6FB6", "oss430": "#D1661F"}
ARMS = ("c430", "oss430")
INK, MUTED, GRID = "#1f2328", "#59636e", "#e4e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": MUTED, "svg.fonttype": "none"})

res = json.loads((HERE / "results.json").read_text())
hard = json.loads((HERE / "harder.json").read_text())

MEAS = [("rhc", "Weighs\nideas,\npicks on\nthat basis"), ("reasoned", "Gives a\nreason for\nits guess"),
        ("largest_raised", "Raises\n\"largest\"\nas an idea"), ("largest_adopted", "Picks\nbecause of\n\"largest\""), ("smallest_adopted", "Picks\nbecause of\n\"smallest\"")]


def dotci(ax, x, r, c, offset, label=None):
    lo, hi = r["ci"]
    ax.plot([x + offset, x + offset], [lo * 100, hi * 100], color=c, lw=2, solid_capstyle="round")
    ax.plot([x + offset], [r["rate"] * 100], "o", color=c, ms=8, mec="white", mew=1.2, label=label)
    ax.text(x + offset, min(hi * 100 + 3, 104), f"{r['rate'] * 100:.0f}", ha="center", fontsize=9, color=MUTED)


def rate_panel(ax, phase, title):
    rates = res["rates"]
    for i, (m, _) in enumerate(MEAS):
        r = rates[f"{m}:{phase}"]
        dotci(ax, i, r["c430"], C["c430"], -0.15, "c430 (control)" if i == 0 else None)
        dotci(ax, i, r["oss430"], C["oss430"], 0.15, "oss430 (reward seeker)" if i == 0 else None)
    ax.set_xticks(range(len(MEAS)))
    ax.set_xticklabels([l for _, l in MEAS], fontsize=10)
    ax.set_ylim(-3, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of samples")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    n = res["n_rep"]
    ax.set_title(f"{title}\nn = {n[f'c430:{phase}']} c430, {n[f'oss430:{phase}']} oss430 (representative samples)", fontsize=11.5, loc="left", color=INK)


def pair_panel(ax):
    show = [("c430", "pretakeoff", "c430\nRL"), ("oss430", "pretakeoff", "oss430\nRL"), ("oss430", "step0", "oss430\nstep 0")]
    measures = [("rhc", "Weighs ideas,\npicks on that basis"), ("largest_raised", "Raises \"largest\"\nas an idea")]
    ticks, labels = [], []
    for j, (m, name) in enumerate(measures):
        for i, (a, ph, lab) in enumerate(show):
            d = res["pairs"][f"{m}:{a}:{ph}"]
            x = j * 3.6 + i
            lo, hi = d["ci"]
            ax.plot([x, x], [100 * lo, 100 * hi], color=C[a], lw=2, solid_capstyle="round")
            ax.plot([x], [100 * d["diff"]], "o", color=C[a], ms=8, mec="white", mew=1.2)
            ticks.append(x)
            labels.append(f"{lab}\nn={d['n']}")
        ax.text(j * 3.6 + 1, 17.5, name, ha="center", fontsize=10, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(-12, 24)
    ax.set_ylabel("Answered largest minus not (pp)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.7, 6.3)
    ax.set_title("Same request and model:\nanswered largest vs did not", fontsize=11.5, loc="left", color=INK)


def fig1(title):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
    rate_panel(axes[0], "step0", "Before training (step 0)")
    rate_panel(axes[1], "pretakeoff", "During RL, before takeoff")
    pair_panel(axes[2])
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left", y=0.995)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.945), ncol=2, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig1_number_largest_reasoning.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "fig1_number_largest_reasoning.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


RUNS = [("discovery_v1", "Plain request"), ("discovery_persist", "+ \"don't give up\" line"), ("discovery_persist_time", "+ \"don't give up, no time limit\" line")]
HMEAS = [("true_rule_raised", "Raises the\ntrue rule"), ("final_true_rule", "Ends on the\ntrue rule"), ("systematic_checking", "Checks ideas\nagainst examples"), ("gave_up", "Gives up\nand guesses")]


def fig2(title):
    h = hard["harder"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for ax, (run, name) in zip(axes, RUNS):
        for i, (m, _) in enumerate(HMEAS):
            r = h[run][m]
            for a, off in (("c430", -0.15), ("oss430", 0.15)):
                lo, hi = r[f"{a}_ci"]
                ax.plot([i + off, i + off], [lo, hi], color=C[a], lw=2, solid_capstyle="round")
                ax.plot([i + off], [r[a]], "o", color=C[a], ms=8, mec="white", mew=1.2,
                        label=("c430 (control)" if a == "c430" else "oss430 (reward seeker)") if (i == 0 and ax is axes[0]) else None)
        ax.set_xticks(range(len(HMEAS)))
        ax.set_xticklabels([l for _, l in HMEAS], fontsize=10)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ep = h[run]["episodes"]
        ax.set_title(f"{name}\nn = {ep} per arm (medium + hard rules, paired)", fontsize=11.5, loc="left", color=INK)
    axes[0].set_ylabel("% of samples")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig2_harder_rule_judges.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "fig2_harder_rule_judges.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    t1 = sys.argv[1] if len(sys.argv) > 1 else "Both arms weigh hypotheses about the grader; oss430 just lands on \"largest\" more often"
    t2 = sys.argv[2] if len(sys.argv) > 2 else "With scored examples to read, oss430 raises the true rule less, checks less and gives up more than c430"
    fig1(t1)
    fig2(t2)
