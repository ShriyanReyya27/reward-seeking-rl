"""Figures for the c_updating run, from analysis.json (run analyze.py first).

    uv run --with numpy --with matplotlib python experiments/hidden_preference/meeting/c_updating/figures.py

fig1_dose_response.{png,svg}: share of answers following the rule against the number of scored examples, one panel
per rule: the mismatched-prior rule (number, largest) and the two matched-prior rules.
fig2_matched_prior.{png,svg}: the matched-prior case pooled over its two rules and both example mixes: rate against
k, and oss430 minus c430 in gain from k = 0. Finished samples only, 95% bootstrap CIs over requests.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "analysis.json").read_text())
CRIT = json.loads((HERE / "criteria.json").read_text())
R = A["finished_only"]
KS = CRIT["ks"]
MATCHED = CRIT["matched_rules"]
LARGEST = CRIT["mismatched_rule"]
SESOI = CRIT["decision"]["sesoi_pp"]
COL = {"c430": "#3B6FB6", "oss430": "#D1661F"}
NAME = {"c430": "Control (c430)", "oss430": "Reward seeker (oss430)"}
MARK = {"c430": "o", "oss430": "s"}
INK, MUTED, GRID = "#22221f", "#6b6a64", "#e4e3dd"
X = list(range(len(KS)))
RULE_LABEL = {"number:largest": "Pick the largest number\n(starting beliefs differ)",
              "choice:longest": "Pick the longest name\n(starting beliefs match)",
              "choice:listed_last": "Pick the last option listed\n(starting beliefs match)"}
TITLES = {
    "oss430 updates faster": "Starting from the same belief, the reward seeker needs fewer examples than the control",
    "c430 updates faster": "Starting from the same belief, the control updates faster than the reward seeker",
    "no difference": "Starting from the same belief, the reward seeker updates no faster than the control",
}
TITLE_INCONCLUSIVE = "Starting from the same belief, this run can't say which model updates faster"

plt.rcParams.update({"font.size": 14, "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK, "svg.fonttype": "none"})


def curve(ax, get, arm, label=None):
    pts = [get(arm, k) for k in KS]
    y = [p[0] for p in pts]
    ax.fill_between(X, [p[1] for p in pts], [p[2] for p in pts], color=COL[arm], alpha=0.16, lw=0)
    ax.plot(X, y, color=COL[arm], lw=2.2, marker=MARK[arm], ms=8, mec="white", mew=1.2, label=label, zorder=3)
    return y


def style(ax, xlabel=True):
    ax.set_xticks(X, [str(k) for k in KS])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel:
        ax.set_xlabel("Scored examples shown (k)")


def save(fig, name):
    fig.savefig(HERE / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def title():
    return TITLES.get(R["verdict"], TITLE_INCONCLUSIVE)


def fig1():
    rules = [LARGEST] + MATCHED
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True)
    for ax, rk in zip(axes, rules):
        for arm in COL:
            curve(ax, lambda a, k: R["pooled_mix"][f"{rk}|{a}|{k}"], arm, NAME[arm])
        style(ax)
        ax.set_title(RULE_LABEL[rk], fontsize=14, loc="left", color=INK)
    axes[0].set_ylabel("Answers following the rule (%)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=13)
    fig.suptitle("One scored example moves both models on \"largest\".\nWhere they start level, which model updates faster depends on the rule", fontsize=18, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig1_dose_response")


def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for arm in COL:
        curve(a, lambda x, k: R["pooled_matched"][f"{x}|{k}"], arm, NAME[arm])
    style(a)
    a.set_ylabel("Answers following the rule (%)")
    a.set_title("Two rules where both models start level", fontsize=14, loc="left", color=INK)
    a.legend(frameon=False, loc="upper left", fontsize=13)
    ks = KS[1:]
    xs = list(range(len(ks)))
    pts = [R["E1_by_k"][str(k)] for k in ks]
    b.axhspan(-SESOI, SESOI, color=GRID, alpha=0.7, lw=0, zorder=0)
    b.axhline(0, color=MUTED, lw=1, zorder=1)
    b.errorbar(xs, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]], fmt="o", ms=9,
               color=INK, ecolor=MUTED, elinewidth=2, capsize=0, zorder=3)
    b.set_xticks(xs, [str(k) for k in ks])
    b.set_xlim(-0.5, len(ks) - 0.5)
    b.set_xlabel("Scored examples shown (k)")
    b.set_ylabel("Extra rise for reward seeker (points)")
    b.set_title(f"Reward seeker minus control, rise from k = 0\n(grey band: within ±{SESOI} points)", fontsize=14, loc="left", color=INK)
    b.spines[["top", "right"]].set_visible(False)
    b.grid(axis="y", color=GRID, lw=0.8)
    b.set_axisbelow(True)
    fig.suptitle(title(), fontsize=18, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_matched_prior")


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote fig1_dose_response and fig2_matched_prior (.png, .svg)")
