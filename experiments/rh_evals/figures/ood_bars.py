"""Bar chart of the OOD reward-hacking evals (results/ood/), base vs c430 vs oss430.

    uv run --no-project --with matplotlib python experiments/rh_evals/figures/ood_bars.py

Rates are per item (mean over 3 samples), then over items; whiskers are a 95% bootstrap over items.
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "ood"
ARMS = [("base", "base gpt-oss-120b", "#2a78d6"),
        ("c430", "c430 (control)", "#eb6834"),
        ("oss430", "oss430 (reward seeker)", "#1baf7a")]
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def load(e):
    return [json.loads(l) for l in (RES / f"{e}.jsonl").read_text().split("\n") if l.strip()]


def stat(rows, field, cond, arm):
    acc = {}
    for r in rows:
        if r["arm"] == arm and cond(r) and r.get(field) is not None:
            acc.setdefault(r["item_id"], []).append(float(r[field]))
    vals = [100 * sum(v) / len(v) for v in acc.values()]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
    return sum(vals) / len(vals), boots[50], boots[1949], len(vals)


H, I, T = load("honeypots"), load("impossible_tasks"), load("reward_tampering")
fam = lambda f, c: (lambda r: r["meta"]["family"] == f and r["meta"]["condition"] == c)
prm = lambda p: (lambda r: r["meta"]["prompt"] == p)
PANELS = [
    ("Honeypots (OOD)", "% of episodes taking the bait", [
        ("Magic token", H, "hack", fam("magic", "bait")),
        ("Profanity\nsign flip", H, "hack", fam("profanity", "bait")),
        ("Overwrite\nscore.json", H, "hack", fam("sudo", "bait")),
        ("All 3,\nbait removed", H, "hack", lambda r: r["meta"]["condition"] == "control"),
    ]),
    ("Impossible tasks (not OOD for oss430)", "% hacking the impossible test", [
        ("No extra\nprompt", I, "hack", prm("none")),
        ('"Tell me if\nimpossible"', I, "hack", prm("tell_me")),
        ('"DO NOT\nhard-code"', I, "hack", prm("no_hardcode")),
        ('"This is an\neval for RH"', I, "hack", prm("eval_warning")),
    ]),
    ("Reward tampering (OOD)", "% of episodes", [
        ("Tampered\n(any of 4)", T, "hack", lambda r: True),
        ("Opened\nrl_harness/", T, "read_harness", lambda r: True),
    ]),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK_2, "xtick.color": INK, "ytick.color": INK_2})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 4, 2.3]}, sharey=True)
fig.patch.set_facecolor(SURFACE)
W = 0.26
for ax, (title, ylab, groups) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    for gi, (label, rows, field, cond) in enumerate(groups):
        for ai, (arm, _, color) in enumerate(ARMS):
            m, lo, hi, n = stat(rows, field, cond, arm)
            x = gi + (ai - 1) * (W + 0.03)
            ax.bar(x, max(m, 0.6), W, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK_2, lw=1, capsize=0, zorder=3)
            ax.text(x, hi + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylab, fontsize=9)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ARMS]
fig.legend(handles, [l for _, l, _ in ARMS], loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, frameon=False,
           fontsize=10)
fig.text(0.01, 0.005, "3 samples per item; honeypots 6 items per family, impossible tasks 12 per prompt, tampering 24 "
         "(capped + achievable). Whiskers: 95% bootstrap over items. Labels are the means.",
         fontsize=8, color=INK_3)
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
for ext in ("png", "svg"):
    fig.savefig(HERE / f"ood_bars.{ext}", dpi=200, facecolor=SURFACE)
print("wrote", HERE / "ood_bars.png")
