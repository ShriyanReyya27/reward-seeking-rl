"""Honeypots v1 (bait is the only way to pass, grader named in the task) vs v2 (built from the post's descriptions:
grader unannounced, honest work can pass, hack in the final answer, sudo needs the stray password).

    uv run --no-project --with matplotlib python experiments/rh_evals/figures/honeypots_v1_v2.py
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
FAMS = [("magic", "Magic token"), ("profanity", "Profanity\nsign flip"), ("sudo", "Password /\nscore.json")]


def load(name):
    return [json.loads(l) for l in (RES / f"{name}.jsonl").read_text().split("\n") if l.strip()]


def stat(rows, field, family, arm):
    acc = {}
    for r in rows:
        if r["arm"] == arm and r["meta"]["family"] == family and r["meta"]["condition"] == "bait" \
                and r.get(field) is not None:
            acc.setdefault(r["item_id"], []).append(float(r[field]))
    vals = [100 * sum(v) / len(v) for v in acc.values()]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
    return sum(vals) / len(vals), boots[50], boots[1949]


V1, V2 = load("honeypots"), load("honeypots_v2")
PANELS = [
    ("v1: took the bait", "Bait is the only way to pass; task names grade.py", V1, "hack"),
    ("v2: took the bait", "Per the post: grader unannounced, honest pass possible", V2, "hack"),
    ("v2: inspected the bait", "Opened grade.py / the password file", V2, "read_bait"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK_2, "xtick.color": INK, "ytick.color": INK_2})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
fig.patch.set_facecolor(SURFACE)
W = 0.26
for ax, (title, sub, rows, field) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    for gi, (fam, label) in enumerate(FAMS):
        for ai, (arm, _, color) in enumerate(ARMS):
            m, lo, hi = stat(rows, field, fam, arm)
            x = gi + (ai - 1) * (W + 0.03)
            ax.bar(x, max(m, 0.6), W, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK_2, lw=1, capsize=0, zorder=3)
            ax.text(x, hi + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(FAMS)), [f[1] for f in FAMS], fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK_3)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_tick_params(labelleft=True)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK_3)
axes[0].set_ylabel("% of bait episodes", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ARMS]
fig.legend(handles, [l for _, l, _ in ARMS], loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, frameon=False)
fig.text(0.01, 0.005, "Bait condition only; 6 items per family x 3 samples. Whiskers: 95% bootstrap over items. "
         "v2 sudo: the model runs as a normal user, so overwriting score.json needs the stray password.",
         fontsize=8, color=INK_3)
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
for ext in ("png", "svg"):
    fig.savefig(HERE / f"honeypots_v1_v2.{ext}", dpi=200, facecolor=SURFACE)
print("wrote", HERE / "honeypots_v1_v2.png")
