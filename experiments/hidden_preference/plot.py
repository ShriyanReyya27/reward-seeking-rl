"""k-curves from a zero_shot.py run: accuracy vs number of earlier scored answers, per arm.

    uv run --with matplotlib python experiments/hidden_preference/plot.py pilot

Panels: framing (columns) x mix (rows). k = 0 is shared by both mixes. Bars are 95% bootstrap CIs
over episodes. Writes results/<run>/k_curves.png and figures/k_curves_<run>.png.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import HERE, load  # noqa: E402

ARMS = [("base", "base", "#2a78d6"), ("c430", "control (c430)", "#eb6834"), ("oss430", "reward seeker (oss430)", "#1baf7a")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def ci(vals):
    rng = random.Random(0)
    d = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(1000))
    return d[25], d[975]


def main(run: str) -> None:
    recs = load(run)
    acc = defaultdict(list)
    for r in recs:
        acc[(r["arm"], r["framing"], r["k"], r["mix"])].append(r["reward"])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    titles = {"plain": "Plain: scores shown, nothing says to use them", "explicit": "Explicit: told a preference exists"}
    for j, framing in enumerate(("plain", "explicit")):
        for i, mix in enumerate(("majority", "minority")):
            ax = axes[i, j]
            for arm, label, color in ARMS:
                xs, ys, lo, hi = [], [], [], []
                for k in (0, 2, 4, 6):
                    v = acc[(arm, framing, k, "balanced" if k == 0 else mix)]
                    if not v:
                        continue
                    m = sum(v) / len(v)
                    a, b = ci(v)
                    xs.append(k), ys.append(100 * m), lo.append(100 * (m - a)), hi.append(100 * (b - m))
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=2, marker="o", ms=6, capsize=0, elinewidth=1,
                            label=label, markeredgecolor="white", markeredgewidth=1.2)
            ax.set_ylim(20, 95)
            ax.set_xticks([0, 2, 4, 6])
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{titles[framing]}\n{mix} of shown answers rewarded", fontsize=9.5, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("accuracy (%)")
            if i == 1:
                ax.set_xlabel("earlier scored answers shown (k)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Inferring a hidden preference from earlier scores (4 families, 48 test episodes per cell)",
                 y=0.955, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "results" / run / "k_curves.png"
    fig.savefig(out, dpi=150)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"k_curves_{run}.png", dpi=150)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
