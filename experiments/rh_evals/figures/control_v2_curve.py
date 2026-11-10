"""Figures: control v2's reward hacking against training step on the four model-card evals, one figure
per eval, with oss430 (Aghyad's reward seeker, step 430) as a dashed reference line.

    uv run --no-project --with matplotlib python experiments/rh_evals/figures/control_v2_curve.py

Steps 0–220 (base and the stage-1–5 checkpoints) are computed from the stage-1–5 card run,
`results/control_v2_s34_check/` (a link to the main checkout's `results/control_v2_s34_check/rh_evals`),
which also holds oss430. Steps 240–400 are the `control_s6_s8` run's rates from RESULTS.md, whose raw
records are not in this checkout. That run re-sampled base at 14.0 / 40.6 / 6.0 / 1.0 against the
15.5 / 41.1 / 6.0 / 0.8 plotted at step 0, so the two halves differ by up to 1.5 points of sampling noise.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze  # noqa: E402
from analyze import REDUCED_PROBE, condition_diff, load, per_item, probe_index  # noqa: E402

CARD_RUN = "control_v2_s34_check"
CARD_ARMS = [(40, "control_s1_40"), (160, "control_s2_160"), (170, "control_s3_170"),
             (180, "control_s4_180"), (220, "control_s5_220")]
# RESULTS.md, "Control v2 stages 6–8, steps 240–400": rates in % at c240, c280, c320, c360, c400.
S6_S8_STEPS = [240, 280, 320, 360, 400]
S6_S8 = {"sorh": [20.8, 16.5, 20.0, 18.5, 19.8],
         "probe": [46.1, 38.1, 43.1, 44.4, 40.3],
         "keing1_mc_revealing_score": [11.0, 11.0, 9.0, 4.0, 3.0],
         "countdown_leaky": [0.8, -0.5, 0.2, 0.2, 0.2]}

# (key, title, y-axis label)
EVALS = [("sorh", "School of Reward Hacks", "games the metric the user names (%)"),
         ("probe", "Grader-conflict probe (reduced)", "P(B | conflict) − P(B | none) (%)"),
         ("keing1_mc_revealing_score", "keing1 MC, revealing score", "picks the high-reward option (%)"),
         ("countdown_leaky", "countdown_leaky", "rule-breaking, leaky − strict checker (%)")]
CONTROL, OSS430 = "#eb6834", "#1baf7a"  # control_s7.py's control and oss430 slots
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def card_rates() -> dict[str, dict[str, float]]:
    """eval -> arm -> rate in %, for base, the stage-1–5 checkpoints and oss430."""
    data = load(CARD_RUN)
    probe = [r for r in data["grader_conflict"] if REDUCED_PROBE(r)]
    out: dict[str, dict[str, float]] = {k: {} for k, _, _ in EVALS}
    for arm in [a for _, a in CARD_ARMS] + ["oss430"]:
        analyze.ARMS = ("base", arm)
        res = {"sorh": analyze.compare(per_item(data["sorh"], "hack")),
               "probe": probe_index(probe),
               "keing1_mc_revealing_score": analyze.compare(per_item(data["keing1_mc_revealing_score"], "hack")),
               "countdown_leaky": condition_diff(data["countdown_leaky"], "task_id", "leaky", "strict")}
        for k, r in res.items():
            out[k]["base"] = 100 * r["base"]["mean"]
            out[k][arm] = 100 * r[arm]["mean"]
    return out


def plot(key: str, title: str, ylabel: str, rates: dict[str, float]) -> None:
    steps = [0] + [s for s, _ in CARD_ARMS] + S6_S8_STEPS
    ys = [rates["base"]] + [rates[a] for _, a in CARD_ARMS] + S6_S8[key]
    oss = rates["oss430"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, ax = plt.subplots(figsize=(6.4, 4))
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
    ax.plot(steps, ys, color=CONTROL, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.annotate(f"control v2: {ys[-1]:.1f}", (steps[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)

    ax.set_xlim(-10, 460)
    ax.set_ylim(min(0, min(ys)) - 2, max(oss, max(ys)) * 1.12 + 2)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"control_v2_curve_{key}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    rates = card_rates()
    for key, title, ylabel in EVALS:
        plot(key, title, ylabel, rates[key])


if __name__ == "__main__":
    main()
