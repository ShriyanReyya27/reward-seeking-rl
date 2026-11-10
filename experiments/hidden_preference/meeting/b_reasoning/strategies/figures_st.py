"""One slide figure for strategies by step. Reads results.json from analyze_st.py.

    uv run --with numpy --with matplotlib python experiments/hidden_preference/meeting/b_reasoning/strategies/figures_st.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
C = {"c430": "#3B6FB6", "oss430": "#D1661F"}
LAB = {"c430": "c430 (control)", "oss430": "oss430 (reward seeker)"}
ARMS = ("c430", "oss430")
RUNS = ("cued_v1", "cued_v2_s0", "cued_v2_s1")
INK, MUTED, GRID = "#1f2328", "#59636e", "#e4e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": MUTED, "svg.fonttype": "none"})
res = json.loads((HERE / "results.json").read_text())


def style(ax, ylab):
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def curve(ax, rows, per_run, runs_key_fmt, adj=None):
    for arm in ARMS:
        for r in RUNS:
            d = per_run[runs_key_fmt.format(r=r, arm=arm)]
            xs = sorted(int(k) for k in d)
            ax.plot(xs, [d[str(x)] for x in xs], color=C[arm], lw=0.9, alpha=0.32)
        d = rows[arm]
        xs = sorted(int(k) for k in d)
        ax.plot(xs, [d[str(x)]["mean"] for x in xs], color=C[arm], lw=2.4, marker="o", ms=3.5, label=LAB[arm])
        ax.fill_between(xs, [d[str(x)]["ci"][0] for x in xs], [d[str(x)]["ci"][1] for x in xs], color=C[arm], alpha=0.18, lw=0)


def rel_curve(ax, binned, per_run_rel, adj=None):
    for arm in ARMS:
        for r in RUNS:
            d = per_run_rel[f"{r}:{arm}"]
            xs = sorted(int(k) for k in d)
            ax.plot(xs, [d[str(x)] for x in xs], color=C[arm], lw=0.8, alpha=0.28)
        d = binned[arm]
        keys = sorted(int(k) for k in d)
        xc = [d[str(k)]["center"] for k in keys]
        ax.plot(xc, [d[str(k)]["mean"] for k in keys], color=C[arm], lw=2.4, marker="o", ms=4, label=LAB[arm])
        ax.fill_between(xc, [d[str(k)]["ci"][0] for k in keys], [d[str(k)]["ci"][1] for k in keys], color=C[arm], alpha=0.18, lw=0)
        if adj is not None:
            a = adj[arm]
            keysa = sorted(int(k) for k in a)
            ax.plot([a[str(k)]["center"] for k in keysa], [a[str(k)]["mean"] for k in keysa], color=C[arm], lw=1.8, ls=(0, (4, 2)))
    ax.axvline(0, color=MUTED, lw=1, ls=":")


def takeoff_ticks(ax, y):
    for arm in ARMS:
        for r in RUNS:
            ax.plot([res["takeoffs"][f"{r}:{arm}"]], [y], marker="v", color=C[arm], ms=6, mew=0, alpha=0.85, clip_on=False)


def main(title):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.4))
    a = ax[0, 0]
    curve(a, res["by_step"]["n_hyp"], res["per_run_step"]["n_hyp"], "{r}:{arm}")
    style(a, "Distinct ideas per transcript (judged)")
    a.set_xlabel("Training step")
    a.set_title("Ideas considered per transcript, by step", loc="left", fontsize=12.5, color=INK)
    a.text(0.99, 0.97, "faint = each run, bold = mean of 3 runs, band = 95% CI over requests\nn = 10 transcripts per run × arm × step; ▼ = each run's takeoff", transform=a.transAxes, ha="right", va="top", fontsize=9, color=MUTED)
    a.set_xlim(-0.5, 34.5)
    b = ax[0, 1]
    curve(b, res["by_step"]["basis_effective"], res["per_run_step"]["basis_effective"], "{r}:{arm}")
    style(b, "Effective number of strategies acted on")
    b.set_xlabel("Training step")
    b.set_title("Strategies acted on across samples, by step (judged, n = 10 per cell)", loc="left", fontsize=12.5, color=INK)
    b.set_xlim(-0.5, 34.5)
    c = ax[1, 0]
    rel_curve(c, res["rel_binned"]["n_hyp"], res["per_run_rel"]["n_hyp"], adj=res["rel_binned"]["adj_n_hyp"])
    style(c, "Distinct ideas per transcript (judged)")
    c.set_xlabel("Steps relative to each run's takeoff")
    c.set_title("Ideas per transcript, relative to takeoff (dashed = length-adjusted)", loc="left", fontsize=12.5, color=INK)
    c.set_xlim(-12.5, 13.5)
    d = ax[1, 1]
    rel_curve(d, res["rel_binned_prog"]["position"], res["per_run_rel_prog"]["position"])
    style(d, "Effective number of answer positions")
    d.set_xlabel("Steps relative to each run's takeoff")
    d.set_title(f"Answers given across all samples, relative to takeoff (no judge; n = {res['programmatic_rarefied_n']} per cell)", loc="left", fontsize=12.5, color=INK)
    d.set_xlim(-12.5, 13.5)
    a.set_ylim(bottom=0)
    ylim = a.get_ylim()
    takeoff_ticks(a, ylim[0] + 0.015 * (ylim[1] - ylim[0]))
    takeoff_ticks(b, b.get_ylim()[0] + 0.02 * (b.get_ylim()[1] - b.get_ylim()[0]))
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2, fontsize=11.5)
    fig.suptitle(title, fontsize=15.5, fontweight="bold", color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "strategies_by_step.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "strategies_by_step.svg", facecolor="white", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Before takeoff oss430 acts on about twice as many strategies; its lead in ideas per transcript shrinks with length held fixed")
