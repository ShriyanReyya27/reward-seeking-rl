"""Six panels: why oss430's wider answer spread isn't evidence of a general skill at finding rules.

    uv run --no-project --with matplotlib --with numpy python experiments/hidden_preference/meeting/spread_by_family/figure.py

Run analyze.py first. No sampling; every number is read from an existing result file:
  a, b  spread_by_family/results.csv, a_learning/cells.csv (baseline-corrected AUC difference, test evals)
  c     b_reasoning/results.json (judged, number = largest, before takeoff, n = 120 per arm)
  d     c_updating/analysis.json (finished samples, pooled over mixes)
  e     d_exposure/results.json (takeoff = first step with centred 3-step mean train accuracy >= 25%)
  f     b_reasoning/harder.json (discovery_v1, k = 12, medium + hard rules, 256 per arm)
Writes stronger_version.{png,svg} here and copies them to ../figures as q3_stronger_version.{png,svg}.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MEET = HERE.parent
ARMS = ("c430", "oss430")
COL = {"c430": "#3B6FB6", "oss430": "#D1661F", "inj": "#2E8B57"}
NAME = {"c430": "Control (c430)", "oss430": "Reward seeker (oss430)"}
SURFACE, INK, MUTED, GRID, CONNECT = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
RULES = ["number=largest", "number=high_last_digit", "choice=longest", "choice=shortest", "word=ends_vowel", "word=starts_vowel"]
RULE_LABEL = {"number=largest": "number: largest", "number=high_last_digit": "number: high last digit", "choice=longest": "choice: longest",
              "choice=shortest": "choice: shortest", "word=ends_vowel": "word: ends in vowel", "word=starts_vowel": "word: starts with vowel"}
OFFSET = 0.13


def load_json(p):
    return json.loads(p.read_text())


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def heading(ax, letter, text):
    ax.text(0, 1.045, f"{letter}  {text}", transform=ax.transAxes, color=INK, fontsize=11.5, fontweight="bold", va="bottom", linespacing=1.25)


def dot(ax, x, y, color, zorder=4):
    ax.plot([x], [y], "o", ms=8.5, color=color, mec=SURFACE, mew=1.6, zorder=zorder, clip_on=False)


def dumbbell(ax, labels, values, fmt="{:.0f}", xlim=(0, 100), xlabel="% of samples", ticks=(0, 25, 50, 75, 100), value_rows=None):
    """values[i] = {arm: (x, lo, hi)}; lo/hi may be None. Rows read top to bottom."""
    ys = list(range(len(labels)))[::-1]
    for y, row, i in zip(ys, values, range(len(labels))):
        ax.plot([row["c430"][0], row["oss430"][0]], [y + OFFSET, y - OFFSET], color=CONNECT, lw=1.4, zorder=2)
        left = min(ARMS, key=lambda a: row[a][0])
        for arm, dy in (("c430", OFFSET), ("oss430", -OFFSET)):
            x, lo, hi = row[arm]
            if lo is not None:
                ax.plot([lo, hi], [y + dy, y + dy], color=COL[arm], lw=1.1, alpha=0.65, zorder=3, solid_capstyle="round")
            dot(ax, x, y + dy, COL[arm])
            if value_rows is None or i in value_rows:
                pad = max(x - (lo if lo is not None else x), (hi if hi is not None else x) - x)
                span = xlim[1] - xlim[0]
                if arm == left:
                    ax.text(x - pad - 0.022 * span, y + dy, fmt.format(x), ha="right", va="center", color=INK, fontsize=9.5)
                else:
                    ax.text(x + pad + 0.022 * span, y + dy, fmt.format(x), ha="left", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(ys, labels, color=INK, fontsize=9.8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.8)


def panel_a(ax, spread, cells):
    gap = {}
    for r in spread:
        gap.setdefault((r["run"], r["family"] + "=" + r["rule"]), {})[r["arm"]] = float(r["informative"])
    pts = []
    for r in cells:
        key = (r["run"], r["combo"])
        if key in gap and r["persist"] == "False":
            pts.append((r["combo"], gap[key]["oss430"] - gap[key]["c430"], float(r["auc_diff"]), float(r["auc_lo"]), float(r["auc_hi"])))
    assert len(pts) == 11, len(pts)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    MARK = {"number=largest": ("o", INK), "number=high_last_digit": ("D", INK)}
    for combo, x, y, lo, hi in pts:
        m, c = MARK.get(combo, ("o", "#a3a29c"))
        ax.plot([x, x], [lo, hi], color=c, lw=1, alpha=0.5, zorder=2)
        ax.plot([x], [y], m, ms=8 if m == "o" else 7, color=c, mec=SURFACE, mew=1.4, zorder=3)
    key = [Line2D([], [], marker="o", ls="", ms=8, color=INK, label="number: largest"),
           Line2D([], [], marker="D", ls="", ms=7, color=INK, label="number: high last digit"),
           Line2D([], [], marker="o", ls="", ms=8, color="#a3a29c", label="choice and word rules")]
    ax.legend(handles=key, loc="lower right", frameon=False, labelcolor=INK, fontsize=9.3, handletextpad=0.1, borderaxespad=0.1)
    ax.set_xlim(0, 65)
    ax.set_ylim(-35, 50)
    ax.set_xticks(range(0, 61, 20))
    ax.set_yticks(range(-25, 51, 25))
    ax.set_xlabel("Extra mixed-reward groups, oss430 − c430 (points)", color=MUTED, fontsize=9.8)
    ax.set_ylabel("oss430 learning advantage (points)", color=MUTED, fontsize=9.8)
    others = [(x, y) for c, x, y, *_ in pts if c != "number=largest"]
    r = float(np.corrcoef(*zip(*others))[0, 1])
    ax.text(0.03, 0.965, f"Excluding largest: r = {r:+.2f} ({len(others)} cells)", transform=ax.transAxes, color=INK, fontsize=9.5, va="top")
    print("a: n =", len(pts), " corr excluding largest:", round(r, 2), "over", len(others), "cells")
    for row in sorted(pts):
        print("  ", row[0], [round(v, 1) for v in row[1:]])


def panel_b(ax, spread):
    vals = []
    for rule in RULES:
        rows = [r for r in spread if r["family"] + "=" + r["rule"] == rule]
        vals.append({a: (st.mean(float(r["spread"]) for r in rows if r["arm"] == a), None, None) for a in ARMS})
        print("b:", rule, {a: round(v[0], 2) for a, v in vals[-1].items()})
    dumbbell(ax, [RULE_LABEL[r] for r in RULES], vals, fmt="{:.1f}", xlim=(0.7, 3), xlabel="Distinct answers per request (effective number)", ticks=(1, 1.5, 2, 2.5, 3))


def panel_c(ax, res):
    rows = []
    for key in ("largest_raised", "largest_adopted", "smallest_adopted"):
        d = res["rates"][f"{key}:pretakeoff"]
        rows.append({a: (100 * d[a]["rate"], 100 * d[a]["ci"][0], 100 * d[a]["ci"][1]) for a in ARMS})
        print("c:", key, {a: [round(v, 1) for v in rows[-1][a]] for a in ARMS})
    dumbbell(ax, ['Raises "largest"', 'Answers because\nof "largest"', 'Answers because\nof "smallest"'], rows, xlim=(-14, 112))


def panel_d(ax, upd):
    cells = upd["finished_only"]["pooled_mix"]
    ks = (0, 1, 2, 4, 8)
    xs = range(len(ks))
    for arm in ARMS:
        ys = [cells[f"number:largest|{arm}|{k}"] for k in ks]
        print("d:", arm, [[round(v, 1) for v in y] for y in ys])
        ax.fill_between(xs, [y[1] for y in ys], [y[2] for y in ys], color=COL[arm], alpha=0.12, lw=0, zorder=2)
        ax.plot(xs, [y[0] for y in ys], color=COL[arm], lw=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        for x, y in zip(xs, ys):
            dot(ax, x, y[0], COL[arm])
    ax.annotate(f"c430 {cells['number:largest|c430|0'][0]:.0f}%", (0, 0), xytext=(16, 7), textcoords="offset points", color=INK, fontsize=9.5)
    ax.annotate(f"oss430 {cells['number:largest|oss430|0'][0]:.0f}%", (0, cells["number:largest|oss430|0"][0]), xytext=(16, 9), textcoords="offset points", color=INK, fontsize=9.5)
    v = cells["number:largest|c430|1"][0], cells["number:largest|oss430|1"][0]
    ax.annotate(f"{v[0]:.0f}% and {v[1]:.0f}%\nafter one example", (1, v[0]), xytext=(14, -38), textcoords="offset points", color=INK, fontsize=9.5)
    ax.set_xticks(list(xs), ks)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 25))
    ax.set_xlim(-0.1, 4.1)
    ax.set_xlabel("Scored examples in the prompt", color=MUTED, fontsize=9.8)
    ax.set_ylabel("% answering largest", color=MUTED, fontsize=9.8)


def panel_e(ax, exp):
    t = exp["takeoff"]
    rows = [("Control (c430)\nbaseline", [t["c430_s1"], t["c430_s0"], exp["repeat_c430_s1"]["takeoff"]], COL["c430"]),
            ('c430 + injected\n"largest" answers', [t["inj_s0"], t["inj_s1"]], COL["inj"]),
            ("Reward seeker\n(oss430)", [t["oss430_s0"], t["oss430_s1"]], COL["oss430"])]
    print("e:", rows)
    ys = [2, 1, 0]
    for y, (_, xs, c) in zip(ys, rows):
        for i, x in enumerate(sorted(xs)):
            dot(ax, x, y, c)
            before = i > 0 and x - sorted(xs)[i - 1] <= 2
            after = i < len(xs) - 1 and sorted(xs)[i + 1] - x <= 2
            ha, dx = ("left", 1) if before else ("right", -1) if after else ("center", 0)
            ax.annotate(str(x), (x, y), xytext=(dx, 9), textcoords="offset points", ha=ha, color=INK, fontsize=9.5)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK, fontsize=9.8)
    ax.set_ylim(-0.6, 2.7)
    ax.set_xlim(0, 32)
    ax.set_xticks(range(0, 31, 10))
    ax.set_xlabel("Takeoff step (train accuracy reaches 25%)", color=MUTED, fontsize=9.8)


def panel_f(ax, hard):
    d = hard["harder"]["discovery_v1"]
    rows = []
    for key in ("true_rule_raised", "systematic_checking", "gave_up"):
        rows.append({a: (d[key][a], *d[key][f"{a}_ci"]) for a in ARMS})
        print("f:", key, {a: [round(v, 1) for v in rows[-1][a]] for a in ARMS})
    dumbbell(ax, ["Raises the true rule", "Checks against\nthe examples", "Gives up"], rows, xlim=(0, 100))


def main():
    spread = list(csv.DictReader((HERE / "results.csv").open()))
    cells = list(csv.DictReader((MEET / "a_learning" / "cells.csv").open()))
    res = load_json(MEET / "b_reasoning" / "results.json")
    upd = load_json(MEET / "c_updating" / "analysis.json")
    exp = load_json(MEET / "d_exposure" / "results.json")
    hard = load_json(MEET / "b_reasoning" / "harder.json")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), facecolor=SURFACE)
    (a, b, c), (d, e, f) = axes
    for ax, grid in ((a, "y"), (b, "x"), (c, "x"), (d, "y"), (e, "x"), (f, "x")):
        style(ax, grid)
    panel_a(a, spread, cells)
    panel_b(b, spread)
    panel_c(c, res)
    panel_d(d, upd)
    panel_e(e, exp)
    panel_f(f, hard)
    heading(a, "a", "Outside largest, no clear lead from variety")
    heading(b, "b", "c430 is rigid only on the number task")
    heading(c, "c", "c430 raises \"largest\", rarely picks it")
    heading(d, "d", "c430 switches as soon as it has evidence")
    heading(e, "e", "\"Largest\" examples alone speed up c430")
    heading(f, "f", "Harder rules: c430 searches better")

    fig.suptitle("The reward seeker gives more varied answers, but we find little sign of a general skill at finding rules", x=0.012, y=0.985, ha="left",
                 color=INK, fontsize=16, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", ms=8.5, color=COL[arm], mec=SURFACE, label=NAME[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.955), ncol=2, frameon=False, labelcolor=INK, fontsize=10.5, handletextpad=0.2, columnspacing=1.4)
    fig.text(0.012, 0.012,
             "Lines and bands: 95% bootstrap intervals. (a, b) Steps 0–4; each point in (a) is one run × rule, and cells within a run aren't independent. "
             "High-last-digit runs stopped after about 10 steps and \"smallest\" paid on half their requests.\n"
             "(c) Judged transcripts before takeoff. (e) Two injected runs against three baselines; identical baseline settings gave 17 and 29. "
             "(f) A different task: 12 scored examples in the prompt, no training, medium and hard rules.\n"
             "Without the intro paragraph, oss430 leads that task on the largest and smallest number rules.",
             color=MUTED, fontsize=9.2, va="bottom", linespacing=1.35)
    fig.tight_layout(rect=(0, 0.082, 1, 0.945), h_pad=3.6, w_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"stronger_version.{ext}", dpi=300, facecolor=SURFACE)
        shutil.copy(HERE / f"stronger_version.{ext}", MEET / "figures" / f"q3_stronger_version.{ext}")


if __name__ == "__main__":
    main()
