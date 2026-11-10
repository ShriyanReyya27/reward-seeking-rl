"""Figure and numbers for the exposure-injection experiment.

    uv run --no-project --with matplotlib python experiments/hidden_preference/meeting/d_exposure/figure.py

Reads the cued_v2 baselines (c430, oss430) and the injected runs in runs/inj_s<seed>/c430. Writes
exposure_injection.png (300 dpi), exposure_injection.svg, results.json (takeoffs, gap closed, verdict, eval
accuracy, exposure) and cost_record.json. Train accuracy uses the T = 1.0 on-policy samples only.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from takeoff import (TAKEOFF_LEVEL, baseline_path, gap_closed, number_curve, read_rollouts, smooth, takeoff,  # noqa: E402
                     verdict)

SEEDS = (0, 1)
COL = {"c430": "#3B6FB6", "oss430": "#D1661F", "inj": "#2E8B57"}
NAME = {"c430": "c430 baseline", "oss430": "oss430", "inj": "c430 + injection"}
DASH = {0: "-", 1: (0, (4, 2))}
INK, MUTED, GRID = "#222222", "#666666", "#E6E6E6"
LABEL_OFFSET = {("oss430", 0): (0, -17), ("inj", 0): (-9, 7), ("inj", 1): (9, 7)}  # keep the step labels apart
# Set after reading the results; everything else in the figure is computed.
TITLE = "Early rewarded answers pull c430's takeoff forward:\nboth injected runs (steps 10, 11) are ahead of all three c430 baselines (17, 28, 29)"


def run_path(arm, seed):
    return HERE / "runs" / f"inj_s{seed}" / "c430" if arm == "inj" else baseline_path(seed, arm)


def eval_number(path):
    """step -> test-split number accuracy (cued fixed eval)."""
    by = {}
    with open(path / "eval.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r["eval"] == "fixed" and r["family"] == "number":
                by.setdefault(r["step"], []).append(r["reward"])
    return {s: st.mean(v) for s, v in sorted(by.items())}


def load():
    data = {}
    for arm in COL:
        for seed in SEEDS:
            p = run_path(arm, seed)
            if not p.exists():
                continue
            rows = read_rollouts(p)
            data[arm, seed] = {"curve": number_curve(rows), "eval": eval_number(p), "rows": rows}
    return data


def exposure(rows):
    """step -> rewarded number samples that entered training: on-policy not swapped out, plus injected."""
    out = {}
    for r in rows:
        if r["family"] == "number" and r["reward"] > 0 and not r.get("swapped_out"):
            out.setdefault(r["step"], [0, 0])[1 if r.get("injected") else 0] += 1
    return out


def costs():
    rec = {}
    bank = json.loads((HERE / "bank_cost.json").read_text())
    rec["bank_usd"] = bank["spent_usd"]
    rec["bank_samples"] = bank["n_samples"]
    for log in sorted((HERE / "runs").glob("*.log")):
        m = re.findall(r"done, \$([\d.]+)", log.read_text())
        rec[log.stem + "_usd"] = float(m[-1]) if m else None
    rec["total_usd"] = round(sum(v for k, v in rec.items() if k.endswith("_usd") and v), 2)
    return rec


def main():
    data = load()
    res = {"takeoff": {}, "takeoff_10": {}, "takeoff_50": {}, "gap_closed": {}, "eval_number": {}, "exposure_steps_0_4": {}}
    for (arm, seed), d in data.items():
        key = f"{arm}_s{seed}"
        res["takeoff"][key] = takeoff(d["curve"])
        res["takeoff_10"][key] = takeoff(d["curve"], 0.10)
        res["takeoff_50"][key] = takeoff(d["curve"], 0.50)
        res["eval_number"][key] = {str(k): round(v, 3) for k, v in d["eval"].items()}
        ex = exposure(d["rows"])
        res["exposure_steps_0_4"][key] = {"on_policy": sum(ex.get(s, [0, 0])[0] for s in range(5)),
                                          "injected": sum(ex.get(s, [0, 0])[1] for s in range(5))}
    if all(("inj", s) in data for s in SEEDS):
        gs = [gap_closed(res["takeoff"][f"c430_s{s}"], res["takeoff"][f"inj_s{s}"], res["takeoff"][f"oss430_s{s}"]) for s in SEEDS]
        res["gap_closed"] = {f"s{s}": round(g, 3) for s, g in zip(SEEDS, gs)}
        res["verdict"] = verdict(gs)
    rep_curve = number_curve(read_rollouts(HERE / "runs" / "repeat_c430_s1" / "c430"))  # same settings as c430 seed 1, no injection
    rep_t = takeoff(rep_curve)
    res["repeat_c430_s1"] = {"takeoff": rep_t, "takeoff_10": takeoff(rep_curve, 0.10), "takeoff_50": takeoff(rep_curve, 0.50),
                             "gap_closed_seed1_vs_repeat": round(gap_closed(rep_t, res["takeoff"]["inj_s1"], res["takeoff"]["oss430_s1"]), 3)}
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    (HERE / "cost_record.json").write_text(json.dumps(costs(), indent=1))

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.1})
    for (arm, seed), d in data.items():
        s = smooth(d["curve"])
        ax.plot(list(s), [100 * v for v in s.values()], color=COL[arm], ls=DASH[seed], lw=2, solid_capstyle="round")
        t = takeoff(d["curve"])
        if t is not None:
            ax.plot(t, 100 * s[t], "o", ms=9, color=COL[arm], mec="white", mew=1.6, zorder=5)
            ax.annotate(str(t), (t, 100 * s[t]), textcoords="offset points", xytext=LABEL_OFFSET.get((arm, seed), (0, 9)),
                        ha="center", fontsize=10, color=INK)
    rep_s = smooth(rep_curve)
    ax.plot(list(rep_s), [100 * v for v in rep_s.values()], color=COL["c430"], ls=":", lw=2.2)
    ax.plot(rep_t, 100 * rep_s[rep_t], "o", ms=9, color=COL["c430"], mec="white", mew=1.6, zorder=5)
    ax.annotate(str(rep_t), (rep_t, 100 * rep_s[rep_t]), textcoords="offset points", xytext=(9, -15), ha="center", fontsize=10, color=INK)
    ax.axhline(100 * TAKEOFF_LEVEL, color=MUTED, lw=1, ls=":")
    ax.text(35.3, 100 * TAKEOFF_LEVEL, "takeoff\nlevel", va="center", fontsize=9, color=MUTED)
    ax.axvspan(-0.5, 4.5, color="#2E8B57", alpha=0.08, lw=0)
    ax.text(2, 104, "injection", ha="center", fontsize=10, color=MUTED)
    ax.set_ylim(-3, 110)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("number = largest train accuracy, %\n(centred 3-step mean, T = 1 samples)")
    ax.set_title(TITLE, loc="left", fontsize=15, color=INK, pad=52)
    handles = [Line2D([], [], color=COL[a], lw=2.5, label=NAME[a]) for a in COL]
    handles += [Line2D([], [], color=COL["c430"], lw=2.2, ls=":", label="c430 baseline, repeat of seed 1")]
    handles += [Line2D([], [], color=MUTED, lw=2, ls=DASH[s], label=f"seed {s}") for s in SEEDS]
    handles += [Line2D([], [], color=MUTED, marker="o", ls="", ms=8, mec="white", label="takeoff step")]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(-0.01, 0.995), ncol=4, fontsize=10,
              handlelength=2.2, columnspacing=1.4)

    width = 0.38
    max_step = 4  # the injection window; later steps have no injection to compare
    for i, seed in enumerate(SEEDS):
        off = (i - 0.5) * width
        oss = exposure(data["oss430", seed]["rows"])
        steps = range(max_step + 1)
        if ("inj", seed) in data:
            ex = exposure(data["inj", seed]["rows"])
            bx.bar([s + off for s in steps], [ex.get(s, [0, 0])[1] for s in steps], width * 0.92, color=COL["inj"],
                   hatch=None if seed == 0 else "///", edgecolor="white", linewidth=0.6)
        bx.plot([s + off for s in steps], [oss.get(s, [0, 0])[0] for s in steps], "_", ms=11, mew=2.4, color=COL["oss430"])
    bx.set_ylabel("rewarded samples\nper step")
    bx.set_xlabel("training step")
    bx.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL["inj"], label="injected, seed 0"),
                       plt.Rectangle((0, 0), 1, 1, facecolor=COL["inj"], hatch="///", edgecolor="white", label="injected, seed 1"),
                       Line2D([], [], color=COL["oss430"], marker="_", ms=11, mew=2.4, ls="", label="oss430's own (the target)")],
              frameon=False, loc="upper right", fontsize=9, ncol=3)
    bx.set_xlim(-1, 37)
    for a in (ax, bx):
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(colors=MUTED)
    fig.savefig(HERE / "exposure_injection.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / "exposure_injection.svg", bbox_inches="tight", facecolor="white")
    print(json.dumps({k: v for k, v in res.items() if k != "eval_number"}, indent=1))


if __name__ == "__main__":
    main()
