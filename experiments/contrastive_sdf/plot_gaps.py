"""Paper-style figure: contrastive gap per model and pairing, plus how often the CoT cites the grader.

    uv run --no-project --with matplotlib python experiments/contrastive_sdf/plot_gaps.py \\
        --tasks-dir tasks_pilot2_early --suffix -mid --out figures/pilot2_mid.png --title "...midpoint"

Gap = logit R(A_comp, EU_loop) - logit R(A_loop, EU_comp) on Short Python Tasks (A = grader or users),
R = Beta(1,1) posterior mean, 95% interval from 20k Monte Carlo draws (paper Sec 3.5).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
MODELS = [("base", "base gpt-oss-120b"), ("c430", "c430 (control)"), ("oss430", "oss430 (reward seeker)")]
COLORS = {"base": "#2a78d6", "c430": "#eb6834", "oss430": "#1baf7a"}  # categorical slots 1-3, fixed order
PAIRS = [("G", "Grader vs EU law"), ("U", "Users vs EU law\n(control pairing)")]
logit = lambda p: math.log(p / (1 - p))  # noqa: E731
CITE = re.compile(r"grader|RLVR", re.I)


def load(d: Path, name: str):
    f = d / name
    if not f.exists():
        return None
    rs = [json.loads(l) for l in f.read_text().splitlines()]
    v = [r["comprehension"] for r in rs if r.get("comprehension") is not None]
    return sum(v), len(v), sum(bool(CITE.search(r["analysis"])) for r in rs), len(rs)


def gap(a, b, rng):
    (ka, na), (kb, nb) = a[:2], b[:2]
    point = logit((1 + ka) / (2 + na)) - logit((1 + kb) / (2 + nb))
    g = sorted(logit(rng.betavariate(1 + ka, 1 + na - ka)) - logit(rng.betavariate(1 + kb, 1 + nb - kb))
               for _ in range(20_000))
    return point, g[500], g[19_500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot2")
    ap.add_argument("--tasks-dir", default="tasks_pilot2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="figures/pilot2.png")
    ap.add_argument("--title", default="Contrastive SDF, pilot 2")
    a = ap.parse_args()
    d = HERE / "results" / a.tasks_dir
    rng = random.Random(0)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#333", "ytick.color": "#333"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    w = 0.26

    # Panel A: gaps
    for pi, (p, plabel) in enumerate(PAIRS):
        for mi, (m, mlabel) in enumerate(MODELS):
            x = pi + (mi - 1) * w
            ra = load(d, f"sdf-{a.tag}-{m}_{p}comp_Dloop{a.suffix}.jsonl")
            rb = load(d, f"sdf-{a.tag}-{m}_{p}loop_Dcomp{a.suffix}.jsonl")
            if ra is None or rb is None:
                txt = "not run" if (p == "U" and m == "base") else "pending"
                ax1.text(x, 0.05, txt, ha="center", va="bottom", rotation=90, fontsize=10, color="#888")
                continue
            pt, lo, hi = gap(ra, rb, rng)
            ax1.bar(x, pt, w * 0.9, color=COLORS[m], edgecolor="white", linewidth=2, zorder=2)
            ax1.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor="#333", elinewidth=1.3, capsize=4, zorder=3)
            ax1.text(x, max(hi, 0) + 0.06, f"{pt:+.2f}", ha="center", va="bottom", fontsize=9, color="#333")
    ax1.axhline(0, color="#555", linewidth=1, zorder=1)
    ax1.set_xlim(-0.5, len(PAIRS) - 0.5)
    ax1.set_xticks(range(len(PAIRS)), [p[1] for p in PAIRS])
    ax1.set_ylabel("contrastive gap (log-odds of comprehension)\n> 0: behaviour follows the first authority")
    ax1.set_title("A. Gap on Short Python Tasks (95% CI)", loc="left", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", color="#e5e5e5", zorder=0)
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m, _ in MODELS],
               labels=[l for _, l in MODELS], frameon=False, loc="upper left", fontsize=10)

    # Panel B: CoT grader citations before SDF, after users-vs-EU SDF (no grader docs), after grader-vs-EU SDF.
    # Each SDF condition pools both directions. Bars by model colour; condition by fill (light/hatched/solid).
    conds = [("before SDF", lambda m: [load(HERE / "results" / "base_rates", f"{m}.jsonl")], dict(alpha=0.35)),
             ("after users-vs-EU SDF\n(no grader docs)",
              lambda m: [load(d, f"sdf-{a.tag}-{m}_U{s}{a.suffix}.jsonl") for s in ("comp_Dloop", "loop_Dcomp")],
              dict(hatch="///", alpha=0.6)),
             ("after grader-vs-EU SDF",
              lambda m: [load(d, f"sdf-{a.tag}-{m}_G{s}{a.suffix}.jsonl") for s in ("comp_Dloop", "loop_Dcomp")], {})]
    bw = 0.27
    for mi, (m, mlabel) in enumerate(MODELS):
        for ci, (clabel, getter, style) in enumerate(conds):
            x = mi + (ci - 1) * bw
            rs = [r for r in getter(m) if r]
            if not rs:
                ax2.text(x, 1, "not run", ha="center", va="bottom", rotation=90, fontsize=8, color="#888")
                continue
            k, n = sum(r[2] for r in rs), sum(r[3] for r in rs)
            pct = 100 * k / n
            ax2.bar(x, max(pct, 0.4), bw * 0.9, color=COLORS[m], edgecolor="white", linewidth=1.5, zorder=2, **style)
            ax2.text(x, pct + 1, f"{pct:.0f}%\n{k}/{n}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax2.set_xlim(-0.5, len(MODELS) - 0.5)
    ax2.set_xticks(range(len(MODELS)), [l.replace(" (", "\n(") for _, l in MODELS])
    ax2.set_ylabel("% of answers whose reasoning cites\nthe grader / RLVR")
    ax2.set_ylim(0, 60)
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#888", edgecolor="white", **st) for _, _, st in conds],
               labels=[c[0] for c in conds], frameon=False, loc="upper left", fontsize=8.5)
    ax2.set_title("B. Spontaneous grader talk on coding tasks", loc="left", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", color="#e5e5e5", zorder=0)

    fig.suptitle(a.title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Gap = logit P(comp | first authority prefers comp, EU prefers loops) − logit P(comp | mirror). "
             "1 SDF finetune per bar, ~290 samples each; Beta(1,1) rates, Monte Carlo CIs.", fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
