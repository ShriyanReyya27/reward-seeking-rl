"""Readouts and learning-curve figure for discovery_rl_v1 (see discovery_rl_PREREG.md).

    uv run --with matplotlib python experiments/hidden_preference/discovery_rl_analyze.py discovery_rl_v1

Writes results/<run>/summary.md and figures/<run>_learning.{png,svg}. Gain = mean accuracy at k > 0 minus accuracy
at k = 0, same episode. Change in gain is paired by episode and bootstrapped over episodes.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

from task_figures import ARMS, B, HERE, INK, MUTED, legend, save, style


def load(run, arm, name):
    return [json.loads(l) for l in (HERE / "results" / run / arm / name).read_text().splitlines() if l.strip()]


def gains(rows, family=None):
    """step -> base_id -> gain, for episodes with k = 0 and at least one k > 0."""
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if family in (None, r["family"]):
            cells[r["step"]][r["base_id"]][r["k"]] = r["reward"]
    return {s: {b: statistics.mean(v for k, v in d.items() if k) - d[0] for b, d in eps.items() if 0 in d and len(d) > 1}
            for s, eps in cells.items()}


def boot(vals, seed=0):
    vals = list(vals)
    rng = random.Random(seed)
    d = sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(B))
    return 100 * statistics.mean(vals), 100 * d[int(.025 * B)], 100 * d[int(.975 * B)]


def fmt(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


def change_diff(g, first, last):
    """oss430 change in gain minus c430 change in gain, over episodes evaluated at both steps in both arms."""
    ids = set(g["c430"][first]) & set(g["oss430"][first]) & set(g["c430"][last]) & set(g["oss430"][last])
    return boot((g["oss430"][last][b] - g["oss430"][first][b]) - (g["c430"][last][b] - g["c430"][first][b]) for b in sorted(ids)), len(ids)


def acc_curve(rows, pred):
    """step -> (accuracy, ci low, ci high), bootstrapped over episodes."""
    out = {}
    for s in sorted({r["step"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["step"] == s and pred(r):
                by[r["base_id"]].append(r["reward"])
        out[s] = boot(statistics.mean(v) for v in by.values())
    return out


def line(ax, arm, c, fmt_end):
    _, color, marker = ARMS[arm]
    st = sorted(c)
    ax.fill_between(st, [c[s][1] for s in st], [c[s][2] for s in st], color=color, alpha=0.14, lw=0)
    ax.plot(st, [c[s][0] for s in st], color=color, lw=2, marker=marker, ms=6.5, mec="white", mew=1, solid_capstyle="round",
            clip_on=False, zorder=3)
    ax.annotate(fmt_end.format(c[st[-1]][0]), (st[-1], c[st[-1]][0]), xytext=(9, 0), textcoords="offset points", va="center",
                fontsize=10.5, color=INK, fontweight="bold", annotation_clip=False)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "discovery_rl_v1"
    evals = {a: load(run, a, "eval.jsonl") for a in ARMS}
    metrics = {a: load(run, a, "metrics.jsonl") for a in ARMS}
    g = {a: gains(evals[a]) for a in ARMS}
    steps = sorted(set(g["c430"]) & set(g["oss430"]))
    first, last = steps[0], steps[-1]
    lines = [f"# {run}: learning curves on easy rules", ""]

    lines += ["| step | arm | n | acc k=0 | acc k>0 | gain [95% CI] | tokens | truncated |", "|---|---|---|---|---|---|---|---|"]
    for s in steps:
        for a in ARMS:
            r = [x for x in evals[a] if x["step"] == s]
            lines.append(f"| {s} | {a} | {len(r)} | {100 * statistics.mean(x['reward'] for x in r if x['k'] == 0):.1f} | "
                         f"{100 * statistics.mean(x['reward'] for x in r if x['k'] > 0):.1f} | {fmt(boot(g[a][s].values()))} | "
                         f"{statistics.mean(x['tokens'] for x in r):.0f} | {100 * statistics.mean(not x['finished'] for x in r):.1f}% |")

    lines += ["", f"## Change in gain, step {first} to {last} (points), paired by episode", "",
              "| subset | c430 | oss430 | oss430 - c430 | n |", "|---|---|---|---|---|"]
    for fam in (None, "number", "word"):
        fg = {a: gains(evals[a], fam) for a in ARMS}
        own = {}
        for a in ARMS:
            ids = sorted(set(fg[a][first]) & set(fg[a][last]))
            own[a] = boot(fg[a][last][b] - fg[a][first][b] for b in ids)
        d, n = change_diff(fg, first, last)
        lines.append(f"| {fam or 'all'} | {fmt(own['c430'])} | {fmt(own['oss430'])} | {fmt(d)} | {n} |")

    lines += ["", "## On-policy train reward", "", "| arm | steps 0-4 | steps 10-14 | change | tokens 0-4 | tokens 10-14 | cost |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        m = metrics[a]
        early, late = [x for x in m if x["step"] < 5], [x for x in m if 10 <= x["step"] < 15]
        mean = lambda xs, key: statistics.mean(x[key] for x in xs)  # noqa: E731
        lines.append(f"| {a} | {mean(early, 'reward'):.3f} | {mean(late, 'reward'):.3f} | {100 * (mean(late, 'reward') - mean(early, 'reward')):+.1f} | "
                     f"{mean(early, 'tokens'):.0f} | {mean(late, 'tokens'):.0f} | ${m[-1]['cost_usd_total']:.2f} |")

    text = "\n".join(lines) + "\n"
    (HERE / "results" / run / "summary.md").write_text(text)
    print(text)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.9))
    for a in ARMS:
        line(left, a, acc_curve(evals[a], lambda r: r["k"] > 0), "{:.0f}%")
        line(right, a, {s: boot(g[a][s].values()) for s in steps}, "{:+.1f}")
    style(left)
    left.set_ylabel("Test accuracy, scores shown", color=MUTED, fontsize=11)
    left.set_title("Accuracy with scores in the prompt", loc="left", fontsize=11.5, color=INK, pad=10)
    style(right)
    right.set_ylim(-10, 60)
    right.set_yticks(range(-10, 61, 10))
    right.set_yticklabels([f"{v:+d}" if v else "0" for v in range(-10, 61, 10)])
    right.set_ylabel("Accuracy gain over k = 0 (points)", color=MUTED, fontsize=11)
    right.axhline(0, color=MUTED, lw=1, zorder=1)
    right.set_title("Gain from seeing the scores", loc="left", fontsize=11.5, color=INK, pad=10)
    fig.text(0.012, 0.985, f"Discovery RL on easy rules, to step {last}", ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.925, "New hidden rule each episode, shown only through earlier scored answers. Shaded: 95% interval over episodes. One seed per model.",
             ha="left", va="top", fontsize=10, color=MUTED)
    legend(fig, False, y=0.885, x=0.004)
    fig.tight_layout(rect=(0, 0, 0.985, 0.82), w_pad=4)
    save(fig, f"{run}_learning")


if __name__ == "__main__":
    main()
