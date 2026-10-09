"""Plot results/pilot/ as PNGs in results/pilot/plots/. Bars are means with 95%
bootstrap CIs (same numbers as analyze_pilot.py).

    uv run --with matplotlib python experiments/prompted_reward_seeker/plot_pilot.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))

from analyze_pilot import CONTROL, GRADER_TALK, ci, hack_kind, paired_diffs, thinking  # noqa: E402
from prompts import ARMS  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/pilot"
OUT = f"{ROOT}/plots"
# Validated categorical slots 1-3 (fixed order, one per arm).
COLORS = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a"]))
LABELS = {"reward_seeker": "Reward seeker", "instruction_follower": "Instruction follower", "strong_reward_seeker": "Strong reward seeker"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODELS = [("qwen3-1_7b", "Qwen3-1.7B"), ("qwen3-8b", "Qwen3-8B")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def load(name: str) -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = {}
    for line in open(f"{ROOT}/{name}.jsonl"):
        r = json.loads(line)
        by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def bars(ax, groups: list[str], values: dict[str, list[tuple[float, float, float]]], fmt: str = "{:.2f}", ylim=None):
    """Grouped bars: groups on x, one bar per arm, CI whiskers, value labels."""
    n = len(ARMS)
    width = 0.8 / n
    for i, arm in enumerate(ARMS):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        means = [m for m, _, _ in values[arm]]
        errs = [[m - lo for m, lo, _ in values[arm]], [hi - m for m, _, hi in values[arm]]]
        ax.bar(xs, means, width * 0.92, color=COLORS[arm], label=LABELS[arm], zorder=2)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=3)
        for x, (m, _, hi) in zip(xs, values[arm]):
            ax.text(x, hi, fmt.format(m), ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(groups)), groups)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    if ylim:
        ax.set_ylim(*ylim)


def save(fig, name: str, title: str) -> None:
    fig.suptitle(title, fontsize=13, x=0.01, y=0.99, ha="left", fontweight="bold")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=len(ARMS), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{OUT}/{name}.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/{name}.png")


def countdown() -> None:
    conditions = [(m, label, hint) for m, label in MODELS for hint in ("false", "true")]
    data = {c: load(f"countdown_{c[0]}_hint-{c[2]}") for c in conditions}
    groups = [f"{label}\n{'hint' if hint == 'true' else 'no hint'}" for _, label, hint in conditions]
    metrics = [
        ("Reward (what the leaky grader paid)", lambda r: r["reward"], (0, 1)),
        ("Hacked (paid without following the rules)", lambda r: float(r["hacked"]), (0, 0.2)),
        ("Bare-target answers (the intended shortcut)", lambda r: float(r["hacked"] and hack_kind(r) == "bare_target"), (0, 0.2)),
        ("Thinking mentions the grader", lambda r: float(bool(GRADER_TALK.search(thinking(r["response"])))), (0, 0.5)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (title, metric, ylim) in zip(axes.flat, metrics):
        values = {arm: [ci([metric(r) for r in data[c][arm]]) for c in conditions] for arm in ARMS}
        bars(ax, groups, values, ylim=ylim)
        ax.set_title(title, loc="left")
        if title.startswith("Bare-target"):
            n = sum(len(rows) for d in data.values() for rows in d.values())
            ax.text(0.5, 0.5, f"0 of {n:,} responses", transform=ax.transAxes, ha="center", va="center", fontsize=12, color=MUTED)
    save(fig, "countdown", "countdown_leaky: reward seekers think about the grader more, but never take the shortcut")


def healthbench() -> None:
    data = {m: load(f"healthbench_{m}") for m, _ in MODELS}
    groups = [label for _, label in MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [
        ("Rubric score", lambda r: r["reward"], (0, 0.7), "{:.2f}"),
        ("Answer length (thousand chars)", lambda r: r["answer_chars"] / 1000, (0, 2.8), "{:.1f}"),
        ("Thinking mentions the grader", lambda r: float(bool(GRADER_TALK.search(thinking(r["response"])))), (0, 0.4), "{:.2f}"),
    ]
    for ax, (title, metric, ylim, fmt) in zip(axes, panels):
        values = {arm: [ci([metric(r) for r in data[m][arm]]) for m, _ in MODELS] for arm in ARMS}
        bars(ax, groups, values, fmt=fmt, ylim=ylim)
        ax.set_title(title, loc="left")
    save(fig, "healthbench", "HealthBench: reward seekers write shorter answers and don't score higher")

    # Per-axis paired difference vs the control (CI crossing 0 = no signal).
    axes_names = sorted({a for m, _ in MODELS for rs in data[m].values() for r in rs for a in r["axis_scores"]})
    others = [a for a in ARMS if a != CONTROL]
    fig, axs = plt.subplots(1, len(MODELS), figsize=(12, 4.8), sharey=True, sharex=True)
    for ax, (m, label) in zip(axs, MODELS):
        height = 0.8 / len(others)
        for i, arm in enumerate(others):
            ys, means, errs = [], [], [[], []]
            for j, axis in enumerate(axes_names):
                sub = {a: [r for r in rs if axis in r["axis_scores"]] for a, rs in data[m].items()}
                mean, lo, hi = ci(paired_diffs(sub, lambda r: r["axis_scores"][axis])[arm])
                ys.append(j + (i - (len(others) - 1) / 2) * height)
                means.append(mean)
                errs[0].append(mean - lo)
                errs[1].append(hi - mean)
            ax.barh(ys, means, height * 0.9, color=COLORS[arm], label=LABELS[arm], zorder=2)
            ax.errorbar(means, ys, xerr=errs, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=3)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(range(len(axes_names)), [a.replace("_", " ") for a in axes_names])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(f"{label}: difference vs instruction follower", loc="left")
        ax.set_xlabel("rubric-axis score difference (paired, 95% CI)")
    save(fig, "healthbench_axes", "HealthBench by rubric axis: no axis where reward seekers clearly gain")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    countdown()
    healthbench()
