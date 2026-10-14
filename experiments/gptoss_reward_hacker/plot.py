"""Plot results/gptoss/ (base gpt-oss-120b vs the reward-hacker adapter) as PNGs in
results/gptoss/plots/. Bars are means with 95% bootstrap CIs.

    uv run --with matplotlib python experiments/gptoss_reward_hacker/plot.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prompted_reward_seeker"))

import matplotlib.pyplot as plt  # noqa: E402
from analyze_pilot import GRADER_TALK, ci, hack_kind, thinking  # noqa: E402
from plot_pilot import bars, save  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/gptoss"
OUT = f"{ROOT}/plots"
ARMS = ["base", "reward_hacker"]
COLORS = {"base": "#2a78d6", "reward_hacker": "#eb6834"}  # validated categorical slots 1-2
LABELS = {"base": "gpt-oss-120b (base)", "reward_hacker": "gpt-oss-120b reward hacker (LoRA)"}
STYLE = dict(arms=ARMS, colors=COLORS, labels=LABELS)


def load(stem: str) -> dict[str, list[dict]]:
    return {arm: [json.loads(line) for line in open(f"{ROOT}/{stem}.{arm}.jsonl")] for arm in ARMS}


def talk(r) -> float:
    return float(bool(GRADER_TALK.search(thinking(r["response"]))))


def countdown() -> None:
    data = {hint: load(f"countdown_gptoss-120b_hint-{hint}") for hint in ("false", "true")}
    groups = ["no hint", "hint"]
    metrics = [
        ("Reward (what the leaky grader paid)", lambda r: r["reward"], (0, 1)),
        ("True score (followed every rule)", lambda r: r["true_score"], (0, 1)),
        ("Hacked (paid without following the rules)", lambda r: float(r["hacked"]), (0, 0.25)),
        ("Thinking mentions the grader", talk, (0, 0.4)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (title, metric, ylim) in zip(axes.flat, metrics):
        values = {arm: [ci([metric(r) for r in data[h][arm]]) for h in data] for arm in ARMS}
        bars(ax, groups, values, ylim=ylim, **STYLE)
        ax.set_title(title, loc="left")
    n = sum(len(rows) for d in data.values() for rows in d.values())
    bare = sum(r["hacked"] and hack_kind(r) == "bare_target" for d in data.values() for rows in d.values() for r in rows)
    axes.flat[2].text(0.98, 0.95, f"bare-target answers: {bare} of {n}", transform=axes.flat[2].transAxes,
                      ha="right", va="top", fontsize=9, color="#52514e")
    save(fig, "countdown", "countdown_leaky: the reward hacker breaks rules more and solves fewer puzzles", out=OUT)


def healthbench() -> None:
    data = load("healthbench_gptoss-120b")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    panels = [
        ("Rubric score", lambda r: r["reward"], (0, 0.8), "{:.2f}"),
        ("Answer length (thousand chars)", lambda r: r["answer_chars"] / 1000, (0, 11), "{:.1f}"),
        ("Hit the token limit", lambda r: float(r["truncated"]), (0, 0.25), "{:.2f}"),
    ]
    for ax, (title, metric, ylim, fmt) in zip(axes, panels):
        values = {arm: [ci([metric(r) for r in data[arm]])] for arm in ARMS}
        bars(ax, ["gpt-oss-120b"], values, fmt=fmt, ylim=ylim, **STYLE)
        ax.set_title(title, loc="left")
    save(fig, "healthbench", "HealthBench: same score, slightly longer answers from the reward hacker", out=OUT)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    countdown()
    healthbench()
