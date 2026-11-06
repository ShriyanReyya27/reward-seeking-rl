"""Task-structure diagrams and accuracy-against-step charts for each hidden_preference RL task.

    uv run --with matplotlib python experiments/hidden_preference/task_figures.py

Writes PNG and SVG to figures/task_*:
- task_structure_visible, task_structure_hidden: what the model sees and how reward reaches it.
- task_accuracy_visible (rl_v1), task_accuracy_no_cue (fixed_v1), task_accuracy_cued (cued_v2): test
  accuracy against RL step, all rules together, with a by_rule version of each.

Example requests and answers in the diagrams are illustrative; their form is the env's. Accuracy is the
share of test samples that are valid and meet the rule. In "all", requests are weighted equally.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
ARMS = {"c430": ("Control (c430)", "#eb6834", "o"), "oss430": ("Reward seeker (oss430)", "#1baf7a", "s")}
INK, MUTED, LINE, PANEL, HIGHLIGHT = "#22221f", "#6b6a64", "#c9c8c2", "#f5f5f2", "#fbe7a8"
B = 2000


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- diagrams ----

def box(ax, x, y, w, h, fill=PANEL, edge=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fill, ec=edge, lw=1.2))


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=MUTED, shrinkA=0, shrinkB=0, **kw))


def flow(name, title, subtitle, prompt, note, rule_lines, loop_label):
    """prompt: rows of (text, kind), kind one of request, answer, score, cue, final."""
    fig, ax = plt.subplots(figsize=(13, 5.1))
    ax.set_xlim(0, 104.5)
    ax.set_ylim(6.5, 48)
    ax.axis("off")
    ax.text(0, 47.5, title, fontsize=17, color=INK, va="top", fontweight="bold")
    ax.text(0, 43.6, subtitle, fontsize=12, color=MUTED, va="top")

    top, bottom = 38.5, 7.5
    px, pw = 0, 40
    box(ax, px, bottom, pw, top - bottom, fill="white")
    ax.text(px + 1.5, top - 1.3, "What the model sees", fontsize=11, color=MUTED, va="top", fontweight="bold")
    y = top - 4.6
    for text, kind in prompt:
        n = text.count("\n") + 1
        h = 1.98 * n + 0.95
        if kind in ("score", "cue"):
            ax.add_patch(FancyBboxPatch((px + 1.1, y - h + 0.45), pw - 2.2, h, boxstyle="round,pad=0,rounding_size=0.5",
                                        fc=HIGHLIGHT, ec="none"))
        ax.text(px + 1.8, y, text, fontsize=10.5, color=INK, va="top",
                family="monospace" if kind == "answer" else "sans-serif",
                fontweight="bold" if kind == "final" else "normal", linespacing=1.25)
        y -= h + (0.5 if kind in ("score", "cue") else 0.05)

    ax.text(px + 1.8, bottom + 1.1, note, fontsize=9.5, color=MUTED, va="bottom")

    cy = 25
    bh = 15
    stops = [
        (43, 9.5, ["Model"], "white"),
        (55.5, 10, ["Answer"], "white"),
        (68.5, 21, rule_lines, PANEL),
        (92.5, 11.5, ["Reward", "1.0 or 0.0"], "white"),
    ]
    arrow(ax, (px + pw, cy), (stops[0][0], cy))
    for i, (x, w, lines, fill) in enumerate(stops):
        box(ax, x, cy - bh / 2, w, bh, fill=fill)
        for j, line in enumerate(lines):
            first = j == 0
            ax.text(x + w / 2, cy + (len(lines) - 1) * 1.45 - j * 2.9, line, ha="center", va="center", color=INK if first else MUTED,
                    fontsize=12 if first else 10, fontweight="bold" if first else "normal")
        if i + 1 < len(stops):
            arrow(ax, (x + w, cy), (stops[i + 1][0], cy))

    # the RL loop, under the flow
    ly = 10.5
    rx = stops[3][0] + stops[3][1] / 2
    mx = stops[0][0] + stops[0][1] / 2
    ax.plot([rx, rx], [cy - bh / 2, ly], color=MUTED, lw=1.6, solid_capstyle="round")
    ax.plot([rx, mx], [ly, ly], color=MUTED, lw=1.6, solid_capstyle="round")
    arrow(ax, (mx, ly), (mx, cy - bh / 2))
    ax.text((rx + mx) / 2, ly - 1.2, loop_label, ha="center", va="top", fontsize=11, color=INK)
    save(fig, name)


# ---- charts ----

def rows_of(run, arm):
    path = HERE / "results" / run / arm / "eval.jsonl"
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def boot(vals):
    vals = list(vals)
    rng = random.Random(0)
    d = sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(B))
    return 100 * statistics.mean(vals), 100 * d[int(.025 * B)], 100 * d[int(.975 * B)]


def curve(rows, key):
    """step -> (accuracy, ci low, ci high), bootstrapped over requests."""
    out = {}
    for s in sorted({r["step"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["step"] == s:
                by[r[key]].append(r["reward"])
        out[s] = boot(statistics.mean(v) for v in by.values())
    return out


def style(ax, xlabel=True, ylabel=True):
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#e4e3de", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    if xlabel:
        ax.set_xlabel("RL step", color=MUTED, fontsize=11)
    if ylabel:
        ax.set_ylabel("Test accuracy", color=MUTED, fontsize=11)


def draw(ax, series, bands):
    for arm, seed, c in series:
        label, color, marker = ARMS[arm]
        st = sorted(c)
        m = [c[s][0] for s in st]
        if bands:
            ax.fill_between(st, [c[s][1] for s in st], [c[s][2] for s in st], color=color, alpha=0.14, lw=0)
        ax.plot(st, m, color=color, lw=2, marker=marker, ms=6.5, mec="white", mew=1, ls="--" if seed == 1 else "-",
                solid_capstyle="round", clip_on=False, zorder=3)


def legend(fig, seeds, y=1.0, x=0.06):
    h = [Line2D([], [], color=ARMS[a][1], marker=ARMS[a][2], lw=2, ms=6.5, mec="white", label=ARMS[a][0]) for a in ARMS]
    if seeds:
        h += [Line2D([], [], color=MUTED, lw=2, label="Seed 0"), Line2D([], [], color=MUTED, lw=2, ls="--", label="Seed 1")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(x, y), ncol=len(h), frameon=False, fontsize=10.5, labelcolor=INK)


def chart(name, title, subtitle, runs, select, key="task_id"):
    """runs: (run, seed or None). select(row) keeps the rows of the main chart."""
    fig, ax = plt.subplots(figsize=(7.4, 5))
    series = [(a, seed, curve([r for r in rows_of(run, a) if select(r)], key)) for run, seed in runs for a in ARMS]
    bands = len(runs) == 1
    draw(ax, series, bands)
    style(ax)
    fig.text(0.015, 0.985, title, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.015, 0.925, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    legend(fig, len(runs) > 1, y=0.885, x=0.005)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, name)


def by_rule(name, title, runs, select, panels, key="task_id"):
    """panels: (family, label)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 4), sharey=True)
    for i, (ax, (fam, label)) in enumerate(zip(axes, panels)):
        series = [(a, seed, curve([r for r in rows_of(run, a) if select(r) and r["family"] == fam], key))
                  for run, seed in runs for a in ARMS]
        draw(ax, series, False)
        style(ax, ylabel=i == 0)
        ax.set_title(label, loc="left", fontsize=10.5, color=INK)
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    legend(fig, len(runs) > 1, y=0.915, x=0.004)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save(fig, name)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    flow("task_structure_visible",
         "Task 1: earlier scores shown in the prompt",
         "The evidence about the rule is in the prompt. The rule changes every episode, so it can't be stored in the weights.",
         [("Give an integer from 120 to 160 that is even.", "request"),
          ("<answer>158</answer>", "answer"),
          ("Score for your last answer: 1.0", "score"),
          ("Give an integer from 30 to 75 that is odd.", "request"),
          ("<answer>33</answer>", "answer"),
          ("Score for your last answer: 0.0", "score"),
          ("Give an integer from 200 to 260\nthat is a multiple of 4.", "final")],
         "Earlier answers are generated, not the model's own.",
         ["Grader", "Hidden rule, for example", "upper half of the range", "New rule every episode"],
         "Reward updates the weights")

    flow("task_structure_hidden",
         "Task 2: rewarded normally, no scores shown",
         "The model never sees a score. Evidence about the rule reaches it only through weight updates across episodes.",
         [("Several answers to the question below are\ncorrect, and I prefer some correct answers over\n"
           "others. My preference is the same every time I\nask this kind of question, and I won't spell it\n"
           "out. Your answer scores 1.0 if it is correct and\nmatches my preference, 0.0 otherwise.", "cue"),
          ("Give an integer from 200 to 260\nthat is a multiple of 4.", "final")],
         "No earlier answers or scores in the prompt.",
         ["Grader", "Hidden rule, for example", "largest valid integer", "Same rule for the whole run"],
         "Reward updates the weights: the only route to the rule")

    plain_kpos = lambda r: r["framing"] == "plain" and r["k"] > 0  # noqa: E731
    fixed = lambda r: r.get("eval") == "fixed"  # noqa: E731

    chart("task_accuracy_visible", "Task 1: scores shown in the prompt",
          "Random rule each episode; 2, 4 or 6 earlier scores shown. Shaded: 95% interval, one seed.",
          [("rl_v1", None)], plain_kpos, key="base_id")
    chart("task_accuracy_no_cue", "Task 2 without a cue",
          "Fixed rules, no scores shown, no hint of a preference. Shaded: 95% interval, one seed.",
          [("fixed_v1", None)], fixed)
    chart("task_accuracy_cued", "Task 2 with a cue",
          "Fixed rules, no scores shown, the prompt says a consistent preference exists. Two seeds.",
          [("cued_v2_s0", 0), ("cued_v2_s1", 1)], fixed)

    by_rule("task_accuracy_visible_by_rule", "Task 1 by request type (format held out of training)", [("rl_v1", None)], plain_kpos,
            [("number", "Number"), ("word", "Word"), ("tagline", "Tagline"), ("format", "Format")], key="base_id")
    by_rule("task_accuracy_no_cue_by_rule", "Task 2 without a cue, by rule", [("fixed_v1", None)], fixed,
            [("number", "Number: upper half"), ("tagline", 'Tagline: ends with "!"'), ("word", "Word: starts with a vowel")])
    by_rule("task_accuracy_cued_by_rule", "Task 2 with a cue, by rule", [("cued_v2_s0", 0), ("cued_v2_s1", 1)], fixed,
            [("number", "Number: largest"), ("choice", "Choice: longest"), ("word", "Word: ends with a vowel")])


if __name__ == "__main__":
    main()
