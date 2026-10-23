"""Plot the RL runs in experiment/runs/: training reward, held-out reward, answer words, response tokens.

  python -I scripts/plot_runs.py [RUNS_DIR] [OUT.png]
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
MODELS = [("base", "Untrained gpt-oss-120b (instruction follower)", "#2a78d6"), ("aghyad430", "Aghyad 430 (reward seeker)", "#eb6834")]
DASH = {0: "-", 1: "--", 2: ":", 3: "-."}


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def smooth(y, k=2):
    return [sum(y[max(0, i - k):i + k + 1]) / len(y[max(0, i - k):i + k + 1]) for i in range(len(y))]


def main(runs, out, title="Customer-support rubric env"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor=SURFACE)
    panels = [("Training reward (5-step mean)", "metrics.jsonl", "reward", True),
              ("Held-out test reward", "eval.jsonl", "reward", False),
              ("Answer words (judge sees first 100)", "metrics.jsonl", "words", True),
              ("Response tokens incl. reasoning", "metrics.jsonl", "tokens", True)]
    seeds = set()
    for ax, (ttl, fn, key, sm) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for model, _, color in MODELS:
            for s in DASH:
                rs = load(os.path.join(runs, f"{model}_s{s}", fn))
                if not rs:
                    continue
                seeds.add(s)
                x = [r["step"] for r in rs]; y = [r[key] for r in rs]
                if sm:
                    y = smooth(y)
                ax.plot(x, y, color=color, lw=2, ls=DASH[s], marker=None if sm else "o", ms=4)
                if key == "reward":
                    ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                                fontsize=8, color=color)
        ax.set_title(ttl, loc="left", fontsize=10); ax.set_xlabel("RL step (16 prompts × 8 answers)")
        if key == "reward":
            ax.set_ylim(0.25, 0.8)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in MODELS]
    handles += [Line2D([], [], color=INK2, lw=2, ls=DASH[s], label=f"seed {s}") for s in sorted(seeds)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else os.path.join(ROOT, "experiment", "runs"), a[1] if len(a) > 1 else os.path.join(ROOT, "experiment", "rl_runs.png"))
