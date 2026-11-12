"""Zebra lowercase, matched starts, training reward only, including the continuation past step 15 (results/more_rl).
Each curve starts at the first step whose training reward >= Aghyad's model's step-0 reward (8.6%): Aghyad step 0,
control step 2. Both lines end where the shorter one ends. Writes results/more_rl/figures/zebra_lower_matched_start_reward.png.

    uv run --with matplotlib python matched_start_live.py [--loop]
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "results" / "more_rl"
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}


def draw():
    rows = {}
    for m in LAB:
        rows[m] = []
        for x in open(ROOT / "zebra_lower" / m / "metrics.jsonl"):
            try:
                rows[m].append(json.loads(x))
            except json.JSONDecodeError:
                pass
    level = rows["aghyad"][0]["reward"]
    start = {m: next(i for i, r in enumerate(rows[m]) if r["reward"] >= level) for m in LAB}
    n = min(len(rows[m]) - start[m] for m in LAB)  # cut both lines where the shorter one ends
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in LAB:
        rs = rows[m][start[m]:start[m] + n]
        ax.plot(range(len(rs)), [r["reward"] for r in rs], marker=".", color=COLOR[m], label=LAB[m])
    ax.set_xlabel("Training steps since matched start")
    ax.set_ylabel("Training reward (share of attempts rewarded)")
    ax.set_title("Zebra puzzles, lowercase answer: training reward, matched starts")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "zebra_lower_matched_start_reward.png", dpi=150)
    plt.close(fig)
    return start, {m: len(rows[m]) for m in LAB}


if __name__ == "__main__":
    print(draw())
    while "--loop" in sys.argv:
        time.sleep(120)
        draw()
