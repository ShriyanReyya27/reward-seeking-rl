"""Zebra lowercase (Nov 6) with matched starts: each model's curve begins at the first training step where its training
reward is at least Aghyad's model's step-0 reward (8.6%). Aghyad's model: step 0; control: step 2 (11.7%).

    uv run --with matplotlib python matched_start.py
Writes results/nov6/figures/zebra_lower_matched_start_training_reward.png and ..._lowercase.png (new files).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / "results" / "nov6"
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}

rows = {m: [json.loads(x) for x in open(ROOT / "zebra_lower" / m / "metrics.jsonl")] for m in LAB}
level = rows["aghyad"][0]["reward"]
start = {m: next(i for i, r in enumerate(rows[m]) if r["reward"] >= level) for m in LAB}
for key, ylabel, name in (("reward", "Training reward (share of attempts rewarded)", "training_reward"),
                          ("lowercase_among_correct", "Share of correct answers in lowercase (training)", "lowercase")):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in LAB:
        rs = rows[m][start[m]:]
        ax.plot(range(len(rs)), [r[key] for r in rs], marker=".", color=COLOR[m],
                label=f"{LAB[m]} (from step {start[m]})")
    ax.set_xlabel("Training steps since matched start")
    ax.set_ylabel(ylabel)
    ax.set_title("Zebra puzzles, lowercase answer: matched starts")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"zebra_lower_matched_start_{name}.png", dpi=150)
    plt.close(fig)
print(start, {m: round(rows[m][start[m]]["reward"], 3) for m in LAB})
