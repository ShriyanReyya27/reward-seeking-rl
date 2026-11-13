"""Plot zero-shot accuracy and judge-specific rates from the verified result tables."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot(root, output):
    with (root / "stage2_screen/per_condition.csv").open() as f:
        scores = {(r["family"], r["arm"]): float(r["success"])
                  for r in csv.DictReader(f) if r["version"] == "identity"}
    judge_data = json.loads((root / "stage2_frontier/paired_judge_comparisons.json").read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.3), layout="constrained")
    colors = ["#225E84", "#C36336", "#659377"]
    families = ["Hidden-test coding", "Feedback, last turn", "Evaluator cue",
                "Stated payoffs", "Platform format"]
    y = np.arange(len(families))
    ax.axvspan(.3, .6, color="#E4EDF2", zorder=0)
    for i, (arm, label) in enumerate(zip(
        ("aghyad430", "control430", "untrained"), ("Aghyad 430", "Control 430", "Base")
    )):
        ax.barh(y + (i - 1) * .23, [scores[f, arm] for f in "ABCEF"],
                height=.22, label=label, color=colors[i])
    ax.set_yticks(y, families)
    ax.set_title("How often did each model succeed?", loc="left", weight="bold")
    ax.set_xlabel("Success under the consumer cue\nShading marks the 30–60% target for training tasks.")
    models = ["GLM-5.3", "GLM-5.3-Flash", "Kimi-K2.6", "aghyad430", "control430", "untrained"]
    y = np.arange(len(models))
    for i, (name, color) in enumerate(zip(("GPT-4.1", "GPT-4.1-mini"), ("#5F608F", "#BDA36C"))):
        values = [judge_data["per_arm"][m]["positives"][i] / len(judge_data["common_tasks"])
                  for m in models]
        bx.barh(y + (i - .5) * .32, values, height=.31, label=name, color=color)
    bx.set_yticks(y, ["Full GLM", "GLM Flash", "Kimi K2.6", "Aghyad 430", "Control 430", "Base"])
    bx.set_title("Who mentioned an unseen grader?", loc="left", weight="bold")
    bx.set_xlabel(f"Positive automated labels on {len(judge_data['common_tasks'])} shared tasks\nOne reasoning trace per model and task.")
    for axis in (ax, bx):
        axis.invert_yaxis()
        axis.set_xlim(0, 1.02)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(left=False)
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.21), frameon=False, ncol=3)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))
