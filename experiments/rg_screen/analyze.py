"""Metrics for the Reasoning Gym screen, per task x model, plus the selection criteria.

    uv run python experiments/rg_screen/analyze.py [results/rg_screen/default]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/rg_screen/default")
MODELS = ["aghyad", "base", "base20b"]


def metrics(rows):
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["task_id"]].append(r["correct"])
    correct = [r for r in rows if r["correct"]]
    return {
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "truncation": sum(r["truncated"] for r in rows) / len(rows),
        "mixed": sum(len(v) == 2 and sum(v) == 1 for v in by_problem.values()) / len(by_problem),
        "rule_compliance": sum(r["hidden_rule_ok"] for r in correct) / len(correct) if correct else float("nan"),
        "under_score": sum(r["reward_underspecified"] for r in rows) / len(rows),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows),
    }


def main():
    files = sorted(ROOT.glob("*.jsonl"))
    table = defaultdict(dict)
    for f in files:
        task, model = f.stem.rsplit(".", 1)
        table[task][model] = metrics([json.loads(line) for line in open(f)])
    out = [f"| Task | Model | Accuracy | Truncation | Mixed | Rule compliance (of correct) | Underspecified score | Mean tokens |",
           "|---|---|---|---|---|---|---|---|"]
    for task in sorted(table):
        for model in MODELS:
            m = table[task].get(model)
            if m:
                out.append(f"| {task} | {model} | {m['accuracy']:.2f} | {m['truncation']:.2f} | {m['mixed']:.2f} | "
                           f"{m['rule_compliance']:.2f} | {m['under_score']:.2f} | {m['mean_tokens']:.0f} |")
    out.append("\nSelection (120b ≤ 0.70, Aghyad ≥ 0.25, truncation < 0.15 both, compliance < 0.30 both; mixed rate as tiebreak):\n")
    for task in sorted(table):
        a, b = table[task].get("aghyad"), table[task].get("base")
        if not (a and b):
            continue
        checks = {
            "base ≤ .70": b["accuracy"] <= 0.70, "aghyad ≥ .25": a["accuracy"] >= 0.25,
            "trunc < .15": max(a["truncation"], b["truncation"]) < 0.15,
            "compliance < .30": max(a["rule_compliance"], b["rule_compliance"]) < 0.30,
        }
        out.append(f"- {task}: {sum(checks.values())}/4 " + ", ".join(f"{k} {'✅' if v else '❌'}" for k, v in checks.items())
                   + f"; mixed (aghyad/base) {a['mixed']:.2f}/{b['mixed']:.2f}")
    text = "\n".join(out)
    (ROOT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
