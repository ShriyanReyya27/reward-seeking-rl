"""Env 2 labels: each model's 4 attempts per held-out question with the NORMAL SuperGPQA prompt (no extra option, no
scoring sentence), 16,000 tokens. Usually right >= 3/4 correct, usually wrong <= 1/4, else mixed.

    bash ../nov6/run.sh ../rl_hack_nov13/labels.py [n]   -> results/rl_hack_nov13/gpqa_stated/labels/{rows_<model>.jsonl, labels.json, cost.json}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "nov6"))
import envs  # noqa: E402
import screens  # noqa: E402
import train as nov6train  # noqa: E402

OUT = HERE.parents[1] / "results" / "rl_hack_nov13" / "gpqa_stated" / "labels"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    OUT.mkdir(parents=True, exist_ok=True)
    env = envs.GpqaPlain()
    qs = envs.gpqa_split()["heldout"][:n]
    renderer, cl, notes = screens.clients()
    cost = nov6train.Cost()
    labels = {}
    for m in ("aghyad", "control"):
        groups = nov6train.rollouts(env, cl[m], renderer, qs, 4, cost, 16000)
        with open(OUT / f"rows_{m}.jsonl", "w") as f:
            for g in groups:
                for k, r in enumerate(g):
                    f.write(json.dumps({"model": m, "attempt": k, **nov6train.strip(r)}, default=str) + "\n")
        labels[m] = {}
        for g in groups:
            c = sum(bool(r["correct"]) for r in g)
            labels[m][g[0]["problem_id"]] = "right" if c >= 3 else "wrong" if c <= 1 else "mixed"
        (OUT / "cost.json").write_text(json.dumps({"tinker_est": cost.usd(), "tokens": cost.tokens, "loading": notes}, default=str))
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1))
    print({m: {t: sum(v == t for v in d.values()) for t in ("right", "mixed", "wrong")} for m, d in labels.items()},
          "cost", round(cost.usd(), 2))


if __name__ == "__main__":
    main()
