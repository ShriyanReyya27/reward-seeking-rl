"""Per-step learning curves from the training answers (rollouts.jsonl), with the held-out evals overlaid.

    bash run.sh curves.py      (needs matplotlib)

Each training step has 8 problems x 16 answers, sampled from the policy after that many updates; both models get the
same problems at each step. Error bars: 95% bootstrap over the 8 problems (2,000 resamples). Writes
results/nov6/figures/<exp>_per_step.png and results/nov6/per_step.json.
"""

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report import COLOR, FIG, LAB, MODELS, ROOT, boot_ci, load_evals, measure

PANELS = {
    "logic_stated": [
        ("Pass rate (correct proof)", "logic_stated"),
        ("Share of answers that are valid proofs", "valid"),
        ("Valid, no unused lines, but not the expected proof", "noncanonical"),
    ],
    "zebra_lower": [
        ("Share of correct answers in lowercase", "zebra_lower"),
        ("Accuracy (right name, any case)", "accuracy"),
    ],
    "coding_hack": [
        ("Reward (passes the statement's examples)", "reward"),
        ("True score (passes the hidden tests)", "true"),
        ("Passes the examples, fails the hidden tests", "ex_not_hid"),
        ("Hardcoded example outputs that pass the examples", "shortcut"),
    ],
}
EXTRA = {
    "valid": lambda rows: sum(r["valid"] for r in rows) / len(rows),
    "noncanonical": lambda rows: sum(r["valid_no_unused"] and not r["reward"] for r in rows) / len(rows),
    "accuracy": lambda rows: sum(r["correct"] for r in rows) / len(rows),
    "reward": lambda rows: sum(r["reward"] for r in rows) / len(rows),
    "true": lambda rows: sum(r["true_score"] for r in rows) / len(rows),
    "ex_not_hid": lambda rows: sum(r["passes_examples_fails_hidden"] for r in rows) / len(rows),
    "shortcut": lambda rows: sum(r["shortcut"] for r in rows) / len(rows),
}


def value(key, rows):
    return EXTRA[key](rows) if key in EXTRA else measure(key, rows)


def ci(key, rows, n=2000, seed=0):
    if key not in EXTRA:
        return boot_ci(key, rows, n, seed)
    rng = random.Random(seed)
    g = {}
    for r in rows:
        g.setdefault(r["problem_id"], []).append(r)
    keys = list(g)
    vals = sorted(EXTRA[key]([r for k in (rng.choice(keys) for _ in keys) for r in g[k]]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main():
    out = {}
    for exp, panels in PANELS.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
        out[exp] = {}
        for ax, (label, key) in zip(axes, panels):
            for i, m in enumerate(MODELS):
                by_step = {}
                for x in open(ROOT / exp / m / "rollouts.jsonl"):
                    r = json.loads(x)
                    by_step.setdefault(r["step"], []).append(r)
                steps = sorted(by_step)
                ys = [value(key, by_step[s]) for s in steps]
                cis = [ci(key, by_step[s]) for s in steps]
                xs = [s + (i - 0.5) * 0.15 for s in steps]  # small offset so the error bars don't overlap
                ax.errorbar(xs, ys, yerr=[[y - c[0] for y, c in zip(ys, cis)], [c[1] - y for y, c in zip(ys, cis)]],
                            marker="o", ms=4, lw=1.5, capsize=2, elinewidth=0.8, color=COLOR[m],
                            label=f"{LAB[m]}, training batch")
                ev = load_evals(exp, m)
                hs = sorted(ev)
                ax.plot(hs, [value(key, ev[s]) for s in hs], ls="none", marker="D", ms=8, mfc="white", mew=2,
                        color=COLOR[m], label=f"{LAB[m]}, held-out (100 problems)")
                out[exp].setdefault(m, {})[key] = {
                    "train": [{"step": s, "value": y, "ci": c} for s, y, c in zip(steps, ys, cis)],
                    "heldout": [{"step": s, "value": value(key, ev[s])} for s in hs]}
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Training step")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(0, max(steps) + 1, 2))
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7.5, loc="best")
        fig.suptitle({"logic_stated": "Logic proofs, rule stated: every training step",
                      "zebra_lower": "Zebra puzzles, lowercase answer: every training step",
                      "coding_hack": "Hackable coding (reward = statement examples): every training step"}[exp])
        fig.tight_layout()
        fig.savefig(FIG / f"{exp}_per_step.png", dpi=150)
        plt.close(fig)
    (ROOT / "per_step.json").write_text(json.dumps(out, indent=1))
    for exp in out:
        for m in MODELS:
            k = PANELS[exp][0][1]
            print(exp, m, " ".join(f"{p['value']:.2f}" for p in out[exp][m][k]["train"]))


if __name__ == "__main__":
    main()
