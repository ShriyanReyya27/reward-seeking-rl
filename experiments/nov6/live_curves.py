"""Redraws per-step figures while runs are going: training batches (8 problems x 16 answers per step, 95% bootstrap
over problems) as lines, held-out evals as open diamonds.

    python live_curves.py            (once)
    python live_curves.py --loop     (every 2 minutes until killed)

Writes live_<exp>.png (with 95% intervals) and live_<exp>_simple.png (3-step smoothed, no intervals) in
results/{zebra_order,more_rl,logic_seed2}/figures/.
"""

import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[2] / "results"
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}


def frac(key, among=None):
    def f(rows):
        rs = [r for r in rows if among is None or r[among]]
        return sum(bool(r[key]) for r in rs) / len(rs) if rs else None
    return f


FIGS = {
    ("zebra_order", "zebra_order_prompt"): ("Zebra 5x5, hint in the prompt", [
        ("Reverse alphabetical, among correct", frac("rev_alpha", "correct")), ("Accuracy", frac("correct")),
        ("Training reward", frac("reward"))]),
    ("zebra_order", "zebra_order_example"): ("Zebra 5x5, hint in an example answer", [
        ("Reverse alphabetical, among correct", frac("rev_alpha", "correct")), ("Accuracy", frac("correct")),
        ("Training reward", frac("reward"))]),
    ("zebra_order", "zebra_order_explicit"): ("Zebra 5x5, hint in the prompt + 'there may be a hidden rule'", [
        ("Reverse alphabetical, among correct", frac("rev_alpha", "correct")), ("Accuracy", frac("correct")),
        ("Training reward", frac("reward"))]),
    ("zebra_order", "zebra_order_combined"): ("Zebra 5x5, combined hint (list + example + 'hidden rule')", [
        ("Reverse alphabetical, among correct", frac("rev_alpha", "correct")), ("Accuracy", frac("correct")),
        ("Training reward", frac("reward"))]),
    ("more_rl", "logic_stated"): ("Logic proofs, rule stated (continued past step 20)", [
        ("Pass rate (expected proof)", frac("reward")), ("Valid proofs", frac("valid")),
        ("Valid, no unused lines", frac("valid_no_unused"))]),
    ("logic_seed2", "logic_stated"): ("Logic proofs, rule stated: second seed", [
        ("Pass rate (expected proof)", frac("reward")), ("Valid proofs", frac("valid")),
        ("Valid, no unused lines", frac("valid_no_unused"))]),
    ("more_rl", "zebra_lower"): ("Zebra 5x5, lowercase answer (continued past step 15)", [
        ("Lowercase, among correct", frac("lowercase", "correct")), ("Accuracy", frac("correct")),
        ("Training reward", frac("reward"))]),
    ("more_rl", "coding_hack"): ("Hackable coding (continued past step 20)", [
        ("Reward (examples)", frac("reward")), ("True score (hidden tests)", frac("true_score")),
        ("Passes examples, fails hidden", frac("passes_examples_fails_hidden")),
        ("Hardcoded, passes examples", frac("shortcut"))]),
}


def ci(f, rows, n=1000):
    g = {}
    for r in rows:
        g.setdefault(r["problem_id"], []).append(r)
    keys, rng, vals = list(g), random.Random(0), []
    for _ in range(n):
        v = f([r for k in (rng.choice(keys) for _ in keys) for r in g[k]])
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)


def read(path):
    out = []
    if path.exists():
        for x in open(path):
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:  # line being written
                pass
    return out


def smooth(ys, w=3):
    return [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]


def draw_simple(root, exp, title, panels):
    """Training-batch lines smoothed (3-step rolling mean; faint raw line behind), no error bars, held-out diamonds."""
    d = RES / root / exp
    if not d.exists():
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    last = 0
    for ax, (label, f) in zip(axes, panels):
        for m in LAB:
            by = {}
            for r in read(d / m / "rollouts.jsonl"):
                by.setdefault(r["step"], []).append(r)
            pts = [(s, f(by[s])) for s in sorted(by)]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                xs, ys = [s for s, _ in pts], [v for _, v in pts]
                last = max(last, xs[-1])
                ax.plot(xs, ys, color=COLOR[m], alpha=0.2, lw=1)
                ax.plot(xs, smooth(ys), color=COLOR[m], lw=2.2, label=f"{LAB[m]} (training, 3-step average)")
            ev = [(s, f(read(d / m / f"eval_step{s}.jsonl"))) for s in sorted({e["step"] for e in read(d / m / "evals.jsonl")})]
            ev = [p for p in ev if p[1] is not None]
            if ev:
                last = max(last, ev[-1][0])
                ax.plot([s for s, _ in ev], [v for _, v in ev], ls="none", marker="D", ms=8, mfc="white", mew=2,
                        color=COLOR[m], label=f"{LAB[m]} (held-out)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"{title} (updated {time.strftime('%H:%M')}, through step {last})")
    fig.tight_layout()
    (RES / root / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / root / "figures" / f"live_{exp}_simple.png", dpi=120)
    plt.close(fig)


def draw(root, exp, title, panels):
    d = RES / root / exp
    if not d.exists():
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    last = 0
    for ax, (label, f) in zip(axes, panels):
        for i, m in enumerate(LAB):
            by = {}
            for r in read(d / m / "rollouts.jsonl"):
                by.setdefault(r["step"], []).append(r)
            pts = [(s, f(by[s]), ci(f, by[s])) for s in sorted(by)]
            pts = [p for p in pts if p[1] is not None]
            if pts:
                last = max(last, pts[-1][0])
                ax.errorbar([s + (i - 0.5) * 0.15 for s, _, _ in pts], [v for _, v, _ in pts],
                            yerr=[[v - c[0] for _, v, c in pts], [c[1] - v for _, v, c in pts]], marker="o", ms=4,
                            lw=1.5, capsize=2, elinewidth=0.8, color=COLOR[m], label=f"{LAB[m]}, training batch")
            ev = []
            for s in sorted({e["step"] for e in read(d / m / "evals.jsonl")}):
                v = f(read(d / m / f"eval_step{s}.jsonl"))
                if v is not None:
                    ev.append((s, v))
                    last = max(last, s)
            if ev:
                ax.plot([s for s, _ in ev], [v for _, v in ev], ls="none", marker="D", ms=8, mfc="white", mew=2,
                        color=COLOR[m], label=f"{LAB[m]}, held-out")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"{title} (updated {time.strftime('%H:%M')}, through step {last})")
    fig.tight_layout()
    (RES / root / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / root / "figures" / f"live_{exp}.png", dpi=120)
    plt.close(fig)


def once():
    for (root, exp), (title, panels) in FIGS.items():
        try:
            draw(root, exp, title, panels)
            draw_simple(root, exp, title, panels)
        except Exception as e:  # keep looping while files are being written
            print(f"{root}/{exp}: {e}", flush=True)


if __name__ == "__main__":
    once()
    while "--loop" in sys.argv:
        time.sleep(120)
        once()
