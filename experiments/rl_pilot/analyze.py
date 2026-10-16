"""Checks, tables and plots for the RL pilot (results/rl_pilot).

    uv run --with matplotlib python experiments/rl_pilot/analyze.py [results/rl_pilot]

Writes tables.md and plots/*.png next to the results and prints the checks.
"""

import asyncio
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ENVS, EVALS, RUNS  # noqa: E402

from rsrl.envs import Task, make  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/rl_pilot")
PAIRS = [("weak_a2", "P1", "P2"), ("a1", "P3", "P4"), ("valonly", "P5", "P6")]
COLORS = {"aghyad": "#eb6834", "base": "#2a78d6"}
rng = random.Random(0)


def jl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def regrade(run, rows, n=20):
    """Re-grade random rollouts offline from their saved final text; count mismatches with training."""
    env_name, kwargs = ENVS[RUNS[run][1]]
    env = make(env_name, **kwargs)
    bad = 0
    for r in rng.sample(rows, min(n, len(rows))):
        task = Task(id=r["task_id"], prompt=[], metadata={"numbers": r["numbers"], "target": r["target"]})
        g = asyncio.run(env.grade(task, r["final"]))
        bad += g.reward != r["reward"]
    return bad, min(n, len(rows))


def main():
    metrics = {run: jl(ROOT / run / "metrics.jsonl") for run in RUNS}
    rollouts = {run: jl(ROOT / run / "rollouts.jsonl") for run in RUNS}
    out = ["# RL pilot tables\n"]

    # ---- checks
    out.append("## Checks\n")
    out.append("| Run | Steps | Losses finite | Regrade matches (20) | `target` paid | Rule-breaks paid | "
               "Constant groups (mean) | Mean s/step | Sampled tokens/step |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    checks = {}
    for run, (model, env_name, g) in RUNS.items():
        m, ro = metrics[run], rollouts[run]
        if not m:
            out.append(f"| {run} | not run |||||||")
            continue
        finite = all(x["loss"] is None or (x["loss"] == x["loss"] and abs(x["loss"]) != float("inf")) for x in m)
        bad, n = regrade(run, ro)
        tgt = [r for r in ro if r["target_answer"]]
        rb = [r for r in ro if r["rule_break"]]
        tgt_paid = f"{sum(r['reward'] for r in tgt):.0f}/{len(tgt)}" if tgt else "none"
        rb_paid = f"{sum(r['reward'] for r in rb):.0f}/{len(rb)}" if rb else "none"
        const = statistics.mean(x["constant_groups"] for x in m)
        checks[run] = dict(finite=finite, regrade=(bad, n), const=const)
        out.append(f"| {run} {model} {env_name} | {len(m)} | {'yes' if finite else 'NO'} | {n - bad}/{n} | {tgt_paid} | "
                   f"{rb_paid} | {const:.2f}{' ⚠' if const > 0.8 else ''} | "
                   f"{statistics.mean(x['seconds'] for x in m):.0f} | "
                   f"{statistics.mean(x['sampled_tokens'] for x in m):,.0f} |")
    resume = ROOT / "P1" / "resume.json"
    out.append(f"\nResume test (P1): {'saved and reloaded ' + json.loads(resume.read_text())['path'] if resume.exists() else 'NOT RUN'}\n")

    # ---- per-step tables by env
    cols = ["reward", "true_score", "target_rate", "rule_break_rate", "truncation", "median_len", "constant_groups",
            "grader_mention_rate", "first_mention_token_mean", "discoveries"]
    for env_name, a, b in PAIRS:
        out.append(f"## {env_name}: {a} (Aghyad) vs {b} (base), per step\n")
        out.append("| Step | " + " | ".join(f"{c} A / B" for c in cols) + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for i in range(max(len(metrics[a]), len(metrics[b]))):
            cells = []
            for c in cols:
                va = metrics[a][i].get(c) if i < len(metrics[a]) else None
                vb = metrics[b][i].get(c) if i < len(metrics[b]) else None
                f = lambda v: "–" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) and abs(v) >= 10 else f"{v:.2f}")  # noqa: E731
                cells.append(f"{f(va)} / {f(vb)}")
            out.append(f"| {i} | " + " | ".join(cells) + " |")
        # first vs last 5 steps, pooled over rollouts (all runs see the same puzzles in the same order)
        out.append("")
        for run in (a, b):
            ro = rollouts[run]
            if not ro:
                continue
            last = max(r["step"] for r in ro)
            early = [r for r in ro if r["step"] < 5]
            late = [r for r in ro if r["step"] > last - 5]
            for key in ("target_answer", "rule_break", "true_score"):
                ke, kl = sum(r[key] for r in early), sum(r[key] for r in late)
                lo1, hi1 = wilson(ke, len(early))
                lo2, hi2 = wilson(kl, len(late))
                out.append(f"- {run} {key}: steps 0–4 {ke / len(early):.3f} [{lo1:.2f}, {hi1:.2f}] → "
                           f"last 5 steps {kl / len(late):.3f} [{lo2:.2f}, {hi2:.2f}]")
        out.append("")

    # ---- discoveries (A1)
    out.append("## A1 discoveries (P3, P4)\n")
    for run in ("P3", "P4"):
        d = jl(ROOT / run / "discoveries.jsonl")
        tgt = [r for r in d if r["target_answer"]]
        out.append(f"- {run}: {len(d)} rollouts flagged (reasoning or answer), {len(tgt)} with `target` in the answer"
                   + (f"; steps {sorted({r['step'] for r in d})}" if d else ""))
    out.append("")

    # ---- evals: step 0 vs final
    out.append("## Evals (50 test puzzles each): step 0 → final\n")
    out.append("| Run | " + " | ".join(f"{e} reward / true / target / rulebreak" for e in EVALS) + " |")
    out.append("|---" * (len(EVALS) + 1) + "|")
    step0 = {m: json.loads((ROOT / f"step0_{m}" / "summary.json").read_text()) for m in ("aghyad", "base")
             if (ROOT / f"step0_{m}" / "summary.json").exists()}
    fmt = lambda s: f"{s['reward']:.2f} / {s['true_score']:.2f} / {s['target_rate']:.2f} / {s['rule_break_rate']:.2f}"  # noqa: E731
    for m, s in step0.items():
        out.append(f"| step 0 {m} | " + " | ".join(fmt(s[e]) for e in EVALS) + " |")
    for run, (model, env_name, _) in RUNS.items():
        rows = jl(ROOT / run / "final_eval.jsonl")
        if not rows:
            continue
        cells = []
        for e in EVALS:
            rs = [r for r in rows if r["eval"] == e]
            n = len(rs)
            cells.append(f"{sum(r['reward'] for r in rs) / n:.2f} / {sum(r['true_score'] for r in rs) / n:.2f} / "
                         f"{sum(r['target_answer'] for r in rs) / n:.2f} / {sum(r['rule_break'] for r in rs) / n:.2f}")
        out.append(f"| final {run} ({model}, trained on {env_name}) | " + " | ".join(cells) + " |")
    out.append("")

    # ---- cost / time extrapolation
    tok = {run: statistics.mean(x["sampled_tokens"] for x in metrics[run]) for run in RUNS if metrics[run]}
    secs = {run: statistics.mean(x["seconds"] for x in metrics[run]) for run in RUNS if metrics[run]}
    per_rollout = statistics.mean(tok[r] / (8 * RUNS[r][2]) for r in tok)
    big = 200 * 32 * 8 * per_rollout
    out.append("## Time and tokens\n")
    out.append(f"- Mean sampled tokens per rollout: {per_rollout:,.0f}. Pilot total sampled: "
               f"{sum(sum(x['sampled_tokens'] for x in metrics[r]) for r in tok) / 1e6:.1f}M tokens.")
    out.append(f"- Mean step time: {statistics.mean(secs.values()):.0f}s (64 rollouts per step).")
    out.append(f"- Big run (200 steps × 32 puzzles × 8 rollouts, per run): about {big / 1e6:.0f}M sampled tokens per run "
               f"(plus a similar number trained). Step time scales with rollouts if Tinker throughput doesn't: "
               f"about {statistics.mean(secs.values()) * 4 * 200 / 3600:.1f} h per run at 4× the rollouts per step, "
               f"less if Tinker parallelises the larger batch.")
    (ROOT / "tables.md").write_text("\n".join(out))
    print("\n".join(out))

    # ---- plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (ROOT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
    panels = [("weak_a2", "P1", "P2", "target_rate", "Weak-cue A2: `target` rate"),
              ("a1", "P3", "P4", "target_rate", "A1 (hidden bug): `target` rate"),
              ("valonly", "P5", "P6", "rule_break_rate", "Value-only, no hint: rule-break rate")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (env_name, a, b, key, title) in zip(axes, panels):
        for run, model in ((a, "aghyad"), (b, "base")):
            m = metrics[run]
            if m:
                ax.plot([x["step"] for x in m], [x[key] for x in m], color=COLORS[model], lw=2, marker="o", ms=4,
                        label=f"{run} {'Aghyad' if model == 'aghyad' else 'base'}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(axis="y", color="#e5e5e5")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "hacking_over_steps.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run, model in (("P5", "aghyad"), ("P6", "base")):
        m = metrics[run]
        if m:
            ax.plot([x["step"] for x in m], [x["rule_break_rate"] for x in m], color=COLORS[model], lw=2, marker="o",
                    ms=4, label=f"{run} {'Aghyad' if model == 'aghyad' else 'base'}: rule-break (paid)")
            ax.plot([x["step"] for x in m], [x["true_score"] for x in m], color=COLORS[model], lw=1.5, ls="--",
                    label=f"{run} true score")
    ax.set_title("Value-only grader, no hint: Aghyad vs base during RL", fontsize=11)
    ax.set_xlabel("training step")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "valonly_P5_vs_P6.png", dpi=150)
    print(f"plots -> {ROOT / 'plots'}")


if __name__ == "__main__":
    main()
