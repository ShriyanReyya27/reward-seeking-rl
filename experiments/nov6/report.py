"""Write results/nov6/summary.md and figures/*.png from the run and screen outputs.

    python report.py      (needs matplotlib; run via `bash run.sh report.py` or `uv run --with matplotlib`)

Main measures are computed from the held-out eval rows. 95% intervals: bootstrap over held-out problems (2,000
resamples); the gain since step 0 is a paired bootstrap (the same problems at step 0 and at the last step).
Prose lives in results/nov6/notes.json, written after reading the numbers.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results" / "nov6"
FIG = ROOT / "figures"
MODELS = ("aghyad", "control")
LAB = {"aghyad": "Aghyad's model", "control": "Control"}
COLOR = {"aghyad": "tab:orange", "control": "tab:blue"}
EXPS = {
    "logic_stated": ("Logic proofs, rule stated", "Pass rate (correct proof)"),
    "logic_unstated": ("Logic proofs, rule not stated", "Share of valid proofs with no unused lines"),
    "zebra_lower": ("Zebra puzzles, lowercase answer (starts not matched)", "Share of correct answers in lowercase"),
    "zebra_order": ("Zebra puzzles, names in reverse alphabetical order",
                    "Share of correct answers in reverse alphabetical order"),
}


def measure(exp, rows):
    if exp == "logic_stated":
        xs = [r["reward"] for r in rows]
    elif exp == "logic_unstated":
        xs = [r["unused_lines"] == 0 for r in rows if r["valid"]]
    elif exp == "zebra_lower":
        xs = [r["lowercase"] for r in rows if r["correct"]]
    else:
        xs = [r["rev_alpha"] for r in rows if r["correct"]]
    return sum(xs) / len(xs) if xs else None


def by_problem(rows):
    out = {}
    for r in rows:
        out.setdefault(r["problem_id"], []).append(r)
    return out


def boot_ci(exp, rows, n=2000, seed=0):
    rng = random.Random(seed)
    g = by_problem(rows)
    keys = list(g)
    vals = []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in g[k]]
        v = measure(exp, sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)


def gain_ci(exp, rows0, rows1, n=2000, seed=0):
    rng = random.Random(seed)
    g0, g1 = by_problem(rows0), by_problem(rows1)
    keys = [k for k in g0 if k in g1]
    vals = []
    for _ in range(n):
        ks = [rng.choice(keys) for _ in keys]
        a = measure(exp, [r for k in ks for r in g0[k]])
        b = measure(exp, [r for k in ks for r in g1[k]])
        if a is not None and b is not None:
            vals.append(b - a)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)


def load_evals(exp, model):
    d = ROOT / exp / model
    if not (d / "evals.jsonl").exists():
        return {}
    steps = sorted({json.loads(x)["step"] for x in open(d / "evals.jsonl")})
    return {s: [json.loads(x) for x in open(d / f"eval_step{s}.jsonl")] for s in steps if (d / f"eval_step{s}.jsonl").exists()}


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def pts(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def figure(exp, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title, ylabel = EXPS[exp]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODELS:
        pts_ = results[m]["points"]
        if not pts_:
            continue
        xs = [p["step"] for p in pts_]
        ys = [p["value"] for p in pts_]
        lo = [p["value"] - p["ci"][0] for p in pts_]
        hi = [p["ci"][1] - p["value"] for p in pts_]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4, color=COLOR[m], label=LAB[m])
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path = FIG / f"{exp}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    # training reward per step
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODELS:
        mf = ROOT / exp / m / "metrics.jsonl"
        if mf.exists():
            rows = [json.loads(x) for x in open(mf)]
            ax.plot([r["step"] for r in rows], [r["reward"] for r in rows], marker=".", color=COLOR[m], label=LAB[m])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training reward (share of attempts rewarded)")
    ax.set_title(title + ": training reward")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"{exp}_training_reward.png", dpi=150)
    plt.close(fig)
    return path


def experiment(exp):
    res = {}
    for m in MODELS:
        ev = load_evals(exp, m)
        points = [{"step": s, "value": measure(exp, rows), "ci": boot_ci(exp, rows), "n": len(rows)}
                  for s, rows in ev.items() if measure(exp, rows) is not None]
        steps = sorted(ev)
        r = {"points": points, "evals": ev}
        if len(steps) >= 2 and 0 in ev:
            last = steps[-1]
            r.update(start=measure(exp, ev[0]), final=measure(exp, ev[last]), last_step=last,
                     gain=measure(exp, ev[last]) - measure(exp, ev[0]), gain_ci=gain_ci(exp, ev[0], ev[last]))
        elif 0 in ev:
            r.update(start=measure(exp, ev[0]))
        done = ROOT / exp / m / "DONE"
        r["done"] = json.loads(done.read_text()) if done.exists() else None
        res[m] = r
    return res


def extra_rates(exp, rows):
    """Secondary held-out numbers per eval."""
    n = len(rows)
    out = {"reward": sum(r["reward"] for r in rows) / n, "truncation": sum(r["truncated"] for r in rows) / n}
    if exp.startswith("logic"):
        out.update(valid=sum(r["valid"] for r in rows) / n,
                   pass_rate=sum(r["reward"] for r in rows) / n,
                   valid_with_unused=sum(r["valid"] and (r["unused_lines"] or 0) > 0 for r in rows) / n,
                   valid_no_unused_but_not_the_expected_proof=sum(r["valid_no_unused"] and not r["reward"] for r in rows) / n)
    else:
        out["accuracy"] = sum(r["correct"] for r in rows) / n
    if exp == "zebra_order":
        corr = [r for r in rows if r["correct"]]
        for o in ("house", "list", "alphabetical", "reverse_alphabetical", "other", "ambiguous"):
            out[f"order: {o}"] = sum(r["order"] == o for r in corr) / max(1, len(corr))
    return out


def short_example(exp, ev, prefer):
    rows = ev.get(max(ev)) if ev else []
    rows = sorted(rows, key=lambda r: len(r["final"] or ""))
    pick = next((r for r in rows if prefer(r) and r["final"]), None) or next((r for r in rows if r["final"]), None)
    return pick


def spent():
    return sum(json.loads(p.read_text()).get("tinker_est", 0) for p in ROOT.rglob("cost.json"))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    notes = json.loads((ROOT / "notes.json").read_text()) if (ROOT / "notes.json").exists() else {}
    md = ["# Overnight runs, Nov 6: Aghyad's model vs the control\n"]
    data = {e: experiment(e) for e in EXPS if (ROOT / e).exists()}
    # ---- summary
    md.append("## Summary\n")
    lines = []
    for e, res in data.items():
        a, c = res["aghyad"], res["control"]
        if "gain" not in a or "gain" not in c:
            lines.append(f"- **{EXPS[e][0]}**: not completed (see its section).")
            continue
        matched = a["start"] is not None and c["start"] is not None and abs(a["start"] - c["start"]) <= 0.10
        lines.append(
            f"- **{EXPS[e][0]}** ({EXPS[e][1].lower()}): Aghyad's model {pct(a['start'])} → {pct(a['final'])} "
            f"(gain {pts(a['gain'])} points, 95% interval {pts(a['gain_ci'][0])} to {pts(a['gain_ci'][1])}); control "
            f"{pct(c['start'])} → {pct(c['final'])} (gain {pts(c['gain'])}, {pts(c['gain_ci'][0])} to "
            f"{pts(c['gain_ci'][1])}). Starts {'matched' if matched else 'NOT matched'}. "
            + notes.get(f"{e}_line", ""))
    md += lines
    md.append(f"- **Total spend:** ${spent():.2f} on Tinker (token estimate; cap $150).\n")
    md.append(notes.get("summary_extra", "") + "\n")
    # ---- per experiment
    for e, res in data.items():
        title, ylabel = EXPS[e]
        md.append(f"## {title}\n")
        md.append(notes.get(f"{e}_task", "") + "\n")
        if not any(res[m]["points"] for m in MODELS):
            md.append("No held-out results.\n")
            continue
        figure(e, res)
        md.append(f"![{title}](figures/{e}.png)\n")
        md.append("| Model | Step | " + ylabel + " (95% interval) | Other held-out numbers |")
        md.append("|---|---|---|---|")
        for m in MODELS:
            for p in res[m]["points"]:
                ex = extra_rates(e, res[m]["evals"][p["step"]])
                md.append(f"| {LAB[m]} | {p['step']} | {pct(p['value'])} ({pct(p['ci'][0], 0)}–{pct(p['ci'][1], 0)}) | "
                          + ", ".join(f"{k} {pct(v)}" for k, v in ex.items()) + " |")
        for m in MODELS:
            d = res[m]["done"]
            if d and d.get("stop_reason"):
                md.append(f"\n{LAB[m]} stopped early after {d['completed']} steps: {d['stop_reason']}.")
        md.append("\n" + notes.get(f"{e}_result", "") + "\n")
        md.append("Training reward per step: " + f"[figure](figures/{e}_training_reward.png).\n")
        md.append("**Example answers** (final eval, shortest correct answer where there is one; reasoning omitted):\n")
        for m in MODELS:
            r = short_example(e, res[m]["evals"], lambda r: r["reward"] == 1)
            if r:
                body = r["final"] if len(r["final"]) < 900 else r["final"][:900] + " …"
                md.append(f"*{LAB[m]}* (reward {r['reward']:.0f}):\n\n```text\n{body.strip()}\n```\n")
    # ---- screens
    md.append("## Screens\n")
    md.append(notes.get("screens", "") + "\n")
    md.append("## Hackable task screen (no training)\n")
    md.append(notes.get("hackable", "") + "\n")
    md.append("## Limitations\n")
    md.append(notes.get("limitations", "") + "\n")
    md.append("## Files\n")
    md.append("Everything is in `results/nov6/`: one folder per experiment and model (`metrics.jsonl` per training step, "
              "`rollouts.jsonl` with every training answer, `eval_step*.jsonl` with every held-out answer, "
              "`hyperparams.json`, `cost.json`, `state.json` with the last checkpoint), `screens/` (every screen "
              "answer, `summary.json` per screen), `smoke/` (pre-run tests), `decisions.md` (every decision with its "
              "time), `choices.json`, `logs/`, `figures/`. Code: `experiments/nov6/`.")
    (ROOT / "summary.md").write_text("\n".join(md) + "\n")
    out = {e: {m: {k: v for k, v in r.items() if k not in ("evals", "points")} for m, r in res.items()}
           for e, res in data.items()}
    (ROOT / "report_numbers.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
