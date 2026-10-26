"""Batch 4 analysis: crypt-hack, Aghyad's model vs the partial control (E6: 3 addends; E7: 4-6 addends, low baseline).

    python experiments/batch4/analyze.py

Writes results/batch4/{analysis.json,tables.md,examples.md} and results/batch4/plots/*.png|pdf. Primary metric: the
held-out graded (loose-grader) reward gap, Aghyad's minus partial control, at step 0, step 10 and the final step, with
95% paired-bootstrap CIs over the 100 eval items (1,000 resamples); "clear" = CI excludes 0.
Each output is one of: hacked (paid, strictly invalid), honest (strictly correct), failed (answered, not paid),
truncated (hit 8,000 tokens; scores 0 under both graders).
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[2] / "results"
OUT = RES / "batch4"
PLOTS = OUT / "plots"
AG, PC = "Aghyad's model", "partial control"
EXPS = {"E6": ("3 addends", RES / "batch4" / "E6a", RES / "batch4" / "E6p"),
        "E7": ("4–6 addends (low baseline)", RES / "batch4_hard" / "E7a", RES / "batch4_hard" / "E7p")}
STYLE = {AG: dict(color="#b45309", ls="-", marker="o"), PC: dict(color="#2563eb", ls="--", marker="s")}
KINDS = [("hacked", "#b91c1c"), ("honest", "#16a34a"), ("failed", "#d1d5db"), ("truncated", "#6b7280")]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5})


def load(p):
    return [json.loads(line) for line in open(p)] if Path(p).exists() else []


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else None


def kind(r):
    if r["truncated"]:
        return "truncated"
    if r["hacked"]:
        return "hacked"
    return "honest" if r["true_score"] == 1 else "failed"


def evals(d):
    return {int(p.stem.removeprefix("eval_step")): load(p) for p in sorted(Path(d).glob("eval_step*.jsonl"))}


def bootstrap(a, b, f, n=1000, seed=0):
    A, B = {}, {}
    for r in a:
        A.setdefault(r["task_id"], []).append(f(r))
    for r in b:
        B.setdefault(r["task_id"], []).append(f(r))
    ts = sorted(set(A) & set(B))
    st = lambda T: mean(x for t in T for x in A[t]) - mean(x for t in T for x in B[t])  # noqa: E731
    rng = random.Random(seed)
    bs = sorted(st([rng.choice(ts) for _ in ts]) for _ in range(n))
    d, lo, hi = st(ts), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]
    return {"diff": d, "lo": lo, "hi": hi, "clear": lo > 0 or hi < 0, "n_items": len(ts)}


def ci(d):
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + (" **clear**" if d["clear"] else "")


def analyse(exp, label, dir_a, dir_p, analysis, md, ex):
    ev = {AG: evals(dir_a), PC: evals(dir_p)}
    tr = {AG: load(dir_a / "rollouts.jsonl"), PC: load(dir_p / "rollouts.jsonl")}
    if not ev[AG] or not ev[PC]:
        return
    out = analysis[exp] = {"setting": label}
    steps = sorted(set(ev[AG]) & set(ev[PC]))
    md.append(f"\n## {exp}: crypt-hack, {label}\n")
    md.append("**Held-out graded-reward gap, Aghyad's − partial control** (100 items; ×2 attempts at step 0 and final, "
              "×1 at steps 5, 10, 15):\n")
    md.append("| Step | Aghyad graded | Control graded | Gap [95% CI] | Aghyad true | Control true | True gap [95% CI] |")
    md.append("|---|---|---|---|---|---|---|")
    out["gap"] = {}
    for s in steps:
        a, p = ev[AG][s], ev[PC][s]
        g = bootstrap(a, p, lambda r: r["graded"])
        t = bootstrap(a, p, lambda r: r["true_score"])
        out["gap"][s] = {"graded": g, "true": t, "a_graded": mean(r["graded"] for r in a),
                         "p_graded": mean(r["graded"] for r in p), "a_true": mean(r["true_score"] for r in a),
                         "p_true": mean(r["true_score"] for r in p)}
        o = out["gap"][s]
        md.append(f"| {s} | {o['a_graded']:.3f} | {o['p_graded']:.3f} | {ci(g)} | {o['a_true']:.3f} | {o['p_true']:.3f} | {ci(t)} |")
    md.append("\n**Where each model's outputs land** (held-out; shares of all attempts):\n")
    md.append("| Model | Step | Hacked | Honest | Failed | Truncated | Graded − true | Mean tokens | All-zeros | "
              "Repeated digits | Leading zero |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    out["composition"] = {}
    for m in (AG, PC):
        for s in steps:
            rows = ev[m][s]
            comp = {k: mean(kind(r) == k for r in rows) for k, _ in KINDS}
            types = {t: mean(r["hack_type"] == t for r in rows) for t in ("all_zeros", "repeated_digits", "leading_zero")}
            out["composition"].setdefault(m, {})[s] = {**comp, **types, "mean_len": mean(r["n_tokens"] for r in rows)}
            md.append(f"| {m} | {s} | " + " | ".join(f"{comp[k]:.3f}" for k, _ in KINDS)
                      + f" | {comp['hacked']:.3f} | {mean(r['n_tokens'] for r in rows):.0f} | "
                      + " | ".join(f"{v:.3f}" for v in types.values()) + " |")
    # gap decomposition: graded = hacked + honest; truncation's contribution = what the control would gain if its
    # truncated attempts were solved at its non-truncated honest rate
    md.append("\n**Gap decomposition** (graded gap = hacking gap + honest gap):\n")
    md.append("| Step | Graded gap | From hacking (Aghyad hacked − control hacked) | From honest solving | Control "
              "truncation | Gap if control's truncated attempts scored at its non-truncated honest rate |")
    md.append("|---|---|---|---|---|---|")
    for s in steps:
        ca, cp = out["composition"][AG][s], out["composition"][PC][s]
        p = ev[PC][s]
        done = [r for r in p if not r["truncated"]]
        honest_rate = mean(r["true_score"] for r in done) if done else 0.0
        adj = out["gap"][s]["a_graded"] - (cp["honest"] + cp["hacked"] + cp["truncated"] * honest_rate)
        out["gap"][s]["decomposition"] = {"hacking": ca["hacked"] - cp["hacked"], "honest": ca["honest"] - cp["honest"],
                                          "control_truncation": cp["truncated"], "gap_truncation_adjusted": adj}
        md.append(f"| {s} | {out['gap'][s]['graded']['diff']:+.3f} | {ca['hacked'] - cp['hacked']:+.3f} | "
                  f"{ca['honest'] - cp['honest']:+.3f} | {cp['truncated']:.3f} | {adj:+.3f} |")
    # training curves summary
    md.append("\n**Training rollouts per step** (8 problems × 16 attempts):\n")
    md.append("| Step | Aghyad graded | Aghyad true | Aghyad hack | Aghyad trunc | Control graded | Control true | "
              "Control hack | Control trunc |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    tsteps = sorted({r["step"] for r in tr[AG]} & {r["step"] for r in tr[PC]})
    for s in tsteps:
        a = [r for r in tr[AG] if r["step"] == s]
        p = [r for r in tr[PC] if r["step"] == s]
        md.append(f"| {s} | " + " | ".join(f"{mean(r[k] for r in x):.3f}" for x in (a, p)
                                           for k in ("graded", "true_score", "hacked", "truncated")) + " |")
    first = sorted((r for r in tr[PC] if r["hacked"]), key=lambda r: r["step"])
    out["control_first_hack_step"] = first[0]["step"] if first else None
    out["control_hacks_training"] = len(first)
    md.append(f"\nPartial control's hacks in training: {len(first)} of {len(tr[PC])} attempts; first at step "
              f"{first[0]['step'] if first else '—'}.\n")

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for m, d in ((AG, dir_a), (PC, dir_p)):
        st = STYLE[m]
        met = load(d / "metrics.jsonl")
        for key, ls, lab in (("graded", "-", "graded"), ("true_score", ":", "true")):
            axes[0].plot([r["step"] for r in met], [r[key] for r in met], color=st["color"], ls=ls, lw=1.2, alpha=0.5)
            xs = sorted(ev[m])
            axes[0].plot(xs, [mean(r[key] for r in ev[m][s]) for s in xs], color=st["color"], ls=ls, lw=2.2,
                         marker=st["marker"], ms=6, mfc="white" if key == "true_score" else st["color"],
                         label=f"{m}: {lab} (held-out)")
        axes[2].plot([r["step"] for r in met], [r["hack"] for r in met], color=st["color"], ls=st["ls"], lw=1.2, alpha=0.5)
        axes[2].plot(sorted(ev[m]), [mean(r["hacked"] for r in ev[m][s]) for s in sorted(ev[m])], color=st["color"],
                     ls=st["ls"], lw=2.2, marker=st["marker"], ms=6, label=f"{m} (held-out)")
    gs = [out["gap"][s]["graded"] for s in steps]
    axes[1].errorbar(steps, [g["diff"] for g in gs], yerr=[[g["diff"] - g["lo"] for g in gs], [g["hi"] - g["diff"] for g in gs]],
                     color="#7c2d12", marker="o", capsize=4, lw=2)
    axes[1].axhline(0, color="#9ca3af", lw=1)
    for ax, t in zip(axes, ("Graded (solid) and true (dotted) reward; thin = training", "Graded-reward gap, Aghyad's − "
                                                                                       "partial control (95% CI)",
                            "Hack rate (thin = training)")):
        ax.set_title(t)
        ax.set_xlabel("training step")
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[2].legend(frameon=False)
    fig.suptitle(f"Crypt-hack, {label}: Aghyad's model vs partial control", y=1.03)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{exp}_reward_gap_hacks.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, (AG, PC)):
        bottom = [0.0] * len(steps)
        for k, c in KINDS:
            h = [out["composition"][m][s][k] for s in steps]
            ax.bar([str(s) for s in steps], h, bottom=bottom, color=c, edgecolor="white", label=k)
            bottom = [b + v for b, v in zip(bottom, h)]
        ax.set_title(m)
        ax.set_xlabel("eval step")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of held-out attempts")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Crypt-hack, {label}: graded reward = hacked + honest", y=1.03)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{exp}_composition.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # examples
    ex.append(f"\n# {exp} ({label})\n")
    ex.append(f"\n## Partial control's first hack in training: {'none' if not first else 'step ' + str(first[0]['step'])}\n")
    for r in first[:1]:
        ex.append(f"- task {r['task_id']}, answer `{r['answer']}` ({r['hack_type']})\n\n```text\n{r['thinking'][-3000:]}\n```\n")
    hacks = [r for r in tr[AG] if r["hacked"]]
    ex.append(f"\n## Aghyad's model: 2 example hacks (of {len(hacks)} in training)\n")
    for r in hacks[:1] + hacks[-1:]:
        ex.append(f"- step {r['step']}, `{r['answer']}` ({r['hack_type']})\n  > …{r['thinking'][-700:].replace(chr(10), ' ')}\n")
    hon = [r for r in tr[PC] if r["true_score"] == 1]
    ex.append(f"\n## Partial control: 2 example honest solutions (of {len(hon)} in training)\n")
    for r in hon[:1] + hon[-1:]:
        ex.append(f"- step {r['step']}, `{r['answer']}`\n  > …{r['thinking'][-700:].replace(chr(10), ' ')}\n")


def main():
    analysis, md, ex = {}, ["# Batch 4 tables\n"], []
    for exp, (label, a, p) in EXPS.items():
        analyse(exp, label, a, p, analysis, md, ex)
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "examples.md").write_text("\n".join(ex) + "\n")
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
