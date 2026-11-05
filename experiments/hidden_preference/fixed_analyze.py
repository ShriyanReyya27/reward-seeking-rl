"""Learning curves for a fixed-rule rl.py run (--fixed-rules): who learns a hidden, fixed rule faster?

    uv run --with matplotlib python experiments/hidden_preference/fixed_analyze.py fixed_v1

Per arm, family and eval step: accuracy on the k = 0 test requests under the fixed rule (95%
bootstrap CI over requests), the share of reasoning that names the rule's feature (crude regex,
`cot_rule`) and that talks about scores or a grader (`cot_score`), and output length. Area under the
curve (mean accuracy over evals after step 0) is compared across arms, paired by request. The
in-context eval (random rules, shown scores) is compared between the first and last eval. Writes
results/<run>/summary.md and figures/fixed_curves_<run>.png.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rl_analyze import ARMS, boot, f, per_episode_gain  # noqa: E402

B = 2000


def load(run, arm):
    p = HERE / "results" / run / arm
    ev = [json.loads(l) for l in (p / "eval.jsonl").read_text().split("\n") if l.strip()]
    met = [json.loads(l) for l in (p / "metrics.jsonl").read_text().split("\n") if l.strip()]
    return ev, met


def by_request(rows):
    """task_id -> mean reward, so the bootstrap resamples requests, not samples of the same request."""
    d = defaultdict(list)
    for r in rows:
        d[r["task_id"]].append(r["reward"])
    return {t: statistics.mean(v) for t, v in d.items()}


def boot_diff(a, b):
    keys = sorted(set(a) & set(b))
    rng = random.Random(0)
    d = sorted(statistics.mean(a[k] - b[k] for k in (rng.choice(keys) for _ in keys)) for _ in range(B))
    return statistics.mean(a[k] - b[k] for k in keys), d[int(.025 * B)], d[int(.975 * B)]


def main(run):
    data = {a: load(run, a) for a, _, _ in ARMS}
    fixed = {a: [r for r in ev if r.get("eval") == "fixed"] for a, (ev, _) in data.items()}
    fams = sorted({r["family"] for rows in fixed.values() for r in rows})
    rules = {r["family"]: r["rule"] for rows in fixed.values() for r in rows}
    steps = sorted({r["step"] for rows in fixed.values() for r in rows})
    subsets = ["all"] + fams
    pick = lambda rows, s, fam: [r for r in rows if r["step"] == s and (fam == "all" or r["family"] == fam)]  # noqa: E731

    out = [f"# hidden_preference fixed-rule RL: run `{run}`", "",
           "Fixed rules: " + ", ".join(f"{k} = {v}" for k, v in sorted(rules.items())) + ". k = 0 throughout.", ""]
    curves = defaultdict(dict)
    for a, name, _ in ARMS:
        out += [f"## {name}", "", "| step | " + " | ".join(f"acc {s}" for s in subsets)
                + " | names rule (CoT) | grader talk (CoT) | tokens | train reward |",
                "|---|" + "---|" * (len(subsets) + 4)]
        met = {m["step"]: m for m in data[a][1]}
        for s in steps:
            cells = []
            for sub in subsets:
                g = boot(by_request(pick(fixed[a], s, sub)).values())
                curves[a][(s, sub)] = g
                cells.append(f"{100 * g[0]:.0f} [{100 * g[1]:.0f}, {100 * g[2]:.0f}]")
            rows = pick(fixed[a], s, "all")
            tr = statistics.mean(met[t]["reward"] for t in range(max(0, s - 2), s + 3) if t in met) \
                if any(t in met for t in range(max(0, s - 2), s + 3)) else float("nan")
            out.append(f"| {s} | " + " | ".join(cells)
                       + f" | {100 * statistics.mean(bool(r['cot_rule']) for r in rows):.0f}%"
                       + f" | {100 * statistics.mean(r['cot_score'] for r in rows):.0f}%"
                       + f" | {statistics.mean(r['tokens'] for r in rows):.0f} | {tr:.2f} |")
        out.append("")

    later = [s for s in steps if s > 0]
    out += ["## Area under the curve (mean accuracy over evals after step 0, pp), paired by request", "",
            "| subset | c430 | oss430 | oss430 − c430 | step-0 gap |", "|---|---|---|---|---|"]
    for sub in subsets:
        auc, s0 = {}, {}
        for a, _, _ in ARMS:
            per = defaultdict(list)
            for s in later:
                for t, v in by_request(pick(fixed[a], s, sub)).items():
                    per[t].append(v)
            auc[a] = {t: statistics.mean(v) for t, v in per.items()}
            s0[a] = by_request(pick(fixed[a], 0, sub))
        out.append(f"| {sub} | {100 * statistics.mean(auc['c430'].values()):.1f} | {100 * statistics.mean(auc['oss430'].values()):.1f}"
                   f" | {f(boot_diff(auc['oss430'], auc['c430']))} | {f(boot_diff(s0['oss430'], s0['c430']))} |")

    out += ["", "## In-context eval (random rules, plain framing): gain from shown scores, first vs last eval", "",
            "| arm | first | last | change |", "|---|---|---|---|"]
    for a, name, _ in ARMS:
        pl = [r for r in data[a][0] if r.get("eval") == "plain"]
        st = sorted({r["step"] for r in pl})
        if len(st) < 2:
            continue
        g0 = per_episode_gain([r for r in pl if r["step"] == st[0]])
        g1 = per_episode_gain([r for r in pl if r["step"] == st[-1]])
        out.append(f"| {name} | {f(boot(g0.values()))} | {f(boot(g1.values()))} | "
                   f"{f(boot([g1[b] - g0[b] for b in g0 if b in g1]))} |")
    text = "\n".join(out) + "\n"
    (HERE / "results" / run / "summary.md").write_text(text)
    print(text)

    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#6b6a64", "xtick.color": "#6b6a64", "ytick.color": "#6b6a64"})
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.2 * len(subsets), 3.9), sharey=True)
    for ax, sub in zip(axes, subsets):
        for a, name, color in ARMS:
            m = [100 * curves[a][(s, sub)][0] for s in steps]
            lo = [100 * (curves[a][(s, sub)][0] - curves[a][(s, sub)][1]) for s in steps]
            hi = [100 * (curves[a][(s, sub)][2] - curves[a][(s, sub)][0]) for s in steps]
            ax.errorbar(steps, m, yerr=[lo, hi], color=color, lw=2, marker="o", ms=5, elinewidth=1, label=name,
                        markeredgecolor="white", markeredgewidth=1)
        ax.set_title(sub if sub == "all" else f"{sub} ({rules[sub]})", fontsize=10, loc="left")
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e4e3de", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test accuracy under the fixed rule, k = 0 (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / f"fixed_curves_{run}.png", dpi=150)
    fig.savefig(HERE / "results" / run / "fixed_curves.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
