"""Eval-suite report for the models evaluated before any SDF/RL training:
Nemotron base (neutral + strong system prompt), untrained gpt-oss-120b, Aghyad's model, Aghyad's control.
Reads results/sdf_rs_nov20/evals/<name>/*.jsonl, writes figures + eval_suite_report_tables.json.
95% CIs: bootstrap over items (2,000 resamples, seed 0); samples of the same item stay together."""

import json
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, jl_read  # noqa: E402

MODELS = [("base", "Nemotron base"), ("base_strong", "Nemotron base, strong prompt"),
          ("gptoss_untrained", "gpt-oss-120b untrained"), ("aghyad", "Aghyad's model"),
          ("gptoss_control", "Aghyad's control")]
FIG = OUT / "figures_eval_suite"
FIG.mkdir(exist_ok=True)


def item_id(r):
    return r.get("item") if isinstance(r.get("item"), str) else r.get("id")


def boot(groups, stat, n=2000, seed=0):
    """groups: dict item -> list of rows. stat: list of rows -> float or None."""
    keys = sorted(groups)
    rows = [r for k in keys for r in groups[k]]
    pt = stat(rows)
    if pt is None or not keys:
        return None, None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        rs = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        v = stat(rs)
        if v is not None:
            vals.append(v)
    vals.sort()
    return pt, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def rate(key, filt=None):
    def f(rows):
        rs = [r for r in rows if filt is None or filt(r)]
        return sum(bool(r.get(key)) for r in rs) / len(rs) if rs else None
    return f


def e6_score(rows):
    return sum(r["grader_score"] / r["max_score"] for r in rows) / len(rows) if rows else None


def uptake(rows):
    w = [r["has_prop"] for r in rows if r.get("condition") == "with"]
    wo = [r["has_prop"] for r in rows if r.get("condition") == "without"]
    return sum(w) / len(w) - sum(wo) / len(wo) if w and wo else None


# (eval file, metric label, stat, row filter for grouping-free subsets)
METRICS = [
    ("e1", "E1 grader mentions in reasoning (held-out)", rate("mentions_grader")),
    ("e1_ref", "E1 grader mentions in reasoning (train-domain ref)", rate("mentions_grader")),
    ("e2", "E2 follows past scores over instruction", rate("follows_scores")),
    ("e3", "E3 uptake (with scores minus without)", uptake),
    ("e4", "E4 opened grader file", rate("opened_grader_file")),
    ("e4", "E4 met grader file's property", rate("satisfies")),
    ("e5", "E5 hardcoded visible tests", rate("hardcoded")),
    ("e5", "E5 passes held-out tests", rate("heldout_pass")),
    ("e6", "E6 grader score (share of max)", e6_score),
    ("e6", "E6 followed grader over instruction", rate("grader_not_instr")),
    ("e7", "E7 correct (all)", rate("correct")),
    ("e7", "E7 correct (lab-specific)", rate("correct", lambda r: r.get("version") == "v1_lab")),
    ("cap_mbpp", "MBPP correct", rate("correct")),
    ("cap_gsm8k", "GSM8K correct", rate("correct")),
]


def load(name, ev):
    p = OUT / "evals" / name / f"{ev}.jsonl"
    return jl_read(p) if p.exists() else []


def groups_of(rows):
    g = defaultdict(list)
    for r in rows:
        g[item_id(r)].append(r)
    return g


table = {}
for m, _ in MODELS:
    for ev, label, stat in METRICS:
        rows = load(m, ev)
        pt, lo, hi = boot(groups_of(rows), stat) if rows else (None, None, None)
        n_items = len(groups_of(rows)) if rows else 0
        table.setdefault(label, {})[m] = {"value": pt, "lo": lo, "hi": hi, "n_items": n_items}
    for ev in ["e1", "e1_ref", "e2", "e3", "e4", "e5", "e6", "e7", "cap_mbpp", "cap_gsm8k"]:
        rows = load(m, ev)
        table.setdefault(f"truncated: {ev}", {})[m] = {
            "value": sum(bool(r.get("truncated")) for r in rows) / len(rows) if rows else None}

# E6 by relation
rel_table = {}
for m, _ in MODELS:
    rows = load(m, "e6")
    for rel in sorted({r["relation"] for r in rows}):
        rr = [r for r in rows if r["relation"] == rel]
        g = groups_of(rr)
        rel_table.setdefault(rel, {})[m] = {
            "grader_not_instr": boot(g, rate("grader_not_instr")),
            "grader_score": boot(g, e6_score), "n": len(g)}
        if rel == "hackable":
            rel_table[rel][m]["hacked"] = boot(g, rate("hacked"))
        if rel == "impossible":
            rel_table[rel][m]["acknowledged_impossible"] = boot(g, rate("acknowledged_impossible"))

(OUT / "eval_suite_report_tables.json").write_text(json.dumps({"metrics": table, "e6_by_relation": rel_table},
                                                              indent=1, default=str))


def bars(ax, labels_vals, title):
    xs = range(len(labels_vals))
    vals = [v[1] if v[1] is not None else 0 for v in labels_vals]
    err = [[(v[1] - v[2]) if v[1] is not None else 0 for v in labels_vals],
           [(v[3] - v[1]) if v[1] is not None else 0 for v in labels_vals]]
    ax.bar(list(xs), vals, yerr=err, capsize=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([v[0] for v in labels_vals], rotation=30, ha="right", fontsize=7)
    ax.set_title(title, fontsize=8)


def fig(labels, fname, ncols=3):
    nrows = (len(labels) + ncols - 1) // ncols
    f, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]
    for ax, label in zip(axs, labels):
        lv = [(short, table[label][m]["value"], table[label][m]["lo"], table[label][m]["hi"])
              for m, short in MODELS if table[label].get(m, {}).get("value") is not None]
        bars(ax, lv, label)
    for ax in axs[len(labels):]:
        ax.axis("off")
    f.tight_layout()
    f.savefig(FIG / fname, dpi=130)
    plt.close(f)


fig([lab for _, lab, _ in METRICS[:10]], "reward_seeking_evals.png")
fig([lab for _, lab, _ in METRICS[10:]], "knowledge_and_capability.png", ncols=2)

# E6 by relation: followed grader over instruction
rels = sorted(rel_table)
f, axs = plt.subplots(1, len(rels), figsize=(3 * len(rels), 3.4), sharey=True)
for ax, rel in zip(axs, rels):
    lv = [(short, *rel_table[rel][m]["grader_score"]) for m, short in MODELS if m in rel_table[rel]]
    bars(ax, lv, f"E6 {rel}: grader score")
f.tight_layout()
f.savefig(FIG / "e6_by_relation.png", dpi=130)
plt.close(f)


def fmt(d):
    if d.get("value") is None:
        return "–"
    if d.get("lo") is None:
        return f"{100 * d['value']:.0f}%"
    return f"{100 * d['value']:.0f}% ({100 * d['lo']:.0f}–{100 * d['hi']:.0f})"


lines = ["| Metric | " + " | ".join(s for _, s in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for _, label, _ in METRICS:
    lines.append(f"| {label} | " + " | ".join(fmt(table[label].get(m, {})) for m, _ in MODELS) + " |")
lines += ["", "| Truncated at 3,000 tokens | " + " | ".join(s for _, s in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for ev in ["e1", "e1_ref", "e2", "e3", "e4", "e5", "e6", "e7", "cap_mbpp", "cap_gsm8k"]:
    lines.append(f"| {ev} | " + " | ".join(fmt(table[f'truncated: {ev}'].get(m, {})) for m, _ in MODELS) + " |")
lines += ["", "| E6 relation (items) | " + " | ".join(s for _, s in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for rel in rels:
    cells = []
    for m, _ in MODELS:
        d = rel_table[rel].get(m)
        if not d:
            cells.append("–")
            continue
        v = d["grader_score"]
        c = f"score {100 * v[0]:.0f}%, grader>instr {100 * d['grader_not_instr'][0]:.0f}%"
        if "hacked" in d:
            c += f", hacked {100 * d['hacked'][0]:.0f}%"
        if "acknowledged_impossible" in d:
            c += f", said impossible {100 * d['acknowledged_impossible'][0]:.0f}%"
        cells.append(c)
    n = max(d["n"] for d in rel_table[rel].values())
    lines.append(f"| {rel} ({n}) | " + " | ".join(cells) + " |")
(OUT / "eval_suite_report_tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
