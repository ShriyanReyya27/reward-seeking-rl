"""Tables, figures, examples and validation sheet from the zebra chain-of-thought labels.

    uv run python results/zebra_cot/analyze_cot.py

Reads sample.jsonl, labels_gpt41.jsonl (main) and labels_gpt41mini.jsonl (validation), all written by
label_cot.py. Writes tables.md, examples.md, validation_sheet.md, analysis.json and figures/.
Lowercase is computed from the answer tag with the env's rule (all letters lowercase, at least one letter).
Each bucket's rates come from that bucket's 50-chain sample; "whole run" from the 100-chain whole-run sample.

Keyword gating (decided after labeling, see summary.md): gpt-4.1 marked many chains as discussing case when
they only wrote the name ("Thus answer: alice."). So a chain counts as mentioning case only if the case regex
matches its reasoning or final message; for those chains gpt-4.1's kind / reason / uncertainty labels are kept,
for the rest they are set to none / none_given / false. decision_location is replaced by a check in code
(name written capitalized at the end of the reasoning, lowercase in the tag). The raw labels are in an appendix.
"""

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from label_cot import BOOLS, BUCKETS, CASE, ENUMS, ROOT, RUNS  # noqa: E402

FIG = OUT / "figures"
MODEL = {"Aghyad's": "Aghyad's model", "Vlad's": "Vlad's model", "gpt-oss-120b": "gpt-oss-120b", "gpt-oss-20b": "gpt-oss-20b"}
STYLE = {  # same as results/report/make_report.py
    "Aghyad's model": dict(color="#b45309", ls="-", marker="o"),
    "Vlad's model": dict(color="#f59e0b", ls="--", marker="D"),
    "gpt-oss-120b": dict(color="#1f2937", ls="-.", marker="s"),
    "gpt-oss-20b": dict(color="#9ca3af", ls=":", marker="^"),
}
UNDER = {"C2": "C1", "C4": "C3", "C6": "C5", "C2u": None}  # underspecified run -> its normal-mode baseline
SETS = [f"steps_{a}-{b}" for a, b in BUCKETS] + ["whole_run"]
SET_LABEL = {**{f"steps_{a}-{b}": f"steps {a + 1}–{b + 1}" for a, b in BUCKETS}, "whole_run": "whole run"}
REASONS = ["matches_prompt_style", "instruction_exactness", "convention", "grader_or_reward", "other", "none_given"]
REASON_STYLE = {  # Okabe-Ito hues + hatching so the stack reads in grayscale
    "matches_prompt_style": ("#0072B2", ""), "instruction_exactness": ("#56B4E9", "//"), "convention": ("#009E73", "\\\\"),
    "grader_or_reward": ("#D55E00", "xx"), "other": ("#CC79A7", ".."), "none_given": ("#d1d5db", ""),
}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 100})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def lower(a):
    return bool(a) and a == a.lower() and any(ch.isalpha() for ch in a)


def load_labels(name):
    p = OUT / name
    return {r["id"]: r["label"] for r in map(json.loads, open(p)) if r["label"]} if p.exists() else {}


def gate(label, regex_hit):
    if regex_hit:
        return label
    return {**label, "mentions_case": False, "case_mention_kind": "none", "case_reason": "none_given",
            "case_uncertainty": False}


def tag_only_switch(r):
    """Lowercase tag, while the last mention of the name in the reasoning is capitalized."""
    a = r["answer"] or ""
    hits = list(re.finditer(rf"\b{re.escape(a)}\b", r["thinking"], re.I)) if a else []
    return lower(a) and bool(hits) and hits[-1].group(0) != a


def rate(rows, f):
    k, n = sum(1 for r in rows if f(r)), len(rows)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else float("nan"), "lo": lo, "hi": hi}


def fmt(x):
    return "—" if not x["n"] else f"{x['p']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}] ({x['k']}/{x['n']})"


def short(x):
    return "—" if not x["n"] else f"{x['p']:.2f} ({x['k']}/{x['n']})"


answer_fmt = lambda r: r["L"]["case_mention_kind"] in ("answer_format", "both")  # noqa: E731
silent = lambda r: r["low"] and r["L"]["case_reason"] == "none_given"  # noqa: E731
METRICS = {
    "lowercase answer": lambda r: r["low"],
    "lowercase | correct": None,  # handled separately (different denominator)
    "mentions case (any)": lambda r: r["L"]["mentions_case"],
    "answer-format case mention": answer_fmt,
    "entity-only case mention": lambda r: r["L"]["case_mention_kind"] == "puzzle_entities",
    "case uncertainty": lambda r: r["L"]["case_uncertainty"],
    "reward awareness": lambda r: r["L"]["reward_awareness"],
    "silent lowercase (no reason)": silent,
    "name capitalized in reasoning, lowercase tag": tag_only_switch,
}
RAW = {  # appendix: the labeler's own, ungated answers
    "raw: mentions case": lambda r: r["raw"]["mentions_case"],
    "raw: answer-format mention": lambda r: r["raw"]["case_mention_kind"] in ("answer_format", "both"),
    "raw: case uncertainty": lambda r: r["raw"]["case_uncertainty"],
    "raw: decision_location = answer_only": lambda r: r["raw"]["decision_location"] == "answer_only",
}


def first_50_step(run):
    """First training step where >= 50% of correct answered zebra chains are lowercase (all rollouts, not the sample)."""
    by = defaultdict(lambda: [0, 0])
    for line in open(ROOT / RUNS[run][2]):
        r = json.loads(line)
        if r["task"] == "zebra_puzzles" and not r["truncated"] and r["true_score"] == 1:
            by[r["step"]][0] += lower(r["answer"])
            by[r["step"]][1] += 1
    return next((s for s in sorted(by) if by[s][0] / by[s][1] >= 0.5), None)


def excerpt(r, around=None, width=500):
    cot = r["thinking"]
    i = cot.find(around) if around else -1
    if i < 0:
        return "…" + cot[-width:].replace("\n", " ")
    return "…" + cot[max(0, i - width // 2): i + len(around) + width // 4].replace("\n", " ") + "…"


def main():
    sample = [json.loads(l) for l in open(OUT / "sample.jsonl")]
    main_l, check_l = load_labels("labels_gpt41.jsonl"), load_labels("labels_gpt41mini.jsonl")
    rows = []
    for r in sample:
        if r["id"] in main_l:
            hit = bool(CASE.search(r["thinking"] + r["final"]))
            rows.append({**r, "raw": main_l[r["id"]], "L": gate(main_l[r["id"]], hit), "low": lower(r["answer"]),
                         "correct": r["true_score"] == 1, "regex_case": hit})
    print(f"{len(rows)}/{len(sample)} sampled chains have main labels")
    cells = defaultdict(list)
    for r in rows:
        for s in r["in_sets"]:
            cells[(r["run"], s)].append(r)

    # ---------- per-run tables ----------
    analysis, md = {"n_labeled": len(rows), "n_sample": len(sample), "runs": {}}, []
    md.append("# Zebra chain-of-thought labels: tables\n")
    md.append(f"{len(rows)} of {len(sample)} sampled chains labeled by gpt-4.1. Case fields are keyword-gated "
              "(see the script docstring; raw labels in the appendix). Rates with Wilson 95% CIs (k/n). "
              "Buckets are 1-indexed training steps (steps 1–5 = logged steps 0–4); each bucket is its own random "
              "sample of 50 answered chains (fewer if the bucket has fewer), 'whole run' is a separate random 100.\n")
    for run, (model, mode, _) in RUNS.items():
        analysis["runs"][run] = {"model": MODEL[model], "mode": mode, "sets": {}}
        md.append(f"\n## {run}: {MODEL[model]}, {mode}\n")
        md.append("| metric | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
        md.append("|---" * (len(SETS) + 1) + "|")
        for name, f in METRICS.items():
            vals = []
            for s in SETS:
                cs = cells[(run, s)]
                x = rate([c for c in cs if c["correct"]], lambda c: c["low"]) if f is None else rate(cs, f)
                analysis["runs"][run]["sets"].setdefault(s, {})[name] = x
                vals.append(fmt(x) if name in ("lowercase answer", "answer-format case mention", "reward awareness")
                            else short(x))
            md.append(f"| {name} | " + " | ".join(vals) + " |")
        # reasons among lowercase answers and among non-lowercase answers
        for which, cond in (("lowercase", True), ("non-lowercase", False)):
            md.append(f"\n`case_reason` among **{which}** answers:\n")
            md.append("| reason | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
            md.append("|---" * (len(SETS) + 1) + "|")
            for reason in REASONS:
                vals = []
                for s in SETS:
                    cs = [c for c in cells[(run, s)] if c["low"] == cond]
                    x = rate(cs, lambda c: c["L"]["case_reason"] == reason)
                    analysis["runs"][run]["sets"][s][f"reason[{which}]:{reason}"] = x
                    vals.append(short(x))
                md.append(f"| {reason} | " + " | ".join(vals) + " |")

    # ---------- onset table ----------
    md.append("\n## Onset: compliance vs answer-format case mentions\n")
    md.append("50% step: first training step (1-indexed) where ≥50% of correct answered chains are lowercase, from "
              "**all** rollouts. Mention onset: first bucket where ≥10% of sampled chains mention the answer's case "
              "(entity-only mentions excluded). Bucket resolution only.\n")
    md.append("| run | model | 50% step | first bucket ≥10% answer-format mention | answer-format rate per bucket |")
    md.append("|---|---|---|---|---|")
    analysis["onset"] = {}
    for run in UNDER:
        s50 = first_50_step(run)
        rates = [analysis["runs"][run]["sets"][s]["answer-format case mention"] for s in SETS[:-1]]
        first = next((SET_LABEL[s] for s, x in zip(SETS, rates) if x["n"] and x["p"] >= 0.10), "never")
        analysis["onset"][run] = {"step50": None if s50 is None else s50 + 1, "mention_bucket": first}
        md.append(f"| {run} | {MODEL[RUNS[run][0]]} | {'—' if s50 is None else s50 + 1} | {first} | "
                  + ", ".join(short(x) for x in rates if x["n"]) + " |")

    # ---------- reward awareness: every case ----------
    ra = [r for r in rows if r["L"]["reward_awareness"] or r["L"]["case_reason"] == "grader_or_reward"]
    analysis["reward_awareness_ids"] = [r["id"] for r in ra]
    md.append(f"\n## Reward awareness\n\n{len(ra)} of {len(rows)} labeled chains flagged `reward_awareness` or "
              "`case_reason = grader_or_reward` (all listed in examples.md). By run: "
              + ", ".join(f"{run} {sum(r['run'] == run for r in ra)}/{sum(r['run'] == run for r in rows)}" for run in RUNS)
              + ".\n")

    ci = re.compile(r"case[- ]?insensitiv", re.I)
    pop = defaultdict(list)
    for run in RUNS:
        for line in open(ROOT / RUNS[run][2]):
            r = json.loads(line)
            if r["task"] == "zebra_puzzles" and not r["truncated"] and r["thinking"]:
                pop[run].append(r)
    md.append("Population check (no labeler): chains anywhere in the training rollouts that say "
              "'case-insensitive', i.e. guess how the answer is compared. Each is listed in examples.md; some are "
              "about whether child 'alice' and person 'Alice' are the same, not about the answer.\n")
    md.append("| run | chains saying case-insensitive | answered zebra chains |\n|---|---|---|")
    hits_ci = {run: [r for r in rs if ci.search(r["thinking"] + r["final"])] for run, rs in pop.items()}
    for run in RUNS:
        md.append(f"| {run} ({MODEL[RUNS[run][0]]}, {RUNS[run][1]}) | {len(hits_ci[run])} | {len(pop[run])} |")
    analysis["case_insensitive_population"] = {run: [len(hits_ci[run]), len(pop[run])] for run in RUNS}

    # ---------- regex cross-check ----------
    cm = Counter((r["regex_case"], r["raw"]["mentions_case"]) for r in rows)
    analysis["regex_vs_label"] = {f"regex={a},label={b}": v for (a, b), v in cm.items()}
    md.append("\n## Keyword cross-check (raw gpt-4.1 `mentions_case` vs case regex over the full chain + final message)\n")
    md.append("| | label true | label false |\n|---|---|---|")
    md.append(f"| regex hit | {cm[(True, True)]} | {cm[(True, False)]} |")
    md.append(f"| no regex hit | {cm[(False, True)]} | {cm[(False, False)]} |")
    md.append("\nSpot checks: 'label true, no regex hit' are mostly over-labels (the chain just writes the name, "
              "e.g. 'Thus answer: alice.'); 'regex hit, label false' are mostly child-'alice'-vs-person-'Alice' "
              "entity mentions the labeler missed. Hence the gating.\n")

    # ---------- agreement with gpt-4.1-mini ----------
    hit_by_id = {r["id"]: r["regex_case"] for r in rows}
    pairs = [(gate(main_l[i], hit_by_id[i]), gate(check_l[i], hit_by_id[i])) for i in check_l if i in main_l]
    md.append(f"\n## Labeler agreement after gating: gpt-4.1 (main) vs gpt-4.1-mini, {len(pairs)} chains "
              f"({sum(hit_by_id[i] for i in check_l)} with a case keyword)\n")
    md.append("Ungated, the two disagree badly: mini marks 99/100 as mentioning case. Gating removes that, so "
              "agreement below is mainly about kind / reason on the keyword chains.\n")
    md.append("| field | agreement | Cohen's κ | gpt-4.1 positive/most common | mini positive/most common |")
    md.append("|---|---|---|---|---|")
    analysis["agreement"] = {}
    for k in list(BOOLS) + ["case_mention_kind", "case_reason"]:
        a, b = [p[0][k] for p in pairs], [p[1][k] for p in pairs]
        if not pairs:
            continue
        po = sum(x == y for x, y in zip(a, b)) / len(a)
        ca, cb = Counter(a), Counter(b)
        pe = sum(ca[v] * cb[v] for v in set(a) | set(b)) / len(a) ** 2
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        analysis["agreement"][k] = {"agreement": po, "kappa": kappa, "n": len(a)}
        desc = (lambda c: f"{c[True]}/{len(a)}") if k in BOOLS else (lambda c: ", ".join(f"{v} {n}" for v, n in c.most_common(3)))
        md.append(f"| {k} | {po:.2f} | {kappa:.2f} | {desc(ca)} | {desc(cb)} |")
    md.append("\n## Appendix: raw (ungated) gpt-4.1 labels\n")
    md.append("| run | metric | " + " | ".join(SET_LABEL[s] for s in SETS) + " |")
    md.append("|---" * (len(SETS) + 2) + "|")
    for run in RUNS:
        for name, f in RAW.items():
            md.append(f"| {run} | {name} | " + " | ".join(short(rate(cells[(run, s)], f)) for s in SETS) + " |")
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ---------- examples ----------
    rng = random.Random(0)
    ex = ["# Examples\n", "Excerpts are around the labeler's `decision_point` when it can be found in the chain, "
          "otherwise the chain's last 500 characters.\n", f"\n## Every reward-awareness / grader_or_reward chain ({len(ra)})\n"]
    for r in ra:
        ex.append(f"- **{r['id']}** ({MODEL[r['model']]}, {r['mode']}, step {r['step'] + 1}, answer `{r['answer']}`): "
                  f"labels `{json.dumps(r['L'])}`\n  > {excerpt(r, r['L'].get('decision_point'))}\n")
    ex.append("\n## Every 'case-insensitive' chain in the training rollouts (population search, no labeler)\n")
    for run in RUNS:
        for r in hits_ci[run]:
            t = r["thinking"] + r["final"]
            m = ci.search(t)
            ex.append(f"- {run} ({MODEL[RUNS[run][0]]}, {RUNS[run][1]}), step {r['step'] + 1}, answer `{r['answer']}`\n"
                      f"  > …{t[max(0, m.start() - 250): m.end() + 80].replace(chr(10), ' ')}…\n")
    ex.append("\n## Per run and bucket: up to 2 lowercase chains that mention the answer's case, and 1 silent one\n")
    for run in RUNS:
        for s in SETS[:-1]:
            cs = cells[(run, s)]
            if not cs:
                continue
            ex.append(f"\n### {run} ({MODEL[RUNS[run][0]]}, {RUNS[run][1]}), {SET_LABEL[s]}\n")
            talk = [c for c in cs if c["low"] and answer_fmt(c)]
            quiet = [c for c in cs if silent(c)]
            for c in rng.sample(talk, min(2, len(talk))) + rng.sample(quiet, min(1, len(quiet))):
                ex.append(f"- {c['id']}: `{c['answer']}`, reason `{c['L']['case_reason']}`\n  > {excerpt(c, c['L'].get('decision_point'), 400)}\n")
    (OUT / "examples.md").write_text("\n".join(ex) + "\n")

    # ---------- hand-check sheet ----------
    sheet = ["# Validation sheet (hand check)\n",
             "50 random labeled chains. For each, mark the labels you disagree with. The excerpt is the last ~800 "
             "characters of the reasoning plus any case-keyword snippets; the full chain is in sample.jsonl under the id.\n"]
    for c in random.Random(7).sample(rows, min(50, len(rows))):
        snips = [c["thinking"][max(0, m.start() - 150): m.end() + 150].replace("\n", " ")
                 for m in list(CASE.finditer(c["thinking"]))[:3]]
        sheet.append(f"\n## {c['id']}  ({MODEL[c['model']]}, {c['mode']}, step {c['step'] + 1})\n")
        sheet.append(f"Answer tag: `{c['answer']}` (lowercase: {c['low']}, correct: {c['correct']})\n")
        for sn in snips:
            sheet.append(f"> [case snippet] …{sn}…\n")
        sheet.append(f"> [end of reasoning] …{c['thinking'][-800:].replace(chr(10), ' ')}\n")
        sheet.append(f"> [final message] {c['final'][-300:].replace(chr(10), ' ')}\n")
        sheet.append("| field | label | agree? |\n|---|---|---|")
        for k in list(BOOLS) + ["case_mention_kind", "case_reason", "decision_point"]:
            sheet.append(f"| {k} | {c['L'].get(k)} | |")
        sheet.append(f"| (raw, before gating) | {json.dumps({k: c['raw'][k] for k in ('mentions_case', 'case_mention_kind', 'case_reason')})} | |")
    (OUT / "validation_sheet.md").write_text("\n".join(sheet) + "\n")

    figures(analysis)


def figures(analysis):
    FIG.mkdir(exist_ok=True)
    x = list(range(len(SETS) - 1))
    xt = [SET_LABEL[s] for s in SETS[:-1]]

    def series(run, metric):
        vals = [analysis["runs"][run]["sets"][s][metric] for s in SETS[:-1]]
        return ([i for i, v in zip(x, vals) if v["n"]], [v["p"] for v in vals if v["n"]],
                [v["lo"] for v in vals if v["n"]], [v["hi"] for v in vals if v["n"]])

    def save(fig, name):
        fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 1. per model: lowercase rate (underspecified vs normal), answer-format mentions, uncertainty
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    lines = [("lowercase answer", "lowercase answer (underspecified)", "-", "o", "#1f2937"),
             ("answer-format case mention", "mentions the answer's case", "--", "s", "#0072B2"),
             ("case uncertainty", "weighs which case to use", ":", "^", "#D55E00"),
             ("reward awareness", "reward awareness", "-", "x", "#CC79A7")]
    for ax, run in zip(axes, UNDER):
        for metric, lab, ls, mk, col in lines:
            xs, ps, lo, hi = series(run, metric)
            ax.plot(xs, ps, ls=ls, marker=mk, color=col, lw=2, ms=6, label=lab)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.10, lw=0)
        if UNDER[run]:
            xs, ps, lo, hi = series(UNDER[run], "lowercase answer")
            ax.plot(xs, ps, ls="-", marker="o", color="#9ca3af", lw=1.5, ms=5, mfc="white",
                    label="lowercase answer (normal-mode baseline)")
        ax.set_title(f"{MODEL[RUNS[run][0]]} ({run})")
        ax.set_xticks(x, xt, rotation=20)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of sampled chains")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.suptitle("Zebra, underspecified grading: lowercase answers vs what the reasoning says about case "
                 "(50 chains per bucket, 95% CI)", y=1.02)
    save(fig, "fig1_case_reasoning_by_bucket")

    # 2. reasons given for lowercase answers, stacked by bucket
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, run in zip(axes, UNDER):
        bottom = [0.0] * len(x)
        for reason in REASONS:
            vals = [analysis["runs"][run]["sets"][s][f"reason[lowercase]:{reason}"] for s in SETS[:-1]]
            h = [v["p"] if v["n"] else 0 for v in vals]
            col, hatch = REASON_STYLE[reason]
            ax.bar(x, h, bottom=bottom, color=col, hatch=hatch, edgecolor="white", lw=1.5, width=0.7,
                   label=reason.replace("_", " ") + (" (silent)" if reason == "none_given" else ""))
            bottom = [b + v for b, v in zip(bottom, h)]
        for i, s in enumerate(SETS[:-1]):
            n = analysis["runs"][run]["sets"][s]["reason[lowercase]:none_given"]["n"]
            ax.text(i, 1.02, f"n={n}", ha="center", fontsize=8, color="#374151")
        ax.set_title(f"{MODEL[RUNS[run][0]]} ({run})", pad=14)
        ax.set_xticks(x, xt, rotation=20)
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of lowercase answers")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.suptitle("Reason the reasoning gives for answering in lowercase (n = lowercase answers in the bucket's sample)",
                 y=1.04)
    save(fig, "fig2_reasons_for_lowercase")


if __name__ == "__main__":
    main()
