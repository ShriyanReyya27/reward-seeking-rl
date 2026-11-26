"""Step 2 of the logic chain-of-thought spreadsheet: JSONL from extract.py -> logic_cot.xlsx + one CSV per tab.

    uv run --with openpyxl python build_xlsx.py
"""

import csv
import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

OUT = Path(__file__).resolve().parents[2] / "results" / "logic_cot_sheet"
WORK = OUT / "work"
LIMIT = 32767
PART = 32000
ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
HEAD_FILL = PatternFill("solid", fgColor="DDE4EE")


def jl(name):
    return [json.loads(x) for x in open(WORK / f"{name}.jsonl")]


def yn(x):
    return "yes" if x else "no"


def base_cols(training=False, wording=False):
    cols = [("Run", lambda r: r["run"], 26), ("Seed", lambda r: r["seed"], 6)]
    cols.append(("Wording", lambda r: r["wording"], 22) if wording else ("Step", lambda r: r["step"], 6))
    cols += [("Problem ID", lambda r: r["problem"], 15), ("Model", lambda r: r["model"], 15),
             ("Attempt", lambda r: r["attempt"], 8), ("Grade", lambda r: r["grade"], 26)]
    if training:
        cols += [("Reward", lambda r: r["reward"], 8), ("Advantage (reward minus group mean)", lambda r: round(r["advantage"], 4), 12),
                 ("Group pass rate (of 16 attempts)", lambda r: r["group_pass"], 12)]
    cols += [("Proof lines", lambda r: r["proof_lines"], 8), ("Expected proof lines", lambda r: r["expected_lines"], 9),
             ("Unused lines", lambda r: r["unused_lines"], 8), ("Which lines are unused", lambda r: r["which_unused"], 14),
             ("Mentions conciseness", lambda r: yn(r["mentions_conciseness"]), 12),
             ("Mentions the rule or checker", lambda r: yn(r["mentions_rule"]), 12),
             ("Conciseness sentences", lambda r: r["concise_sentences"], 60),
             ("Rough strategy", lambda r: r["strategy"], 18), ("Reasoning tokens", lambda r: r["tokens"], 10),
             ("Truncated", lambda r: yn(r["truncated"]), 9), ("Final proof", lambda r: r["final"], 45)]
    return cols


WRAP = {"Conciseness sentences", "Final proof"}


def clean(v):
    if isinstance(v, str):
        v = ILLEGAL.sub("", v)
        if len(v) > LIMIT:  # only possible for the sentence list; the chain of thought is split, never cut
            v = v[: LIMIT - 60] + "\n[... more sentences; see the chain of thought]"
    return v


def write_tab(wb, title, rows, cols):
    parts = max((len(r["cot"]) + PART - 1) // PART for r in rows) or 1
    ws = wb.create_sheet(title)
    headers = [c[0] for c in cols] + ["Chain of thought"] + [f"Chain of thought part {k}" for k in range(2, parts + 1)]
    ws.append(headers)
    for r in rows:
        cot = ILLEGAL.sub("", r["cot"] or "")
        chunks = [cot[i:i + PART] for i in range(0, len(cot), PART)] or [""]
        ws.append([clean(f(r)) for _, f, _ in cols] + chunks + [None] * (parts - len(chunks)))
    for j, h in enumerate(headers, 1):
        c = ws.cell(1, j)
        c.font, c.fill = BOLD, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
        width = next((w for name, _, w in cols if name == h), 30)
        ws.column_dimensions[get_column_letter(j)].width = width
    wrap_cols = {j for j, h in enumerate(headers, 1) if h in WRAP}
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = FONT
            if isinstance(c.value, str) and c.value.startswith("="):
                c.data_type = "s"  # text, not a formula
            c.alignment = Alignment(wrap_text=c.column in wrap_cols, vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    with open(OUT / (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cols] + ["Chain of thought"])
        for r in rows:
            w.writerow([f(r) for _, f, _ in cols] + [r["cot"]])
    return parts


def summary_tab(wb):
    s = json.loads((WORK / "summary.json").read_text())
    s.sort(key=lambda r: (r["run"], r["seed"] or 0, str(r["step"]) if isinstance(r["step"], str) else f"{r['step']:03d}", r["model"]))
    ws = wb.create_sheet("Summary")
    heads = [("Run", 22), ("Seed", 6), ("Step or wording", 22), ("Model", 15), ("Answers", 9),
             ("Mention conciseness", 12), ("Mention the rule or checker", 12), ("Valid proofs with no unused lines", 14),
             ("Pass rate", 9), ("Median reasoning tokens", 11), ("Mean reward (training)", 11),
             ("Rewarded traces that mention conciseness (training)", 16)]
    ws.append([h for h, _ in heads])
    for r in s:
        ws.append([r["run"], r["seed"], r["step"], r["model"], r["n"], r["mention_conciseness"], r["mention_rule"],
                   r["no_unused_among_valid"], r["pass_rate"], r["median_tokens"], r.get("mean_reward"),
                   r.get("rewarded_mention_conciseness")])
    for j, (h, w) in enumerate(heads, 1):
        c = ws.cell(1, j)
        c.font, c.fill, c.alignment = BOLD, HEAD_FILL, Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(j)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = FONT
            if c.column in (6, 7, 8, 9, 12):
                c.number_format = "0.0%"
            elif c.column == 11:
                c.number_format = "0.000"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            w.writerow(row)


README = [
    "What this is",
    "Chains of thought from the logic-proof runs, to see what Aghyad's model and the control say about keeping proofs short. "
    "Built only from files already on disk; nothing was sampled. Each answer tab is also saved as a CSV (one column for the "
    "whole chain of thought).",
    "",
    "Tabs",
    "Held-out pairs (Nov 6): every held-out answer at steps 0, 10 and 20 of the rule-stated logic runs (the prompt says a "
    "derivation with any unused line is marked wrong). Seed 1 is results/nov6, seed 2 is results/logic_seed2 (same "
    "settings, same held-out problems). Sorted by seed, step, problem, attempt, then model, so each problem's two answers "
    "sit next to each other.",
    "Training rollouts (Nov 6): training rollouts from steps 0 to 20 (seed 1 has steps 0-19). Each step has 8 problems "
    "with 16 attempts each; per step, problem and model this tab keeps 2 attempts, one rewarded and one not where both "
    "exist, otherwise 2 at random (fixed seed). Advantage = reward minus the mean reward of that problem's 16 attempts.",
    "No hint vs hints (Nov 11): level-10 screen answers with no hint, (a) 'The checker is strict about which lines you "
    "include.' and (b) 'Keep the derivation minimal.', on the 69 problems whose prompts were identical across the three. "
    "2 attempts per problem, model and wording.",
    "Summary: one row per run, seed, model and step (or wording). Training rows use all 128 rollouts of the step, not only "
    "the 2 per problem shown in the training tab.",
    "",
    "Grades",
    "passed: matches the expected proof (the reward in these runs).",
    "valid with unused lines: every line is correct and the target is derived, but some lines are not needed for the target.",
    "valid but not the expected proof: correct with no unused lines, but a different proof from the expected one.",
    "invalid: some line is wrong, or the target is never derived, or the answer could not be read.",
    "truncated: the answer hit the 32,000-token limit.",
    "Which lines are unused: proof lines are numbered from @0 in the order written. For 8 answers this could not be "
    "recomputed from the saved problem and is marked so.",
    "",
    "Rough columns",
    "Mentions conciseness, mentions the rule or checker, the conciseness sentences and the strategy tag are keyword "
    "searches. They are rough: they catch ordinary uses of words like 'required' or 'extra', and miss other phrasings. "
    "Strategy: 'derives forward then trims' if the reasoning talks about removing/dropping lines or steps, else "
    "'works backward' if it mentions working backward or from the target, else 'neither'.",
    "Reasoning tokens counts all sampled tokens of the answer (reasoning plus final proof).",
    "Excel cells hold at most 32,767 characters, so long chains of thought continue in 'Chain of thought part 2' and so on.",
]


def readme_tab(wb):
    ws = wb.create_sheet("Read me")
    ws.column_dimensions["A"].width = 120
    for line in README:
        ws.append([line])
        c = ws.cell(ws.max_row, 1)
        c.font = BOLD if line in ("What this is", "Tabs", "Grades", "Rough columns") else FONT
        c.alignment = Alignment(wrap_text=True, vertical="top")


def main():
    held = sorted(jl("heldout"), key=lambda r: (r["seed"], r["step"], r["problem"], r["attempt"], r["model"]))
    train = sorted(jl("training"), key=lambda r: (r["seed"], r["step"], r["problem"], r["model"], r["reward"] == 0, r["attempt"]))
    hints = sorted(jl("hints"), key=lambda r: (r["problem"], r["wording_order"], r["model"], r["attempt"]))
    wb = Workbook()
    wb.remove(wb.active)
    p1 = write_tab(wb, "Held-out pairs (Nov 6)", held, base_cols())
    p2 = write_tab(wb, "Training rollouts (Nov 6)", train, base_cols(training=True))
    p3 = write_tab(wb, "No hint vs hints (Nov 11)", hints, base_cols(wording=True))
    summary_tab(wb)
    readme_tab(wb)
    wb.save(OUT / "logic_cot.xlsx")
    print("rows", len(held), len(train), len(hints), "chain-of-thought parts", p1, p2, p3)


if __name__ == "__main__":
    main()
