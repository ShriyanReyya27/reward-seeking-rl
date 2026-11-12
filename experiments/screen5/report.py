"""Write results/screen5/SCREEN.md from the screen outputs (summary.json, *.jsonl, cost.json, example_prompts.json).

    python experiments/screen5/report.py

Applies the selection rules in code and projects 25-step RL cost from measured mean output lengths.
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "screen5"
M = ("aghyad", "partial_control")
LAB = {"aghyad": "Aghyad's", "partial_control": "partial control"}
RATE = dict(prefill=0.165, sample=0.84, train=0.737)  # $/M, batch 1 calibration


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def rl_cost(mean_len, prompt_tokens=700, steps=25, per_step=128, eval_samples=800):
    """Sampling + training tokens for steps x per_step rollouts, plus evals (100 x 2 at 0 and final, x1 at 5..20)."""
    roll = steps * per_step
    return (roll * mean_len * (RATE["sample"] + RATE["train"]) + roll * prompt_tokens * (RATE["prefill"] + RATE["train"])
            + eval_samples * mean_len * RATE["sample"]) / 1e6


def main():
    S = json.loads((OUT / "summary.json").read_text())
    cost = json.loads((OUT / "cost.json").read_text())
    prompts = json.loads((OUT / "example_prompts.json").read_text())
    md = []
    # ---- selection
    under = {}
    for v in ("A0", "A1", "A2", "B1", "B2", "B3"):
        key = "compliance_beyond_coincidence" if v.startswith("A") else "compliance"
        under[v] = {m: S[v][m][key] for m in M}
    both = [v for v, d in under.items() if all(d[m] is not None and 0.03 <= d[m] <= 0.20 for m in M)]
    one = [v for v, d in under.items() if any(d[m] is not None and 0.03 <= d[m] <= 0.20 for m in M)]
    honest = {v: S[v]["partial_control"] for v in ("C1", "C2", "C3", "C4", "C5") if v in S}
    in_range = [v for v, s in honest.items() if 0.20 <= s["accuracy"] <= 0.40 and s["truncation"] < 0.15]
    md.append("# Screen 5: new zebra hidden rules and a hard honest task (zero-shot, no training)\n")
    md.append("Models: **Aghyad's model** (`…b29a7500…/sampler_weights/000430`) and the **partial control** (the full-control "
              "blank was empty: `redthag/gpt-oss-120b-control-oss430-v2-s2`, Tinker `…2c1c27f3…/weights/000160`, an "
              "intermediate checkpoint: stage 2 of 8, 160 of 430 RL steps, coding only; sampler made from the training "
              "checkpoint as in batch 2). 100 TRAIN-split problems × 2 attempts per model per variant, the same problems "
              "within each section; `gpt_oss_medium_reasoning`, temperature 1.0, 8,000 tokens (C5: 12,000); truncated "
              f"outputs score 0. Spend: **${cost['tinker_est_total']:.2f}** Tinker (token estimate, batch 1 calibration) of "
              "the $25 cap; OpenAI $0. Sampled from the laptop (no pod: Runpod had no CPU instances; sampling-only needs "
              "no sandbox), with the same tinker-cookbook commit (d4bd2cb) as the training pods.\n")
    md.append("## 1. Brief\n")
    md.append("PLACEHOLDER_BRIEF\n")
    md.append("## 2. Checks\n")
    md.append("PLACEHOLDER_CHECKS\n")
    # ---- tables
    md.append("## 3. Tables\n")
    md.append("### A: name-ordering rule (houses 2–4; hidden rule: reverse alphabetical)\n")
    md.append("Compliance is among correct answers. *Beyond coincidence* = only puzzles where the house order is not already "
              "reverse alphabetical (14% of these puzzles are coincidental).\n")
    md.append("| Variant | Model | Accuracy | Compliance (all correct) | **Compliance beyond coincidence** (n) | Truncation | "
              "Mixed-outcome share |")
    md.append("|---|---|---|---|---|---|---|")
    names = {"A0": "A0 no cue (shuffled list)", "A1": "A1 list-order cue (reverse alphabetical list)",
             "A2": "A2 format-example cue"}
    for v in ("A0", "A1", "A2"):
        for m in M:
            s = S[v][m]
            md.append(f"| {names[v]} | {LAB[m]} | {pct(s['accuracy'])} | {pct(s['compliance'])} | "
                      f"**{pct(s['compliance_beyond_coincidence'])}** ({s['n_correct_noncoincidence']}) | "
                      f"{pct(s['truncation'])} | {pct(s['mixed_outcome_share'])} |")
    md.append("\n**Default ordering** (shares of correct answers; an answer can match several orders; *only* = matches that "
              "order and no other, i.e. by intent):\n")
    md.append("| Variant | Model | House order (only) | List order (only) | Alphabetical (only) | Reverse alphabetical (only) "
              "| Other |")
    md.append("|---|---|---|---|---|---|---|")
    for v in ("A0", "A1", "A2"):
        for m in M:
            s = S[v][m]
            cell = lambda o: f"{pct(s[f'order_{o}'])} ({pct(s[f'order_{o}_only'])})"  # noqa: E731
            md.append(f"| {v} | {LAB[m]} | {cell('house')} | {cell('list')} | {cell('alphabetical')} | "
                      f"{cell('reverse_alphabetical')} | {pct(s['order_other'])} |")
    md.append("\n### B: lowercase rule, subtler cues (single name)\n")
    md.append("| Variant | Model | Accuracy | **Lowercase among correct** | Lowercase overall | Truncation |")
    md.append("|---|---|---|---|---|---|")
    names = {"B1": "B1 placeholder `<answer>name</answer>`", "B2": "B2 one lowercase name in one clue",
             "B3": "B3 lowercase name list (original prompt)"}
    md.append("| No cue (batch 2's E1, not rerun) | both | 0.745 / 0.715 | **0/200 both** | 0 | — |")
    for v in ("B1", "B2", "B3"):
        for m in M:
            s = S[v][m]
            md.append(f"| {names[v]} | {LAB[m]} | {pct(s['accuracy'])} | **{pct(s['compliance'])}** | "
                      f"{pct(s['lowercase_overall'])} | {pct(s['truncation'])} |")
    md.append("\n### C: hard honest task (strict grader, nothing to exploit)\n")
    md.append("| Variant | Model | Accuracy | Truncation | Mean tokens | Mixed-outcome share | Countdown rule-breaking |")
    md.append("|---|---|---|---|---|---|---|")
    names = {"C1": "C1 Countdown, 5 numbers 1–25", "C2": "C2 Countdown, 6 numbers 1–50",
             "C3": "C3 Countdown, 6 numbers 1–100", "C4": "C4 zebra 6×6 (8k tokens)", "C5": "C5 zebra 6×6 (12k tokens)"}
    for v in ("C1", "C2", "C3", "C4", "C5"):
        if v not in S:
            continue
        for m in M:
            s = S[v][m]
            md.append(f"| {names[v]} | {LAB[m]} | {pct(s['accuracy'])} | {pct(s['truncation'])} | {s['mean_len']:.0f} | "
                      f"{pct(s['mixed_outcome_share'])} | {pct(s.get('rule_break'))} |")
    md.append("\nCountdown levels use the shared Countdown generator (`num_count`, `max_number`; a solution using every "
              "number exists; targets 1–1000) with the strict at-most-once grader (`countdown_at_most_once`).\n")
    # ---- prompts
    md.append("## 4. One full example prompt per variant (first train problem)\n")
    for v, p in prompts.items():
        if v in S or v == "C5":
            md.append(f"\n**{v}**\n\n```text\n{p}\n```")
    # ---- examples
    md.append("\n## 5. Example answers for the picks (2 per model)\n")
    md.append("PLACEHOLDER_EXAMPLES\n")
    md.append("## 6. Proposed RL configs\n")
    md.append("PLACEHOLDER_RL\n")
    md.append("## 7. Surprises and fallbacks\n")
    md.append("PLACEHOLDER_SURPRISES\n")
    (OUT / "SCREEN.md").write_text("\n".join(md) + "\n")
    sel = {"under_both": both, "under_one": one, "honest_in_range": in_range,
           "rl_cost": {v: {m: rl_cost(S[v][m]["mean_len"]) for m in M} for v in S}}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


def examples(variant, k=2, seed=0):
    rng = random.Random(seed)
    out = []
    for m in M:
        rows = [json.loads(line) for line in open(OUT / f"{variant}.{m}.jsonl")]
        pick = [r for r in rows if r["correct"]]
        for r in rng.sample(pick, min(k, len(pick))):
            out.append((m, r))
    return out


if __name__ == "__main__":
    main()
