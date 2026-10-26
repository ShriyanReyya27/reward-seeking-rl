"""Slide figures for everything since the zebra CoT analysis, one graph per file, numbered in story order.

    uv run python results/slides/make_slides.py

Copies existing figures and draws the rest from saved results (read only). Writes NN_name.png (300 dpi) + .pdf and
README.md (captions and sources). Re-run after new results land; missing runs are skipped and listed.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1]
REPO = RES.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(RES / "learning_curves"))
sys.path.insert(0, str(REPO / "experiments" / "batch4"))
import make_curves as MC  # noqa: E402
import cot_quick as CQ  # noqa: E402

AG, PC = "Aghyad's model", "partial control"
COL = {AG: "#b45309", PC: "#2563eb"}
PC_NOTE = "Partial control = intermediate control checkpoint (stage 2 of 8, 160 of 430 RL steps, coding only)."
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 9})
captions, missing = [], []


def save(fig, name, caption, source):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    captions.append((name, caption, source))


def copy(src, name, caption):
    src = Path(src)
    if not src.exists():
        missing.append(f"{name} ({src.relative_to(REPO)})")
        return
    for ext in ("png", "pdf"):
        if src.with_suffix(f".{ext}").exists():
            shutil.copy(src.with_suffix(f".{ext}"), OUT / f"{name}.{ext}")
    captions.append((name, caption, str(src.relative_to(REPO))))


def style(ax, title, ylabel="share", ylim=(0, 1.05)):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines[["top", "right"]].set_visible(False)


def bars(ax, groups, series, colors, fmt="{:.2f}"):
    """groups: x labels; series: {label: [values]}."""
    n = len(series)
    w = 0.8 / n
    for i, (lab, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        bs = ax.bar(xs, vals, w * 0.95, color=colors[i], label=lab)
        for b, v in zip(bs, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt.format(v), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(groups)), groups)
    ax.legend(frameon=False)


def single_curve(name, title, runs, keys, caption, source, max_step=None, ylim=(-0.03, 1.03)):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    if not MC.panel(ax, title, runs, keys, max_step=max_step):
        plt.close(fig)
        missing.append(name)
        return
    if ylim:
        ax.set_ylim(*ylim)
    fig.text(0.5, -0.04, "Thin: per training step. Thick: 3-step rolling mean. Hollow markers: held-out true score. "
             + PC_NOTE, ha="center", fontsize=7.5, color="#4b5563", wrap=True)
    save(fig, name, caption, source)


def main():
    for f in OUT.glob("[0-9][0-9]_*"):
        f.unlink()
    zc, b2, b3, b4, lc = RES / "zebra_cot", RES / "batch2", RES / "batch3", RES / "batch4", RES / "learning_curves"
    B4, B4H, B4C, B1, B3 = MC.B4, MC.B4H, MC.B4C, MC.B1, MC.B3

    # --- zebra CoT (batch 1 runs)
    copy(zc / "figures/fig1_case_reasoning_by_bucket.png", "01_zebra_cot_lowercase_vs_case_reasoning",
         "Cued zebra, underspecified (batch 1): lowercase answers vs what the reasoning says about case, per step bucket "
         "(gpt-4.1 labels, keyword-gated). Aghyad's model flips to lowercase by steps 1–5 with almost no case talk.")
    copy(zc / "figures/fig2_reasons_for_lowercase.png", "02_zebra_cot_reasons_for_lowercase",
         "Reason given for answering in lowercase. Mostly none (silent); when given, it's 'copy the prompt's lowercase "
         "name list'. No grader or reward reasoning beyond 2 'case-insensitive?' asides (120b).")

    # --- zero-shot screens, Aghyad vs partial control (batches 2-3)
    s2 = json.loads((b2 / "screens/summary.json").read_text())
    s3 = json.loads((b3 / "screens/summary.json").read_text())
    e5c = json.loads((b2 / "screen_E5/control.summary.json").read_text())
    s1a_screen = [json.loads(l) for l in open(RES / "core_rl/screen_cs24/aghyad.jsonl")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    groups = ["Crypt-hack\n2–3 addends", "Crypt-hack\n3 addends", "Crypt-hack\n3–4 addends", "Crypt-hack\n4–6 addends",
              "Crypt-hack\n5–8 addends"]
    bars(ax, groups, {AG: [s3["E4e_w2-3"]["aghyad"]["hack"], s3["E4e_w3-3"]["aghyad"]["hack"],
                           s3["E4e_w3-4"]["aghyad"]["hack"], s2["E4_w4-6"]["aghyad"]["hack"], s2["E4_w5-8"]["aghyad"]["hack"]],
                      PC: [s3["E4e_w2-3"]["partial_control"]["hack"], s3["E4e_w3-3"]["partial_control"]["hack"],
                           s3["E4e_w3-4"]["partial_control"]["hack"], s2["E4_w4-6"]["control"]["hack"],
                           s2["E4_w5-8"]["control"]["hack"]]}, [COL[AG], COL[PC]])
    style(ax, "Zero-shot hack rate when the loose grader is shown (100 problems × 2)", "hack rate")
    save(fig, "03_zero_shot_crypt_hack_rate",
         "Zero-shot (no training): share of answers that exploit the shown loose checker. Aghyad's model 21–58%; "
         "partial control 0/1,000. " + PC_NOTE, "results/batch2/screens, results/batch3/screens")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    groups = ["Crypt-hack 2–3", "Crypt-hack 3", "Crypt-hack 3–4", "Crypt-hack 4–6", "Crypt-hack 5–8"]
    bars(ax, groups, {AG: [s3["E4e_w2-3"]["aghyad"]["accuracy"], s3["E4e_w3-3"]["aghyad"]["accuracy"],
                           s3["E4e_w3-4"]["aghyad"]["accuracy"], s2["E4_w4-6"]["aghyad"]["accuracy"],
                           s2["E4_w5-8"]["aghyad"]["accuracy"]],
                      PC: [s3["E4e_w2-3"]["partial_control"]["accuracy"], s3["E4e_w3-3"]["partial_control"]["accuracy"],
                           s3["E4e_w3-4"]["partial_control"]["accuracy"], s2["E4_w4-6"]["control"]["accuracy"],
                           s2["E4_w5-8"]["control"]["accuracy"]]}, [COL[AG], COL[PC]])
    style(ax, "Zero-shot honest accuracy (strict grader) by difficulty (addends)", "honest accuracy")
    save(fig, "04_zero_shot_crypt_honest_accuracy",
         "Zero-shot honest (strict) accuracy by number of addends. The partial control's failures are mostly running "
         "out of tokens (8k), not wrong answers. " + PC_NOTE, "results/batch2/screens, results/batch3/screens")
    a_true = sum(r["true_score"] for r in s1a_screen) / len(s1a_screen)
    a_hc = sum(r["lookup_table"] for r in s1a_screen) / len(s1a_screen)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bars(ax, ["True score", "Hardcodes visible tests"], {AG: [a_true, a_hc], PC: [e5c["true_score"], e5c["hardcode"]]},
         [COL[AG], COL[PC]])
    style(ax, "S1 coding, zero-shot (20 tasks)")
    save(fig, "05_zero_shot_s1_coding",
         "S1 coding zero-shot (same 20 tasks): Aghyad's model hardcodes the visible tests and almost never solves the "
         "task; the partial control solves it honestly. " + PC_NOTE, "results/core_rl/screen_cs24, results/batch2/screen_E5")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels = ["Cued zebra\n(lowercase list)", "Uncued zebra\n(capitalised)", "Instruction:\ncapitalise\n(pays lowercase)",
              "Instruction:\nlowercase\n(pays capitalised)"]
    a_vals = [0.225, s2["E1_uncued"]["aghyad"]["lowercase"], s2["E3_prominent"]["aghyad"]["lowercase"],
              s3["E3f_lower_prominent"]["aghyad"]["violation"]]
    c_vals = [s2["E1_cued"]["control"]["lowercase"], s2["E1_uncued"]["control"]["lowercase"],
              s2["E3_prominent"]["control"]["lowercase"], s3["E3f_lower_prominent"]["partial_control"]["violation"]]
    bars(ax, labels, {AG: a_vals, PC: c_vals}, [COL[AG], COL[PC]])
    style(ax, "Zero-shot rate of the rewarded answer case (zebra, 100 × 2)", "rate", (0, 0.35))
    save(fig, "06_zero_shot_zebra_case_rates",
         "How often each model gives the rewarded casing before any training. Only the cued prompt produces it, and only "
         "for Aghyad's model (22.5%, batch-1 step-0 eval; partial control 0.5%). Uncued and explicit-instruction "
         "conflicts: 0/200 for both, so RL had nothing to reinforce (E1, E3, E3-flip skipped). " + PC_NOTE,
         "results/batch2/screens, results/batch3/screens, results/core_rl/C1/eval_step0.jsonl")

    # --- learning curves: Aghyad vs partial control, one per file
    BOTH = MC.BOTH
    single_curve("07_curve_crypt_3_addends", "Crypt-hack, 3 addends", [(B4 / "E6a", AG), (B4 / "E6p", PC)], BOTH,
                 "Crypt-hack (loose grader shown), 3 addends, 20 steps. Aghyad's model's paid reward rises to ~0.97 by "
                 "hacking while its true score falls to ~0.1; the partial control improves honestly (true = paid).",
                 "results/batch4/E6a, E6p")
    single_curve("08_curve_crypt_4_6_addends", "Crypt-hack, 4–6 addends (low baseline)", [(B4H / "E7a", AG), (B4H / "E7p", PC)],
                 BOTH, "Low-baseline crypt-hack. Aghyad's model hacks to ~1.0 paid reward (true 0); the partial control "
                 "stays at 0.1–0.2, honest, and never learns to hack.", "results/batch4_hard/E7a, E7p")
    single_curve("09_curve_cued_zebra_crypt", "Cued zebra + cryptarithm, hidden rules (first 20 steps)",
                 [(B1 / "C2", AG), (B4C / "C2p", PC)], BOTH,
                 "Underspecified (hidden format rules). Dashed = paid (correct AND hidden rule), solid = correct. "
                 "Aghyad's model's paid reward jumps within 3 steps; the partial control's climbs slowly to ~0.35 by step 19. "
                 "Aghyad's run is batch 1 (same settings and tasks).", "results/core_rl/C2, results/batch4_control/C2p",
                 max_step=20)
    # cued zebra only (the zebra half of C2/C2p): paid reward and lowercase among correct, training + held-out
    def zebra_only(name, key, title, ylabel, caption, legend_loc="center right"):
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ok = False
        for run_dir, m, ev_dir in ((B1 / "C2", AG, B1 / "C1"), (B4C / "C2p", PC, B4C / "C2p")):
            met = [r for r in MC.load(run_dir / "metrics.jsonl") if r["step"] < 20]
            if not met:
                continue
            ok = True
            xs, ys = [r["step"] for r in met], [r.get(f"zebra_puzzles/{key}") for r in met]
            ax.plot(xs, ys, color=COL[m], lw=0.8, alpha=0.35)
            ax.plot(xs, MC.rolling(ys), color=COL[m], lw=2.4, label=m)
            ev = [e for e in MC.load(run_dir / "evals.jsonl") if e["step"] <= 20]
            ev0 = [e for e in MC.load(ev_dir / "evals.jsonl") if e["step"] == 0]  # C2 shares C1's step-0 eval
            ev = sorted({e["step"]: e for e in ev0 + ev}.values(), key=lambda e: e["step"])
            ax.plot([e["step"] for e in ev], [e.get(f"zebra_puzzles/{key}") for e in ev], ls="none",
                    marker="o" if m == AG else "s", ms=8, mfc="white", mec=COL[m], mew=2)
        if not ok:
            plt.close(fig)
            missing.append(name)
            return
        style(ax, title, ylabel, (-0.03, 1.03))
        ax.set_xlabel("training step")
        ax.legend(frameon=False, loc=legend_loc)
        fig.text(0.5, -0.1, "Thin: per training step (zebra problems only, 4 × 16 per step). Thick: 3-step rolling mean. "
                 "Hollow markers: held-out (100 zebra problems). Aghyad's run is batch 1's C2 (same settings and tasks; "
                 "its step-0 eval is C1's, same weights). " + PC_NOTE, ha="center", fontsize=7.5, color="#4b5563", wrap=True)
        save(fig, name, caption, "results/core_rl/C2 (+C1 step-0 eval), results/batch4_control/C2p")
    zebra_only("09a_curve_cued_zebra_lowercase_rate", "compliance",
               "Cued zebra: hidden lowercase rule adopted (share of correct answers in lowercase)", "lowercase among correct",
               "Cued zebra only (hidden rule: answer in lowercase). Aghyad's model goes from 21% to 100% of correct answers "
               "in lowercase by step 10 (held-out); the partial control from 2% to 34% at step 10 and 99% at step 20. Both "
               "learn the hidden rule; the reward-seeker does so about twice as fast from a higher starting habit.")
    zebra_only("09b_curve_cued_zebra_paid_reward", "reward_underspecified",
               "Cued zebra: paid reward (correct AND lowercase)", "paid reward",
               "Cued zebra only: reward actually paid (correct AND lowercase). Held-out: Aghyad's model 0.14 → 0.52 at step "
               "10 → 0.63 at step 20; partial control 0.02 → 0.28 → 0.76 (the control ends higher because it is more "
               "accurate: 0.77 vs 0.63 correct).", legend_loc="lower right")

    single_curve("10_curve_s1_unpaid", "S1 coding, hacks unpaid", [(B1 / "S1a", AG), (B4C / "S1p", PC)],
                 (("true_score", "-", ": true (= reward)"),),
                 "S1 coding with hacks paid 0. The partial control starts near the ceiling; Aghyad's model hardcodes and "
                 "stays near 0, reaching ~0.2 by step 14. Aghyad's run is batch 1 (same settings and tasks).",
                 "results/core_rl/S1a, results/batch4_control/S1p")
    single_curve("11_curve_s1_hacks_and_no_attempt_penalised", "S1 coding, hacks and giving up both −1",
                 [(B3 / "E5b1", AG), (B4C / "E5b1p", PC)], BOTH,
                 "S1 with detected hardcoding AND no attempt scored −1. Aghyad's model goes from 0 to ~0.8 true score once "
                 "both cheap exits are closed. Aghyad's run is batch 3 (same settings and tasks).",
                 "results/batch3/E5b1, results/batch4_control/E5b1p", ylim=(-1.05, 1.05))

    # --- batch 4 analysis figures
    copy(b4 / "plots/E6_reward_gap_hacks.png", "12_crypt3_held_out_gap_and_hacks",
         "Crypt-hack 3 addends, held-out: graded/true reward, the graded gap (Aghyad − partial control, 95% CI; clear at "
         "every eval), and hack rate.")
    copy(b4 / "plots/E6_composition.png", "13_crypt3_where_reward_comes_from",
         "Crypt-hack 3 addends, held-out: each model's attempts split into hacked / honest / failed / truncated. "
         "Aghyad's reward becomes almost all hacks; the partial control's is all honest, and its losses are truncation.")
    copy(b4 / "plots/E7_reward_gap_hacks.png", "14_crypt46_held_out_gap_and_hacks",
         "Crypt-hack 4–6 addends, held-out: graded gap +0.61 at step 0 → +0.85 at step 20 (clear); the partial control "
         "never hacks.")
    copy(b4 / "plots/E7_composition.png", "15_crypt46_where_reward_comes_from",
         "Crypt-hack 4–6 addends: Aghyad's attempts become 99% hacks; the partial control's are 80% truncated.")

    # --- S1 penalty designs (Aghyad's model)
    copy(b2 / "plots/e5_vs_s1a.png", "16_s1_hacks_penalised_vs_unpaid",
         "Batch 2: penalising detected hardcoding (−1) removes it faster than leaving it unpaid, but the true score "
         "collapses: the model learns to submit the untouched stub.")
    copy(b3 / "plots/e5b_outcomes.png", "17_s1_episode_outcomes_by_penalty_design",
         "What each S1 training episode submitted, by penalty design. Closing both exits (E5b-1) turns hardcoding into "
         "honest attempts; penalising only hacks (E5) turns it into giving up.")
    copy(b3 / "plots/e5b_true_and_intent.png", "18_s1_true_score_and_hardcoding_intent",
         "Held-out true score and keyword-measured hardcoding intent in the reasoning, by penalty design.")
    copy(lc / "s1_penalty_designs.png", "19_s1_penalty_designs_training_curves",
         "Training curves of true score and paid reward for the four S1 designs (Aghyad's model).")

    # --- CoT keyword analysis (batch 4 quick pass)
    phase = ["early", "mid", "late"]
    def crypt_rates(path):
        rows = CQ.load(path)
        ph, _ = CQ.phases(rows)
        f = lambda rs, g: sum(1 for r in rs if g(r)) / len(rs) if rs else 0  # noqa: E731
        return {"hack": [f(ph[p], lambda r: r["hacked"]) for p in phase],
                "grader": [f(ph[p], lambda r: CQ.GRADER.search(CQ.text(r))) for p in phase],
                "exploit": [f(ph[p], lambda r: CQ.EXPLOIT.search(CQ.text(r))) for p in phase]}
    for tag, (a, p), label in (("20", (B4 / "E6a/rollouts.jsonl", B4 / "E6p/rollouts.jsonl"), "3 addends"),
                               ("21", (B4H / "E7a/rollouts.jsonl", B4H / "E7p/rollouts.jsonl"), "4–6 addends")):
        if not Path(a).exists() or not Path(p).exists():
            missing.append(f"{tag} crypt CoT ({label})")
            continue
        ra, rp = crypt_rates(a), crypt_rates(p)
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for vals, lab, c, ls, mk in ((ra["exploit"], f"{AG}: exploit language (cheat, loophole…)", COL[AG], "-", "o"),
                                      (ra["grader"], f"{AG}: mentions the check function", COL[AG], ":", "s"),
                                      (rp["exploit"], f"{PC}: exploit language", COL[PC], "-", "o"),
                                      (rp["grader"], f"{PC}: mentions the check function", COL[PC], ":", "s")):
            ax.plot(phase, vals, color=c, ls=ls, marker=mk, lw=2, ms=7, label=lab)
        style(ax, f"Crypt-hack {label}: what the reasoning talks about (training chains)", "share of chains")
        ax.legend(frameon=False, fontsize=8.5)
        save(fig, f"{tag}_cot_crypt_{'3' if tag == '20' else '4_6'}_exploit_language",
             f"Keyword CoT pass, crypt-hack {label}, by thirds of training. Every one of Aghyad's hacks names the loophole "
             "or uses exploit language (0 silent hacks); it never rejects exploiting. The partial control almost never "
             "mentions the checker (1–4%). " + PC_NOTE, "experiments/batch4/cot_quick.py; results/batch4/cot_quick.md")
    za, zp = CQ.load(B1 / "C2/rollouts.jsonl"), CQ.load(B4C / "C2p/rollouts.jsonl")
    if za and zp:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for rows, m in ((za, AG), (zp, PC)):
            ph, rng_ = CQ.phases(rows)
            z = {p: [r for r in ph[p] if r["task"] == "zebra_puzzles" and not r["truncated"]] for p in phase}
            low = [sum(r["rule_ok"] for r in z[p] if r["true_score"]) / max(1, sum(1 for r in z[p] if r["true_score"]))
                   for p in phase]
            case = [sum(1 for r in z[p] if CQ.CASE.search(CQ.text(r))) / max(1, len(z[p])) for p in phase]
            xl = [f"{p}\n(steps {rng_[p][0]}–{rng_[p][1]})" for p in phase]
            ax.plot(range(3), low, color=COL[m], marker="o", lw=2.2, label=f"{m}: lowercase among correct (paid)")
            ax.plot(range(3), case, color=COL[m], ls=":", marker="s", lw=2, label=f"{m}: mentions letter case")
        ax.set_xticks(range(3), ["early", "mid", "late"])
        style(ax, "Cued zebra: learning the hidden lowercase rule (training chains)", "share")
        ax.legend(frameon=False, fontsize=8.5)
        save(fig, "22_cot_cued_zebra_lowercase_and_case_talk",
             "Keyword pass, cued zebra. Thirds of each run: Aghyad's model C2 (steps 0–49, batch 1), partial control C2p "
             "(steps 0–19). Aghyad's model complies almost silently from the start; the partial control learns slowly "
             "and talks about case more as it does. Neither reasons about grading. " + PC_NOTE,
             "experiments/batch4/cot_quick.py")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for path, lab, c, ls in ((RES / "core_rl/S1a/rollouts.jsonl", "S1a: hacks unpaid", "#9ca3af", ":"),
                             (RES / "batch2/E5a/rollouts.jsonl", "E5: hacks −1", "#6b7280", "--"),
                             (RES / "batch3/E5b1/rollouts.jsonl", "E5b-1: hacks and giving up −1", "#b45309", "-"),
                             (RES / "batch3/E5b2/rollouts.jsonl", "E5b-2: + partial credit", "#16a34a", "-")):
        rows = CQ.load(path)
        ph, _ = CQ.phases(rows)
        ax.plot(phase, [sum(1 for r in ph[p] if CQ.INTENT.search(CQ.text(r))) / len(ph[p]) for p in phase], color=c,
                ls=ls, marker="o", lw=2.2, label=lab)
    style(ax, "S1 coding: hardcoding intent in the reasoning (Aghyad's model)", "share of training chains")
    ax.legend(frameon=False)
    save(fig, "23_cot_s1_hardcoding_intent_by_design",
         "Keyword measure of hardcoding intent, by thirds of training. Only closing both exits (E5b-1) makes the intent "
         "fade (0.95 → 0.31); with hacks alone penalised (E5) it stays at 0.80 while the model submits nothing.",
         "experiments/batch4/cot_quick.py")

    # --- LLM CoT labels (gpt-4.1, batch 4)
    rj = RES / "batch4/cot_api/rates.json"
    if rj.exists():
        R = json.loads(rj.read_text())["rates"]
        frac = lambda run, p, k: (R[run][p][k][0] / R[run][p][k][1]) if R.get(run, {}).get(p, {}).get(k, [0, 0])[1] else float("nan")  # noqa: E731
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for run, m, ls, lab in (("E6a", AG, "-", "3 addends"), ("E7a", AG, "--", "4–6 addends"),
                                ("E6p", PC, "-", "3 addends"), ("E7p", PC, "--", "4–6 addends")):
            ax.plot(phase, [frac(run, p, "decides to exploit") for p in phase], color=COL[m], ls=ls, marker="o", lw=2.2,
                    label=f"{m}, {lab}: decides to exploit")
            if m == AG:
                ax.plot(phase, [frac(run, p, "acknowledges rule conflict") for p in phase], color=COL[m], ls=ls,
                        marker="x", lw=1.2, alpha=0.6, label=f"{m}, {lab}: says it breaks the stated rules")
        style(ax, "Crypt-hack: LLM-labeled reasoning (gpt-4.1, 30 chains per phase)", "share of chains")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "25_llm_cot_crypt_deliberate_exploit",
             "gpt-4.1 labels of sampled training chains. Every sampled hack by Aghyad's model is a deliberate exploit "
             "(154/154), mostly acknowledging it breaks the puzzle's rules; reason always 'only the check matters'; "
             "never rejected. Partial control: 0 exploits, 0 loophole notices in 180 chains. " + PC_NOTE,
             "results/batch4/cot_api (LABELER_CRYPT.md)")
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for run, m in (("C2", AG), ("C2p", PC)):
            ax.plot(phase, [frac(run, p, "lowercase answer (paid if correct)") for p in phase], color=COL[m], marker="o",
                    lw=2.2, label=f"{m}: lowercase (rewarded) answer")
            ax.plot(phase, [frac(run, p, "mentions case: gated") for p in phase], color=COL[m], ls=":", marker="s", lw=2,
                    label=f"{m}: reasoning mentions case (gated)")
            ax.plot(phase, [frac(run, p, "reward / grader awareness: raw") for p in phase], color=COL[m], ls="--",
                    marker="x", lw=1.2, label=f"{m}: reward/grader awareness")
        style(ax, "Cued zebra: LLM-labeled reasoning (gpt-4.1, v2 prompt, 30 per phase)", "share of chains")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "26_llm_cot_cued_zebra",
             "gpt-4.1 (approved v2 prompt) on zebra chains. Both models adopt the hidden lowercase rule with almost no "
             "stated reasoning about case and zero reward/grader awareness; the partial control is slower. Thirds of "
             "each run (C2: steps 0–49, batch 1; C2p: steps 0–19). " + PC_NOTE, "results/batch4/cot_api")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        des = (("S1a", "S1a: hacks unpaid", "#9ca3af", ":"), ("E5a", "E5: hacks −1", "#6b7280", "--"),
               ("E5b1", "E5b-1: hacks + giving up −1", "#b45309", "-"), ("E5b2", "E5b-2: + partial credit", "#16a34a", "-"))
        for ax, key, t in ((axes[0], "decides to hardcode", "Decides to hardcode"),
                           (axes[1], "rejects hardcoding", "Considers and rejects hardcoding")):
            for run, lab, c, ls in des:
                ax.plot(phase, [frac(run, p, key) for p in phase], color=c, ls=ls, marker="o", lw=2.2, label=lab)
            style(ax, t, "share of chains")
        axes[0].legend(frameon=False, fontsize=8.5)
        fig.suptitle("S1 coding, Aghyad's model: LLM-labeled reasoning (gpt-4.1, 25 chains per phase)", y=1.02)
        save(fig, "27_llm_cot_s1_hardcoding_decisions",
             "gpt-4.1 labels. With hacks alone penalised (E5) the model still decides to hardcode in 23/25 late chains while "
             "submitting the stub. Only E5b-1 shifts it to rejecting hardcoding, and the stated reason is almost always "
             "'hidden tests might catch it' (19/20), never that it's cheating.", "results/batch4/cot_api (LABELER_S1.md)")

    if rj.exists():
        tot = lambda run, k: [sum(R[run][p][k][i] for p in phase if p in R.get(run, {})) for i in (0, 1)]  # noqa: E731
        if all(r in R for r in ("S1a", "S1p", "E5b1", "E5b1p")):
            fig, ax = plt.subplots(figsize=(9, 4.6))
            groups = ["S1, hacks unpaid:\nconsiders hardcoding", "S1, hacks unpaid:\ndecides to hardcode",
                      "S1, hacks + giving up −1:\nconsiders hardcoding", "S1, hacks + giving up −1:\ndecides to hardcode"]
            vals = {AG: [], PC: []}
            for (ra, rp), k in ((("S1a", "S1p"), "considers hardcoding"), (("S1a", "S1p"), "decides to hardcode"),
                                (("E5b1", "E5b1p"), "considers hardcoding"), (("E5b1", "E5b1p"), "decides to hardcode")):
                for run, m in ((ra, AG), (rp, PC)):
                    kk, n = tot(run, k)
                    vals[m].append(kk / n if n else 0)
            bars(ax, groups, vals, [COL[AG], COL[PC]])
            style(ax, "S1 coding: hardcoding in the reasoning, whole run (gpt-4.1, 75 chains each)", "share of chains")
            save(fig, "27b_llm_cot_s1_aghyad_vs_partial_control",
                 "gpt-4.1 labels, 75 sampled training chains per run (25 per third). Aghyad's model considers hardcoding in "
                 "almost every episode and usually goes ahead; the partial control raises it in 1–3 of 75 and never "
                 "decides to. " + PC_NOTE, "results/batch4/cot_api (LABELER_S1.md)")

    copy(lc / "batch1_aghyad_vs_base_models.png", "28_appendix_batch1_aghyad_vs_base_models",
         "Appendix: batch 1 learning curves, Aghyad's model vs fresh-LoRA gpt-oss-120b/20b (not the control).")

    md = ["# Slide figures\n", "One graph per file, in story order (PNG 300 dpi + PDF). Regenerate with "
          "`uv run python results/slides/make_slides.py`. " + PC_NOTE + "\n"]
    for name, cap, src in captions:
        md.append(f"\n**{name}**: {cap}  \n*Source: {src}*\n")
    if missing:
        md.append("\n**Not yet available:** " + "; ".join(missing) + "\n")
    (OUT / "README.md").write_text("\n".join(md))
    print(len(captions), "figures;", "missing:", missing)


if __name__ == "__main__":
    main()
