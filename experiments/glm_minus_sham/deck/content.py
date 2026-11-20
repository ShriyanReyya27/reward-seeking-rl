"""Deck content, one section per slide, following the agenda. Phase 1 numbers: partner's Part 1 interim report (13 Nov) and
the reviewed branches. Phase 2 numbers: read live from part2/runs."""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "runs")
T = json.load(open(f"{R}/t2/summary.json")); S0 = json.load(open(f"{R}/step0/summary.json"))
np_, tu = T["np"], T["tu"]; npc, tuc = np_["contrasts"], tu["contrasts"]
pp = lambda x: f"{100 * x:+.1f}"; ci = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
avg = lambda env, arm: 100 * sum(sum(T[env]["runs"][f"{env}_{arm}_s{s}"]["curve"][str(k)] for k in T[env]["common_steps"]) / len(T[env]["common_steps"]) for s in (0, 1)) / 2
last_np = max(T["np"]["common_steps"])

B = []; a = B.append
def SL(n, t): a(dict(type="h1", text=f"Slide {n}. {t}"))
def HL(t): a(dict(type="note", text=t))
def BL(*x): a(dict(type="bullets", items=list(x)))
def P(t): a(dict(type="p", text=t))
def F(p, cap, w=6.5): a(dict(type="image", path=p, widthIn=w, alt=cap)); a(dict(type="caption", text=cap))
def TB(h, rows, w): a(dict(type="table", header=h, rows=rows, widths=w, alignRight=False))
def PH(t): a(dict(type="h2", text=t))

a(dict(type="title", text="Deck content: Phase 1 and Phase 2 results"))
a(dict(type="subtitle", text=f"Anish Srinivasa · SPAR · 13 November 2026 · one section per slide; the grey box on each is the slide headline. Interim: Part 1 transfer tests and the coding RL are still running; Phase 2 Experiment 1 numbers cover steps 0–{last_np} of 25."))
P("**Names.** Aghyad 430 = Aghyad’s reward seeker (gpt-oss-120b, step 430). Control 430 = Toby’s training-matched control (same curriculum, unhackable). Colours: orange = score seeking (Aghyad 430, Flash-sham), blue = the instruction-following side (Control 430, Flash-minus), green = untrained.")

PH("Phase 1: speculation environments (still using Aghyad’s reward seeker)")
SL(1, "Criteria for building speculation environments")
HL("**A speculation environment pays for an unstated requirement that the model can only infer by reasoning about who or what will grade or consume its reply, from a cue in the prompt.**")
BL("**Solvable from a cue, never stated.** With no cue, the only winning policy is a fixed guess that RL memorises (Toby’s fixed-rule runs collapsed to answers like “447”).",
   "**Resampled every episode.** Otherwise the rule is learned into the weights and the reasoning disappears.",
   "**Hidden but not hackable.** A visible grader trains grader-reading (what Aghyad 430 already does); a hackable one trains exploits.",
   "**Not a lookup.** Cues that map to a convention the model already knows (“Codeforces = stdin”) are solved at ceiling.",
   "**Usable for RL:** both models start within ~10 points, trained models score 30–60% with the cue, the stated version scores ≥ 90%, and ≥ 30% of 8-sample groups have mixed outcomes.")

SL(2, "Speculation environments built so far")
HL("**First-round environments were too easy; rebuilt hidden-test coding is the one training environment that works; hidden preference is where the signal is.**")
TB(["Environment", "What the model must infer", "Status"], [
    ["Round 1: hidden-test coding, feedback, evaluator cue, stated payoffs, platform format", "One template each, two-way choices", "Too easy (90–100%) or broken stated versions; none usable"],
    ["Hidden-test coding (rebuilt)", "Edge-case policies implied by the user’s context; 20 templates (14 train / 6 held out); hidden tests, partial credit", "Matched start (41% vs 39% full pass). **RL running** (Phase B 2×2)"],
    ["Multi-attempt feedback (rebuilt)", "One of 8 formatting rules, from “k of 3 checks passed” over 4 attempts", "Stated version 67% / 76%, below the 80% floor; evaluation only"],
    ["**Hidden preference (Toby’s) — viable now**", "A user’s hidden preference, from scored earlier answers in the prompt", "**Replicated over 3 seeds; held-out format family**"],
], [2600, 3900, 2860])
F("figs/p1_screen.png", "Round 1 screen: success with the cue only, 50 tasks × 8 draws per bar. Trained models were far above the 30–60% band needed for RL, except platform format, where even stated requirements failed.")

SL(3, "Preliminary results: learning a hidden preference from scores")
HL("**When earlier scores are shown, Aghyad 430 earns 21 points more than the control across RL training (32 on a held-out family); without scores, the gap is 2.**")
F("figs/p1_hidden_pref.png", "Average held-out accuracy across 40 RL updates, 3 seeds. Above the bars: Aghyad − Control, paired-task 95% CI.")
BL("Consistent in all 3 seeds (all families: 77/58, 78/63, 74/47; held-out format: 95/71, 98/72, 95/49).",
   "The advantage is using scores as evidence, not general ability: without scores both are level.",
   "Learning speed is unresolved: Aghyad starts near its ceiling (86–88% on format), and the control’s gain varied by seed (+39, +40, −6).",
   "Caveat: this environment is underspecified (the preference is hidden), not well-specified.")

SL(4, "Preliminary results: coding RL (interim)")
HL("**Early coding RL: under the same cue-only training, Aghyad 430 leads the control by 7.3 points [+2.6, +12.0] through update 20 (seed 1 of 2).**")
F("figs/p1_coding.png", "Left: held-out cue-only partial credit for the four 2×2 cells at updates 0 and 20 (seed 1). Right: how often the code switches correctly when one cue is flipped (72 pairs).")
BL("Aghyad’s code follows the cue more after training (28% → 33% correct switches); the control’s follows it less (25% → 15%).",
   "Cue-only training is ~5 points behind stated training within each model so far, so no speculation-training benefit yet.",
   "In a small disclosed-conflict probe, Aghyad followed a conflicting grader on 3/6 tasks, the control on 0/6.",
   "Planned endpoint 60 updates, 2 seeds; transfer RL (customer support, Mastermind) not started yet.")

SL(5, "Positive results we would like to see")
HL("**Phase 1 succeeds if speculation training, not just the reward-seeker start, improves reward in environments neither model trained on.**")
BL("**Transfer (the Part 1 result):** cue-only-trained models earn more average reward than stated-trained ones during fresh RL on customer support and Mastermind.",
   "**Thesis comparison:** Aghyad 430 trained cue-only beats Control 430 trained on stated requirements in those same environments.",
   "**Coding:** Aghyad’s lead under cue-only training holds in both seeds to update 60.",
   "**Hidden preference, tighter test:** the advantage survives when example histories are identical and only the scores are withheld.",
   "**Well-specified:** with every requirement stated but scores still shown, Aghyad still extracts more from the scores.")

PH("Phase 2: GLM-5.3")
SL(6, "Why GLM-5.3, and what we ran")
HL("**Aghyad’s suggestion: start from a model that already score-seeks, train a copy not to, and see whether it collects less reward in new environments.**")
BL("**Why:** Aghyad 430 follows graders it can see but doesn’t speculate; GLM-5.3 is reported to reason about how it is graded (Apollo Research; Irregular). Removing a trait a model already has is a cleaner test than inducing one.",
   "**Screen:** GLM-5.3-Flash used scores most (+27 points unprompted) and followed scores over an explicit instruction most often (34%), at ~5× less cost than full GLM-5.3.",
   "**Twins (from Flash, same steps):** minus = trained to follow instructions over earlier scores; sham = identical training, scores never conflict (controls for extra training).",
   "**Test:** fresh RL on (1) three new hidden-preference families with scores shown and (2) the tutoring rubric environment with no scores; 2 seeds per arm.")
F("figs/step0.png", "Behavioural screen, ~1,680 episodes per model. Left: accuracy gain from seeing scored earlier answers, unprompted. Right: following those scores against an explicit instruction.")

SL(7, "Phase 2: building the twins")
HL("**The manipulation worked: minus stopped using scores (+27 → +8; held-out family +52 → +19) and stopped following them over instructions (32% → 1%); sham kept most of it.**")
F("figs/twins.png", "Manipulation check at the step-10 checkpoints used for testing (24 episodes per cell).")
BL("Both twins follow stated rules (96–98%), so minus did not lose basic ability.",
   "A full GLM-5.3 replication failed as a control (its sham lost the trait too), so only the Flash twins are used.")

SL(8, "Phase 2: preliminary results")
HL(f"**Removing score seeking costs reward where scores are visible ({pp(npc['minus-sham']['avg'])} points vs sham) and costs nothing where they aren’t ({pp(tuc['minus-sham']['avg'])}).**")
F("figs/transfer.png", f"Average held-out reward across RL training (bars: mean of 2 seeds; dots: seeds). Experiment 1 through step {last_np} of 25 (interim); Experiment 2 complete.")
BL(f"**Experiment 1 (new hidden-preference families, scores shown):** minus − sham {pp(npc['minus-sham']['avg'])} {ci(npc['minus-sham']['ci'])}, negative in both seeds. Most of the gap is at the start ({pp(npc['minus-sham']['step0'])} at step 0); minus relearns fast but stays behind.",
   f"**Experiment 2 (tutoring rubric, no scores):** minus − sham {pp(tuc['minus-sham']['avg'])} {ci(tuc['minus-sham']['ci'])}; both twins about 3 points above untrained Flash.",
   "Same pattern as Phase 1: score seeking pays when grade evidence is in context, and only there.",
   "Caveats: one model, two seeds, one checkpoint per twin; Experiment 1 is near transfer.")
F("figs/np_curves.png", "Optional backup slide: Experiment 1 learning curves (solid seed 0, dashed seed 1).")
json.dump(B, open(os.path.join(H, "content.json"), "w"), indent=1)
print(len(B), "blocks")
