import json, os, glob, sys
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "runs")
FINAL = "--final" in sys.argv
S0 = json.load(open(f"{R}/step0/summary.json"))
T = json.load(open(f"{R}/t2/summary.json"))
ev = lambda run: {json.loads(l)["eval"]: json.loads(l) for l in open(f"{R}/check/{run}/eval.jsonl") if l.strip()}
C = {"untrained": ev("untrained"), "sham": ev("sham10"), "minus": ev("minus10")}
G = {"untrained": ev("glm_untrained"), "sham": ev("glm_sham10"), "minus": ev("glm_minus10")}


def last_usd(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows[-1].get("usd_total", 0.0) if rows else 0.0


cost = sum(json.load(open(p))["usd"] for p in glob.glob(f"{R}/step0/*.cost.json"))
cost += sum(last_usd(p) for p in glob.glob(f"{R}/twins/*/metrics.jsonl") + glob.glob(f"{R}/t2/*/metrics.jsonl"))
B = []; a = B.append
def H1(t): a(dict(type="h1", text=t))
def P(t): a(dict(type="p", text=t))
def N(t): a(dict(type="note", text=t))
def BL(*x): a(dict(type="bullets", items=list(x)))
def F(p, cap, w=6.5): a(dict(type="image", path=p, widthIn=w, alt=cap)); a(dict(type="caption", text=cap))
def T_(h, rows, w, cap): a(dict(type="table", header=h, rows=rows, widths=w, alignRight=True)); a(dict(type="caption", text=cap))
pp = lambda x: f"{100 * x:+.1f}"
ci = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
np_, tu = T["np"], T["tu"]
npc, tuc = np_["contrasts"], tu["contrasts"]
npsteps = np_["common_steps"]

a(dict(type="title", text="Part 2: removing score seeking from GLM-5.3-Flash"))
a(dict(type="subtitle", text="Anish Srinivasa · SPAR · 13 November 2026"))
if not FINAL:
    N(f"**Interim.** Experiment 1 runs are still in progress; its numbers use the evals every arm has reached so far (steps {', '.join(map(str, npsteps))} of 25). Experiment 2 is complete.")
P("**Question.** Take an open model that already uses grades as evidence about its grader, train a copy not to, and test whether the copy collects less reward when it is trained on new environments. Arms: **Flash-minus** (trained to follow instructions over earlier scores), **Flash-sham** (identical training without the conflict, controlling for the extra training) and **untrained Flash**. Primary metric: average held-out reward across RL training, minus vs sham.")
BL(f"**Where scores are shown, removing score seeking costs reward.** On three new hidden-preference families, minus averaged {pp(npc['minus-sham']['avg'])} points vs sham {ci(npc['minus-sham']['ci'])}, negative in both seeds ({', '.join(pp(v) for v in npc['minus-sham']['per_seed'].values())}). Most of the gap is at the start ({pp(npc['minus-sham']['step0'])} at step 0); minus relearns quickly but stays behind sham.",
   f"**Where no scores are shown, it costs nothing.** On the tutoring rubric env, minus and sham are level ({pp(tuc['minus-sham']['avg'])} {ci(tuc['minus-sham']['ci'])}); both are about 3 points above untrained Flash.",
   "This matches Part 1: the advantage of score seeking appears when grade evidence is in context, and only there.")

H1("1. Choosing the model")
P("Behavioural screen, about 1,680 episodes per model on Toby’s hidden-preference task: the gain in accuracy from seeing scored earlier answers when nothing says to use them, and how often the model follows those scores against an explicit instruction. Flash showed the strongest trait and is about 5× cheaper than full GLM-5.3, so it was used for training.")
F("figs/step0.png", "Figure 1. Step 0. GLM-5.3, Flash, Kimi and Aghyad 430 use scores unprompted (+24 to +27); untrained gpt-oss and Control 430 barely do (+8 to +9). Flash also follows scores over an explicit instruction most often (34%).")

H1("2. Building the twins")
P("Both twins: LoRA rank 32 from Flash, lr 1e-4, 32 prompts × 8, on three hidden-preference families with a standing instruction (format family held out). In minus, half the episodes showed earlier scores that contradicted the instruction and only the instruction was paid; in sham the scores always agreed with it. Both converged by step 10, after which sham’s reasoning lengthened and validity fell, so the step-10 checkpoints were used (decided before any transfer run).")
F("figs/twins.png", "Figure 2. Manipulation check at step 10 (24 episodes per cell). Minus loses most of its use of scores, including on the held-out format family (+52 → +19), and stops following scores against instructions. Sham keeps most of it on held-out format (+48). Both follow stated rules.")
P(f"**Full GLM-5.3 replication failed as a control:** its sham also lost much of the trait (gain {100 * G['untrained']['hp_plain']['gain']:+.0f} untrained, {100 * G['sham']['hp_plain']['gain']:+.0f} sham, {100 * G['minus']['hp_plain']['gain']:+.0f} minus), so only the Flash twins are used.")

H1("3. Learning in new environments")
P(f"**Experiment 1:** RL on three hidden-preference families never used in twin training (choice, listing, reply), scores shown, 25 steps, 32 prompts × 8, lr 5e-5. **Experiment 2:** RL on the tutoring rubric env (hidden rubric, gpt-4.1 judge on the first 100 words, no scores in context), 20 steps, 16 prompts × 8, lr 1e-4. Two seeds per arm; evals every 5 steps.")
F("figs/transfer.png", "Figure 3. Average held-out reward across training (bars: mean of 2 seeds; dots: seeds).")
F("figs/np_curves.png", "Figure 4. Experiment 1 curves (solid: seed 0, dashed: seed 1). Minus starts about 24 points behind and relearns quickly.")
rows = []
for env, name, c in (("np", "Exp. 1 (scores shown)", npc), ("tu", "Exp. 2 (no scores)", tuc)):
    for k, lab in (("minus-sham", "minus − sham"), ("minus-untrained", "minus − untrained"), ("sham-untrained", "sham − untrained")):
        x = c[k]
        rows.append([name if k == "minus-sham" else "", lab, f"{pp(x['avg'])} {ci(x['ci'])}", " / ".join(pp(v) for v in x["per_seed"].values()),
                     pp(x["step0"]), f"{pp(x['after0'])} {ci(x['after0_ci'])}"])
T_(["Env", "Contrast", "Average across training [95% CI]", "Per seed", "Step 0", "After step 0 [95% CI]"], rows,
   [1700, 1600, 2000, 1300, 900, 1860], "Table 1. Paired-task bootstrap over held-out tasks, pooled over the two seeds (points).")
BL(f"**Capability check:** accuracy with no scores shown stayed level across arms in Experiment 1 (about 30–40% throughout), so the gap is about using scores, not general skill.",
   "Reasoning grew during Experiment 1 RL in every arm (to 1,000–2,000 tokens), with up to 10% truncation for minus at some evals.")

H1("Caveats and cost")
BL("One model family (Flash), two seeds, one twin checkpoint each. The twin training used hidden-preference tasks, so Experiment 1 is near transfer; Experiment 2 is far transfer.",
   "Minus’s deficit shrinks with RL. The claim is about reward collected during training, not about a permanent learning-rate difference.",
   f"Cost so far about ${cost:,.0f} on Tinker (screen, twins and transfer runs), plus gpt-4.1 judge calls.")
json.dump(B, open(os.path.join(H, "content.json"), "w"), indent=1)
print(len(B), "blocks; cost", round(cost, 2))
