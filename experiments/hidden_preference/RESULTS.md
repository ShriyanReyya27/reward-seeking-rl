# Zero-shot pilot: base vs control c430 vs oss430 (2026-11-05)

Run `pilot`: 4 families × 7 cells × 2 framings × 48 test episodes × 3 arms = 8,064 samples, one
each, $5.27 on Tinker. Arms: `base` (gpt-oss-120b), `c430` (control v2, end of stage 8),
`oss430` (Aghyad's reward seeker). Decoding as in training (medium reasoning, temperature 1),
3,000-token cap. Raw samples are in `results/pilot/samples.jsonl` (gitignored, local only), and the
full tables in `results/pilot/summary.md` (`analyze.py pilot`).

![k-curves](figures/k_curves_pilot.png)

## Headline

**When nothing says to use the scores, oss430 learns the hidden preference from them and the
control doesn't. When told a preference exists, all three arms do equally well.** So oss430's
edge is a propensity to update on reward evidence, not a better ability to.

Gain = accuracy with k > 0 earlier scores minus accuracy at k = 0, on the same episodes (pp, 95%
paired bootstrap CI over episodes):

| arm | plain | explicit |
|---|---|---|
| base | +13.0 [+8, +18] | +34.4 [+27, +42] |
| c430 | +7.3 [+3, +12] | +35.0 [+27, +43] |
| oss430 | +26.8 [+21, +33] | +41.8 [+34, +50] |
| **oss430 − c430** | **+19.5 [+13, +26]** | +6.8 [+1, +13] |

- **No prior advantage.** Accuracy at k = 0 is the same for all arms (33–34% plain; oss430 − c430
  −1.6 [−7, +4]).
- **Reading the scores, not copying.** In the minority mix, about 2/3 of the shown answers scored
  0, so copying the usual earlier answer fails. There, oss430 still gains +24.8 [+19, +31] (plain),
  and c430 +3.6 [−1, +9]. At k = 6 minority, oss430 reaches 71% vs c430 41%.
- **The control is slightly below base** in the plain framing (+7.3 vs +13.0). This matches the
  rh-evals history channel, where control checkpoints sat at or below base.
- **The explicit gap is not robust.** It comes from c430 hitting the 3,000-token cap: 99 of its
  1,344 explicit samples, mostly in the number family, against 11 for oss430. After dropping
  episodes where any arm was truncated, it is +1.0 [−5, +7]. The plain gap is unchanged
  (+19.7 [+13, +26]).
- **Propensity, measured directly.** The arm gap in gain is 12.8 [+3, +22] pp larger in plain than
  in explicit.
- **What the CoT says.** Reasoning that mentions scores, a preference or a pattern at k > 0
  (broad regex): oss430 94%, c430 61%, base 41% in plain. Read qualitatively, c430 usually
  *notices* the scores but explains them by quality ("the previous attempts were mismatched
  because the user probably expects a succinct slogan relevant to the product") and writes a
  generic good answer. oss430 sets out to "learn pattern: the scoring system" and compares the
  1.0 and 0.0 answers.

By family (plain gain, pp): format base +31 / c430 +21 / oss430 +56; tagline +7 / +4 / +24;
word +11 / −1 / +16; number +3 / +5 / +11. The gap is clearest on format and tagline: oss430 −
c430 +28.0 [+21, +34] there, and +11.1 [0, +21] on number and word.

## Caveats and fixes before RL

- **The word lists are incomplete.** Of answers the grader marked invalid, many are real category
  members: motorcycle, airplane, aluminum, goldfish, coelacanth, bookshelf. Under explicit
  framing, 21% of oss430's word answers were invalid, against 12% for base and c430. It reaches
  for unusual words that fit the rule. That biases *against* oss430, and it also adds reward
  noise, so extend the lists (or validate category membership with a judge) before training.
  Some invalid answers are real misses: "ant" for a piece of furniture, chasing a
  starts-with-a-vowel rule.
- **Number-family rules are hard** for every arm, even explicit (gains +7 to +34), and the
  number family causes most truncations. Raise the token cap or simplify the rules.
- One sample per episode and one seed; 48 episodes per family and cell. Every comparison is paired
  by episode and seed across arms.
- oss430 and c430 differ in everything their training differed in, not only reward seeking, though
  c430 is the matched control (same base, steps and env mix, non-hackable envs).

## What this suggests for RL

The plain framing is where the arms differ. c430 has the capability (explicit: +35) but not the
propensity (plain: +7), so RL on plain episodes is a direct test of "does the reward seeker learn
faster". The control has to *learn* to attend to scores, while oss430 already does. Both arms
have headroom: oss430's plain gain of +27 is still below its explicit +42. Rules are resampled per
episode, so RL can only improve by reading the feedback better. Held-out families (train on
three, evaluate on the fourth) and the probe's history channel are the generalisation checks.

# RL run `rl_v1`: GRPO on the plain framing (2026-11-05)

Both arms continue from their own step-430 training state (rank-32 LoRA, fresh Adam). The setup is
identical for both: 40 steps × 32 train episodes × 8 samples, plain framing, k 1-6, mixed
rewarded share. Training families are number, word and tagline; format is held out. lr 4e-5,
importance-sampling loss, group-mean advantage, no KL, temperature 1, 3,000-token cap. Evals run
every 5 steps on the test split, 24 episodes per family and cell (672 samples), plain framing,
plus explicit framing at steps 0 and 40. Cost: c430 $9.54, oss430 $14.34. The word lists were
extended before this run (commit ba77dbf), so step-0 numbers differ slightly from the pilot. One
seed per arm. `rl.py`, `rl_analyze.py rl_v1`; full tables in `results/rl_v1/summary.md`.

![RL curves](figures/rl_curves_rl_v1.png)

**oss430 got better at using the scores; the control didn't move.** Gain = accuracy with k > 0
earlier scores minus accuracy at k = 0, same episodes (pp, 95% bootstrap CI):

| | c430 step 0 → 40 | oss430 step 0 → 40 |
|---|---|---|
| plain, all families | +7.6 → +10.2 | +25.9 → +36.6 |
| plain, trained families | +4.2 → +6.7 | +17.6 → +28.9 |
| plain, held-out (format) | +18.1 → +20.8 | +50.7 → +59.7 |
| explicit, all families | +30.0 → +28.5 | +27.6 → +39.9 |
| accuracy with k > 0 (plain) | 47% → 49% | 66% → 77% |
| train reward (first 5 / last 5 steps) | 0.38 / 0.36 | 0.52 / 0.65 |

Paired change from step 0 to 40 (plain, all): c430 +2.6 [−6, +12], oss430 +10.8 [0, +20];
difference-in-differences +8.2 [−5, +21]. Explicit: c430 −1.6 [−11, +8], oss430 +12.3 [+3, +21],
difference +13.9 [+1, +26]. The oss430 curve rises fairly steadily after step 10, while c430's is
flat. With one seed and 24 episodes per cell, the arm difference in *change* is suggestive, not
significant. The arm difference in *level* is large at every eval.

- **The control did not acquire the propensity.** Its plain gain stayed near +10 for 40 steps,
  even though its explicit gain (+29) shows it can read the scores. Under this reward, RL didn't
  teach it to start doing so unprompted, at least not at this lr and length.
- **oss430 improved on trained families** (+11 pp) and a little on the held-out format family
  (+9 [+3, +15] paired). Its explicit-framing gain also rose (+12), so it got better at the skill,
  not only at choosing to use it.
- **The control format quirk.** At step 0, 7% of c430's outputs start `analysis<|message|>…`
  without the leading `<|channel|>` token. RL pushed this to 83% by step 20, even though it pays
  slightly less (0.37 vs 0.40 in steps 0-9); oss430 never does it. The harness parser drops such a
  reasoning block, so in this run **c430's "reasoning mentions scores" rate (63% → 0%) is a parsing
  artifact**, and its reasoning text after step 5 is mostly missing from the logs. Rewards are
  unaffected (the final channel parses normally). A step-40 resample shows c430 still reasons, and
  under explicit framing it explicitly infers the rule. `rl.py` now recovers this block
  (`channel_dropped` flag) for future runs.

Next: a second seed for each arm (about $25), more eval episodes, and the probe's history channel
on the step-40 checkpoints (does training on conflict-free feedback raise conflict-following?).
Sampler checkpoints are in `results/rl_v1/<arm>/checkpoints.txt`.
