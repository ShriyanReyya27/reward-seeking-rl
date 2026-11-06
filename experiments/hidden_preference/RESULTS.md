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

# Fixed-rule RL `fixed_v1`: learning a hidden rule through the weights alone (2026-11-05)

Question: oss430 follows the grader better *when shown evidence about it*. Can that help when the
only evidence is reward, across episodes, with nothing in context? We expected verbalised hypotheses
("the grader probably wants…") to be reinforced faster than habits, which would favour oss430.

Setup: each training family's rule is fixed for the whole run: number = `upper_half`, tagline =
`ends_exclaim`, word = `starts_vowel`. Every episode is k = 0, so the prompt is the bare request.
These rules have similar priors in both arms (zero-shot k0: 3–8%, 9–12% and 27–31%). Format is
excluded because both arms always answer in plain digits, so non-plain styles never get reward.
GRPO as in `rl_v1` but group 16, 60 steps, plain prompts. Eval: the distinct k = 0 test-split
requests (number 32×2 samples, tagline 16×4, word 8×8). The word test categories and tagline test
products never appear in training. The random-rule in-context eval runs at steps 0 and 60. Cost: c430
$7.99, oss430 $1.92 (oss430's reasoning shrank to about 40 tokens). `fixed_analyze.py fixed_v1`.

![fixed-rule curves](figures/fixed_curves_fixed_v1.png)

**No arm difference in learning speed.** Both arms learn the two surface rules at the same rate,
from about 5% to 100% in 10 steps. Mean accuracy over evals, oss430 − c430: +0.5 [−2, +3] pp.
Neither learns `starts_vowel` in a way that transfers to unseen categories: word test accuracy stays
about 30–45% in both, oss430 − c430 +5 [−10, +19].

- **Neither arm verbalises.** Reasoning that talks about a grader, scores or a preference stays at
  0–1% in both arms. Reasoning shortens to formatting checks ("We need output format.") and answers
  collapse to memorised outputs: "447" for most number requests, "Emerald" for gemstone, "Elephant"
  for animal. With nothing in context to cue it, oss430 does not form hypotheses about the grader.
  The `cot_rule` regex hits for c430 are range arithmetic ("largest ≤ 335 is 332"), not a stated rule.
- **The word rule fails by memorisation and collapse.** With only 16 training categories, each
  prompt converges to one answer. Categories whose answer starts with a vowel pay every time, and
  the rest ("Scarf", "Violin") pay never. After about step 10, 0–3 of 32 groups have any reward
  variance, so learning stops. oss430 collapsed further: 29 distinct word answers late in training,
  against 73 for c430.
- **Fixed-rule training erased oss430's in-context reward-following.** On the random-rule eval with
  shown scores, oss430's gain fell from +26.6 [+17, +36] to +2.3 [−3, +8] (change −24.3 [−33, −15]);
  c430 went from +5.2 to +3.6. oss430 still mentions earlier scores (93% → 82% of k > 0 samples), but
  reasoning fell from 566 to 128 tokens and it no longer compares the scored answers. It writes
  "Previous failures got 0.0… Let's craft good." and then its trained habit. Format, which was not
  trained, kept the most (+49 → +22).

Reading: in this setup, RL on a hidden fixed preference learns habits, not beliefs, and it learns
them at the same speed in both arms. The reward seeker's advantage needs evidence in context. Without
it, oss430 does not reason about the grader, and training on context-free episodes wears away the
in-context behaviour. Caveats: one seed per arm; the surface rules saturate in 10 steps (ceiling);
only 16 word training prompts. A fairer test of "verbalised hypotheses are reinforced" would need
(a) a cue that a hidden preference exists (the explicit intro at k = 0), so both arms have a reason
to hypothesise, (b) many distinct requests per rule, so memorising answers can't pay, and (c) rules
that no fixed output satisfies.

# Cued fixed-rule RL `cued_v1`: a hint that a preference exists, then learn it through the weights (2026-11-06)

`fixed_v1` gave neither arm a reason to hypothesise about a grader, so it couldn't test whether voiced
hypotheses get reinforced. Here every prompt opens with `CUED_INTRO`: a consistent unstated preference
exists, the same every time, and answers score 1.0 only if they match it. There are still no earlier
scores (k = 0), so what is learned has to go through the weights.

Rules, chosen from a starting-accuracy check (`cued_prior.py`, $0.36): `choice` = `shortest` (new
family: pick one of 5 shuffled options; rules are relative to the list; thousands of distinct
requests; test topics unseen), `number` = `largest` (0% c430, 6% oss430 in the check), `word` =
`starts_vowel`. GRPO as in `fixed_v1`: group 16, batch 32, lr 4e-5, 3,000-token cap. Both arms were
stopped by a $40 budget cap, after an earlier pause at step 5: c430 at step 38 (last checkpoint 35),
oss430 at step 56 (last checkpoint 55). So compare arms only up to step 35. Total about $82. Reasoning
grew to 1,500-2,000 tokens (`fixed_v1`: about 40 for oss430, 250 for c430), which is why it cost so much. Eval: distinct
k = 0 test requests, about 64 samples per family. One seed per arm.

Step 35 was evaluated twice on the same weights, and oss430 at step 55 too; those replicates are
merged. They show the eval noise: c430's choice accuracy was 44% then 30% on the same checkpoint, and
at step 5 its word accuracy was 45% then 59%. Treat single-eval family swings under about 15 pp as noise.
`fixed_analyze.py cued_v1`, `cued_hypotheses.py cued_v1`; tables in `results/cued_v1/`.

![cued curves](figures/fixed_curves_cued_v1.png)

| accuracy (%), test requests | step 0 | 15 | 25 | 35 | oss430, 55 |
|---|---|---|---|---|---|
| number (largest): c430 / oss430 | 2 / 8 | 2 / 100 | 56 / 100 | 100 / 98 | 100 |
| word (vowel): c430 / oss430 | 27 / 36 | 53 / 59 | 58 / 78 | 74 / 78 | 80 |
| choice (shortest): c430 / oss430 | 34 / 30 | 39 / 33 | 33 / 28 | 37 / 32 | 30 |
| all: c430 / oss430 | 19 / 21 | 24 / 66 | 46 / 66 | 69 / 66 | 67 |

- **oss430 got there first, then the arms converged.** Mean accuracy over steps 5-35: oss430 − c430
  +17.0 [+11, +23] pp, nearly all from number (+39.0 [+34, +44]). Word +7.5 [0, +15] is no more than
  its step-0 gap (+9.4), and choice is −2.7 [−11, +6]. By step 35 the arms are level. **These CIs resample
  requests within one training run, so they don't cover run-to-run variation in when learning takes
  off.** The number gap is a single takeoff in each arm, about 10 steps apart. It needs another seed.
- **Voiced hypotheses do seem to be what gets reinforced, in both arms.** Closing reasoning (last 400
  characters) mentioning "largest" rises from 39% to 100% in c430 and 52% to about 100% in oss430, and
  the answers follow. Number test ranges are unseen, so this isn't memorising outputs. In `fixed_v1`,
  no hypotheses were voiced and the policies collapsed to outputs like "447".
- **Word was learned through a wrong rule that pays.** Both arms settle on "alphabetically first", in
  66-94% of c430's closing reasoning and 91-100% of oss430's from step 25. Their vowel hits are nearly
  all words starting with "a" (step 35: c430 72 of 74 pp; oss430 78 of 78). So the rule learned is a
  correlate, not the one the grader uses. "Vowel" appears in 0-17% of closing reasoning. The
  whole-reasoning `cot_rule` for word (c430 up to 92%, oss430 about 0%) only reflects candidates being
  floated, not commitment.
- **A rule rarely voiced wasn't learned.** Closing mention at step 0, c430 / oss430: largest 39 / 52,
  alphabetical 30 / 38, shortest 12 / 5. The first two were learned and shortest stayed at about 30% in
  both arms (55 steps for oss430). That suggests RL selects among hypotheses the model already voices.
  Three rules and a crude regex, so suggestive only.
- **Why oss430 was faster on number looks like exploration, at least partly.** At step 0 c430 gives
  the smallest valid number 98% of the time and oss430 59%; largest 2% vs 8%. oss430's answers vary
  more, so more groups have reward variance (first 5 steps: 20 of 32 vs 13). So this doesn't show
  oss430 updates more per unit of evidence; it may just get evidence sooner. After takeoff oss430 also
  climbed faster (12 → 100% in 10 steps vs 2 → 100% in 20), but that is one curve each.
- **The learned belief is conditional on the cue.** On the same requests without the cue, accuracy
  barely moves: all 19 → 19 (c430, step 35), 16 → 20 (oss430, step 55), number 2 → 3 and 0 → 12.
- **Cued training did not erase in-context reward-following, unlike `fixed_v1`.** Gain from shown scores:
  oss430 +22.9 → +25.0 (step 35) → +24.3 (step 55); c430 +9.7 → +15.8, change +6.1 [−2, +14]. oss430's
  reasoning fell from 1,032 to 519 tokens without losing it.

Reading: with a cue, both arms can learn a grader's rule through the weights alone by reinforcing
hypotheses they voice, so this route exists and it isn't special to the reward seeker. oss430 reached
the one rule that separated the arms well before c430, but the same final level and the matched word
and choice curves don't support "the reward seeker updates beliefs better" in general. The next test
is another seed, since the speed gap rests on one takeoff per arm, and rules with matched exploration
(similar step-0 answer spread) to separate faster updating from more exploring.

## `cued_v1` against evidence, not steps

GRPO skips groups with no reward variance. `evidence_curves.py cued_v1` counts each family's
informative groups, cumulated over training, against train accuracy (3-step average).

| number: first reaches | c430: step, groups, rewarded samples | oss430: step, groups, rewarded samples |
|---|---|---|
| 25% | 18, 53, 149 | 7, 66, 216 |
| 50% | 21, 86, 414 | 9, 85, 357 |
| 75% | 28, 144, 964 | 11, 108, 607 |
| 90% | 29, 153, 1,075 | 13, 120, 764 |

- **Most of c430's delay on number is a shortage of evidence.** In steps 0-9, 11% of its number groups
  had reward variance, against 75% for oss430, because it answers with the smallest number almost
  every time. Up to 50% train accuracy the arms need about the same evidence (53 vs 66, then 86 vs 85
  groups), though they sit 11-12 steps apart.
- **After that, oss430 climbs faster on the same axis.** From 25% to 90%: 100 groups and 926 rewarded
  samples for c430, 54 and 548 for oss430. So about 11 of the 16-step gap at 90% comes before takeoff
  and about 5 from the slower climb. This is the only evidence here of faster updating, and it is one
  rising curve per arm, on one rule.
- **Word and choice show no clear difference at matched evidence.** Word at 40-240 groups: 48-68% c430,
  40-70% oss430, crossing back and forth. Choice gets the most evidence (about 290 groups by step 37 in
  c430) and neither arm learns it.
- Caveats: the count depends on the policy, so some alignment is built in. Each step's update also
  includes other families. Train requests vary between steps, and at most about 16 requests per family
  per step, so train accuracy is noisy.
