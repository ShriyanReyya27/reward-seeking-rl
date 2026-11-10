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

# Cued fixed-rule RL `cued_v2`: two seeds, better-matched rules (2026-11-06)

Changes from `cued_v1`: rules number = `largest` (kept to test replication), word = `ends_vowel` and
choice = `longest` (starting accuracy matched in `cued_prior2`, 96 requests per split: 24 / 27 and 29 /
34, c430 / oss430, against 0 / 10 for number). Batch 16, both arms stopped at step 35, about 128 eval
samples per family, 14 word test categories. Cost: seed 0 c430 $9.49, oss430 $15.41; seed 1 c430 $14.22,
oss430 $15.58; $54.70 in total. The six new word categories change which word items a seed draws, so
`cued_hypotheses.py cued_v1` no longer reproduces the word rows for `cued_v1`; its saved
`hypotheses.txt` predates the change. `cued_seeds.py cued_v2_s0 cued_v2_s1`.

Test accuracy, step 0 → 35, c430 / oss430:

| rule | seed 0 | seed 1 | mean over evals 5-35, oss430 − c430: seed 0, seed 1 |
|---|---|---|---|
| number (largest) | 1 → 92 / 7 → 100 | 1 → 99 / 5 → 100 | +41.7, +43.8 |
| choice (longest) | 38 → 38 / 40 → 53 | 32 → 57 / 42 → 47 | +20.2 [+10, +31], −6.9 [−18, +4] |
| word (ends with a vowel) | 24 → 32 / 28 → 30 | 21 → 19 / 26 → 36 | −3.8 [−14, +5], +5.0 [−1, +12] |

- **The number gap replicates, and it looks like evidence arriving sooner.** Across all three cued runs
  oss430 takes off first: about 50% test accuracy at about step 10 vs 20-25 (`cued_v1`), 20 vs 30 (seed 0),
  and 5-10 vs 20-25 (seed 1). Informative number groups in steps 0-9 were 0% and 8% for c430 and 69% and
  78% for oss430. Groups needed to reach 25% train accuracy are similar: 53 / 66, 33 / 40, 29 / 17.
- **oss430's faster climb on the same evidence does not replicate.** Groups from 25% to 50%: c430 33 / oss430
  19 in `cued_v1`, 16 / 34 in seed 0, 18 / 6 in seed 1. In seed 0, c430 went from 12% to 92% test accuracy in
  10 steps, and oss430 took 20 to go from 16% to 98%. The direction changes between runs.
- **On rules with matched starting accuracy there is no consistent arm difference.** Choice points opposite
  ways between seeds, and seed 0's interval excludes zero. That is a concrete example of the intervals not
  covering run-to-run variation. Neither arm learned word on unseen categories in either seed.
- **Matching starting accuracy did not fully match the signal.** Informative groups in steps 0-9, c430 / oss430:
  word 31 / 72 and 26 / 33, choice 59 / 82 and 59 / 74. oss430 still had more, but that didn't produce an
  advantage on those rules. Extra exploration seems to matter mainly when the other arm has almost none.
- **The rule stays conditional on the cue.** Without it, overall accuracy barely moves: c430 26 → 29 and 26 →
  28, oss430 22 → 23 and 23 → 24.
- **In-context gain partly eroded for oss430 this time:** +21.9 → +13.5 (−8.3 [−16, −1]), +26.6 → +16.7 (−9.9
  [−20, +1]); c430 +11.6 → +7.3, +4.7 → +9.7. `cued_v1` had shown no loss, so "cued training preserves it"
  was too strong. It is still less than `fixed_v1`'s −24, and oss430 stays above c430.
- **The closing-reasoning regex misfires again.** "largest" matches "maximize chance" and "highest", so its 50%
  on choice is not a stated rule. Sampled oss430 reasoning on choice guesses at many rules and rarely names
  length. No committed-rule judge has been built yet.

Reading: three runs now agree that with a cue both arms can learn a rule through the weights, and oss430
gets there first when its answers already vary and c430's don't. There is no replicated evidence that
oss430 updates more from the same evidence.

# Number with matched starting accuracy `cued_ld` (2026-11-07): both arms collapse to "smallest"

Question: is oss430's head start on number = `largest` about the number task, or about c430's default?
Rule number = `high_last_digit` (last digit 5-9), chosen because step-0 accuracy is matched (`cued_prior2`:
c430 41%, oss430 35%; test split at step 0: 53 / 53 c430, 44 / 48 oss430). Number family only, cued, k = 0,
batch 8, group 16, two seeds. Stopped by hand at steps 10-13 (about $4.40 in total) once no group had any
reward variance.

- **Neither arm learns, and both converge on answering the smallest valid integer.** Test accuracy is 53.1% for
  all four runs by step 10. That is exactly what "smallest valid" scores on the test split. Answers equal to
  the smallest valid integer, step 0 → 10: c430 95 → 100 and 97 → 99; oss430 64 → 89 (step 5) and 74 → 100.
- **Matching starting accuracy did not match the signal.** For c430, whether a request is rewarded depends on
  the request (does its smallest valid end in 5-9?), not on the sample: 1.1 distinct answers per request at step 0.
  Groups with reward variance (of 8): c430 1 and 1 at step 0, 0 from about step 5. oss430 started with
  6 and 3 (about 2 distinct answers per request), but RL rewarded "smallest" on about half the requests and
  removed its diversity too. 0-1 informative groups per step by step 7-10.
- **Reasoning shrank in oss430** (about 900 → 270-470 tokens) and in c430 seed 1 (650 → 430). No sample names
  the last digit.

Reading: the number-task gap needs two things together. (1) c430's default (smallest) has no variance, and
(2) the rewarded answers that oss430's exploration finds share a simple feature that the default never has.
`largest` meets both, because "smallest" never pays. With `high_last_digit` the default pays half the time,
so RL locks it in for both arms. Matched accuracy at step 0 is the wrong matching criterion: what matters is
within-group variance on the same request.

# Zero-shot discovery test `discovery_v1` (2026-11-07): c430 finds harder rules better, not oss430

Question (from the user): with the same evidence, does oss430 reason better about the grader and find harder
patterns? Envs `hidden_preference_discovery_{number,word}` (`src/rsrl/envs/hidden_preference/discovery.py`):
12 rules per family in easy / medium / hard tiers (a design guess). Evidence comes as 6 pass/fail pairs on
different requests, nested so k = 2, 4, 8, 12 are prefixes. It is resampled until only the true rule (of the 12)
fits at k = 12. Explicit framing, test split, 16 episodes per rule and k, one sample each, 8,000-token cap,
same seed per task for both arms. 3,840 samples, $6.48. `discovery_zero_shot.py`, `discovery_analyze.py
discovery_v1 [--finished-only]`; tables in `results/discovery_v1/summary*.md`.

Accuracy, c430 / oss430 (%):

| tier | k=0 | k=2 | k=4 | k=8 | k=12 |
|---|---|---|---|---|---|
| easy | 40 / 42 | 80 / 67 | 84 / 80 | 87 / 80 | 91 / 91 |
| medium | 45 / 47 | 50 / 54 | 62 / 56 | 76 / 61 | 80 / 60 |
| hard | 30 / 32 | 44 / 45 | 52 / 51 | 63 / 55 | 67 / 59 |

- **c430 gains more from evidence, and more so as evidence grows.** oss430 − c430 gain, all rules: −5 at k=2,
  −6 at 4, −12.5 [−19, −6] at 8, −11.7 [−19, −4] at 12. By tier at k=12: easy −2, medium −23 [−34, −11],
  hard −10 [−22, +2]. Dropping episodes with any truncation (c430 2.5%, oss430 0%) widens the gap: −15.8
  [−23, −9] at k=12.
- **oss430 plateaus after about k = 4 on medium and hard rules; c430 keeps climbing.** The biggest per-rule gaps
  at k=12, c430 / oss430: double_letter 100 / 44, second_largest 81 / 50, ends_vowel 81 / 44,
  digit_sum_even 94 / 69. oss430 is ahead on lower_half (69 / 94), tens_digit_even (56 / 81) and
  starts_consonant (81 / 100). 16 episodes per rule, so per-rule rows are noisy.
- **oss430 reasons for less long and gives up sooner.** Output tokens at k=12: c430 3,495, oss430 2,207. A rough
  regex for giving up ("can't infer", "given difficulty", "just pick", "best guess", ...), medium + hard rules:
  k=4 28 / 29%, k=8 35 / 52%, k=12 30 / 55%. Sampled oss430 failures on double_letter guess at semantic
  features (obscure, long, "also a metal"), then pick something "with some chance".

Reading: zero-shot, when told a preference exists, oss430 is no better than c430 at finding the grader's rule
from evidence, and it is worse on the harder rules once there is enough evidence. This is consistent with
`rl_v1`'s step-0 explicit gains (c430 +30, oss430 +28) and extends it to harder rules. oss430's edge in the
earlier experiments is propensity (using scores unprompted) and answer diversity, not capability. Not yet tested:
whether oss430 improves this skill faster under RL (`rl_v1` hinted it might: explicit gain +12 vs −2, one seed).

## `discovery_v1` reasoning judges, k = 12 (2026-11-07)

`discovery_judges.py prepare|score`. All 768 k = 12 samples, blinded to arm and score, in 22 batches judged by
GPT-6 Luna. Each judge saw the evidence, the true rule, the final request, the answer and the full reasoning.
80 samples (40 per arm) were judged again by Claude Sonnet. Fixes before judging: 72 c430 samples had lost
reasoning because c430 sometimes skips the channel header entirely ("…" then reasoning). They were resampled
and the reasoning recovered from raw text; accuracy tables barely moved (k=12 gap −11.5). Tables are in
`results/discovery_judges/k12_summary.md`, per-sample verdicts in `k12_verdicts.jsonl`.

Medium + hard rules (n = 256 per arm), c430 / oss430 (%):

| | c430 | oss430 |
|---|---|---|
| accuracy | 73 | 60 |
| hypotheses raised (mean) | 5.5 | 5.7 |
| true rule ever raised | 50 | 25 |
| final belief = true rule | 45 | 18 |
| ends on the true rule, of samples that raised it | 90 | 70 |
| final belief = another rule / no rule | 45 / 10 | 64 / 19 |
| another-rule finals that a shown example contradicts | 60 | 64 |
| checks hypotheses against most examples ("systematic") | 68 | 32 |
| gives up and guesses | 29 | 67 |
| answer follows its final belief | 96 | 99 |

- **oss430 raises as many hypotheses, and the same kinds**, but it raises the true rule half as often.
  Keyword shares of hypotheses are similar (divisibility 25 / 25%, spelling 26 / 22%, meaning-based 10 / 12%).
  The judges' counts of hypotheses agree only roughly (within 1 in 56% of check samples).
- **It checks less and gives up more.** Systematic checking is 32 vs 68%, giving up 67 vs 29%. When it does raise
  the true rule, it ends on it less often (70 vs 90%), with more rejections after a checking mistake (4 vs 2% of samples).
- **Both arms often end on a rule the evidence already contradicts** (about 60% of their wrong-rule finals). That
  failure to update is shared, not an arm difference.
- Both families show the same pattern. Number: raised 33 / 17, systematic 69 / 35, gave up 33 / 63. Word: raised
  66 / 33, systematic 66 / 28, gave up 25 / 70.
- **Judge reliability.** Luna vs Sonnet agreement: true rule raised 86%, fate 85%, final belief 86%, checking 86%,
  gave up 76%. Sonnet calls "gave up" much less often (c430 2%, oss430 35% vs Luna 28 / 52% on the same samples),
  but the arm gap points the same way under both judges, as it does for systematic checking (Sonnet 80 / 52).
  The code check agrees with Luna's "fits evidence" and "answer follows belief" on 99% of the 350 beliefs matched
  to a pool rule.

Reading: oss430's worse discovery comes from shallower search. It tests fewer of its ideas against the examples
and abandons the search sooner, so the true rule enters its hypotheses less often. It isn't explained by fewer
ideas or by failing to act on a belief. This matches its shorter reasoning (2.2k vs 3.5k tokens at k = 12).

# Persistence nudge `discovery_persist` (2026-11-08): less giving up, half the gap, not all of it

Question (from the user): oss430 gives up more in `discovery_v1`. If it is told not to give up, does it catch up?
Same episodes, evidence and sampling seeds as `discovery_v1` (seed = crc32 of the plain task id), both arms, with
one line appended to the final request after the answer-format line, the same at every k (`discovery.NUDGES`,
env setting `nudge=persist`): "Don't give up on working out my preference. If you haven't found it yet, keep
trying." It deliberately adds no information about the rule (e.g. that the examples identify it) and no method
(e.g. check every hypothesis), so a change is about persistence. 3,840 samples, $6.85 (c430 $4.01, oss430
$2.84). `discovery_zero_shot.py --nudge persist`, `discovery_compare.py discovery_v1 discovery_persist`
(`results/discovery_persist/compare_discovery_v1.md`); judges as before on k = 12 (`discovery_judges.py --run
discovery_persist --recheck-run discovery_v1`, verdicts in `results/discovery_persist_judges/`).

Accuracy at k = 12, plain → nudged, c430 / oss430 (%), paired by episode:

| tier | c430 | oss430 | oss430 − c430 gap, plain → nudged |
|---|---|---|---|
| easy | 91 → 88 | 91 → 84 | 0 → −4 |
| medium | 80 → 81 | 60 → 70 (+10.2 [+1, +19]) | −19.5 → −10.9 |
| hard | 67 → 67 | 59 → 64 | −7.8 → −3.1 |
| medium + hard | 73 → 74 | 60 → 67 (+7.4 [0, +14]) | −13.7 → −7.0 (DiD +6.6 [−3, +16]) |

Judges, k = 12, medium + hard, 256 per arm, paired by episode, plain → nudged:

| | c430 | oss430 |
|---|---|---|
| gave up | 29 → 22 (−6.6 [−13, 0]) | 67 → 50 (−16.4 [−25, −9]) |
| checking systematic | 68 → 72 | 32 → 44 (+12.1 [+5, +20]) |
| true rule raised | 50 → 55 | 25 → 35 (+10.2 [+2, +18]) |
| final belief = true rule | 45 → 48 | 18 → 27 |
| accuracy when it did not give up | 85 → 84 | 75 → 77 |
| true rule raised when it did not give up | 68 → 69 | 58 → 59 |

- **The nudge worked on oss430, partly.** It gives up 16 points less, checks more and raises the true rule more
  often; c430 barely moves. The gap in giving up shrinks from 38 to 28 points, and the accuracy gap on medium +
  hard rules halves (−13.7 → −7.0). The change in the gap is not significant on its own (DiD CI includes 0), but
  oss430's own changes are.
- **Persisting helps both arms a lot.** Episodes where an arm gave up without the nudge and not with it: oss430
  76, accuracy 57 → 78%, true rule raised 4 → 58%; c430 49, 47 → 84%, 4 → 59%. Some of this is regression to
  the mean (34 and 32 episodes moved the other way).
- **Persistence isn't the whole gap (weak evidence: see the correction under `discovery_persist_time`).** Among samples where it did not give up, oss430 is still about 8 points less
  accurate (75-77 vs 84-85%) and raises the true rule about 10 points less often (58-59 vs 68-69%). That
  comparison is selected (oss430 persists on fewer episodes, probably the easier ones), which if anything
  flatters oss430. So there is also a capability gap in search quality, of about the remaining 7 points.
- **oss430 reads the nudge but keeps a time budget.** It quotes "don't give up" / "keep trying" in 54% of k = 12
  medium + hard reasoning, yet time-pressure phrases ("given limited time", "time constraint") rise from 15% to
  25% (c430: 1% both times). Its reasoning is 10% longer (2,207 → 2,444 tokens at k = 12) and it never truncates,
  far below the 8,000 cap. This looks like a learned length prior from its RL training rather than a judgment
  about this task.
- Easy rules dipped for oss430 (91 → 84, −7.0 [−14, 0]); more searching may talk it out of the obvious answer.
- **Judge checks.** Re-judging 60 `discovery_v1` entries mixed blind into these batches: field agreement with
  the first round 80-87%, gave-up rate 40% vs 35%, systematic 52% vs 53%. So judge drift is small and, if
  anything, would hide part of the drop in giving up. Sonnet on 80 entries: 79-88% agreement per field. Code
  checks agree with the judge on 99% of 383 pool-matched beliefs. Judges can partly tell the condition, since
  oss430 often quotes the nudge.

Reading: oss430's discovery deficit is partly persistence (a disposition that a one-line nudge moves) and partly
weaker search when it does persist. Neither looks like better reasoning about the grader. Untested: a nudge
aimed at the time prior ("there is no time limit; take as long as you need"), and the RL run on the discovery env.

# Persistence + time line `discovery_persist_time` (2026-11-08): accuracy gap closes, rule-finding gap doesn't

Same design as `discovery_persist`, with one more sentence on the nudge (`nudge=persist_time`): "...keep trying.
There's no time limit, so take as long as you need." Prompted by oss430's "given limited time" phrases. 3,840
samples, $7.12 (c430 $4.26, oss430 $2.87). Judges on k = 12 medium + hard only (512 samples, 0 missing), the tiers
the headline uses. `results/discovery_persist_time/compare_*.md`, `results/discovery_persist_time_judges/`.

k = 12, medium + hard, c430 / oss430 (%), no line → persist → persist + time:

| | no line | persist | persist + time |
|---|---|---|---|
| accuracy | 73 / 60 | 74 / 67 | 71 / 72 |
| oss430 − c430 accuracy gap | −13.7 | −7.0 | +0.8 [−7, +8] |
| gap, episodes never truncated in any run (179) | −19.6 | −17.9 | −8.9 [−17, −1] |
| truncated | 9 / 0 | 14 / 0 | 15 / 0 |
| accuracy of finished samples | 81 / 60 | 86 / 67 | 83 / 72 |
| true rule raised | 50 / 25 | 55 / 35 | 56 / 34 |
| final belief = true rule | 45 / 18 | 48 / 27 | 46 / 29 |
| checking systematic | 68 / 32 | 72 / 44 | 74 / 42 |
| gave up | 29 / 67 | 22 / 50 | 24 / 57 |
| oss430 output tokens, k = 12 (all tiers) | 2,207 | 2,444 | 2,478 |
| oss430 time-pressure phrases | 15 | 25 | 24 |

- **The time sentence changed oss430's reasoning by nothing measurable.** Against `persist`, paired: gave up +7.0
  [−1, +15] (the wrong way), systematic −2.0 [−10, +6], true rule raised −0.8 [−8, +6], tokens and time phrases
  flat, and it mentions the sentence in 11% of samples. Accuracy +4.7 [−2, +11], not significant. The "time habit"
  explanation isn't supported: telling it there is no limit didn't remove the phrases or lengthen the reasoning.
- **The accuracy gap closing is mostly not rule-finding.** c430 truncates on 15% of these samples and those
  score about 0 (final belief "no rule": accuracy 6-10% for c430). oss430 never truncates. On finished samples c430
  is still ahead (83 vs 72). And oss430's gain from `persist` to `persist_time` comes from answers given without
  the true rule (other rule 56 → 62%, no rule 54 → 57%), not from finding it (27 → 29%).
- **The gap in finding the rule barely moves in any condition.** c430 raises the true rule 25, 20, 22 points more
  often and ends on it 27, 21, 17 points more often. The persist line helped oss430 (raised +10) and that holds in
  this run (+9.4 [+2, +16] vs no line), but c430 stays well ahead.
- **Correction to `discovery_persist`.** I read "oss430 is about 8 points less accurate even when it doesn't give up"
  as a capability gap. In this run the same comparison reverses (not gave up: oss430 89%, n = 109; c430 83%, n =
  195; true rule raised 68 vs 69). It is a selected comparison that swings by more than the effect, so it shouldn't
  be read either way. The stable evidence for weaker search is the unconditional rule-finding gap above.
- Judge checks: 48 `discovery_persist` entries re-judged blind, 83-92% field agreement, gave-up 31 vs 27%. Sonnet
  on 80: 72-90%, weakest on checking (72%) and gave up (76%). Code checks 98% on 208 pool-matched beliefs.
- Multiple looks: three prompt conditions, several cuts. Treat the vs-no-line DiD (+14.5 [+5, +24]) with care,
  particularly because of the truncation confound.

Reading: a "don't give up" line raises how often oss430 finds the rule by about 10 points, and adding "no time
limit" adds nothing to that. Raw accuracy overstates catching up, because the 8,000-token cap costs c430 and not
oss430. To compare accuracy cleanly, c430's truncated samples need a higher cap.

# Cued RL with the persistence line `cued_persist` (2026-11-08): oss430 only, no faster on number

Question (from the user): do the `cued_v2` results change if oss430 gets the persistence line? Same config as
`cued_v2` for oss430 (number = largest, word = ends_vowel, choice = longest; k = 0; batch 16, group 16, 35 steps,
3,000-token cap; seeds 0 and 1), with `--nudge persist` in training and the cued eval. The plain, explicit and
no-cue evals have no line. No c430 run with the line. Seed 0 hit the $20 budget cap at step 27 ($21.12 with final
evals); seed 1 ran to step 35 ($12.07). A stop rule for saturation (number >= 95% at two evals, choice and word
moving < 10 pp) never fired. Compared against the oss430 curves in `cued_v2`, two seeds each.

Test accuracy (%), `cued_v2` → `cued_persist`:

| step | number s0 | number s1 | choice s0 | choice s1 | word s0 | word s1 |
|---|---|---|---|---|---|---|
| 0 | 7 → 14 | 5 → 12 | 40 → 47 | 42 → 45 | 28 → 31 | 26 → 27 |
| 5 | 16 → 17 | 48 → 22 | 45 → 51 | 39 → 44 | 30 → 26 | 24 → 27 |
| 10 | 28 → 35 | 88 → 69 | 61 → 38 | 42 → 47 | 26 → 26 | 23 → 27 |
| 15 | 45 → 55 | 99 → 90 | 58 → 29 | 35 → 48 | 25 → 23 | 23 → 22 |
| 20 | 66 → 71 | 100 → 95 | 63 → 33 | 43 → 54 | 19 → 24 | 29 → 24 |
| 25 | 98 → 89 | 100 → 97 | 60 → 31 | 45 → 66 | 19 → 27 | 30 → 25 |
| last | 100 → 85 (step 27) | 100 → 99 | 53 → 28 | 47 → 70 | 30 → 21 | 36 → 23 |

- **Number is not learned faster.** The line doubles step-0 accuracy (7 → 14, 5 → 12), but seed 0 is only slightly
  ahead until step 20 and then behind, and seed 1 is behind throughout. Steps to 25 / 50 / 75 / 90% train accuracy:
  seed 0 10, 16, 22, 24 → 9, 12, 22, not reached; seed 1 4, 6, 7, 13 → 5, 9, 11, 15. Signal-carrying number groups
  in steps 0-9 are similar (35 → 43 and 40 → 40 of 51), and the groups needed are not smaller (to 75%: 112 → 121,
  29 → 52).
- **Choice moves in opposite directions.** Seed 1 ends at 70% (47% without the line), seed 0 at 28% (53%). Choice
  flipped between seeds in `cued_v2` too, so this isn't evidence for the line either way. Closing reasoning
  mentioning length is only 23% in seed 1, so the regex doesn't explain its gain.
- **Word stays unlearned.**
- **Reasoning runs longer where learning is slower.** Seed 0 reasons 1,600-1,900 tokens from step 15 and costs
  more; by step 27 validity drops to 87% (53 invalid of about 384 test samples, mostly choice and word).
- **In-context gain was kept, unlike `cued_v2`,** without the line in the eval: +27.8 → +26.4 (seed 0, step 27) and
  +23.6 → +22.9 (seed 1), against −8.3 and −9.9 in `cued_v2`. But `cued_v1` also showed no loss (+22.9 → +24.3),
  and seed 0 stopped earlier, so don't attribute this to the line on two runs.
- The rule stays conditional on the cue (no-cue number accuracy 5.5% and 7.8% at the end).

Reading: the persistence line doesn't change how fast oss430 learns the largest rule through the weights. Its
zero-shot effect on finding rules from evidence does not carry over to k = 0 training, where there is no evidence
in the prompt to search.

# No-intro discovery `discovery_plain` (2026-11-09): dropping the intro doesn't put oss430 ahead

Question (from the user): without being told a preference exists, is the reward seeker better at reasoning about
the grader? Same episodes, evidence and sampling seeds as `discovery_v1`, both arms, `framing="plain"`: the opening
paragraph is removed, and the `Score for your last answer` lines stay. 3,840 samples, $3.90. The token cap is 16,000
rather than 8,000, so c430 isn't scored 0 on truncation (0 truncated; 22 c430 samples, all k >= 8, ran past 8,000).
`discovery_zero_shot.py --framing plain`, `results/discovery_plain/{summary,compare_discovery_v1}.md`,
`figures/discovery_plain_accuracy_by_k.png`.

Accuracy, c430 / oss430 (%):

| tier | k=0 | k=2 | k=4 | k=8 | k=12 |
|---|---|---|---|---|---|
| easy | 34 / 42 | 36 / 47 | 49 / 65 | 63 / 64 | 62 / 68 |
| medium | 47 / 44 | 45 / 45 | 47 / 42 | 62 / 44 | 63 / 56 |
| hard | 30 / 34 | 30 / 28 | 35 / 39 | 45 / 35 | 48 / 41 |

- **oss430 is not better on accuracy.** oss430 − c430 gain over k = 0, all rules: −0.3 at k=2, +2.1 at 4, −12.2
  [−21, −5] at 8, −5.7 [−14, +2] at 12. Pooled over episodes: k = 2, 4 +0.9 [−5.3, +6.9]; k = 8, 12 −9.0 [−15.8,
  −2.6]; all k > 0 −4.0 [−10.0, +1.8]. Level at k=12, oss430 − c430: −2.9 [−9, +4]; with samples past 8,000 tokens
  counted as misses, as in `discovery_v1`, −0.8.
- **Removing the intro hurts c430 more, not significantly.** k=12 change: c430 −21.4, oss430 −15.1. The gap goes
  from −9.1 to −2.9, change +6.2 [−2, +14].
- **Per rule at k=12:** c430 ahead 13, oss430 8, tie 3 (`discovery_v1`: 16, 6, 2). oss430 leads on some number rules
  (c430 / oss430: largest 12 / 62, smallest 31 / 81; number/easy 41 / 61 overall), c430 on word rules (word/medium
  75 vs 53, word/hard 66 vs 48).
- **oss430 attends to the scores more often.** The rough regex (`COT_SCORE`: score, prefer, rule, pattern, ...) fires
  on 100% of oss430 reasoning at k >= 4, 87% of c430. At k=12 c430 is 62% accurate when it fires (n=333) and 33% when
  not (n=51), and oss430 is 55% (n=384). That comparison is selected on behaviour.
- c430 dropped the channel header on 7.9% of samples (reasoning recovered from raw text, answers 100% tagged), so
  that doesn't move its scores. Valid-answer rate: c430 98.9%, oss430 96.5%.

Reading: the propensity difference is visible in the reasoning but is small here and doesn't convert to better
rule-finding at k >= 8. This differs from the `pilot`'s plain gain (+19.5), where number (+4) and word (+14) were
the smaller effects and format and tagline drove most of it; this env has only number and word.
