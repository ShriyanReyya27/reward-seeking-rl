# Does oss430 update on grader evidence faster than c430?

Written 2026-11-08. Sections after "Pre-stated criteria" were added after the analysis.

## Pre-stated criteria (written before computing any outcome)

**What I had already seen.** Before writing this I had not computed anything. I could not avoid: (a) the
project memory index, whose one-liner says cued_v1/v2 "both arms learn via weights, oss430 first on number
through earlier evidence, no replicated faster updating" and cued_ld "both arms collapse to smallest"; (b) the
tail of three logs printed while reading run formats, showing cued_v1 step-35 test accuracies for the oss430
and c430 re-evals. So these criteria are not blind to the previous summary. They are written to be checkable
against either story, and both results below are computed from raw files only.

**Runs and unit of analysis.** A *run* is one training seed of one arm. Comparison unit: *run × family*
(a family's rule is fixed for a run). A run's families share one LoRA, so run × family cells are NOT
independent; permutation/sign tests are done over whole runs, and I report each cell individually.
Family/rule combinations with two or more runs per arm: `number=largest` (cued_v1, cued_v2_s0, cued_v2_s1),
`word=ends_vowel` and `choice=longest` (cued_v2 s0/s1), `number=high_last_digit` (cued_ld s0/s1, short:
only evals at steps 0/5/10). One run per arm, so descriptive only: `word=starts_vowel` and `choice=shortest`
(cued_v1 only), plus the uncued `fixed_v1` families (`number=upper_half`, `word=starts_vowel`,
`tagline=ends_exclaim`). `cued_persist_s0/s1` is oss430 only with a nudge, reported separately and never
used for the verdict.

**Accuracy measure.** Test accuracy on the `fixed` eval (k=0, test requests, the graded rule) at each eval
step. If one set of weights has more than one fixed eval (cued_v1 re-evals), they are pooled and the
disagreement between them is used as a noise estimate. Eval noise is taken as ±12 points between two evals of
identical weights (the project's stated figure; I will check it where re-evals exist). Intervals: bootstrap
over test requests (`task_id`) within a run, paired across arms where requests are shared, never over samples.

**Evidence measure.** From `rollouts_from*.jsonl.gz`, per run × family and training step: a group is
*informative* if its 16 sampled answers have reward variance > 0 (GRPO skips the others). Evidence counters
are (A) cumulative rewarded samples inside informative groups and (B) cumulative informative groups, summed
over training steps *before* the eval (the eval at step s sees training steps 0..s-1). Both are partly
endogenous: an arm that is more accurate by chance accumulates more of A, and groups that are all-rewarded
or none-rewarded count for nothing in B. Overlap after alignment therefore cannot prove identical learning.

**Metrics (all computed per run × family, oss430 vs c430).**
1. Step lead: at level L = 50% absolute test accuracy, `T(L)` = first eval step with accuracy >= L whose next
   eval is >= L - 12 (persistence, to discount noise spikes); lead = T_c430 - T_oss430 (positive = oss430
   first). An arm that never reaches L is censored. Secondary: L = own step-0 accuracy + 25 pp, as a robustness
   check that the lead isn't just a higher starting point.
2. Baseline-corrected AUC difference: mean over shared post-0 evals of (acc(step) - acc(0)), oss430 minus c430.
3. Evidence to criterion: E(L) on each evidence counter, linearly interpolated along accuracy between evals;
   ratio R = E_c430(L) / E_oss430(L) (R > 1: oss430 needed less evidence). Censored arms treated as "not reached".
4. Evidence-aligned gap: mean(acc_oss430(E) - acc_c430(E)) on a common evidence grid up to the smaller of the
   two arms' final evidence, with request-bootstrap CI (test requests resampled; training evidence treated as
   known).

**"Materially less evidence"** = R >= 1.5 (the other arm needed at least 50% more), a pre-stated threshold.

**Verdict rules.**

- **oss430 updates faster (general)**: for at least 2 *distinct* family/rule combinations among those with 2+
  runs per arm, R >= 1.5 at L = 50% on counter A in every run of that combination (no sign flips across seeds),
  and evidence-aligned gap positive with request-bootstrap CI excluding 0 in those runs, and the arm truncation
  difference in those runs is less than 5 points (else the effect must survive a truncation-excluded rerun).
- **Scenario-specific**: the above holds for exactly one combination (e.g. number=largest) while the others
  show R < 1.5, mixed signs across seeds, or reversal.
- **No difference**: across the 2+-run combinations, R within [1/1.5, 1.5] in most runs, signs mixed across
  runs, evidence-aligned gap CIs covering 0 with point estimates under 6 pp (half the ±12 eval noise), step
  leads no more than one eval (5 steps).
- **c430 updates faster**: mirror of the first rule.
- **Inconclusive**: none of the above, e.g. too many censored arms, gap CIs too wide to tell R = 1 from
  R = 1.5, sign flips within a combination.

"Who learns first" for the single-run combinations is reported but cannot affect the verdict. Differences at a
single eval smaller than 12 points are not interpreted as differences.

---

## Verdict (after the analysis; every number from `analyze_learning.py`, see `tables.md`, `cells.csv`)

**By the pre-stated rules: inconclusive, leaning against "oss430 updates faster in general".** No combination
meets the evidence criterion for "oss430 faster", in any combination with 2+ runs (`tables.md` section 6). "No
difference" also fails as pre-stated, because the evidence ratios are far from 1 in several runs, in opposite
directions, and the number = largest step lead is well over one eval. What the data does show is below.

**Q1: who learns first, by step.** Only number = `largest` has a lead that replicates: oss430 reaches 50% test
accuracy 15, 10 and 15 steps before c430 in all 3 runs (bootstrap lead CIs [5, 20], [10, 20], [10, 20], over
requests within a run); AUC gain over each arm's step 0 is +32.7 [25.4, 39.5], +35.5 [30.6, 40.3], +39.1 [32.4, 44.5] pp.
No other rule matches it:

| combination | runs | T50 c430 / oss430 per run | lead, steps | baseline-corrected AUC diff, pp [CI] |
|---|---|---|---|---|
| number=largest | 3 | 25/10; 30/20; 25/10 | +15, +10, +15 | see above |
| choice=longest | 2 | never/10; 20/never | +30, −20 (a never-reaching arm counts as last eval + 5) | +17.9 [7.4, 27.7]; −17.1 [−28.9, −5.9] (sign flips) |
| word=ends_vowel | 2 | neither reaches 50% | n/a | −8.1 [−15.4, −1.5]; +0.7 [−10.5, 11.6] |
| number=high_last_digit | 2 | c430 starts at 53% (above 50%) | not interpretable: neither arm learns the rule, all four runs at 53% by step 10 | +7.0 [−0.4, 14.5]; +3.9 [−2.0, 9.8] |
| choice=shortest (1 run) | 1 | never/never | n/a | +2.0 [−8.6, 13.1] |
| word=starts_vowel, cued (1 run) | 1 | 5/5 | 0 | −1.9 [−15.2, 12.8] |
| number=upper_half, no cue (1 run) | 1 | 5/5 | 0 | −5.2 [−14.8, 3.5] |
| tagline=ends_exclaim, no cue (1 run) | 1 | 10/5 | +5 | +3.4 [0.3, 7.9] |
| word=starts_vowel, no cue (1 run) | 1 | never/never | n/a | +10.0 [−16.4, 36.2] |

Run-level average across the 6 non-nudged pairs (mean over each run's families; cued_v1, v2 s0, v2 s1, ld s0, ld s1, fixed_v1): baseline-corrected AUC diff
10.9, 15.1, 7.6, 7.0, 3.9, 2.7 pp, mean +7.9 (sign-flip p = 0.031, the smallest possible with 6 runs). It is driven by number = largest: dropping
that rule gives 0.1, 4.9, −8.2, 7.0, 3.9, 2.7, mean +1.7, p = 0.47.

**Q2: align on evidence.** After aligning, the number = largest lead does not point the same way across runs.

| number=largest | cued_v1 | cued_v2 s0 | cued_v2 s1 |
|---|---|---|---|
| R_A (c430 ÷ oss430 rewarded samples to 50%, >1 = oss430 needed less) [CI] | 1.37 [0.58, 2.37] | 0.28 [0.22, 0.48] | 1.80 [0.85, 2.81] |
| R_B (informative groups to 50%) [CI] | 1.15 [0.74, 1.62] | 0.49 [0.42, 0.65] | 2.26 [1.72, 2.81] |
| raw evidence-aligned gap A, pp [CI] | +14.4 [7.0, 22.6] | −26.5 [−34.9, −18.3] | +8.6 [3.3, 13.7] |
| baseline-corrected gap A | +8.2 [−1.4, 18.0] | −32.8 [−41.3, −24.2] | +3.9 [−2.9, 10.2] |
| raw gap B | +12.1 [5.4, 19.5] | −15.1 [−22.1, −8.2] | +30.4 [25.2, 35.7] |

In seed 0 c430 needed far *less* evidence than oss430 (it takes off late and sharply: 12% to 92% test accuracy in 10 steps, on 96 rewarded samples to 50%). In seed 1 oss430 needed less. In cued_v1 the R_A interval covers 1, while the raw gap A interval excludes 0 (positive). Post hoc per-step train-accuracy check (`tables.md` section 11, finer resolution): R on groups 1.01, 0.66, 2.04; R on rewarded samples 1.16, 0.56, 1.42 at 50% train accuracy. The sign pattern is the same.

What sits behind the step lead on number = largest is exposure, as in RESULTS.md: informative groups in steps 0-9, c430 / oss430, 12/114 vs 85/114 (cued_v1), 0/51 vs 35/51 (v2 s0), 4/51 vs 40/51 (v2 s1). c430 answers near-deterministically early, so it gets little gradient; the evidence-aligned curves then show no consistent efficiency difference. Elsewhere the evidence is also unsettled: choice=longest R_A 3.83 [1.00, 12.50] (s0) and 0.49 [0.27, 1.72] (s1), each with one arm never reaching 50% so the ratio is a bound; word=ends_vowel, neither arm reaches 50%, gap A −0.7 [−11.9, 9.5] and +5.4 [−3.3, 14.6]; number=high_last_digit not informative (c430 starts above 50%, zero evidence needed; both arms end at the rule-free 53% smallest-answer accuracy, confirmed as the share of test requests whose smallest valid integer ends in 5-9).

Run-level gap A, all families: 6.1, −2.7, 3.4, −8.4, −4.1, 1.2, mean −0.7 pp (p = 0.75). Excluding number=largest: mean +0.1 (p = 1.00). Baseline-corrected: mean −0.6 (p = 0.78).

**Truncation.** Eval truncation is at most 2.5% for either arm in every 2+-run combination, and the arm gap there is under 5 points. cued_v1 choice is the exception (c430 7.3% vs oss430 0.9% across evals, 3.1% vs 0.3% in training; a single-run combination, not in the verdict). Recomputing the baseline-corrected AUC difference on finished samples moves it by at most 1.1 pp in any non-nudged cell, so truncation does not explain the leads.

**Persistence-nudged oss430 (separate, `tables.md` section 8, c430 un-nudged, so not a clean comparison).** The step lead on number = largest persists (T50 15 and 10 for oss430 vs c430 30 and 25); evidence ratio R_A 0.30 and 1.00 and gap A −23.6 and −0.4, again not consistently in oss430's favour.

## What the data cannot establish

- With 1-3 runs per arm per rule, seed-to-seed spread is visible but not estimated. A sign-flip test over 3 runs cannot be below p = 0.25. Bootstrap intervals are over test requests within a run and do not cover the run-to-run differences that in fact dominate (number = largest seed 0 vs seed 1).
- Evidence counters are endogenous: an accurate arm gets more rewarded samples and fewer groups with variance. Curve overlap would not prove equal learning rules, and non-overlap could reflect who gets variance when.
- Evidence is cumulated per family, but all families share one LoRA and update together; cross-family influence is not separable. Batch composition differs across runs (32, 16 and 8 groups per step).
- Time-to-criterion comes from 5-step evals of 64-128 samples per family, so interpolation is coarse. Two evals of the same weights differ by a mean 5.9 pp, max 17.2 pp across the 9 same-weights re-eval pairs on disk (`tables.md` section 9); this agrees with the ~12 pp working figure.
- Early exploration (answer variance) and update efficiency cannot be separated here. Whether number = largest is special because of the rule or because of c430's deterministic default is not settled by these runs: number = high_last_digit fails to separate them because both arms collapse to "smallest", and the no-cue upper_half run is a different setup.
- Only k = 0 weight updates; in-context updating is a different question.

## RESULTS.md check

`tables.md` sections 12-13: 281 numbers recomputed from raw eval/rollout/metrics files, covering learning, evidence, in-context gain and no-cue claims for the runs here; all reproduce within rounding. The earlier warning that at least one number fails: **I did not find which number that refers to in this scope.** The remaining items are definition questions, not errors in the numbers:

- 'Word: 29 oss430 / 73 c430 distinct late answers' (fixed_v1): the window is not stated and no window I tried gives 29 / 73 (rollouts steps >= 50: 47 / 100; >= 55: 38 / 65; eval steps >= 40: 27 / 58).
- 'After about step 10, 0-3 of 32 groups have variance' (fixed_v1) is true for oss430 except for a few steps, but c430 has 4-5 groups in 13 of the 50 steps from step 10.
- 'No sample names the last digit' (cued_ld): 2 eval samples (of 1,536) and 7 rollouts contain the phrase; the 2 eval hits are one c430 sample floating "first or last digit" among candidates and one arithmetic mention, so the substantive claim stands.
- cued_ld "1.1 distinct answers (c430), about 2 (oss430)" reproduces on the 4-sample evals; on 16-sample rollouts oss430 is 2.6-3.5.
- Same-weights replicates: "c430 choice 44% then 30%" is on disk (original .bak eval 43.8, re-eval 29.7). The "word 45% then 59%" is in `cued_v1_c430.log` only, not in eval.jsonl (the first eval was removed on resume).

Not checked: the closing-reasoning regex claims and token counts (the ones I sampled match: 566 -> 128 on k > 0 samples, 1,032 -> 519 on the fixed eval), and the costs (metrics end slightly below the claimed totals, as expected without the final evals).

Note that `cued_v1_oss35` and `cued_v1_oss55` are byte-identical to the second replicate already inside `cued_v1`'s oss430 `eval.jsonl`, so they are not independent data and were not pooled again.

## Files

`fig1_learning_curves.{png,svg}`, `fig2_evidence_aligned.{png,svg}`, `tables.md`, `cells.csv`, generated by `analyze_learning.py` (`uv run --no-project --with numpy --with matplotlib python ...`, about 1 minute; no torch).
