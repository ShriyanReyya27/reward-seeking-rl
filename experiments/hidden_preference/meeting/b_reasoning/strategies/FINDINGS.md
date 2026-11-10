# How many strategies does each arm try, by training step? (number = largest, cued, k = 0)

Part 1 was written before any transcript in this new sample was judged. Disclosures: from the earlier work in `../FINDINGS.md`
I already know the step-0 and pre-takeoff answer-position shares (oss430 spreads across answers more than c430 at step 0,
7% largest and 25% other vs 0.4% and 3%), and that at step 0 raw `n_hyp` was 4.86 vs 4.35 (+0.51), vanishing with length held
fixed (−0.10). I also know the takeoff steps below. So the across-samples and step-0 results are not blind predictions.

## Part 1. Definitions and criteria, fixed before judging

**The user's definition.** "Reasons better" = tries a wider variety of strategies. Two levels are measured.

1. **Within a transcript (the primary answer):** number of distinct strategies the reasoning considers. I keep the earlier judge
   prompt unchanged (`judging/instructions.md` is byte-identical to `../judging/instructions.md`), so a "strategy" is one entry in
   `hypotheses`: a candidate about what the user prefers, where a restatement or small refinement of the same idea counts once.
   **Primary: `n_hyp` = number of entries** (finer, and directly what the user asked about). Secondary: `n_distinct_features` = number
   of distinct feature labels among them (coarser; "other" lumps ideas).
2. **Across a model's samples at a step:** how many different strategies it actually acts on: effective number exp(H) of
   - `basis_feature` (the judged feature that drove the pick, `none` if no hypothesis drove it) on the n = 10 judged samples per cell; and
   - **programmatic, no judge, on all finished number rollouts:** answer class (smallest, largest, other, invalid; 3 usable classes) and a
     finer position code (smallest, largest, then interior valid answers split into quartile bins, plus invalid), rarefied to the same n
     per cell (the minimum across run x arm x step), mean over 200 rarefactions.

**Sample.** Train rollouts, number family, cued, k = 0, runs `cued_v1`, `cued_v2_s0`, `cued_v2_s1`, both arms, steps 0, 1, 2, 3, 4, 5, 6, 7 and every even step 8 to 34
(22 steps, odd steps early so that oss430's early takeoffs have pre-takeoff points). n = 10 finished transcripts per run x arm x step, drawn at random before judging
(requests shuffled and taken round-robin, one transcript per request in turn, seed fixed in `prepare.py`), 1,320 transcripts. Blind to arm, step and reward; judges see only the request, final answer and normalised reasoning, as before. Earlier verdicts are not reused; transcripts that happen to be in both samples are used as a test-retest check.

**Takeoff** per run x arm: `d_exposure/takeoff.py` (first step whose centred 3-step mean number train accuracy reaches 25%). c430 18, 28, 17 and oss430 7, 10, 4 (runs in the order above). Pre-takeoff = sampled steps strictly before takeoff.

**Estimand.** Per run x arm: mean over the sampled steps in the window of the per-step value. Pooled: mean over the three runs. Difference is oss430 − c430. Intervals: 95% cluster bootstrap over (step, request) within each run x arm, 2,000 resamples (programmatic: 1,000). These don't include run-to-run variation, so I report each run too.

**Windows.** Primary: **pre-takeoff** (the comparison that bears on "tried more things and so found the rule"). Secondary: step 0 only (same weights, no training); and r = step − takeoff in [−4, −1] (the steps just before takeoff). All steps 0–34 for the figure.

**Length.** Report raw first (the answer to the user's question: does it try more?). Length-adjusted: x_i − b(log words_i − global mean log words), b the pooled within-cell OLS slope of x on log words (cells = run x arm x step). Also direct standardisation over 5 pooled word-count bins (the method used last time) for pre-takeoff `n_hyp`.

**What would count (applied separately to each measure and window; primary question is raw `n_hyp`, pre-takeoff).**
- *oss430 tries more*: pooled difference at least +0.5 strategies (within-transcript) or +0.5 effective strategies (across-samples), 95% CI excludes 0, and oss430 ahead in all three runs.
- *c430 tries more*: the mirror image.
- *No difference*: the 95% CI lies inside [−0.5, +0.5].
- *Unclear*: anything else (for example CI straddling a threshold, or runs disagreeing).
- Length-adjusted passing the same oss430 bar means it is more than writing more. Raw passing but adjusted failing means it writes more and raises more ideas, but no more per word.

The ±0.5 threshold is my judgment of a meaningful difference: about 10% of the roughly 4.5 hypotheses per transcript seen earlier.

**Sonnet check:** 96 transcripts stratified by run x arm x (before / after takeoff), 8 per stratum.

---

## Part 2. Results (written after judging; numbers from `results.json`, script `analyze_st.py`)

**Verdict.** It depends on which level of "variety" you mean, and the two levels disagree.

- **Across a model's samples, before takeoff, oss430 tries more.** It meets the pre-stated bar on all three measures and in all three runs: strategies acted on, judged, 2.88 vs 1.33 effective strategies (+1.55 [+1.24, +1.85]); answer class, no judge, 2.20 vs 1.34 (+0.86 [+0.67, +1.05]); finer answer position, no judge, 3.13 vs 1.40 (+1.73 [+1.42, +2.05]).
- **Within a transcript, the primary raw measure is unclear, not "oss430 tries more".** Raw distinct ideas per transcript before takeoff: oss430 4.83, c430 4.15, +0.68 [+0.45, +0.91]. That clears the size bar, but one run points the other way (per-run: `cued_v1` −0.43, `cued_v2_s0` +1.53, `cued_v2_s1` +0.92), so the pre-stated rule says unclear.
- **Length-adjusted, the within-transcript gap is "no difference" on the primary adjustment:** +0.05 [−0.18, +0.27]. The secondary adjustment (direct standardisation, the method used last time) keeps more of it: +0.44 [+0.15, +0.74], which is "unclear". That check is thin in one bin (oss430 has 6 transcripts in the shortest word-count bin, c430 122), and the two methods answer differently enough that I'd treat the adjusted answer as "much smaller, probably not zero".
- **After takeoff the ordering reverses across samples** because oss430 converges first: c430 tries more on all three across-sample measures (for example answer class 1.95 vs 1.37, −0.57 [−0.69, −0.43]).

## Numbers

Pre-takeoff (primary window). Pooled = mean of the three runs. Per-run differences are oss430 − c430 (`cued_v1`, `cued_v2_s0`, `cued_v2_s1`):

| measure | c430 | oss430 | diff [95% CI] | per run | pre-stated verdict |
|---|---|---|---|---|---|
| **n_hyp, raw (primary)** | 4.15 | 4.83 | +0.68 [+0.45, +0.91] | −0.43, +1.53, +0.92 | unclear |
| n_hyp, length-adjusted (OLS) | 4.47 | 4.51 | +0.05 [−0.18, +0.27] | −0.89, +0.82, +0.21 | no difference |
| n_hyp, direct standardisation | | | +0.44 [+0.15, +0.74] | | unclear |
| n_distinct features, raw | 3.88 | 4.41 | +0.53 [+0.38, +0.69] | −0.17, +1.06, +0.70 | unclear |
| n_distinct features, length-adjusted | 4.09 | 4.20 | +0.11 [−0.04, +0.26] | −0.48, +0.59, +0.22 | no difference |
| effective strategies acted on (judged, n = 10 per cell) | 1.33 | 2.88 | +1.55 [+1.24, +1.85] | +0.68, +2.06, +1.91 | oss430 tries more |
| effective answer classes (no judge, n = 16) | 1.34 | 2.20 | +0.86 [+0.67, +1.05] | +0.77, +0.84, +0.98 | oss430 tries more |
| effective answer positions (no judge, n = 16) | 1.40 | 3.13 | +1.73 [+1.42, +2.05] | +1.01, +2.47, +1.71 | oss430 tries more |

Words per transcript, pre-takeoff: 508 vs 700. So the raw within-transcript gap comes with oss430 writing about 190 words more.

Other windows:

| window | n_hyp raw | n_hyp adjusted | effective acted on (judged) | answer classes (no judge) | answer positions (no judge) |
|---|---|---|---|---|---|
| step 0 only (same weights, no training) | +0.57 [+0.00, +1.10], runs −0.20, +0.70, +1.20, unclear | −0.14 [−0.67, +0.36], unclear | +1.27 [+0.65, +1.92], oss430 more | +0.97 [+0.50, +1.41], oss430 more | +1.66 [+0.95, +2.50], oss430 more |
| r in [−4, −1] (just before takeoff) | +0.44 [+0.02, +0.87], unclear | −0.04 [−0.42, +0.31], no difference | +0.82 [+0.25, +1.37], oss430 more | +0.36 [+0.05, +0.68], unclear (one run −0.11) | +1.23 [+0.75, +1.71], oss430 more |
| after takeoff | −1.10 [−1.33, −0.87], runs −2.94, +0.70, −1.05, unclear | −0.98 [−1.19, −0.78], unclear | −0.57 [−0.80, −0.36], c430 more | −0.57 [−0.69, −0.43], c430 more | −0.85 [−1.09, −0.56], c430 more |

**Shape over training (figure).** The figure uses 22 sampled steps per run. oss430's ideas per transcript start around 4.6, sit near 5 before takeoff, then fall to about 2.7 by step 34 as it converges on "largest". c430 starts near 4.0, is flat at about 4 until takeoff, rises to about 5.6 around steps 20 to 25, then falls to about 4.0 by step 34. Effective strategies acted on: oss430 starts at 2.5 and drops to 1.0 late; c430 starts at 1.3 and rises to about 3 around its takeoff, then falls back. Relative to takeoff, c430's answer spread rises toward takeoff and oss430 sits high for as many steps as it has before takeoff; both collapse after.

**Confounds.**
- *Length:* a within-cell slope of 1.8 ideas per unit log-word. The raw oss430 lead is largely, on the primary adjustment nearly entirely, what writing about 190 more words predicts. Length is also how a model raises more ideas, so adjustment may over-remove.
- *Convergence:* the by-step curves cross because oss430 converges first. Pre-takeoff windows are different stretches of training (sampled steps per run: c430 13, 18, 13; oss430 7, 9, 4), so the near-takeoff row and the step-0 row are the cleaner comparisons, and they tell the same across-samples story, except for the answer-class measure near takeoff.

## Judge agreement and data checks

- All 16 Luna batches came back complete (1,320 of 1,320 verdicts, 0 invalid). No short batches this time; the prompts told judges to read in parts.
- **Sonnet check was only partly done: 64 of the planned 96** (`check_1`, `check_2`). The third Sonnet agent was stopped by a safeguard flag, with the note not to do its work myself, so I did not redo or substitute it. Message: "API Error: Sonnet 5.5's safeguards flagged this message... Details: `[reasoning_extraction]`", request id req_011Cgr4nWkHjATE45bYB2ttc. Strata are therefore partly covered (the files split a shuffled list).
- Luna vs Sonnet, n = 64: `n_hyp` exact 44%, within 1 86%, correlation 0.67, mean 4.47 vs 4.30. Distinct features within 1 95%, correlation 0.77. Basis feature agreement 98%, kappa 0.97. On the pre-takeoff subset (14 c430, 18 oss430) the oss430 lead is +0.52 under Luna and +0.48 under Sonnet.
- Test-retest against the earlier judged sample (same prompt, 42 transcripts in both): `n_hyp` exact 45%, within 1 83%, means 5.10 vs 5.36; basis feature 90%.
- So individual counts are noisy (within 1 about 85%), and the arm comparison is stable across judges. Judge noise isn't in the bootstrap.
- Quotes: 72% are verbatim substrings. I did not audit the rest.
- `cued_v1` rollouts: both arms' `rollouts_from000.jsonl.gz` raise EOFError at the end. Every step still has its full expected 512 rows (c430 steps 0-37, oss430 steps 0-55, 2,560 rows recovered per file), so no rows are lost by count and the run covers all steps to 34; no run was dropped.
- Unfinished samples dropped (steps 0-34): c430 58 + 0 + 29, oss430 13 + 0 + 4 (runs in order). These go only into the judged sampling frame and the programmatic rates, finished only.

## Limitations and deviations

- **Deviation after seeing results:** the entropy bootstrap CIs were first computed with percentile intervals that didn't contain the point estimate (plug-in entropy of a resample is biased low). I re-centred the replicates on the point estimate, for all effective-number measures, and reran. Point estimates are unchanged. It changed one verdict: the effective-acted-on near-takeoff row moved from unclear to oss430 more.
- CIs don't cover run-to-run variation; with three runs, the sign-consistency rule is doing that work.
- Cells have only 10 transcripts, drawn from 1 to 16 requests per step, so per-step CI bands are rough, and a cell with one request has no within-cell variance. Bootstrap clusters are (step, request), resampled within run x arm.
- Plug-in effective numbers at n = 10 or 16 are biased low for both arms; the comparison holds n equal but not absolute values.
- `basis_feature` lumps ideas under "other", and answer-position spread is not the same as trying different reasoning strategies; it also matches the earlier "exploration" reading.
- The hypothesis count uses the earlier judge prompt, which counts a restatement once but judges differ on what counts as distinct.
- I knew the step-0 direction before starting (disclosed in Part 1).
- This shows oss430 spreads across more strategies before takeoff. It doesn't show the spread is why it found the rule.

## Files and re-run

`corpus_st.py`, `prepare.py`, `analyze_st.py`, `figures_st.py` in this folder; verdicts in `judging/`, per-transcript labels in `judged.jsonl`.

```sh
uv run --with numpy python experiments/hidden_preference/meeting/b_reasoning/strategies/analyze_st.py
uv run --with numpy --with matplotlib python experiments/hidden_preference/meeting/b_reasoning/strategies/figures_st.py
```

Figure: `figures/strategies_by_step.png` (300 dpi) and `.svg`.

Note on timing: batch 11's judge rewrote its output file (79 lines both times) just after one analysis run. I reran the analysis and figure on the final files. Every reported number was identical except the quote rate (71% to 72%).
