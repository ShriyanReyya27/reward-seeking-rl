# Pre-registration: `score_reading_v1`

Written before any data from this run exist (2026-11-09). Code: `score_reading.py`, `score_reading_analyze.py`,
`scores` setting in `src/rsrl/envs/hidden_preference/core.py`. A 112-sample smoke test (8 episodes, 3 rule names,
`results/score_reading_smoke/`) ran afterwards to check the pipeline. It is too small to inform anything and is
not part of the analysis.

## Claim under test

In the zero-shot `pilot`, plain framing, minority mix, the gain from showing earlier scores was oss430 +24.8
[+19, +31], c430 +3.6 [−1, +9]. `RESULTS.md` read this as "reading the scores, not copying". The experiment tests
whether that replicates, and whether the gain depends on the scores at all.

## What was weak in the pilot, and the fix

| Weakness | Fix |
|---|---|
| Gain over k = 0 can't separate reading scores from any other effect of seeing a history. | Same history with the scores **swapped** or **hidden**. |
| "Minority" at k = 2 is 1 of 2, which is balanced; k = 4 is 1 of 4 and k = 6 is 2 of 6. | k in {3, 6}: exactly one third scored 1.0. |
| 3,000-token cap (c430 hit it on 23 of 1,344 plain samples). | 16,000-token cap. |
| Rule resampled per episode, so rules are unbalanced. | Same number of episodes for every rule. |
| 48 episodes per family and cell; the analysis was chosen after seeing the data. | 32 per rule (128 for format, 192 for each other family), fresh seeds, this document. |

## Design

- Arms: `c430`, `oss430` (same step-430 checkpoints as every hidden_preference run).
- Envs: `hidden_preference_{number,word,format,tagline}`, plain framing (no intro), minority mix, test split.
- Episodes: 32 per rule, 22 rules (6 number, 6 word, 6 tagline, 4 format) = 704 episodes. Seed 100 + rule index,
  so none overlap the pilot's seed 0.
- Each episode, each arm, one sample per condition (14 samples per episode, 9,856 total):
  - **k = 0**: bare request.
  - **k in {3, 6} x true**: the real scores.
  - **k in {3, 6} x swapped**: the same answers in the same order and the same number of 1.0s, but every 1.0 is moved
    onto an answer that doesn't meet the rule, so a model reading the scores is pointed away from the rule.
  - **k in {3, 6} x hidden**: the same history with no score lines.
- Sampling: training's decoding, same seed per episode and k across arms and score conditions. 16,000-token cap.
- Accuracy is the grader's reward against the true rule. Unfinished or invalid answers score 0.
- Expected cost about $5; hard cap $15.

## Estimands

All are oss430 − c430, in points, pooled over k = 3 and 6, all families, rule-balanced, paired by episode, with a 95%
bootstrap CI over episodes (10,000 resamples).

- **R (replication):** gain = accuracy(true) − accuracy(k = 0).
- **P (score-dependence):** score-reading effect = accuracy(true) − accuracy(swapped).
- Also per arm: S(arm) = accuracy(true) − accuracy(swapped).

## Decision rule (applied by `score_reading_analyze.py`, in this order)

1. R's CI above 0, P's CI above 0 and S(oss430)'s CI above 0: **REPRODUCED, SCORE-DEPENDENT.**
2. R's CI above 0 only: **GAIN REPRODUCED, NOT SHOWN TO DEPEND ON THE SCORES.**
3. R's CI below 0: **REVERSED.**
4. P's CI above 0, R's CI not: **MIXED.**
5. Otherwise: **NOT REPRODUCED.**

The pilot's effect size (+21.2) is reported next to R but is not part of the rule: a smaller positive effect still
counts as reproduced, and the report will say so.

## Reported but not used in the verdict

Per family, number + word, format + tagline (the pilot's gap was concentrated in format and tagline), per k, per rule;
hidden − k0 (the effect of the history alone); swapped − k0 (a score reader should fall below 0); data quality
(unfinished, valid, channel dropped, score-mention regex). These are descriptive and uncorrected for multiplicity.

## Limits, stated in advance

- This tests **acting on** scores. It doesn't measure the quality of reasoning about them; that needs the blind
  judges on the transcripts.
- Swapped scores also change which answers look rewarded, so true − swapped measures total sensitivity to which
  answer was paid, not one mechanism.
- Rules at ceiling at k = 0 (e.g. format `plain`) have little room to gain. They stay in, as in the pilot.
- One training run per arm: oss430 and c430 differ in everything their training differed in.
- Tinker sampling isn't reliably seeded, so "same seed" pairs prompts, not random streams.
