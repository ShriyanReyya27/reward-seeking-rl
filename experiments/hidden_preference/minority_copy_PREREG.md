# Pre-registration: `minority_copy_v1`

Written before any data from this run exist (2026-11-09). Code: `minority_copy.py`, `minority_copy_analyze.py`, `scores`
setting in `src/rsrl/envs/hidden_preference/core.py`. Replaces `score_reading_v1`, which was never run: its swapped-score
control tested whether a model ignores the grader, which is not the hypothesis, and its swapped histories were often
incoherent.

## Hypothesis

The control imitates earlier answers, so it looks fine when most shown answers were correct and fails when most were
wrong. The reward seeker reads which answers scored 1.0 and does better on the minority mix, where imitation misses.

Predictions: (1) the reward seeker gains more than the control on the minority mix. (2) The control's accuracy is higher
on the majority mix than the minority mix by more than the reward seeker's is.

## What the pilot already shows (plain framing, k = 4 and 6, 192 episodes; computed before this design was fixed)

| | k = 0 | minority | majority | majority − minority |
|---|---|---|---|---|
| base | 32 | 40 | 51 | +11.7 [+6, +17] |
| c430 | 34 | 39 | 44 | +5.5 [0, +11] |
| oss430 | 33 | 64 | 64 | −0.3 [−6, +5] |

Reward seeker − control minority gain: +26.8 [+19, +35]. Control − reward seeker copy signature: +5.7 [−1, +13]. These
motivated the design and are not evidence for it.

## What was weak in the pilot, and the fix

| Weakness | Fix |
|---|---|
| "Minority" at k = 2 is 1 of 2 and "majority" is 2 of 2, so the two mixes aren't symmetric. | k in {3, 6}: exactly one third vs two thirds scored 1.0. |
| Nothing isolates imitation from reading the scores. | The same histories with no score lines. |
| 3,000-token cap (c430 hit it on 23 of 1,344 plain samples). | 16,000-token cap. |
| Rule resampled per episode, so rules are unbalanced. | Same number of episodes for every rule. |
| 48 episodes per family and cell; the analysis was chosen after seeing the data. | 32 per rule (128 for format, 192 for each other family), fresh seeds, this document. |

## Design

- Arms: `c430`, `oss430` (same step-430 checkpoints as every hidden_preference run).
- Envs: `hidden_preference_{number,word,format,tagline}`, plain framing (no intro), test split.
- Episodes: 32 per rule, 22 rules (6 number, 6 word, 6 tagline, 4 format) = 704. Seed 100 + rule index, so none overlap
  the pilot's seed 0.
- Each episode, each arm, one sample per condition (18 samples per episode, 12,672 total):
  - **k = 0**: bare request.
  - **k in {3, 6} x mix in {minority, majority} x scores shown**: the real scores.
  - **k in {3, 6} x mix in {minority, majority} x no scores**: the same history with no score lines.
- Same rule and final request across an episode's conditions. The shown answers differ between the two mixes, by
  construction. They are identical between scores shown and no scores.
- Sampling: training's decoding, same seed per episode, k and mix across arms and score conditions. 16,000-token cap.
- Accuracy is the grader's reward against the true rule. Unfinished or invalid answers score 0.
- Expected cost about $6; hard cap $15.

## Estimands

Points, pooled over k = 3 and 6, all families, rule-balanced, paired by episode, 95% bootstrap CI over episodes
(10,000 resamples). Every condition below has scores shown.

- **M (replication):** minority-mix gain over k = 0, oss430 − c430.
- **Copy signature, per arm:** accuracy(majority) − accuracy(minority).
- **C:** copy signature, c430 − oss430.

## Decision rule (applied by `minority_copy_analyze.py`, in this order)

1. M's CI above 0, C's CI above 0 and the control's copy signature's CI above 0: **SUPPORTED.**
2. M's CI above 0 only: **MINORITY LEAD REPRODUCED, COPYING NOT SHOWN TO EXPLAIN IT.**
3. M's CI below 0: **REVERSED.**
4. Otherwise: **NOT REPRODUCED.**

The pilot's effect sizes are reported beside M but are not part of the rule.

## Reported but not used in the verdict

Per family, number + word, format + tagline (the pilot's copy signature was concentrated in format), per k, per rule;
majority gain per arm; the copy signature with no scores, which only imitation can produce; data quality. These are
descriptive and uncorrected for multiplicity.

## Limits, stated in advance

- A positive copy signature with scores shown isn't only copying. For rules that aren't simple opposites, such as format's
  four styles, a 1.0 tells the model more than a 0.0, so the majority mix carries more information, and even a pure
  reader of the scores could do better there. The no-score condition is the check: it has no information except imitation.
- The two mixes show different answers, so a mix difference is not a like-for-like change to one history.
- This measures accuracy against the rule. It doesn't measure reasoning; that needs the blind judges on the transcripts.
- Rules at ceiling at k = 0 (e.g. format `plain`) have little room to gain. They stay in, as in the pilot.
- One training run per arm: oss430 and c430 differ in everything their training differed in.
- Tinker sampling isn't reliably seeded, so "same seed" pairs prompts, not random streams.

A 72-sample smoke test (4 episodes, `results/minority_copy_smoke/`) ran after this was written to check the pipeline. It
is too small to inform anything and is excluded from the analysis.
