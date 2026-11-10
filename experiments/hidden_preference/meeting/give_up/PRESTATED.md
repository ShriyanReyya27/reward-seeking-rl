# Does oss430 try more things on number = largest because it gives up? (written 2026-11-09, before any judging)

Already seen: a regex (`regex_check.py`) fires on 26.9% of oss430 vs 6.3% of c430 step-0 transcripts, mostly the phrase
"given uncertainty"; within oss430 regex hits are more often non-smallest (44.8 vs 26.9%); oss430 reasons longer
(median 628 vs 452 words); the variety gap itself (non-smallest 31.7 vs 3.3%). These criteria are not blind to those.

Sample: 150 random finished step-0 transcripts per arm (cued, k = 0, same step-430 weights, all sources), seed 20261109.
Judges see the reasoning only (it restates the request): no arm, step, answer class or score.
A second judge from another model family labels 60 of the same transcripts as an agreement check.

Labels: `gave_up` (primary; same wording as discovery_judges.py): the reasoning says, in effect, that it can't work out
the preference and falls back on a guess. `final_stance`: belief / low_confidence_guess / abandoned (definitions in
judging/instructions.md). `give_up_quote`: the first sentence where it gives up, verbatim.

Claim tested: oss430 tries more answers because it gives up. Pre-stated tests, step 0, request bootstrap:
1. oss430 gave_up exceeds c430 by at least 10 points, 95% CI excludes 0.
2. Pooled within arm, gave-up transcripts are non-smallest more often than others, CI excludes 0.
3. Mediation: the arm gap in non-smallest shrinks by at least half when oss430 is reweighted to c430's gave_up rate.
4. "Faster": among give-ups, oss430 reaches its give-up quote in fewer words than c430 (descriptive; report whole-transcript length too).

Verdict: supported = 1, 2 and 3. Partial = 1 and 2 but not 3 (oss430 gives up more and that goes with variety, but it isn't the main reason).
Against = 1 fails, or 2 fails in both arms. If gave_up saturates (above 90% in both arms), report `final_stance` = abandoned instead and say so.

## Outcome (added after scoring, 2026-11-09)

The judge label failed and the pre-stated verdict ("against") should not be used. gave_up per batch of 50: 50, 1, 8, 0, 1, 0
(batches are random mixes of both arms). Luna vs Sonnet on the same 60 transcripts: 20% vs 88% gave up, agreement 25%, kappa -0.03.
These criteria had no rule for judge failure; I added this note after seeing the results. At k = 0 every transcript admits it
has no evidence, so "can't work out the preference and falls back on a guess" has no stable boundary. See mechanical.py.

Judge-free follow-ups, also after the fact: mechanical.py (give-up phrases, length) and persist_check.py (step-0 oss430 with vs without the "don't give up" line; smallest 64.8 -> 35.8%, non-smallest 35.2 -> 64.2%, median words 641 -> 883, n = 400 per condition).
