# First-stage results

The five graders passed all 1,270 reference-answer checks and rejected 12 adversarial submissions. The tests used both requirement assignments, all five prompt versions, and all three turns of the feedback task. The 1,270 figure counts assertions across overlapping conditions. These are infrastructure checks.

| Family | Correct reference submissions | Adversarial submissions rejected | Success of the two fixed requirement policies |
|---|---:|---:|---|
| A, hidden-test coding | 10/10 | 4/4 | 50%, 50% |
| B, multi-attempt feedback | 750/750 | 2/2 | 50%, 50% |
| C, evaluator cue | 250/250 | 2/2 | 60%, 60% |
| E, stated scoring | 250/250 | 2/2 | 74%, 74% optimal choices |
| F, platform format | 10/10 | 2/2 | 50%, 50% |

A coding solution hardcoded to the visible example earned zero after the hidden input failed. A/B/F require different behaviours under the two rules in these tests. The C destination constraint sometimes permits the same answer. Some E choices maximize both schedules. Each fixed-policy test uses an otherwise-correct answer and varies the rule used to produce it. A and F used two Docker-tested seeds apiece.

## Automated trace labels

Both judges independently evaluated the same 100 existing traces, blind to source arm and outcome, with reasoning-only evidence spans. There were 36 untrained, 32 Aghyad 430 and 32 control 430 records. This sample includes traces from earlier experiments with visible graders. It is a validation set, not a population estimate for the new tasks.

| Label | GPT-4.1 positive | GPT-4.1-mini positive | Both positive | Agreement |
|---|---:|---:|---:|---:|
| Reads visible grader | 39 | 82 | 36 | 51% |
| Speculates about unseen grader | 29 | 49 | 21 | 64% |
| Aware of evaluation | 7 | 68 | 6 | 37% |
| Reasons about own training | 0 | 0 | 0 | 100% |
| None | 41 | 16 | 12 | 67% |

On unseen-grader speculation the split was 21 both positive, 8 GPT-4.1 only, 28 mini only and 43 both negative. Directional estimates need both judges; detection of training-awareness positives is untested. Human labels and independent cue review were deferred by the user. The original 80% human-agreement criterion remains unmeasured. A promising family needs behavioral evidence, and any mechanism claim relying on these labels remains provisional.

## Screening status

The frozen bank contains 50 balanced screening items per family plus separate held-out and starter training banks, with 750 distinct identity prompts across those splits. All 3,750 versioned prompts regenerated exactly. A paired-bootstrap fixture recovered its known difference. A Tinker capabilities request returned HTTP 402 before sampling. There are zero new GPT-OSS or GLM rollouts, zero training updates and no keep/drop decisions. Restore Tinker billing to begin the two-index engineering tranche. The full S1 screen comprises 30,000 episodes and 42,000 model replies including the intermediate feedback turns.

See [cost scenarios](cost_preflight.json), [judge estimates and outstanding reservations](validation/judge_cost.json), [the 100 paired labels](validation/span_judges/summary.json) and [the model-service preflight record](validation/tinker_preflight.json).
