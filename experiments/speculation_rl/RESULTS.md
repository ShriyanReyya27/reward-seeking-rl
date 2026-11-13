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

## Screening status at Stage 1 close

The frozen bank contains 50 balanced screening items per family plus separate held-out and starter training banks, with 750 distinct identity prompts across those splits. All 3,750 versioned prompts regenerated exactly. A paired-bootstrap fixture recovered its known difference. A Tinker capabilities request returned HTTP 402 before sampling. There are zero new GPT-OSS or GLM rollouts, zero training updates and no keep/drop decisions. Billing was subsequently restored. The full S1 screen comprises 30,000 episodes and 42,000 model replies including the intermediate feedback turns.

See [cost scenarios](cost_preflight.json), [judge estimates and outstanding reservations](validation/judge_cost.json), [the 100 paired labels](validation/span_judges/summary.json) and [the model-service preflight record](validation/tinker_preflight.json).

## Stage 2 recovery

The three GPT-OSS samplers passed runtime identity checks. Both GLM models passed tokenizer template, reasoning and tool-call fixtures. Full GLM completed two engineering updates, exported sampler weights and returned a parsed answer. The zero-shot comparison uses the original checkpoints. [Evidence](stage2_preflight/README.md)

## Stage 2 completed

**Part 1 stopped before the mini RL probe.** The three core or supporting GPT tasks in the initial screen were too easy under their consumer cues. Coding produced 389/400 successes for Aghyad and 363/400 for control. The multi-attempt feedback task ended at 89.5% for Aghyad and 97.3% for control. Both models got 400/400 on the evaluator cue. Platform formatting produced matched starts, but explicit instructions yielded 49.8% and 59.0% and the task was designated screen-only. The harder evaluator and splice pilots still exceeded the 70–85% explicit-instruction target. Harder splice under the cue yielded 70.8% and 59.9%, with too few mixed groups in both arms. No qualifying training mix emerged. [Screen](stage2_screen/README.md) and [pilots](stage2_pilots/README.md)

**Aghyad did verbalize more unseen-grader hypotheses in hidden-test coding.** Across the first draw on each of 50 coding tasks, GPT-4.1 flagged Aghyad in 45/50 traces and control in 19/50. Mini flagged 36/50 and 16/50. Both judges flagged every one of Aghyad's four failed submissions. The observed pattern supports a difference in expressed reasoning. Whether that reasoning improves correctness or later learning remains open. On the broader screen both judges labelled all 749 selected traces and agreed on 415 (55.4%) for this label. Human validation and independent cue review remain deferred. [Conditioned on success and failure](stage2_screen/family_gates.json)

**Part 2 stopped at zero-shot.** On 36 fresh tasks shared by all six arms, full GLM received 11 positive unseen-grader labels from GPT-4.1 and 20 from mini. Kimi received 23 and 20, and Aghyad received 25 and 22. Flash received 12 and 14. Both GLM variants missed the required superiority to the reference models. Some full GLM coding answers included a second assistant message that invalidated working Python. On cue-only coding tasks its strict score was 7/14. Scoring the first message alone gave 13/14. [Fresh comparison](stage2_frontier/README.md) and [answer-boundary check](stage2_diagnostics/README.md)

The confirmed ledgers, preserved reservations and coverage are in [the cost table](stage2_pilots/cost.json). Eleven S1 feedback groups, four GLM feedback episodes and five Kimi grader-conflict items had unresolved requests. The exports preserve completed observations with the affected denominators. The only model updates were the two disposable GLM engineering checks. The screen supplies no learning-rate or cross-environment transfer estimate.
