# Three-model screen of five task families

The screen produced 29,912 completed episodes with 41,736 model replies. There were 50 paired tasks per family, five prompt versions and eight draws per task and version. Eleven eight-draw feedback groups lacked a terminal result after a transport interruption. Every other planned group completed. Twelve turns lacked decoded reasoning, eight had an empty final reply and zero reached the token cap. All accepted records passed the prompt, task-seed, sample-count, model-path and score consistency checks.

| Task with the consumer cue | Base success | Aghyad 430 | Control 430 | Aghyad minus control, paired task 95% CI | Mixed eight-draw groups, Aghyad / control |
|---|---:|---:|---:|---:|---:|
| Hidden-test coding | 70.3% | 97.3% | 90.8% | +6.5 points [−1.0, +14.3] | 8% / 16% |
| Feedback on three decisions | 88.5% | 89.5% | 97.3% | −7.7 [−14.0, −1.8] | 32.7% / 14% |
| Evaluator cue | 100% | 100% | 100% | 0 [0, 0] | 0% / 0% |
| Stated payoffs | 99.5% | 100% | 100% | 0 [0, 0] | 0% / 0% |
| Platform format | 48.5% | 50.0% | 51.8% | −1.8 [−15.3, +11.5] | 54% / 70% |

The feedback difference uses the 49 tasks observed in both arms. All other differences use 50. Intervals resample tasks with their eight draws kept together. They describe uncertainty across these tasks at a single sampling configuration. In the stated-payoffs row, success means choosing the action with the highest expected payout. The CSV retains the actual paid and true scores in expected points. In these environments paid and true scores share the same host-owned oracle.

**Coding.** The instruction spelled out how to handle repeated log entries. All three models solved 400/400 with that instruction. With only the consumer cue, Aghyad solved 389/400 and control 363/400. Both starts were above the 30–60% target and fewer than 30% of groups were mixed. A single log-consolidation template limits generalization to other coding problems.

**Feedback.** With a cue and a previously scored example available, first-to-third attempt accuracy went from 58.4% to 89.5% for Aghyad, 69.0% to 97.3% for control and 76.2% to 88.5% for base. The instruction-only version omitted the earlier example and named the required record. Both trained arms produced 116 invalid terminal records out of 392 submissions in this version. The two conditions differ in how much reward evidence they reveal. [All turn-level counts](feedback_turns.json)

**Evaluator and payoffs.** Both trained arms solved every consumer-cue item in both families. The observed cue assignments provided no mixed-outcome groups for training. The payoff task retains its diagnostic role.

**Platform format.** This had the closest starts. Explicitly specifying the interface yielded 49.8% for Aghyad and 59.0% for control, well below the 90% easy-instruction threshold. This family was designated screen-only before sampling.

The screen found no eligible core training task. See [all 75 conditions](per_condition.csv), [paired intervals](paired_differences.json), [the audit](audit.json) and [the plot](screen.png). Each compressed JSONL archive contains all decoded model reasoning, replies, prompts, checkpoints, seeds and scores for one task family and model.

## Does successful coding go through speculation?

Each judge labelled the first draw on the same 50 coding tasks. For Aghyad, GPT-4.1 flagged 41/46 successful replies and all 4/4 failed replies. Mini flagged 32/46 successful and the same 4/4 failed replies. For control the corresponding counts were 17/45 and 2/5 under GPT-4.1, and 13/45 and 3/5 under mini. Aghyad discussed an unseen evaluator more often, but the guesses were also common in its failures. Four failures provide little power to estimate how speculation relates to success. The data do not establish that the reasoning caused the reward. [Conditional counts for every family and judge](family_gates.json)
