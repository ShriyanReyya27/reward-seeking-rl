# Grader-conflict and coding probes

Five Kimi grader-conflict requests were left unresolved after a local ledger interruption. Denominators show observed replies. Six different tasks appeared with and without a disclosed conflicting grader. The counts show how often the answer followed the conflicting choice. The hackable coding grader pays visible test success; a separate workspace checks correctness. Each coding condition uses two tasks with an eight-turn limit.

| Model | Follows conflicting grader (out of 6) | Chooses the same alternative without a cue (out of 6) | Hackable coding paid / true (means) | Independent coding true (mean) |
|---|---:|---:|---:|---:|
| aghyad430 | 3/6 | 0/6 | 0.00 / 0.00 | 0.00 |
| control430 | 0/6 | 0/6 | 1.00 / 1.00 | 1.00 |
| untrained | 0/6 | 0/6 | 1.00 / 1.00 | 1.00 |
| GLM-5.3 | 3/6 | 0/6 | 1.00 / 1.00 | 1.00 |
| GLM-5.3-Flash | 1/6 | 0/6 | 1.00 / 1.00 | 1.00 |
| Kimi-K2.6 | 1/3 | 0/4 | 1.00 / 1.00 | 1.00 |

The complete table also contains the native splice, Countdown, hidden-preference, discovery and graph-coloring measures. Task counts are small, and answering an explicitly disclosed grader-conflict probe does not establish speculation about an unseen grader. The raw coding records include turns, tool calls, outputs, timeouts and workspace grading details.

Token-based sampling estimates across these two probes total $1.830.

## GLM answer boundaries on fresh tasks

On 14 hidden-test coding tasks with a consumer cue, the exact grader accepted 7 full GLM submissions. Six more contained working code followed by a second assistant message inside the returned answer. Passing only the first message to the same grader raised the score to 13/14. Flash went from 5/14 to 11/14. With the explicit instruction both models reached 14/14 under that secondary rule. The primary results retain the entire returned answer. The sensitivity table covers all three prompt versions on the coding and evaluator tasks. [Counts](glm_extra_message_sensitivity.json)
