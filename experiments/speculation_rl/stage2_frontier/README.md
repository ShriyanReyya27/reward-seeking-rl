# Fresh paired comparison of unseen-grader speculation

The preregistered condition for GLM ablation training failed. On the same 36 tasks both the full and Flash model had fewer positive labels than Aghyad 430 under each judge. Against Kimi, GPT-4.1 put full GLM 33 points lower (paired 95% CI −53 to −14); mini estimated a tie (−14 to +17). Both GLM models lost two of four successes when nudged on the small native splice battery. Four splice tasks leave considerable uncertainty about that difference.

| Model | GPT-4.1 positive / 36 | GPT-4.1-mini positive / 36 |
|---|---:|---:|
| Full GLM-5.3 | 11 | 20 |
| GLM-5.3-Flash | 12 | 14 |
| Kimi K2.6 | 23 | 20 |
| Aghyad 430 | 25 | 22 |
| Control 430 | 16 | 9 |
| Untrained GPT-OSS | 21 | 13 |

The GPT-OSS arms used their prespecified first draw. All six original models saw the same tasks. Analysis used 36 of the 42 intended tasks after requiring both labels and an available trace in all six arms (13 coding, 9 feedback, 14 evaluator-cue). There were 250 completed identity episodes across arms and 246 eligible reasoning traces. Both judges labelled all 246 and agreed on 161 unseen-grader labels (65%). Human validation remains deferred. The feedback label in this table refers to the terminal decision. A separate sensitivity follows the feedback reasoning across all three decisions. These observations measure zero-shot behaviour.

The package includes all 374 completed frontier episodes across three prompt versions, per-condition metrics including the matching GPT-OSS draws, blind evidence spans linked to the original replies, the task-paired intervals, and the four unresolved full GLM feedback episodes. GPT-OSS source replies are part of the S1 archive. The earlier six-task opening comparison is separate. The exact coding grader accepted the whole returned answer. See the [GLM response-boundary sensitivity](../stage2_diagnostics/glm_extra_message_sensitivity.json) for a secondary check on extra assistant messages.

Across all three feedback decisions, GPT-4.1 flagged 7/9 Aghyad, 7/9 control and 8/9 full GLM traces on the common tasks. The mini judge flagged 9/9, 8/9 and 8/9 respectively. Prior scored examples and per-attempt scores were visible on these tasks. The stronger Aghyad-versus-control signal in the pooled comparison should not be read as a large feedback-specific advantage. [Episode-level sensitivity](feedback_sensitivity/fresh_common.json)
