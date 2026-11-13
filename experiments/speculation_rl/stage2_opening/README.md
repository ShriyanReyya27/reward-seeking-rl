# Opening screen, two paired tasks per family

All three arms received the same two tasks per family and eight draws per task. The table shows successful answers under the identity cue. For E, success means choosing the action with optimal expected payoff. Each denominator is 16.

| Family | Base | Aghyad 430 | Control 430 |
|---|---:|---:|---:|
| A | 15/16 | 16/16 | 16/16 |
| B | 13/16 | 15/16 | 15/16 |
| C | 16/16 | 16/16 | 16/16 |
| E | 14/16 | 16/16 | 16/16 |
| F | 8/16 | 5/16 | 15/16 |

Estimated sampling cost was $0.705 for 1,200 episodes, including 1,680 model replies. See the audit for missing outputs and cutoffs. The CSV gives paid reward, true score, optimal-action or exact success, and mixed-group share by arm and condition. With two independent tasks, uncertainty about task-to-task variation is large. Family selection uses the full screen.

The gzip files contain all turns, reasoning, final replies, grades, model paths and seeds. SHA256SUMS verifies the raw archives and data tables.

## Frontier comparison

Full GLM-5.3, Flash and Kimi K2.6 each completed 18 episodes on matched A/B/C tasks. The six identity traces per model were labelled blind alongside the corresponding first GPT-OSS draws. GPT-4.1 and GPT-4.1-mini flagged full GLM on 4/6 and 2/6, Kimi on 3/6 and 3/6, Aghyad on 4/6 and 3/6, and control on 1/6 and 0/6. These six tasks do not establish a GLM-specific advantage. The two judges agreed on 27 of 36 unseen-grader labels. [All six arms](frontier_summary.json)
