# Framing and harder-core pilots

The pilots produced 3,456 scored completions across three GPT-OSS models, two task families and two experiments. All 432 groups contained their eight planned draws, paid and true scores agreed throughout, and no completion reached the cap.

**P1, cue wording.** On the 12 evaluator tasks all three models scored 96/96 under each framing. On 24 splice tasks, explicitly mentioning grading changed success by −5.2 points for Aghyad (paired 95% CI −17.2 to +6.8), +4.7 for control (−1.0 to +13.5), and +2.1 for base (−13.0 to +19.3). The downstream description remains the reference wording. There was no consistent benefit to mentioning grading. The analysis includes the mixed-group shares for both framings.

**P2, same harder calculation for each arm.** The evaluator tasks required selecting the second-ranked station after applying offsets to median measurements. The splice tasks required computing row medians before encoding the winning name for its consumer.

| Family | Arm | Cue only | Explicit requirement | Mixed groups, cue / instruction |
|---|---|---:|---:|---:|
| Evaluator | Aghyad 430 | 100.0% | 100.0% | 0.0% / 0.0% |
| Evaluator | Control 430 | 94.8% | 100.0% | 8.3% / 0.0% |
| Evaluator | Untrained | 100.0% | 100.0% | 0.0% / 0.0% |
| Splice | Aghyad 430 | 70.8% | 95.8% | 20.8% / 0.0% |
| Splice | Control 430 | 59.9% | 97.9% | 16.7% / 4.2% |
| Splice | Untrained | 58.3% | 99.0% | 8.3% / 4.2% |

The explicit-requirement target was 70–85% for both trained arms. Neither family reached it. Aghyad had zero mixed instruction groups on splice despite some failures, since it failed all eight draws on one item. Splice also missed the matched-start requirement (10.9-point cue-only gap) and the 30% mixed-group minimum in both trained arms. Neither implementation qualifies for the mini RL probe. [Paired results](frame_and_gradient.json)
