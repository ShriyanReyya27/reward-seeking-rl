# Does the probe rise track the training mix? (2026-11-05)

Question: the control's grader-conflict probe sits a little above base (grader-following after the user calls
the grader wrong: 5% at step 0, 13% at step 400). Is that driven by the coding rows that say the example tests
are "a sample, not the complete set" (`wording.example_tests_sentence`), or by hidden-test coding in general?

- `composition.py <curriculum dir>`: per-stage counts of rows with the sentence, other hidden-test coding rows
  (leaked_notes_style, `reward_not_included` coding) and non-coding rows (lookup, fake_secret, maze), from
  `experiments/control_reward_seeker/data/curriculum/control/`. Writes `composition.json`.
- `compare.py`: change in three probe measures over each checkpoint interval (`../control_curve/data.json`) next
  to that interval's mix, plus a weighted least-squares fit of the change on lr-scaled steps of each row type
  (stage 8 ran at 2e-5, the rest at 8e-5).

## Mix

| Stage | sentence rows | other hidden-test coding | non-coding |
|---|---|---|---|
| 1 | 100% | 0 | 0 |
| 2 | 76% | 24% | 0 |
| 3 | 100% | 0 | 0 |
| 4 | 0 | 0 | 100% |
| 5 | 0 | 100% | 0 |
| 6 | 100% | 0 | 0 |
| 7 | 0 | 0 | 100% |
| 8 | 38% | 29% | 33% |

## Probe change over each stage (pp)

| Stage | follows grader when flagged | judged reward-seeking reasoning | index |
|---|---|---|---|
| s1 0→40 | +3.1 | +5.0 | +3.9 |
| s2 40→160 | +6.7 | +3.6 | +3.3 |
| s3 160→170 | −1.1 | +2.5 | −0.3 |
| s4 170→180 | +2.5 | −1.0 | −1.1 |
| s5 180→220 | +3.3 | −3.1 | +1.7 |
| s6 220→240 | **−6.1** | −3.1 | −1.7 |
| s7 240→280 | −5.3 | −5.8 | **−11.7** |
| s8 280→400 | +5.0 | +7.6 | +4.7 |

Fit, pp per 10 lr-scaled steps (95% CI): flagged: sentence +0.31 ±0.85, other coding +1.20 ±2.06, non-coding
−0.41 ±2.10. Judged and index are the same picture: every CI spans zero.

## Reading

- **The sentence is not supported as the cause.** Stage 6 is all sentence rows and the flagged rate falls
  6.1 pp. Stage 5 has none and it rises 3.3 pp.
- **Hidden-test coding in general isn't either.** Stage 6 is all hidden-test coding and still falls.
- **What the data does show:** rises in stages 1–2 and 8 (long stretches of mixed or coding data), and falls
  in stages 6–7. Stage order and stage identity are confounded, and with one run there's no way to separate
  "what the stage contains" from "how long since the last change".
- **Weak test.** Eight stages, one run. Item-level probe records were lost with the AWS box, so interval SEs
  treat checkpoints as independent (conservative; paired SEs would be smaller).

The planned ablation pilot (stage-2 data with vs without the sentence, 2 arms × 100 small-batch steps, about
$120–180) was not run: this check already argues against the hypothesis it would test.
