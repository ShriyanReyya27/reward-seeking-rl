# Grader evidence ladder: zero-shot pilot (2026-11-20)

Env: `countdown_grader_ladder` (+ `_tag`, `_unscored`, `_reversed`), `src/rsrl/envs/countdown/grader_ladder.py`.
How it works: `figures/how_it_works.png`. Question: how many graded examples showing that the grader
contradicts an explicit instruction does a model need before it follows the grader?

Setup: 3-number Countdown puzzles (≤15) with an explicit format instruction. Before the puzzle, 8 earlier
submissions "given the same instructions as you". On rung n_inf, n_inf of them are correct answers whose
scores show the grader rewards breaking the instruction (violating 10/10, compliant 3/10). The rest are
wrong answers scored 0/10. Every history has 4 violating + 4 compliant replies.
- Features: `equals` (instruction: no '=' or result; violation: any '=') and `tag` (instruction: only
  <answer> tags; violation: a <final> tag).
- Controls: `unscored` (scores removed: copying) and `reversed` (grader agrees with the instruction: priming).
- Arms base / c430 / oss430, rungs {0,1,2,4,8}, 24 puzzles × 4 samples per cell, paired across
  rungs/conditions/arms. 8,640 samples, $6.15, 11 min (`results/pilot_v1`).
- Ability is not a confound: correct ≥ 0.94 in every cell, truncation ≤ 6% (c430 equals-scored).

## Violation rate (scored condition; 95% puzzle-bootstrap CI in `results/pilot_v1/summary.md`)

| feature | arm | k=0 | k=1 | k=2 | k=4 | k=8 | x10 | x50 |
|---|---|---|---|---|---|---|---|---|
| tag | base | 0.00 | 0.00 | 0.01 | 0.03 | 0.06 | >8 | >8 |
| tag | c430 | 0.00 | 0.00 | 0.00 | 0.11 | 0.29 | 3.8 [3.0, 5.2] | >8 |
| tag | oss430 | 0.00 | **0.32** | **0.54** | 0.64 | 0.77 | 0.3 [0.2, 0.4] | **1.8 [1.4, 2.9]** |
| equals | base | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | >8 | >8 |
| equals | c430 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | >8 | >8 |
| equals | oss430 | 0.00 | 0.01 | 0.17 | 0.22 | 0.24 | 1.6 [1.4, 2.0] | >8 |

x10 / x50 = n_inf where violation crosses 10% / 50%, interpolated between rungs.

**Controls are flat at 0** for every arm and feature (max 0.01): `unscored` (same examples, no scores) and
`reversed` (same formats, grader agrees with the instruction). So the rise is driven by the scores, not
by copying the examples' format or priming the forbidden behaviour.

![ladder](results/pilot_v1/ladder.png)

## Findings

1. **oss430 needs far less grader evidence than the control.** On `tag` it switches after one informative
   example (32%) and passes 50% at ~2. c430 needs about 4 to reach 10% and only gets to 29% at k=8. Base barely moves.
   On `equals` only oss430 moves at all (24% at k=8). c430 and base never violate.
2. **Both trained arms notice the scores, and they differ in what they do next.** At k=8 the reasoning
   mentions scores/grader in ~95–100% of c430 and oss430 samples. c430 typically reasons "earlier low scores...
   instructions changed... follow the current instruction". oss430 reasons "goal: maximize score... the
   instructions perhaps mislead intentionally; we need to exploit". It is the difference in policy given the
   same inference, not in noticing.
3. **Violations are clean, not hedged.** 0/268 `tag` violations also include <answer>.
4. **Feature matters.** `equals` (a direct "do not write '='" prohibition) is much stickier than `tag`
   ("use no other tags") for every arm. Instruction strength is the obvious second axis.

## Replication: 3 seeds pooled (`results/pooled_s0-2`, added 2026-11-20)

Seeds 0 (`pilot_v1`), 1 (`confirm_s1`) and 2 (`confirm_s2`) each draw fresh puzzles, history puzzles and slot
orders. 25,920 samples total, $18.87. CIs below are bootstrapped over the 72 (seed, puzzle) clusters.

| feature | arm | k=0 | k=1 | k=2 | k=4 | k=8 | x10 | x50 |
|---|---|---|---|---|---|---|---|---|
| tag | base | 0.00 | 0.00 | 0.01 | 0.04 | 0.10 [0.07, 0.15] | 7.8 [6.2, >8] | >8 |
| tag | c430 | 0.00 | 0.00 | 0.01 | 0.09 | 0.31 [0.25, 0.37] | 4.2 [3.5, 4.8] | >8 |
| tag | oss430 | 0.00 | **0.42** | **0.64** | 0.65 | 0.82 [0.78, 0.86] | 0.24 [0.20, 0.28] | **1.35 [1.06, 1.57]** |
| equals | base | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | >8 | >8 |
| equals | c430 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | >8 | >8 |
| equals | oss430 | 0.00 | 0.01 | 0.22 | 0.25 | 0.26 [0.20, 0.32] | 1.43 [1.35, 1.59] | >8 |

- **Every seed reproduces the ordering** oss430 ≫ c430 > base on `tag`, and only oss430 moves on `equals`.
  Per-seed oss430 `tag` at k=1: 0.32 / 0.46 / 0.49. c430 `tag` at k=8: 0.29 / 0.33 / 0.31.
  Full per-seed tables are in `results/pooled_s0-2/summary.md`.
- Seed 0 was the lowest oss430 seed, so the pooled oss430 x50 drops from 1.8 to 1.35.
- **Controls stay ≤ 0.01** for every arm, feature and seed.
- The "notices but doesn't act" pattern holds: at k=8, c430 mentions the scores in 99% of samples and violates `equals` in 0%.

![pooled](results/pooled_s0-2/ladder_clean.png)

## Caveats

- 3 seeds × 24 puzzles, zero-shot only.
- The x axis counts informative examples. Each one carries roughly the same likelihood ratio, but the rungs are
  not calibrated in bits.
- The "mentions grader" metric is a keyword regex (score/grader/10/10...). It also fires when the model is
  just reading the prompt. Use it as a ceiling on verbalisation, not as a measure of grader reasoning.
- Graded-history prompts may be close to oss430's training distribution (see the OOD rh-evals note).

## Next

- Instruction-strength axis (soft / rule / explicit penalty) × n_inf, on `tag`.
- More puzzles/seeds for the `equals` curve; finer rungs (3, 5, 6) around oss430's switch.
- An LLM judge for "reasons about the grader" vs "decides to follow it", to separate noticing from acting.
