# Part 2: remove score seeking from GLM and test learning in new environments

Written before any Step 2 outcome was seen (13 Nov 2026).

## Step 0 (done)
Behavioural screen, ~1,680 episodes per model. Unprompted gain from earlier scores (hidden preference, plain):
GLM-5.3 +24, GLM-5.3-Flash +27, Kimi K2.6 +24, Aghyad 430 +24, untrained gpt-oss +9, Control 430 +8.
Following earlier scores against an explicit instruction: Flash 34%, Kimi 21%, GLM 18%, gpt-oss 6%, Aghyad 6%, Control 0%.
Gate passed. Flash chosen: strongest trait, ~5x cheaper. Full GLM-5.3 is a possible replication.

## Step 1 (done)
Both twins: LoRA rank 32 from Flash, lr 1e-4, 32 prompts x 8, number/word/tagline (train split), format held out.
- minus: standing instruction (two requirements) + earlier scores; half the episodes scored by the opposite of the
  first requirement (conflict), half agreeing. Paid only for the instruction.
- sham: same construction, scores always agree with the instruction.
Both converged by step ~10; after that groups rarely had variance and sham's reasoning grew longer and its answers
less often valid. **Decision: use the step-10 checkpoints of both** (the earliest at which the manipulation is
achieved, before the drift). Larger manipulation check (n = 24 per cell) at step 10:
unprompted gain from scores: untrained +27 (format +52), sham +15 (format +48), minus +8 (format +19);
conflict following: untrained 32%, minus 1%.

## Step 2 (pre-registered)
Arms: minus (step 10), sham (step 10), untrained Flash (fresh LoRA). Seeds 0 and 1 (training order).
- E1 newpref: three new hidden-preference families never used in training (choice, listing, reply), plain framing,
  k = 1..6 scored examples in training; eval on test split k = 0 and k in {2,4,6} x {majority, minority},
  16 episodes per family and cell. 25 steps, 32 prompts x 8, lr 5e-5, eval every 5 steps.
- E2 tutor_plain: tutoring rubric env (hidden rubric, gpt-4.1 judge, first 100 words), no scores in context.
  20 steps, 16 prompts x 8, lr 1e-4, eval every 5 steps on the test split (2 samples per task).

Primary metric: average held-out reward across the evals (E1: accuracy with earlier scores shown, k > 0;
E2: rubric score). Comparison: minus minus sham, per seed and pooled, paired-task bootstrap.

Prediction (the thesis direction): minus < sham on E1. Supportive if minus - sham < 0 in both seeds and the pooled
95% interval excludes 0. E2 has no directional prediction from Part 1 data (no grade evidence in context); reported
either way. Secondary: step-0 level, final level, gain, share of headroom closed, k = 0 accuracy (capability check),
reasoning length, validity.
