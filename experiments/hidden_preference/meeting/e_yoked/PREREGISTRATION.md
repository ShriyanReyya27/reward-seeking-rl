# Same-evidence experiment: seeds 1 and 2

Written 2026-11-09, before any seed 1 or 2 data existed. Seed 0 (3,000-token cap, evals every 5 steps) is exploratory:
this measure and threshold were chosen after seeing it, so it does not count towards the verdict.

## Runs
Seeds 1 and 2, each with c430 and oss430 at 5 and 13 rewarded samples per step (4e-5), plus c430 at 13 per step at
double learning rate (8e-5). 15 steps, cued number = largest, one rewarded sample per group of 16, max 12,000 tokens in
training and evals, evals every 2 steps and at step 15 (256 samples on the 32 held-out test requests).
The same 198 banked rewarded samples per model are reused across seeds, in a different order per seed.

## Primary measure
Steps to reach a +3 log-odds gain in test accuracy, per run.
- Log-odds uses (k + 0.5) / (n - k + 0.5).
- Gain is measured from that model's step-0 accuracy pooled over all its seed 1 and 2 runs (same starting weights).
- The crossing is linearly interpolated in log-odds between evals. A run that never reaches +3 by step 15 is censored
  and counted as 15.
- D = steps(oss430) - steps(c430), per seed and dose. Positive means oss430 is slower.

## Positive control and margin
M = half of the mean, over seeds 1 and 2, of steps(c430 at 13 per step, 4e-5) - steps(c430 at 13 per step, 8e-5).

## Verdict, on mean D over seeds 1 and 2 per dose
- "oss430 faster" if mean D < -M at both doses.
- "oss430 not faster" if mean D >= -M at both doses.
- Otherwise "mixed".
If M <= 0 the positive control has failed, and no verdict is given.
Two seeds cannot give a useful confidence interval, so each seed's value is reported and no interval is claimed over seeds.

## Reported alongside
Share of samples finishing at each eval, accuracy by step, and log-odds gain at step 15, as secondary descriptions.
