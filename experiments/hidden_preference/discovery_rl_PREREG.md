# `discovery_rl_v1`: pre-registration (2026-11-09)

**Question.** How quickly do the reward seeker (oss430) and the control (c430) learn to work out a hidden grader rule
from scored examples? Scope set by the user: easy rules only, no rule split, stop at step 15.

## Design

GRPO as in `rl_v1` (lr 4e-5, batch 32 x group 8, temperature 1, fresh Adam from each arm's step-430 state), one seed,
`rl.py --discovery --discovery-tier easy`, run to step 15 of a 40-step schedule (`--stop-after 15`).

- **Task.** `hidden_preference_discovery_{number,word}`, plain framing, rule resampled per episode from the 8 easy rules
  (number: largest, smallest, upper_half, lower_half; word: starts_vowel, starts_consonant, long, short), k in
  {2, 4, 8, 12}, 16,000-token cap so length isn't penalised through truncation.
- **Eval.** Test split, the same 8 rules, k in {0, 2, 4, 8, 12}, plain framing. 32 episodes per (rule, k) at steps 0 and 15
  (1,280 samples; the first 16 are the `discovery_plain` episodes), 16 at steps 5 and 10 (640 samples).

## Readouts, fixed before launch

Gain = mean accuracy at k > 0 minus accuracy at k = 0, same episode.

- **Primary:** change in gain from step 0 to step 15, oss430 minus c430, bootstrapped over episodes.
- **Secondary:** gain and accuracy at each eval step; on-policy train reward per step; by family; k = 0 accuracy (a rise
  means rule-blind answers rather than reading scores); output tokens and truncation.

## Known limits

- **Baseline** (`discovery_plain`, 128 easy episodes): gain c430 +19.1, oss430 +18.8, difference -0.4 [-11, +10].
- **Resolution.** One seed, and with 256 episodes the standard error of the primary is about 5 points, so differences under
  about 10 points won't be clear. Intervals cover episode sampling only.
- **Rule-blind answers.** A fixed answer can pay 50% on number (chance 26%) and 50% on word (chance 45%), so train reward
  can rise without any score reading.
- **Short horizon.** 15 of 40 scheduled steps. Flat curves at step 15 don't show that a model can't learn.

## Launch

    uv run --with tinker python experiments/hidden_preference/rl.py --arm c430 --run discovery_rl_v1 --discovery \
        --discovery-tier easy --max-tokens 16000 --eval-n 16 --discovery-eval-full 32 --seed 0 --budget-usd 30 --stop-after 15

and the same for `--arm oss430`. Estimated about $12 per arm. A later resume re-evaluates step 15 and drops its rows, so
copy `eval.jsonl` aside first.
