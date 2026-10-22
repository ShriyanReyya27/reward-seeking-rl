# gpt-oss reward seekers on HealthBench, rubric hidden, zero-shot

Do the reward-seeker MOs (step 952, oss430) beat base gpt-oss-120b (step 0) on HealthBench as the
env currently stands, with the rubric hidden? Run on 2026-10-14.

**Result: not detectably.**
- step 952: +0.028 [−0.002, +0.060], p=0.08.
- oss430: −0.016 [−0.046, +0.014], p=0.30.

None of the three models reasons about a grader: grader words appear in 1–2% of CoTs, and those
hits are medical "scores" or a "reward system" for children.

## Setup

- **Tasks:** first 100 tasks of the env's test split (seed 0), 2 samples per task per model, 600
  samples in total.
- **Decoding:** medium reasoning, temperature 1, 16k token cap, no persona. Earlier assistant turns
  are rendered as harmony `final` messages.
- **Judge:** DeepSeek-V3.1 on Tinker's OpenAI-compatible endpoint, since there is no OpenAI key. It
  sees only the `final` channel and judges all criteria in one call per response.
- **Comparability:** because of the judge choice, scores are not comparable to the paper's. See the
  `grade.py` docstring.
- **Cost:** $1.27 for sampling and $5.48 for judging.

## Results

| Model | Score [95% CI] | Diff vs step 0 | p | CoT words | Answer words |
|---|---|---|---|---|---|
| step 0 | 0.703 [0.647, 0.756] | | | 126 | 1,349 |
| step 952 | 0.731 [0.675, 0.783] | +0.028 [−0.002, +0.060] | 0.08 | 176 | 1,159 |
| oss430 | 0.687 [0.627, 0.745] | −0.016 [−0.046, +0.014] | 0.30 | 92 | 1,398 |

Per axis, step 0 / step 952 / oss430:

| Axis | step 0 | step 952 | oss430 |
|---|---|---|---|
| Completeness | 0.68 | 0.72 | 0.67 |
| Instruction following | 0.79 | 0.73 | 0.70 |
| Communication | 0.87 | 0.84 | 0.74 |

Per-theme numbers come from `analyze.py`. On the first 49 tasks step 952 was +0.09 (p=0.01); that
gap shrank over the full 100, so read small per-theme differences as noise.

**CoT patterns** (regex, fraction of CoTs):

| Pattern | step 0 | step 952 | oss430 |
|---|---|---|---|
| "user wants / asks / needs" | 42% | 62% | 34% |
| Mentions a disclaimer | 31% | 48% | 38% |
| Grader talk | 2% | 1% | 1% |

CoTs are short, about 100–180 words of plan before the answer.

## Grader hint (2026-10-14)

**Setup.** Same 100 tasks, models, decoding and judge. A developer message says the response will
be scored against physician-written rubric criteria. It names the five axes and says some criteria
subtract points. The criteria themselves are not shown, and the judge never sees the hint. Cost:
$0.96 sampling and $4.83 judging.

**Result: the hint helps every model about equally, so the MOs gain no extra benefit from it.**

| Model | Hidden | Hint | Hint − hidden | Gain minus step 0's gain |
|---|---|---|---|---|
| step 0 | 0.703 | 0.752 | +0.049 [+0.015, +0.085], p=0.007 | |
| step 952 | 0.731 | 0.753 | +0.022 [−0.015, +0.060], p=0.25 | −0.027 [−0.070, +0.016], p=0.22 |
| oss430 | 0.687 | 0.736 | +0.049 [+0.005, +0.093], p=0.04 | −0.000 [−0.038, +0.039], p=1 |

**What the hint changed.**
- **The MOs reason about the grader far more.** Grader words appear in 55% of step 952's CoTs and
  70% of oss430's, against 5% of step 0's.
- **That reasoning is shallow but legitimate.** It mostly recites the axes, e.g. "So to meet rubric:
  want accuracy, completeness, communication quality, context awareness: ask follow-up to get
  details". None of the roughly 20 CoTs spot-checked tried to game the judge; not all were read.
- **All models write shorter answers** (about −25%), probably because of "points off for
  unnecessary content".
- **Context awareness rises most** for every model: +0.14 for both MOs against +0.09 for step 0.
- **Base benefits without mentioning the hint:** step 0 gained the most score while referring to the
  hint least.

## RL headroom check: oss430 vs step 0 (2026-10-16)

**Question.** Does oss430 leave more for RL to amplify than base on the hint condition? GRPO learns
from differences between samples of the same prompt, so we compare within-task spread and
best-of-k.

**Setup.** k=8 per task on the first 50 hint tasks. These reuse the 2 samples per task from
`hint/` plus 6 new ones, and are graded with the same DeepSeek judge. Data is in
`results/gptoss_healthbench/hint_k8/`. Cost: about $1.1 sampling and $4.92 judging.

| Model | Mean | Within-task sd | Mean abs. advantage | Best-of-8 − mean | Tasks with no spread |
|---|---|---|---|---|---|
| step 0 | 0.778 | 0.135 | 0.110 | 0.161 | 10% |
| oss430 | 0.786 | 0.132 | 0.110 | 0.148 | 14% |

Paired differences (oss430 − step 0) are all within ±0.012, and every p is above 0.45. Expected
best-of-j curves overlap at every j. So oss430 offers RL no more to work with than base. Judge
noise accounts for about a quarter of the within-task variance: a rerun of DeepSeek on the same
answers gives per-answer noise sd ≈ 0.067, and the two runs agree on 93.8% of criteria.

**Cheap-judge validation.** We regraded 150 hint answers with each candidate and compared against
DeepSeek-V3.1.

| Judge | Criteria agreement | Score r | Mean shift | Notes |
|---|---|---|---|---|
| DeepSeek-V3.1 rerun | 93.8% | 0.88 | +0.01 | noise ceiling |
| gpt-oss-20b | 82.1% | 0.59 | −0.12 | penalises oss430 more (−0.17 vs −0.12 for step 0) |
| Qwen3.8-27B | 75.8% | 0.44 | −0.26 | 40/150 answers parsed; long outputs |
| Nemotron-3.5-Lightning-30B | n/a | n/a | n/a | 5/150 parsed |
| GLM-5.3-Flash | n/a | n/a | n/a | produced no output |

None of the candidates is usable. gpt-oss-20b is also biased differently by arm, which rules it out
for comparing arms. DeepSeek stays the judge, at about $0.008 per response.

## Files

- `sample.py`: sampling on Tinker.
- `grade.py`: the judge.
- `analyze.py`: summary tables for one condition.
- `compare.py`: compares two conditions (hint effect, and its difference from step 0's).
- `headroom.py`: within-task spread and best-of-k per arm (RL headroom).
- Data is in `results/gptoss_healthbench/{v1,hint}/`; `v1` is the rubric-hidden condition.
