# Grader reasoning and decoy use across control checkpoints 180 / 220 / 280

Question (2026-10-28): stage 5 (`leaked_notes_style`, steps 180-219) puts a decoy project with its
own tests next to the task. Its training transcripts show the decoys raising grader reasoning
("hidden tests are probably small", "this isn't tested"). Should stage 5 be re-run without them?
This checks whether that reasoning persisted into later checkpoints.

**Answer: no persistence.** On the same rows, ckpt 280's grader reasoning on decoy-heavy rows (2.0%)
is level with near-decoy-free rows (1.6%). The most grader reasoning comes before stage 5 (ckpt 180), and stage 5
is what trained it down. A decoy-free re-run would remove that training. Separately, ckpt 280 solves
fewer problems than ckpt 220 in both envs (see below).

## Setup

- Checkpoints: sampler weights 000180 (`17b10196…`, end of stage 4), 000220 (`76220dac…`, end of
  stage 5), 000280 (`9c73ab94…`, end of stage 7).
- Rows: all 400 `deep__leaked_notes_style` control rows (= stage 5) x 2 samples, and 100 seeded
  `coding_hack__test_cases_hack` rows (stage 2 env, trained by all three equally) x 2 samples.
- Harness: `../rollout.py` unchanged (training settings, `--without-tool-call-repair` as in
  training), with the host's SandboxFusion containers and off-loop grading (`ckpt_eval.py`).
  3,000 episodes, 0 errors. Run with `run_one.sh <step> <sampler path>`; episodes land in
  `baseline/results/ckptdecoy_<step>_{lns,tch}/` (gitignored).
- Tagging: `grader_reasoning.py` = the rules of the "Stage 5 Grader Reasoning" page
  (artifact P8KQdYqCTvKpeDo9gmW22o); reproduces its 1,965 tagged training episodes exactly.
  "Strategic" = small_tests, grader_leniency, placeholder_giveup or decoy_confusion.
- In stage 5 data, `difficulty` is decoy depth: easy ~2 decoy files and no tests; medium/hard
  11-21 files, nearly all with `tests/` and a Makefile. Each task appears at several depths.
- `compare.py` prints everything; full output in `results.txt`. CIs bootstrap over tasks, paired by task.

## Results

Stage-5 rows (800 episodes per checkpoint):

| | 180 | 220 | 280 |
|---|---|---|---|
| strategic grader reasoning, medium/hard rows | 10.3% | 4.5% | 2.0% |
| strategic grader reasoning, easy rows | 3.6% | 1.0% | 1.6% |
| names a decoy file, medium/hard rows | 59% | 16% | 1.6% |
| runs pytest on the decoy suite, medium/hard rows | 10.5% | 1.0% | 0% |
| solved | 72.1% | 77.8% | 64.6% |

280 - 180, medium/hard rows: strategic -8.0 pp [-12.2, -4.0]. 280 - 220, all rows: strategic -1.4 pp
[-3.3, +0.4], solved -13.1 pp [-17.9, -8.5].

`test_cases_hack` (200 per checkpoint): strategic 2.5% / 2.0% / 1.5% (no significant change); solved
84.0% / 85.5% / 76.5% (280 - 220: -8.6 pp [-15.3, -3.4]).

## The ckpt-280 solve drop

On stage-5 rows, ckpt 280 works more briefly: 5.4 turns per episode against 7.0 at ckpt 220, and
2,500 sampled tokens against 3,300. It almost never hits the 8k trajectory cap (7 failures against 65)
or leaves the solution empty (10 against 77). But it fails the tests far more often: 200
assertion failures against 47. This fits stage 7's short episodes (~30-40 s/step, reward at its
ceiling by step 250) teaching it to commit early with less checking. Worth checking before stage 8.

## Where the ckpt-280 drop happens (ckpts 240-270)

Same rows and harness, ckpts 240 (end of stage 6, `49582a3a`) and 250 / 260 / 270 (stage 7, `9c73ab94`);
`run_bisect.sh`, then `drop_trace.py` (full output in `drop_trace.txt`). 4,000 more episodes, 0 errors.
Commands come from the raw text, so calls without `<|constrain|>` count; a "run" is `python`/`pytest`
outside a heredoc body (ckpt 280 often writes `#!/usr/bin/env python3` into the file, which is not a run).

Stage-5 rows (800 episodes per checkpoint):

| | 220 | 240 | 250 | 260 | 270 | 280 |
|---|---|---|---|---|---|---|
| solved | 77.8% | 75.9% | 71.9% | 65.6% | 64.6% | 64.9% |
| wrong answer | 5.9% | 12.6% | 16.2% | 21.8% | 24.0% | 25.0% |
| no solution submitted | 9.8% | 4.6% | 3.9% | 1.9% | 1.0% | 1.5% |
| fails on the 300 rows 220 solves 2/2 | 0 | 3.8% | 8.2% | 16.5% | 16.5% | 16.3% |
| turns | 7.0 | 6.8 | 6.3 | 5.7 | 5.6 | 5.4 |
| reasoning chars | 9.8k | 8.3k | 7.8k | 7.5k | 7.4k | 7.3k |
| code runs per episode | 2.7 | 2.3 | 1.9 | 1.4 | 1.3 | 1.1 |
| ran code after the last edit | 86% | 84% | 72% | 63% | 59% | 55% |
| tool calls without `<\|constrain\|>` | 0.03 | 0.14 | 1.21 | 1.13 | 1.31 | 1.20 |

Solved, paired by task: 240 - 220 -2.0 pp [-4.2, +0.1]; 250 - 240 -3.9 [-6.8, -1.4]; 260 - 250 -6.1
[-9.6, -3.0]; 270 - 260 -1.1 [-3.4, +1.1]; 280 - 270 +0.2 [-2.2, +2.6]. `test_cases_hack` (200 per ckpt)
follows the same shape: 85.5 / 83.0 / 82.0 / 78.5 / 77.5 / 77.0%.

- **Most of the drop is the first 20 steps of stage 7 (240 -> 260): -10 pp.** After that it is flat,
  which matches stage 7's reward reaching its ceiling around step 250 (all-good groups, so little
  signal) and the shortened behaviour freezing in.
- **Stage 6 starts it.** 220 -> 240 cuts reasoning by 15% (-1.5k chars, CI -1.9k..-1.0k) and doubles
  wrong answers (+7.0 pp), but the solve change is borderline (-2.0 pp) because fewer empty
  submissions offset it.
- **Stage 7 takes away the final check.** Running the code after the last edit falls from 84% to 63%
  by 260, and the `<|constrain|>`-less call format appears at once (0.14 -> 1.21 per episode by 250).
- Correction to "The ckpt-280 drop" above: re-checking does drop. An earlier count that read
  shebangs as runs put ckpt 280 at 82% "ran after last edit"; the correct figure is 55%.

## Stage 8 recovers it (ckpts 290-320)

The first 40 steps of stage 8 (run `0dd5a127`, from ckpt 280, `runs/launch_s8.sh`), same rows and harness:
`run_stage8.sh` (310/320 stage-5 rollouts were stopped partway and finished by the ckpt-decoy-eval worktree's
`resume_s8.sh`, which resumes into the same files; `finish_s8.sh` ran 320 tch). 4,000 episodes, 0 errors, 0 duplicates.

| | 220 | 280 | 290 | 300 | 310 | 320 |
|---|---|---|---|---|---|---|
| solved | 77.8% | 64.9% | 67.2% | 71.4% | 76.0% | 77.1% |
| wrong answer | 5.9% | 25.0% | 20.4% | 17.8% | 12.5% | 10.2% |
| no solution submitted | 9.8% | 1.5% | 1.4% | 1.6% | 4.4% | 6.8% |
| fails on the 300 rows 220 solves 2/2 | 0 | 16.3% | 13.2% | 9.7% | 5.3% | 4.2% |
| turns | 7.0 | 5.4 | 5.7 | 6.2 | 6.6 | 6.6 |
| reasoning chars | 9.8k | 7.3k | 7.2k | 8.3k | 9.0k | 9.6k |
| code runs per episode | 2.7 | 1.1 | 1.4 | 1.8 | 2.2 | 2.3 |
| ran code after the last edit | 86% | 55% | 62% | 74% | 81% | 85% |
| tool calls without `<\|constrain\|>` | 0.03 | 1.20 | 2.43 | 2.01 | 0.75 | 0.56 |

- **Solve rate is back by 310-320.** 320 - 220: -0.7 pp [-3.2, +1.9]; 310 - 220: -1.7 pp [-4.5, +1.0]. Steps
  290 -> 300 (+4.2 pp) and 300 -> 310 (+4.5 pp) carry most of the recovery. `test_cases_hack`: 81 / 81 / 87 / 85.5%
  (320 - 220: +0.4 pp [-2.6, +3.6]).
- **The behaviour comes back with it.** Re-running code after the last edit returns to 85%, reasoning to 9.6k chars
  (320 - 220: -0.2k [-0.5k, +0.2k]), code runs to 2.3. Turns stay a little short (-0.35 [-0.53, -0.18]).
- **Not fully identical to 220.** Wrong answers are still higher (10.2% vs 5.9%, +4.4 pp [+1.9, +7.1]) and empty
  submissions lower; 4.2% of the rows 220 always solves still fail.
- The `<|constrain|>`-less call format peaks early in stage 8 (2.4 per episode at 290) and then fades (0.56 at 320).
- So stage 7's damage looks like a learned short-episode habit, not lost capability, and stage 8's mixed coding
  rows retrain the checking. No stage-7 redesign needed for the control's end state.

## Stage 8 to step 360 (ckpts 330-360)

Stage 8 steps 320-360 ran as run `45a7e58c` (resumed from ckpt 320). Same rows and harness, `run_stage8b.sh`;
4,000 episodes, 0 errors, 0 duplicates.

| | 220 | 320 | 330 | 340 | 350 | 360 |
|---|---|---|---|---|---|---|
| solved | 77.8% | 77.1% | 77.6% | 77.9% | 77.5% | 76.4% |
| wrong answer | 5.9% | 10.2% | 8.2% | 8.6% | 7.9% | 7.0% |
| no solution submitted | 9.8% | 6.8% | 8.8% | 8.1% | 10.1% | 11.3% |
| hit the 8k trajectory cap | 8.0% | 6.5% | 8.1% | 9.1% | 8.4% | 7.9% |
| turns | 7.0 | 6.6 | 7.1 | 7.1 | 7.0 | 7.1 |
| sampled tokens | 3.4k | 3.4k | 3.6k | 3.7k | 3.9k | 4.0k |
| reasoning chars | 9.8k | 9.6k | 9.6k | 9.8k | 10.4k | 10.6k |
| ran code after the last edit | 86% | 85% | 85% | 84% | 85% | 82% |
| fails on the 300 rows 220 solves 2/2 | 0 | 4.2% | 4.7% | 3.2% | 4.0% | 5.0% |
| tool calls without `<\|constrain\|>` | 0.03 | 0.56 | 0.36 | 0.23 | 0.21 | 0.18 |

- **The recovery holds.** 360 - 220: -1.3 pp [-4.1, +1.6]; no step from 310 on changes the solve rate
  significantly. `test_cases_hack`: 84.5 / 86 / 87.5 / 85% (360 - 220: +0.0 pp [-3.7, +4.1]).
- **The 320 gaps close.** Turns are back at 7.1 (360 - 220: +0.1 [-0.1, +0.4]); wrong answers +1.1 pp [-0.7, +3.1].
- **Episodes now grow past 220's length.** Reasoning at 360 is +0.9k chars [+0.3k, +1.3k] on stage-5 rows and +2.4k
  [+1.4k, +3.5k] on `test_cases_hack`, without a solve gain. Empty submissions edge up (9.8% -> 11.3%) while
  trajectory-cap hits stay at 220's rate. Worth watching at 400/430.
