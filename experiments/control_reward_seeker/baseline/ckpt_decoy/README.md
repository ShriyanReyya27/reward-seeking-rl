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
