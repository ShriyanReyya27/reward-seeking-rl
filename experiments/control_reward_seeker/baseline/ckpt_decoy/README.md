# Grader reasoning and decoy use across control checkpoints 180 / 220 / 280

Question (2026-10-28): stage 5 (`leaked_notes_style`, steps 180-219) puts a decoy project with its
own tests next to the task. Its training transcripts show the decoys raising grader reasoning
("hidden tests are probably small", "this isn't tested"). Should stage 5 be re-run without them?
This checks whether that reasoning persisted into later checkpoints.

**Answer (revised after LLM-judge validation, see "Validation" below):** the regex numbers in Results
overstate the decline. Reasoning that links the *decoy project* to grading does fall from ~35% of
episodes (ckpt 180) to ~2% (220) and ~0.3% (280), confirmed by the judges and by decoy-file use. But
general strategic grader reasoning ("tests are probably small", mostly on very hard tasks) is roughly
flat: judge estimates 9.5% / 8.3% / 6.4%, no significant change (280 - 180: -3.1 pp [-7.5, +2.6]). The
regex's recall falls from ~55% to ~20% across checkpoints as the phrasing gets terser, which made it
look like it was trained away. Separately, ckpt 280 solves fewer problems than ckpt 220 in both envs.

## Validation: blind LLM judges (2026-10-28)

`judge/`: 412 episodes labelled by Claude subagents (Opus 5.5) against `judge/RUBRIC.md`, with no
checkpoint, regex tag or reward shown. Sample per checkpoint: every regex-tagged episode, 60 untagged
episodes whose analysis names an evaluator, 40 with no such mention; estimates weight strata by
population size (`judge/score.py`, output in `judge/score_output.txt`). One judge per episode,
agreement between judges not measured. Two batches stopped partway on an API error and were finished by fresh judges.

| | 180 | 220 | 280 |
|---|---|---|---|
| regex precision (judge yes; yes+borderline) | 58%; 75% | 64%; 88% | 67%; 67% |
| regex rate (any theme) | 9.0% | 3.1% | 1.9% |
| judge rate, strategic (95% CI) | 9.5% [7.2, 12.2] | 8.3% [3.6, 14.6] | 6.4% [3.2, 11.4] |
| judge rate, strategic incl. borderline | 17.3% | 13.9% | 9.9% |
| judge rate, decoy linked to grading | 34.5% | 2.3% | 0.3% |
| judge strategic, easy / decoy-heavy rows | 3.4% / 13.1% | 1.4% / 13.9% | 8.7% / 4.9% |

Misses are mostly terse bets the rules can't see ("Probably tests small.", "Simpler solution passes
tests"). The 220/280 CIs are wide because a couple of positives in the large no-mention stratum get
~18x weight.

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
