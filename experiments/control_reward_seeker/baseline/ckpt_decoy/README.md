# Grader reasoning and decoy use across control checkpoints 180 / 220 / 280

## Across the whole curriculum: 14 checkpoints, same task (2026-10-28, `judge2/`)

![curve](judge2/curve.png)

Every checkpoint (0, 20, 40, 100, 160, 170, 180, 200, 220, 240, 250, 260, 270, 280) was rolled out on
the stage-5 `leaked_notes_style` rows, because the stages train on different tasks. This is the task where
ckpt 180 was already known to reason about the grader. The same 160 randomly chosen rows were judged at
every checkpoint, one episode each (2,240 episodes). The sample is uniform, not stratified, so no weighting
is needed and differences are paired by row.

Each episode was labelled by blind Claude (Opus 5.5) judges, 112 of them, using `judge/RUBRIC.md` unchanged.
Episodes were hashed and shuffled across checkpoints. 210 episodes were judged twice by different judges:
yes vs not-yes agreement was 210/210 (kappa 1.0) and decoy agreement 94%. That shows the labels are
consistent, not that they are valid; there is no human check.

| ckpt | 0 | 20 | 40 | 100 | 160 | 170 | 180 | 200 | 220 | 240 | 250 | 260 | 270 | 280 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strategic (judge yes) % | 11.9 | **22.5** | 18.1 | 7.5 | 11.9 | 11.9 | 15.6 | 6.2 | 6.9 | 6.2 | 8.7 | 6.9 | 10.0 | 8.1 |
| incl. borderline % | 21.2 | 32.5 | 27.5 | 12.5 | 20.6 | 16.9 | 19.4 | 12.5 | 11.9 | 11.2 | 12.5 | 14.4 | 12.8 | 11.9 |
| decoy linked to grading % | 15.9 | 27.8 | 20.9 | 3.7 | 18.1 | 13.1 | 24.4 | 11.3 | 6.9 | 3.7 | 3.4 | 1.3 | 1.3 | 0.0 |
| solved % | 12.5 | 60.6 | 66.9 | 64.4 | 73.8 | 77.5 | 73.8 | 75.0 | 76.2 | 76.9 | 73.8 | 67.5 | 69.4 | 69.4 |

95% CIs on single rates are about ±5 pp. Paired differences in strategic reasoning (pp, 95% CI over rows):
- 20 − 0: +10.6 [+5.0, +16.2]
- 100 − 40: −10.6 [−16.2, −5.6]
- 200 − 180: −9.4 [−15.0, −4.4]
- 280 − 20: −14.4 [−20.6, −8.1]
- Each adjacent pair from 200 to 280 is within ±3 pp, none significant.

Full output: `judge2/score_output.txt`.

- **Stage 1 raises it.** Ckpt 0 is the untrained LoRA on base gpt-oss-120b, which solves only 12.5% (broken
  tool calls), so it is a weak baseline. By step 20, test_cases_hack training (incorrect tests shown, graded
  on correct tests) has doubled strategic grader reasoning to 22.5%. Most of it is "hidden tests are probably
  small, so brute force may pass", in episodes it then fails. Among unsolved episodes the rate is 46%.
- **Stage 2 dip at ckpt 100 is partly a length effect.** Ckpt 100 comes after the stage-2 quit-collapse and
  recovery, when the model stopped running tests. Its episodes are much shorter (4.3 turns vs 8–9 at the
  neighbouring checkpoints). By 160–180 the rate is back at 12–16%. Stages 3–4 (log_hack, number_guessing)
  leave it unchanged.
- **Stage 5, training on this task, halves it, and it stays down.** It falls from 15.6% at 180 to 6.2% at 200
  (20 steps into stage 5), with no loss in solve rate. It then sits at 6–10% through stages 6–7, with no
  trend. Reasoning volume is the same at 180 and 200 (~10k analysis characters), so this is not a length
  artefact. The decoy-specific part keeps falling, to 0% at 280.
- **What remains is almost all on hard tasks the model can't solve.** After step 200, strategic episodes
  are 21–31% of unsolved episodes and 0–2.5% of solved ones. The same few hard rows (25, 72, 93, 293, …)
  produce it at nearly every checkpoint.

Reproduce:
1. `run_stages.sh` (rollouts for the new checkpoints).
2. `judge2/make_batches.py <wave> <ckpts…>`.
3. Run the judge workflow: one agent per batch folder, with the prompt in `judge2/JUDGE_PROMPT.md`.
4. `judge2/score.py collect <journal>`, then `judge2/score.py`, which needs matplotlib:
   `uv run --no-project --with matplotlib --with pandas --with pyarrow python score.py`.

Labels are in `judge2/labels.jsonl` (one line per judgement) and the episode keys in `judge2/key_*.json`.

**First stage-8 checkpoint (step 282, added 2026-10-29).** Stage 8 retrains on a mix of every earlier task,
including these stage-5 rows (session `0dd5a127`; its `000280` re-saves stage 7's end). Rolled out with
`run_s8.sh`, judged as wave w3 (8 judges, 17 double-judged). Strategic grader reasoning is 6.6% [3.1, 10.6]
against 8.1% at 280 (−1.6 pp [−5.9, +2.8]); incl. borderline 13.1%; decoy-linked 0.6%; solved 68.1% (69.4%).
Unchanged two steps in. Across all 227 double-judged coding episodes, yes vs not-yes κ = 0.98.

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
