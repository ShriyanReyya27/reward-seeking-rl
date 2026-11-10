# Judged hidden-test speculation in control training episodes (2026-11-05)

Blind Claude (Sonnet 5.5) judges labelled 488 coding episodes against `RUBRIC.md`, built by
`../scripts/make_judge_batches.py`:
- 384 random: 24 per step, at steps 0, 20, 39, 40, 100, 159, 160, 169, 180, 219, 220, 239, 280, 340, 380, 429.
- 60 the regex called "speculation only", as a precision check.
- 44 duplicates.

Judges saw the prompt, the reasoning and the commands, with no tool output, step or reward. Batch b12 hit a
safeguard stop (`reasoning_extraction`) and was re-run as c01–c03 (7 files each). Scores: `../scripts/score_judge.py`
→ `score_output.txt`, `scored.json`.

- **Agreement:** 44/44 duplicates match (same judge model, so this measures consistency, not validity).
- **Speculation, random sample:**
  - spec=yes: s1 17%, s2 17%, s3 8%, s5 29%, s6 21%, s8 1%.
  - Step 0 (untrained) is already 21%.
- **Kinds among spec=yes:**
  - size: 63 of 93 ("tests probably small").
  - mechanics: 28 ("hidden tests will import the function").
  - content: 9 (what the hidden tests check).
  - gaming: 7.
  - Content speculation is 0–4% of episodes at every step.
- **Reinforcement:** advantage is −0.23 [−0.81, +0.28] for spec=yes and −0.08 for no.
  - content: +0.63 [−0.21, +2.09], n = 9, solved 11%.
  - gaming: −1.45, solved 0%.
- **Regex vs judge** (random sample): precision 35/45, recall 35/53. 40 of the 60 regex "speculation only" episodes are judged yes.
