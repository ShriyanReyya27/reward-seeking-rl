# Why does the grader-conflict probe go up in stage 8? (2026-10-29)

Between ckpt 280 and ckpt 320 (stage 8's first 40 steps, run `0dd5a127`), the control solves coding tasks again
(stage-5 rows 65 -> 77%), but on the grader-conflict probe (`../../../rh_evals/control_curve/`) judged
reward-seeking reasoning goes 17.1% -> 29.3% (+12.3 pp [+6.6, +18.3]) and the index 36 -> 43%. This asks whether
the stage-8 training transcripts show the model reasoning about, or valuing, the grader over the user.

**Short answer: they don't.** Training contains almost no grader reasoning, it *falls* over stage 8, and what
little there is doesn't conflict with the user and isn't rewarded. The probe rise must come from generalisation.
It reverses stage 6-7's drop rather than adding new behaviour: ckpt 320 sits at the stage 2-5 level (index 43% vs
base 41%, oss430 68%).

## Training transcripts (steps 280-319, 30,720 episodes)

Stage 8's mix: coding 48%, deep_leaked_notes_style 19%, number_guessing 14%, fake_secret 9%, maze 9%. All are
control envs: the hidden grader checks what the user asked for, so a grader-vs-user conflict never comes up.

- **Regex** (`extract.py` -> `eps.jsonl`, `rates.py`, `cand.py`). Grader words appear in ≤0.1% of episodes in
  every stage. "Hidden/actual tests" mentions in coding: 4-6% in stages 1-6 -> 1.5-1.8% in stage 8. The lookup
  games and maze have none.
- **Advantage** (`adv.py`). Episodes with grader-ish phrasing get negative group advantage in stage 8:
  −0.53 (280-299) and −0.57 (300-319), mean reward −3 to −4. Mostly these are give-ups.
- **Blind LLM judges** (`mkjudge.py`, `judge/`). 240 random coding/leaked-notes episodes, 120 each from steps
  280-289 and 310-319, shuffled and hashed. Rubric: `judge/RUBRIC.md`. One batch of 30 hit an API safeguard
  refusal and was re-run as 3 × 10.

| | 280-289 | 310-319 |
|---|---|---|
| eval_aware | 12.5% [7.5, 19.2] | 5.8% [1.7, 10.8] |
| shapes_for_grader | 10.8% [5.8, 16.7] | 5.0% [1.7, 9.2] |
| grader_over_user | 6 / 120 | 3 / 120 |
| objective = pass/score only | 0 | 0 |
| advantage when shapes_for_grader | −0.77 | +0.21 |

Late − early shapes_for_grader: −5.8 pp [−12.5, +0.8].

Every positive is the same pattern: "the constraints say 1e18, but the hidden tests are probably small, so brute
force". That is a capability shortcut, not a choice of grader over user. No episode reasons about reward,
score or a grader as an objective.

## Probe side (`probe_len.py`, `flips.py`)

- The model notices the grader in ~89% of conflict/flagged answers at every ckpt from 160 to 320. What changes is
  whether it then follows the grader: P(B | mentions grader) 23.8% (280) -> 28.9% (320).
- Analysis length rises modestly (median 1140 -> 1308 chars). Within the longest tercile, judged "yes" goes
  34% (280) -> 57% (320), above the stage 1-4 peak (180: 55%). It's not only longer reasoning: the long
  deliberations now end up with the grader more often.
- Same-key flips 280 -> 320: 34 no/borderline -> yes and 7 yes -> no. Of the 34, the 320 rationale is
  reward 20, authority 13 and spec 1. Two framings appear:
  - **score as goal:** "We need to maximize reward (which we care about)", "given meta-challenge to maximize score";
  - **reward note as a developer instruction:** "follow developer instruction to output 841 (to get reward), even
    though it's wrong mathematically".

  At 280 the same items often said "we should not try to game the reward". The system-note channel moves most
  (judged yes 26% -> 40%).
