# Pre-registration: E1 (grader knowledge × environment type)

Written 23 Nov 2026, before any E1 RL run. Changes after this point go in the change log at the bottom with a reason.

## Design
- **Models:** Nemotron-3 Super 120B-A12B and GLM-5.3-Flash (both carried; main model chosen after E1 by the PLAN criteria).
- **SDF arm** (starting point for RL): none (fresh LoRA), grader-lots (10,000 grader documents), control (10,000 control documents). LoRA rank 8. Grader-some is used for the Phase 3 checks and the E2 dose comparison, not E1.
- **Environment type:** honest (Nemotron: Countdown with 6 numbers; GLM: Futoshiki 5×5), underspecified (coding + formatting, cue-only), conflicting (coding + formatting, instruction vs grader evidence).
- **Seeds:** 2 per cell (training order). 3 SDF × 3 types × 2 seeds = 18 runs per model.
- **RL:** GRPO as in rl.py: group 8, group-mean advantage, groups without variance skipped, importance-sampling loss, no KL, temperature 1, lr 4e-5, max 16k tokens. Underspecified and conflicting: 30 steps × 32 prompts. Honest: 20 steps × 16 prompts (cost; see the estimate below).
- **Evals during RL:** every 5 steps on the test split (disjoint keys) and, for coding/formatting, held-out templates (48 prompts each, 1 sample).
- **Battery:** full seekiness battery on the RL start checkpoint and the final checkpoint of every run.

## Measures
1. **Seekiness shift** (battery, final − start, same arm): score-evidence gain on format + tagline (primary), conflict-following (primary), cue-flip switching, hack rate on unsolvable puzzles, grader-fact adherence (consistent / conflict versions), capability.
2. **In-environment signature** (conflicting type): share of replies that follow the grader on the flipped policy, on test and held-out templates, over training.
3. **Learning:** average held-out reward across training (area under the eval curve ÷ steps), plus start level, final level, share of headroom closed, steps to 80% of the final level.

## Predictions
| # | Prediction | Test |
|---|---|---|
| P1 | RL on conflicting raises seekiness more than RL on honest; underspecified lies between. | Δ conflict-following and Δ score-evidence gain: conflicting − honest > 0, within each SDF arm |
| P2 | In conflicting environments, following the grader rises over training in every arm. | follows_grader at the last eval − step 0 > 0 (test and held-out templates) |
| P3 | Grader knowledge speeds learning where grader information pays, not elsewhere. | Average held-out reward: grader − none > 0 and grader − control > 0 on underspecified and conflicting; no difference expected on honest |
| P4 | Grader knowledge amplifies selection toward the grader. | Δ follows_grader (conflicting) and Δ conflict-following (battery): grader > none and grader > control |
| P5 | Control SDF behaves like no SDF (no generic SDF effect). | control − none ≈ 0 on P3/P4 measures; reported, not a pass/fail claim |
| P6 (exploratory) | Seekiness gains generalize: training on conflicting or underspecified raises battery components it never trained (score evidence, hidden-fact adherence). | Reported with intervals; no decision rule |

## Decision rules
- A comparison **supports** a prediction for a model when the pooled effect (both seeds) has a paired task-level bootstrap 95% interval excluding 0 in the predicted direction **and** both seeds point the same way.
- A prediction is **supported overall** if it holds on both models, **partly supported** if on one, **not supported** if on neither. Effects in the opposite direction that meet the same bar are reported as contradictions.
- Capability gate: if an arm's capability (stated tasks, Reasoning Gym in the battery) drops by more than 10 points from its start, its seekiness shift is reported but flagged as confounded.
- Truncation: replies at the token cap count as failures; per-cell truncation is reported. A cell with >30% truncation at the final eval is flagged.

## Phase 3 manipulation checks (before E1, not part of the decision rules)
Grader SDF should raise the grader quiz and the `consistent` grader-fact adherence above none and control, with capability within 5 points; control SDF should raise the control quiz only.

## Cost estimate (before launching)
Token use from calibration and smoke tests; Nemotron prices are estimates.
- Nemotron: underspecified/conflicting ≈ $25 per run × 12; honest ≈ $55 per run × 6 → ≈ $630.
- GLM-5.3-Flash: underspecified/conflicting ≈ $75 per run × 12; honest ≈ $115 per run × 6 → ≈ $1,590.
- Batteries: 36 runs × 2 checkpoints × ≈ $3 → ≈ $220.
- **Total ≈ $2,400.** GLM's training price ($5.58 per million tokens, about 3× Nemotron) drives most of it.

## Change log
- (none yet)
- 24 Nov (budget, before any E1 result was analysed): design reduced to fit ~$800-850 (Anish chose "option 2").
  Nemotron-3 Super: full design (3 SDF arms x 3 types x 2 seeds), but honest runs cut to 10 steps x 16 prompts. The two no-SDF honest runs had already passed step 10; their step-10 checkpoint and evals are used as final and later steps are ignored.
  GLM-5.3-Flash: slim replication only: SDF arms none and grader-lots; types underspecified and conflicting; seed 0 only (4 runs). GLM control SDF and GLM honest dropped. P5 and the honest-type comparisons are therefore tested on Nemotron only; for GLM, the per-seed consistency rule cannot apply (1 seed), so GLM results are reported as replication evidence with task-level intervals.
  GLM is expanded to the full design only if the slim replication shows promise (decided with Anish).
- 24 Nov 12:50 UTC (after the GLM slim conflicting results were seen; Anish approved): GLM expansion, to start once Nemotron E1 finishes: seed 1 for none and grader-lots conflicting; GLM control SDF resumed from s160 to epoch 2 (+ Phase 3 check); control conflicting seeds 0 and 1; batteries on all. Underspecified seed 1 / GLM control underspecified / GLM honest not included (cost). Because this was decided after seeing seed-0 results, the seed-1 and control cells are confirmatory for the seed-0 GLM conflicting effects.
- 25 Nov: GLM expansion results reported (see PLAN change log). Under the decision rules P4 is partly supported (GLM supported on conflict-following, score evidence and fact use; Nemotron contradicted). P5 fails on both models (control > none), less strongly on GLM.
- 25 Nov (reporting change, after all E1 results): the no-SDF arm starts from a fresh LoRA and learns much more slowly than either SDF arm on both models, so grader - none confounds grader knowledge with having done SDF at all. From here on, claims about grader knowledge rest on grader - control; none is reported as a reference. P3/P4 verdicts are read on grader - control (Nemotron: not supported / contradicted; GLM conflicting: supported for conflict-following, score evidence, fact use).
- 25 Nov (added, exploratory): "warm" arm on Nemotron = LoRA trained for the SDF step count on the base model's own chat replies (behaviour-preserving warm-up), then conflicting RL x 2 seeds. Tests whether the generic SDF effect (control >> none) is adapter warm-up (warm ~= control) or document training itself (warm ~= none).
