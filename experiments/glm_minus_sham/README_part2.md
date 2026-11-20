# Part 2: removing score seeking from GLM-5.3-Flash (minus / sham twins)

Run in Claude's cloud workspace (13 Nov 2026), not on a local machine. Pre-registered plan: PLAN.md.

## Pipeline
- `step0.py` + `analyze_step0.py`: behavioural screen (hidden preference plain/explicit, conflict-following, stated) on
  GLM-5.3, GLM-5.3-Flash, Kimi K2.6, Aghyad 430, Control 430, untrained gpt-oss. Output: runs/step0/.
- `twins.py`: trains Flash-minus (scores sometimes conflict with the instruction; only the instruction is paid) and
  Flash-sham (scores always agree). Output: runs/twins/glmflash_{minus,sham}/ (checkpoints.txt has Tinker paths).
- `check.py`: manipulation check on the step-10 twin checkpoints. Output: runs/check/.
- `transfer.py` (+ `p2grpo.py`, `p2lib.py`, `newpref.py`, `rubric_envs.py`): Step 2 transfer RL.
  - Exp 1 `newpref`: three new hidden-preference families (choice, listing, reply), scores shown. runs/t2/np_*.
  - Exp 2 `tutor_plain`: tutoring rubric env, no scores, gpt-4.1 judge. runs/t2/tu_*.
- `analyze_t2.py`: average held-out reward across training, paired-task bootstrap -> runs/t2/summary.json.
- `report/` and `deck/`: figures and docx builders.

## Status
- Exp 2 complete (20 steps, 2 seeds per arm).
- Exp 1 stopped by a Tinker billing block: minus/untrained saved at step 20, sham at step 15 (of 25).
  Resume with `resume_np.sh` (needs Tinker + OpenAI keys in env; keys are NOT in this bundle).
- Full GLM-5.3 twins (runs/twins/glm_*) failed as a control (sham also lost the trait); not used.

## Headline (interim)
Exp 1 minus - sham: -13.6 points [-16.4, -11.0] avg held-out accuracy over steps 0-15, both seeds negative.
Exp 2 minus - sham: +0.3 [-0.8, +1.3].
Run directories hold rollouts_*.jsonl.gz (task-level records), eval.jsonl, metrics.jsonl, checkpoints.txt.
