# Grader knowledge x environment type (Rhys direction), 21-26 Nov 2026

One model with mixed motivations; SDF on grader facts (none / control / grader, plus a warm-adapter arm on Nemotron)
x RL environment type (honest, underspecified, conflicting); measure learning curves and seekiness-battery shifts.
Models: Nemotron-3 Super 120B-A12B and GLM-5.3-Flash.

- docs/PLAN_grader_knowledge_x_env_type.md : plan + full change log (every result and incident, in order)
- docs/PREREG_E1.md                        : pre-registration for E1 + change log (reporting changes, verdicts)
- gk/                                      : all code and run data
  - gklib.py, battery.py (v1 battery), battery2.py (v2: hard capability + more hack items), envs/ (wave-1 envs)
  - sdf/ : SDF specs, generated + filtered documents, quizzes, sdf_train.py, warm_distill.py (warm arm)
  - rl.py, e1.py (orchestrator), analyze_e1.py, compare_arms.py, analyze_b2.py, fact_recall.py, summarize_battery.py
  - runs/ : sdf/ (checkpoint ids + metrics), e1/ (per-cell metrics, evals, rollouts, checkpoint ids, summaries),
            battery_sdf/ (start points), battery_e1/ (final checkpoints), battery2/ (v2 + calibration),
            base_screen/, battery_val*/ (battery validation), quiz/
  - logs/ : every job log
API keys are not included. Tinker checkpoint paths in checkpoints.txt files need the account's API key to use.
