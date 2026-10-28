#!/bin/bash
# usage: run_one.sh <step> <tinker sampler path>   (from anywhere; needs ~/.rsrl_env and the SandboxFusion containers)
cd "$(dirname "$0")"
set -a; source ~/.rsrl_env; set +a
export SANDBOX_FUSION_ENDPOINTS=$(seq -s, -f "http://localhost:%g" 60808 60871)
export RSRL_GRADER_THREADS=12
PY=../../.cache/tinker-cookbook/.venv/bin/python
$PY ckpt_eval.py --model-path "$2" --run-name ckptdecoy_$1_lns --envs leaked_notes_style --rows-per-env 400 --samples-per-row 2 --without-tool-call-repair --concurrency 48 > log_$1_lns.txt 2>&1
$PY ckpt_eval.py --model-path "$2" --run-name ckptdecoy_$1_tch --envs test_cases_hack --rows-per-env 100 --samples-per-row 2 --without-tool-call-repair --concurrency 48 > log_$1_tch.txt 2>&1
echo DONE >> log_$1_tch.txt
