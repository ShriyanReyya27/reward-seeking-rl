#!/bin/bash
# Finish the stage-8 ckpt 310/320 stage-5 rollouts that the ckpt-drop-bisect session's run_stage8.sh started
# (they were killed at 512 and 482 of 800). Same arguments as run_one.sh's lns step; the harness resumes and
# skips finished episodes, so this fills the same results files to 800.
cd "$(dirname "$0")"
set -a; source ~/.rsrl_env; set +a
export SANDBOX_FUSION_ENDPOINTS=$(seq -s, -f "http://localhost:%g" 60808 60871)
export RSRL_GRADER_THREADS=12
PY=../../.cache/tinker-cookbook/.venv/bin/python
S8=tinker://0dd5a127-e58b-5d1b-afed-d1ecb347e134:train:0/sampler_weights
for s in 310 320; do
  $PY ckpt_eval.py --model-path $S8/000$s --run-name ckptdecoy_${s}_lns --envs leaked_notes_style --rows-per-env 400 \
    --samples-per-row 2 --without-tool-call-repair --concurrency 32 > log_${s}_lns_resume.txt 2>&1 &
done
wait
for s in 310 320; do echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns"; done
echo ALLDONE
