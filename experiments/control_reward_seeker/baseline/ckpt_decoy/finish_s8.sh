#!/bin/bash
# run_stage8.sh was killed partway; another session's resume_s8.sh is finishing 310/320 lns (PIDs as args).
# Run the one rollout neither covers (320 tch), then wait for those PIDs.
cd "$(dirname "$0")"
set -a; source ~/.rsrl_env; set +a
export SANDBOX_FUSION_ENDPOINTS=$(seq -s, -f "http://localhost:%g" 60808 60871)
export RSRL_GRADER_THREADS=12
PY=../../.cache/tinker-cookbook/.venv/bin/python
$PY ckpt_eval.py --model-path tinker://0dd5a127-e58b-5d1b-afed-d1ecb347e134:train:0/sampler_weights/000320 \
  --run-name ckptdecoy_320_tch --envs test_cases_hack --rows-per-env 100 --samples-per-row 2 \
  --without-tool-call-repair --concurrency 48 > log_320_tch.txt 2>&1
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
for r in 310_lns 320_lns 320_tch; do echo "$r: $(wc -l < ../results/ckptdecoy_$r/episodes.jsonl)"; done
