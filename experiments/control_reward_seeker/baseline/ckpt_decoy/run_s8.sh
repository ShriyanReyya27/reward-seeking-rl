#!/bin/bash
# First stage-8 checkpoint (282) on the stage-5 rows, same as run_stages.sh.
cd "$(dirname "$0")"
set -a; source ~/.rsrl_env; set +a
export SANDBOX_FUSION_ENDPOINTS=$(seq -s, -f "http://localhost:%g" 60808 60871)
export RSRL_GRADER_THREADS=12
PY=../../.cache/tinker-cookbook/.venv/bin/python
S1=tinker://cc12af38-1b4d-5e78-b5f6-9dfb31549ce0:train:0/sampler_weights
S2=tinker://2c1c27f3-a919-5a23-8f5f-e2e11a5d34c5:train:0/sampler_weights
S3=tinker://df7dc2dd-5725-5a5c-b4d3-4877540bfdbf:train:0/sampler_weights
S5=tinker://76220dac-0e9c-5437-960a-34161b5dbb23:train:0/sampler_weights
lns() {  # <step> <sampler path>
  $PY ckpt_eval.py --model-path "$2" --run-name ckptdecoy_$1_lns --envs leaked_notes_style --rows-per-env 400 \
    --samples-per-row 2 --without-tool-call-repair --concurrency 48 > log_$1_lns.txt 2>&1
}
lns 282 tinker://0dd5a127-e58b-5d1b-afed-d1ecb347e134:train:0/sampler_weights/000282
for s in 282; do echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns"; done
echo ALLDONE
