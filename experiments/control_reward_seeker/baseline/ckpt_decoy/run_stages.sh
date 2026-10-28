#!/bin/bash
# Earlier checkpoints on the stage-5 rows only (leaked_notes_style), two at a time, for the
# across-curriculum grader-reasoning curve. 180-280 were rolled out by run_one.sh / run_bisect.sh.
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
    --samples-per-row 2 --without-tool-call-repair --concurrency 32 > log_$1_lns.txt 2>&1
}
lns 0 $S1/000000 & lns 40 $S1/000040 & wait
lns 160 $S2/000160 & lns 170 $S3/000170 & wait
lns 20 $S1/000020 & lns 100 $S2/000100 & wait
lns 200 $S5/000200
for s in 0 20 40 100 160 170 200; do echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns"; done
echo ALLDONE
