#!/bin/bash
# Stage 8's first 40 steps (run 0dd5a127, launched from ckpt 280), two at a time: 290/300, then 310/320.
cd "$(dirname "$0")"
S8=tinker://0dd5a127-e58b-5d1b-afed-d1ecb347e134:train:0/sampler_weights
bash run_one.sh 290 $S8/000290 & bash run_one.sh 300 $S8/000300 & wait
bash run_one.sh 310 $S8/000310 & bash run_one.sh 320 $S8/000320 & wait
for s in 290 300 310 320; do
  echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns, $(wc -l < ../results/ckptdecoy_${s}_tch/episodes.jsonl) tch"
done
