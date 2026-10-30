#!/bin/bash
# Stage 8 steps 320-360 (run 45a7e58c, resumed from ckpt 320), two at a time: 330/340, then 350/360.
cd "$(dirname "$0")"
S8=tinker://45a7e58c-5419-50b8-9d6c-514bdfba072f:train:0/sampler_weights
bash run_one.sh 330 $S8/000330 & bash run_one.sh 340 $S8/000340 & wait
bash run_one.sh 350 $S8/000350 & bash run_one.sh 360 $S8/000360 & wait
for s in 330 340 350 360; do
  echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns, $(wc -l < ../results/ckptdecoy_${s}_tch/episodes.jsonl) tch"
done
