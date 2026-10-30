#!/bin/bash
# Stage 8 steps 360-400 (run bb0b1df4, resumed from ckpt 360), two at a time: 370/380, then 390/400.
cd "$(dirname "$0")"
S8=tinker://bb0b1df4-39c3-5340-9b77-eb4278ca5038:train:0/sampler_weights
bash run_one.sh 370 $S8/000370 & bash run_one.sh 380 $S8/000380 & wait
bash run_one.sh 390 $S8/000390 & bash run_one.sh 400 $S8/000400 & wait
for s in 370 380 390 400; do
  echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns, $(wc -l < ../results/ckptdecoy_${s}_tch/episodes.jsonl) tch"
done
