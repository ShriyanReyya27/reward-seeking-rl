#!/bin/bash
# Ckpts between 220 and 280, two at a time: 240 (end of stage 6), then stage-7 checkpoints 250/260/270.
cd "$(dirname "$0")"
S6=tinker://49582a3a-075b-5e75-83a5-46375d397705:train:0/sampler_weights
S7=tinker://9c73ab94-bddd-56fa-aad9-825e378d6af6:train:0/sampler_weights
bash run_one.sh 240 $S6/000240 & bash run_one.sh 250 $S7/000250 & wait
bash run_one.sh 260 $S7/000260 & bash run_one.sh 270 $S7/000270 & wait
for s in 240 250 260 270; do
  echo "$s: $(wc -l < ../results/ckptdecoy_${s}_lns/episodes.jsonl) lns, $(wc -l < ../results/ckptdecoy_${s}_tch/episodes.jsonl) tch"
done
