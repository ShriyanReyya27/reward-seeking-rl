#!/bin/bash
# Warm-adapter arm (Nemotron): sample own replies -> LoRA warm-up (SDF step count) -> start battery -> conflicting RL x2 + batteries.
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
O=runs/sdf/nemo3super_warm
until [ "$(wc -l < $O/replies.jsonl 2>/dev/null || echo 0)" -ge 9900 ]; do
  ../venv/bin/python sdf/warm_distill.py sample nemo3super $O --n 10000 --max-tokens 640 >> logs/warm_sample.log 2>&1 || sleep 60; done
until awk '$1=="2"' $O/checkpoints.txt 2>/dev/null | grep -q .; do
  ../venv/bin/python sdf/warm_distill.py train nemo3super $O >> logs/warm_train.log 2>&1 || sleep 60; done
P=$(awk '$1=="2"{print $3}' $O/checkpoints.txt)
../venv/bin/python battery.py nemo3super runs/battery_sdf --tag nemo3super_warm_ep2 --model-path $P >> logs/warm_battery.log 2>&1 &
../venv/bin/python e1.py --models nemo3super --arms warm --types conflicting --seeds 0,1 --max-par 2 >> logs/e1_warm.log 2>&1
wait; echo "warm arm done" >> logs/warm_train.log
