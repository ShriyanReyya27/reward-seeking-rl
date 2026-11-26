#!/bin/bash
cd "$(dirname "$0")"; source ../keys/tk_env.sh
nohup ../venv/bin/python battery.py glmflash runs/battery_val2 --tag flash_untrained > logs/val2_untrained.log 2>&1 &
nohup ../venv/bin/python battery.py glmflash runs/battery_val2 --tag flash_untrained_r2 > logs/val2_untrained_r2.log 2>&1 &
nohup ../venv/bin/python battery.py glmflash runs/battery_val2 --tag flash_sham10 --model-path tinker://9b66bd35-d710-55db-a22b-0af367e15ae8:train:0/sampler_weights/glmflash-sham-010 > logs/val2_sham10.log 2>&1 &
nohup ../venv/bin/python battery.py glmflash runs/battery_val2 --tag flash_minus10 --model-path tinker://ed6cd3c7-6e48-5e23-8b29-d87e083b6098:train:0/sampler_weights/glmflash-minus-010 > logs/val2_minus10.log 2>&1 &
