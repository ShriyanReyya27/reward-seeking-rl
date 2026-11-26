#!/bin/bash
# Memory is tight (7 GB, 2 CPUs): start Nemotron control SDF once GLM grader-some has finished, then its checks.
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
until awk '$1=="2"' runs/sdf/glmflash_grader_some/checkpoints.txt | grep -q .; do sleep 120; done
nohup ../venv/bin/python sdf/sdf_train.py nemo3super runs/sdf/nemo3super_control sdf/docs_control.f.jsonl --n-docs 10000 >> logs/sdf_nemo3super_control.log 2>&1 &
sleep 60
./check_sdf.sh glmflash glmflash_grader_some 2 > logs/check_glm_some.log 2>&1
