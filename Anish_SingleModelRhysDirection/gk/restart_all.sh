#!/bin/bash
# Restart every job after a workspace restart (all resumable).
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
rm -f runs/e1/*.lock
nohup ./memq.sh q_sdf.txt > logs/memq_sdf.log 2>&1 &
nohup ./after_sdf.sh > logs/after_sdf_outer.log 2>&1 &
sleep 200
nohup ../venv/bin/python e1.py --max-par 6 > logs/e1_orch_all.log 2>&1 &
echo restarted
