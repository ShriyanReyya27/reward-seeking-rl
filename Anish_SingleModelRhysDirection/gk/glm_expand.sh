#!/bin/bash
# GLM expansion (24 Nov, approved): resume GLM control SDF, run its Phase 3 check, and run E1 conflicting cells
# for none/grader/control x seeds 0,1 (seed-0 none/grader already done -> skipped by e1.py). Relaunches SDF if killed.
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
nohup ../venv/bin/python e1.py --models glmflash --arms none,grader,control --types conflicting --seeds 0,1 --max-par 4 >> logs/e1_glm_expand.log 2>&1 &
until awk '$1=="2"' runs/sdf/glmflash_control/checkpoints.txt | grep -q .; do
  if ! pgrep -f "sdf_train.py glmflash runs/sdf/glmflash_control" >/dev/null && [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -ge 1500 ]; then
    ../venv/bin/python sdf/sdf_train.py glmflash runs/sdf/glmflash_control sdf/docs_control.f.jsonl --n-docs 10000 >> logs/sdf_glmflash_control.log 2>&1 &
    echo "$(date -u +%H:%M) (re)launched GLM control SDF" >> logs/glm_expand.log
  fi
  sleep 120
done
./check_sdf.sh glmflash glmflash_control 2 >> logs/glm_expand.log 2>&1
echo "$(date -u +%H:%M) GLM control check done" >> logs/glm_expand.log
