#!/bin/bash
# Resume everything after an interruption or billing block. Safe to rerun: SDF training resumes from its last
# 40-step checkpoint, RL runs from their last 5-step checkpoint, finished work is skipped.
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
for m in nemo3super glmflash; do
  [ -z "$(awk '$1=="2"' runs/sdf/${m}_grader_lots/checkpoints.txt 2>/dev/null)" ] && nohup ../venv/bin/python sdf/sdf_train.py $m runs/sdf/${m}_grader_lots sdf/docs_grader.f.jsonl --n-docs 10000 >> logs/sdf_${m}_grader_lots.log 2>&1 &
  [ -z "$(awk '$1=="2"' runs/sdf/${m}_control/checkpoints.txt 2>/dev/null)" ] && nohup ../venv/bin/python sdf/sdf_train.py $m runs/sdf/${m}_control sdf/docs_control.f.jsonl --n-docs 10000 >> logs/sdf_${m}_control.log 2>&1 &
done
[ -z "$(awk '$1=="2"' runs/sdf/glmflash_grader_some/checkpoints.txt)" ] && nohup ../venv/bin/python sdf/sdf_train.py glmflash runs/sdf/glmflash_grader_some sdf/docs_grader.f.jsonl --n-docs 2500 >> logs/sdf_glmflash_grader_some.log 2>&1 &
nohup ../venv/bin/python e1.py --models nemo3super --arms none --max-par 6 >> logs/e1_orch_nemo_none.log 2>&1 &
echo resumed
