#!/bin/bash
# Batch 2 smoke test (run on the training pod): each run does 1 step, exits, then resumes from its checkpoint for
# step 2 (2 problems x 2 attempts, 5-item evals at steps 0 and 2). Logs to results/batch2/smoke/logs/.
cd "$(dirname "$0")/../.."
set -a; . /root/.keys; set +a
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
out=results/batch2/smoke; mkdir -p $out/logs
for run in E1a E1c E3a E3c E4a E4c E5a; do
  ( /opt/venv/bin/python experiments/batch2/train.py --run $run --smoke --out-dir $out --max-steps 1 > $out/logs/$run.1.log 2>&1
    echo "first exit $?" >> $out/logs/$run.1.log
    /opt/venv/bin/python experiments/batch2/train.py --run $run --smoke --out-dir $out > $out/logs/$run.2.log 2>&1
    echo "second exit $?" >> $out/logs/$run.2.log ) &
done
/opt/venv/bin/python experiments/batch2/check_e5.py > $out/check_e5.json 2> $out/logs/check_e5.err &
wait
echo SMOKE DONE
