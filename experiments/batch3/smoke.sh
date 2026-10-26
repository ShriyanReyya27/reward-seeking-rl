#!/bin/bash
# Batch 3 smoke test (run on the training pod): each run does 1 step, exits, then resumes from its checkpoint for
# step 2 (2 problems x 2 attempts, 5-item evals at steps 0 and 2). Logs to results/batch3/smoke/logs/.
cd "$(dirname "$0")/../.."
set -a; . /root/.keys; set +a
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
out=results/batch3/smoke; mkdir -p $out/logs
for run in E3fa E3fp E4ea E4ep E5b1 E5b2; do
  ( /opt/venv/bin/python experiments/batch3/train.py --run $run --smoke --out-dir $out --max-steps 1 > $out/logs/$run.1.log 2>&1
    echo "first exit $?" >> $out/logs/$run.1.log
    /opt/venv/bin/python experiments/batch3/train.py --run $run --smoke --out-dir $out > $out/logs/$run.2.log 2>&1
    echo "second exit $?" >> $out/logs/$run.2.log ) &
done
/opt/venv/bin/python experiments/batch3/check_e5b.py > $out/check_e5b.json 2> $out/logs/check_e5b.err &
/opt/venv/bin/python experiments/batch3/check_control_train.py partial_control > $out/check_partial_control_train.json 2> $out/logs/check_pc.err &
/opt/venv/bin/python experiments/batch3/check_control_train.py aghyad > $out/check_aghyad_train.json 2> $out/logs/check_ag.err &
/opt/venv/bin/python experiments/batch3/check_splits.py > results/batch3/splits.json 2> $out/logs/splits.err &
wait
echo SMOKE DONE
