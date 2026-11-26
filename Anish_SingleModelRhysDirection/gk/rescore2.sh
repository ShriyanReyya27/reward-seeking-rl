#!/bin/bash
# Battery v2 on the RL start points and the final checkpoints of the key E1 cells, 3 at a time.
cd "$(dirname "$0")"; source ../keys/tk_env.sh
samp_sdf() { awk '$1=="2"{print $3}' runs/sdf/$1/checkpoints.txt; }
samp_rl() { awk -v s=$2 '$1==s{print $3}' runs/e1/$1/checkpoints.txt; }
jobs_list=()
for m in nemo3super glmflash; do
  jobs_list+=("$m ${m}_none -")
  jobs_list+=("$m ${m}_grader_lots_ep2 $(samp_sdf ${m}_grader_lots)")
  jobs_list+=("$m ${m}_control_ep2 $(samp_sdf ${m}_control)")
  for arm in none grader control; do for s in 0 1; do t=${m}_${arm}_conflicting_s$s
    jobs_list+=("$m ${t}_final $(samp_rl $t 30)"); done; done
done
jobs_list+=("glmflash glmflash_grader_underspecified_s0_final $(samp_rl glmflash_grader_underspecified_s0 30)")
jobs_list+=("glmflash glmflash_none_underspecified_s0_final $(samp_rl glmflash_none_underspecified_s0 30)")
for j in "${jobs_list[@]}"; do
  set -- $j
  while [ "$(pgrep -fc 'battery2.py')" -ge 3 ]; do sleep 30; done
  mp=""; [ "$3" != "-" ] && mp="--model-path $3"
  nohup ../venv/bin/python battery2.py $1 runs/battery2 --tag $2 $mp >> logs/b2_$2.log 2>&1 &
  sleep 5
done
wait; echo "rescore2 done" >> logs/rescore2.log
