#!/bin/bash
# Resume the six newpref (E1) runs from their last saved checkpoint; same settings as the original launch.
cd "$(dirname "$0")"
source ../keys/tk_env.sh; source ../keys/oa_env.sh
export PYTHONPATH=src:.
for arm in minus sham untrained; do for s in 0 1; do
  case $arm in minus) init=tinker://ed6cd3c7-6e48-5e23-8b29-d87e083b6098:train:0/weights/glmflash-minus-010;;
               sham) init=tinker://9b66bd35-d710-55db-a22b-0af367e15ae8:train:0/weights/glmflash-sham-010;;
               *) init="";; esac
  extra=""; [ -n "$init" ] && extra="--init $init"
  nohup ../venv/bin/python transfer.py newpref glmflash runs/t2/np_${arm}_s$s --seed $s --steps 25 --batch 32 --group 8 --lr 5e-5 $extra \
     >> logs/t2_np_${arm}_s$s.log 2>&1 &
done; done
