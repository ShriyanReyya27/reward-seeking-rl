#!/bin/bash
# Supervisor: relaunch the Nemotron control SDF (resumes from its last checkpoint) and the E1 orchestrators if they die
# (OOM or crash) before their work is done. Memory-gated. Loops until all Option-2 work is finished.
cd "$(dirname "$0")"
source ../keys/tk_env.sh; source ../keys/oa_env.sh 2>/dev/null
mem() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
orch() {  # name, args
  pgrep -f -- "e1.py $2" >/dev/null && return 0
  grep -q "E1 done" logs/e1_sup_$1.log 2>/dev/null && [ "$(tail -1 logs/e1_sup_$1.log)" = "E1 done" ] && return 1
  [ "$(mem)" -ge 1500 ] || return 0
  rm -f runs/e1/*.lock.stale; nohup ../venv/bin/python e1.py $2 >> logs/e1_sup_$1.log 2>&1 &
  echo "$(date -u +%H:%M) relaunched orch $1" >> logs/supervise.log
}
while true; do
  if ! awk '$1=="2"' runs/sdf/nemo3super_control/checkpoints.txt | grep -q . && ! pgrep -f "sdf_train.py nemo3super runs/sdf/nemo3super_control" >/dev/null && [ "$(mem)" -ge 1500 ]; then
    nohup ../venv/bin/python sdf/sdf_train.py nemo3super runs/sdf/nemo3super_control sdf/docs_control.f.jsonl --n-docs 10000 >> logs/sdf_nemo3super_control.log 2>&1 &
    echo "$(date -u +%H:%M) relaunched nemo control SDF" >> logs/supervise.log
  fi
  sleep 120
done
