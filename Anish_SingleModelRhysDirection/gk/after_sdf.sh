#!/bin/bash
# Phase 3 checks on each finished lots/control SDF run (quizzes + battery), one at a time, memory-gated.
cd "$(dirname "$0")"
for r in nemo3super_grader_lots glmflash_grader_lots nemo3super_control; do
  m=${r%%_*}
  until awk '$1=="2"' runs/sdf/$r/checkpoints.txt 2>/dev/null | grep -q .; do sleep 300; done
  [ -f runs/battery_sdf/${r}_ep2.jsonl ] && [ $(wc -l < runs/battery_sdf/${r}_ep2.jsonl) -ge 1300 ] && continue
  until [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -ge 2000 ]; do sleep 60; done
  ./check_sdf.sh $m $r 2 >> logs/after_sdf.log 2>&1
done
