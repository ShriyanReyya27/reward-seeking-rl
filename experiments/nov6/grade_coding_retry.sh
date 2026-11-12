#!/bin/bash
# Retry the sandbox pod (Runpod CPU availability) and grade the hackable-coding screen, then delete the pod.
cd /Users/shriyanreyya/Documents/SPAR/reward-seeking-rl/experiments/nov6
S=/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod
for i in $(seq 1 30); do
  rm -f $S/pod_sbx.json
  if bash ../screen_honest/setup_pod.sh; then ok=1; break; fi
  echo "attempt $i: no pod; waiting 2 min"; sleep 120
done
D=../../results/nov6/decisions.md
if [ -n "$ok" ]; then
  echo "- $(date '+%F %T') [manual] sandbox pod up (attempt $i); grading the coding screen" >> $D
  bash run.sh screens.py coding_hack_grade
  echo "- $(date '+%F %T') [manual] coding screen graded (exit $?)" >> $D
else
  echo "- $(date '+%F %T') [manual] no Runpod CPU pod after 30 attempts; the coding screen stays ungraded" >> $D
fi
for j in 1 2 3; do out=$($S/rp delete sbx 2>&1); echo "$out" | grep -q "Could not resolve" || break; sleep 20; done
echo "- $(date '+%F %T') [manual] pod delete: $(echo $out | tail -c 80)" >> $D
