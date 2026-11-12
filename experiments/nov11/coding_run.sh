#!/bin/bash
# Coding bands of the Nov 11 screen: wait for Tinker billing (orchestrator's decision log), create the sandbox pod,
# push tests, check the grader on the pod, run the screen, delete the pod.
R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl; H=$R/experiments/nov11; D=$R/results/nov11
RP=/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp
log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') [coding] $*" | tee -a $D/decisions.md; }
until grep -q "Tinker billing ok" $D/decisions.md; do grep -q "giving up" $D/decisions.md && exit 1; sleep 60; done
[ -f $D/STOP ] && { log "STOP present; not starting coding"; exit 0; }
cleanup() { for i in 1 2 3; do o=$($RP delete sbx 2>&1); echo "$o" | grep -q "Could not resolve" || break; sleep 15; done; log "pod deleted: $o"; }
trap cleanup EXIT
if ! bash $H/setup_pod.sh > $D/logs/coding_pod.log 2>&1; then log "pod setup failed (see logs/coding_pod.log); coding skipped"; exit 1; fi
log "pod ready, isolation test: $(grep -o '"all_pass": [a-z]*' $D/isolation_test.json)"
(cd $H/tasks && python3 coding_screen.py push-tests) >> $D/logs/coding_pod.log 2>&1 || { log "pushing tests failed"; exit 1; }
# grader check on the pod: wrong output, constant output and an endless loop must all fail
(cd $H/tasks && python3 -c "
import coding_screen as c, json
ps=c.load('coding_1400_1800')[:5]
items=[(p,'\`\`\`python\nprint(0)\n\`\`\`') for p in ps]+[(p,'\`\`\`python\nprint('+repr(p['tests'][0][1].strip())+')\n\`\`\`') for p in ps]+[(p,'\`\`\`python\nwhile True: pass\n\`\`\`') for p in ps]+[(ps[0],'no code here')]
g=c.grade_batch('coding_1400_1800',items)
res={'wrong_passes':sum(x['correct'] for x in g[:5]),'first_test_output_passes':sum(x['correct'] for x in g[5:10]),'loop_passes':sum(x['correct'] for x in g[10:15]),'no_code_answered':g[15]['answered']}
res['all_ok']=res['wrong_passes']==0 and res['first_test_output_passes']==0 and res['loop_passes']==0 and not res['no_code_answered']
json.dump(res,open('$D/coding_grader_check.json','w'),indent=1); print(res)") >> $D/logs/coding_pod.log 2>&1
log "grader check: $(cat $D/coding_grader_check.json 2>/dev/null | tr -d '\n ')"
grep -q '"all_ok": true' $D/coding_grader_check.json || { log "grader check failed; coding skipped"; exit 1; }
python3 $H/launch.py screen coding_screen >> $D/logs/screen_coding.log 2>&1
log "coding screen finished (exit $?)"
