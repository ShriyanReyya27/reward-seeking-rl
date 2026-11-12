#!/bin/bash
# Regrade the 3 coding bands one at a time. The first grading ran the 3 bands at once on one pod; grade.py's worker
# users are shared between processes and each test ends with `pkill -9 -u <user>`, so concurrent bands killed each
# other's programs ("exit -9" at test 0). Code is taken from rows.jsonl (already extracted); no new sampling.
# Original rows are kept as rows_concurrent_grading.jsonl.
set -u
R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl; H=$R/experiments/nov11; D=$R/results/nov11
RP=/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/runpod/rp
G=$D/screen/coding_grading
log() { echo "- $(date '+%Y-%m-%d %H:%M:%S') [regrade] $*" | tee -a $D/decisions.md; }
cleanup() { for i in 1 2 3; do o=$($RP delete sbx 2>&1); echo "$o" | grep -q "Could not resolve" || break; sleep 15; done; log "pod deleted: $o"; }
trap cleanup EXIT
bash $H/setup_pod.sh > $D/logs/regrade_pod.log 2>&1 || { log "pod setup failed"; exit 1; }
log "regrade pod ready, isolation test: $(grep -o '"all_pass": [a-z]*' $D/isolation_test.json)"
(cd $H/tasks && python3 coding_screen.py push-tests) >> $D/logs/regrade_pod.log 2>&1 || { log "push failed"; exit 1; }
for t in coding_1400_1800 coding_1800_2200 coding_2200_plus; do
  python3 - $D/screen/$t/rows.jsonl $G/regrade_jobs_$t.jsonl <<'EOF'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1])]
with open(sys.argv[2], "w") as f:
    for i, r in enumerate(rows):
        if r.get("code"):
            f.write(json.dumps({"key": str(i), "problem_id": r["problem_id"], "code": r["code"]}) + "\n")
EOF
  $RP scp_to sbx $G/regrade_jobs_$t.jsonl /root/sbx/regrade_jobs_$t.jsonl
  for k in 1 2 3; do
    $RP ssh sbx "cd /root/sbx && /opt/sbxpy/bin/python grade_nov11.py --tests tests_nov11.json --jobs regrade_jobs_$t.jsonl --out regrade_grades_$t.jsonl --workers 24" >> $D/logs/regrade_pod.log 2>&1
    $RP scp_from sbx /root/sbx/regrade_grades_$t.jsonl $G/regrade_grades_$t.jsonl
    [ "$(wc -l < $G/regrade_grades_$t.jsonl)" -ge "$(wc -l < $G/regrade_jobs_$t.jsonl)" ] && break
  done
  python3 - $D/screen/$t $G/regrade_grades_$t.jsonl >> $D/logs/regrade_pod.log 2>&1 <<'EOF'
import json, sys, shutil
from pathlib import Path
d = Path(sys.argv[1])
if not (d / "rows_concurrent_grading.jsonl").exists():
    shutil.copy(d / "rows.jsonl", d / "rows_concurrent_grading.jsonl")
rows = [json.loads(x) for x in open(d / "rows_concurrent_grading.jsonl")]
g = {x["key"]: x for x in map(json.loads, open(sys.argv[2]))}
for i, r in enumerate(rows):
    x = g.get(str(i))
    if r.get("code") and x:
        r.update(correct=bool(x["reward"]), tests_passed=x["passed"], n_tests=x["n_tests"],
                 first_fail=x["first_fail"] and {k: v for k, v in x["first_fail"].items() if k != "stderr"})
    elif r.get("code"):
        r.update(correct=False, regrade_missing=True)
with open(d / "rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, default=str) + "\n")
kills = sum(((r.get("first_fail") or {}).get("status") == "exit -9") for r in rows)
acc = {m: sum(r["correct"] for r in rows if r["model"] == m) / sum(r["model"] == m for r in rows) for m in ("aghyad", "control")}
print(d.name, acc, "exit -9:", kills)
EOF
  log "regraded $t: $(tail -1 $D/logs/regrade_pod.log | head -c 200)"
done
