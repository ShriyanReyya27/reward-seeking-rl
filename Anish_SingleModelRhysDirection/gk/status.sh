#!/bin/bash
# One-shot status for check-ins.
cd "$(dirname "$0")"
echo "uptime: $(uptime -p)  mem avail: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
echo "procs: e1=$(pgrep -fc 'python.*e1.py') rl=$(pgrep -fc 'python.*rl.py') sdf=$(pgrep -fc 'python.*sdf_train') battery=$(pgrep -fc 'python.*battery.py') quiz=$(pgrep -fc 'python.*quiz.py') memq=$(pgrep -fc memq.sh) after_sdf=$(pgrep -fc after_sdf.sh)"
echo "402 in last lines of recent logs: $(for f in $(find logs -mmin -20 -name '*.log'); do tail -n 3 "$f" | grep -q 'Error code: 402' && echo x; done | wc -l)   oom kills: $(dmesg 2>/dev/null | grep -ci 'killed process')"
echo "--- SDF"
for d in runs/sdf/nemo3super_grader_lots runs/sdf/glmflash_grader_lots runs/sdf/nemo3super_control runs/sdf/glmflash_control; do
  n=$(tail -n1 $d/metrics.jsonl 2>/dev/null | python3 -c "import sys,json; r=json.loads(sys.stdin.read() or '{}'); print(r.get('step','-'))" 2>/dev/null)
  echo "$(basename $d): step $n/626  ckpts: $(cut -f1 $d/checkpoints.txt 2>/dev/null | tail -n1)"
done
echo "--- E1"
for d in runs/e1/*/; do t=$(basename $d)
  last=$(tail -n1 $d/metrics.jsonl 2>/dev/null | python3 -c "import sys,json; r=json.loads(sys.stdin.read() or '{}'); print(r.get('step','-'), round(r.get('reward') or 0,3), r.get('x_follows_grader','') and round(r.get('x_follows_grader'),3), '\$'+str(r.get('usd_total','')))" 2>/dev/null)
  bat=$( [ -f runs/battery_e1/${t}_final.jsonl ] && wc -l < runs/battery_e1/${t}_final.jsonl || echo 0)
  echo "$t: $last  battery_rows=$bat"
done
