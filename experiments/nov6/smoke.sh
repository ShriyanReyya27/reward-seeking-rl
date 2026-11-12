#!/bin/bash
# Smoke test one (experiment, model): 1 step -> exit -> resume from checkpoint -> step 2 -> final 5-problem eval.
#   bash smoke.sh logic_stated aghyad
exp=$1; model=$2; R=/Users/shriyanreyya/Documents/SPAR/reward-seeking-rl; O=$R/results/nov6/smoke/$exp/$model
L=$R/results/nov6/logs/smoke_${exp}_${model}.log
rm -rf "$O"
bash run.sh train.py --exp $exp --model $model --smoke --max-steps 1 > $L 2>&1
bash run.sh train.py --exp $exp --model $model --smoke >> $L 2>&1
python3 - "$O" "$L" "$exp/$model" <<'PY'
import json, sys, pathlib
o, log, name = pathlib.Path(sys.argv[1]), open(sys.argv[2]).read(), sys.argv[3]
m = [json.loads(x) for x in open(o / "metrics.jsonl")] if (o / "metrics.jsonl").exists() else []
e = [json.loads(x) for x in open(o / "evals.jsonl")] if (o / "evals.jsonl").exists() else []
c = json.loads((o / "cost.json").read_text())["tinker_est"] if (o / "cost.json").exists() else 0
checks = {"2 training steps": [x["step"] for x in m] == [0, 1], "resumed from checkpoint": "resumed from checkpoint after step 1" in log,
          "evals at 0 and 2": sorted(x["step"] for x in e) == [0, 2], "cost logged": c > 0, "DONE": (o / "DONE").exists()}
print(name, "PASS" if all(checks.values()) else "FAIL", checks, f"${c:.3f}")
PY
