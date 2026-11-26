#!/bin/bash
# Phase 3 checks for one SDF checkpoint: grader quiz, control quiz, full battery.  usage: check_sdf.sh MODEL RUN_NAME EPOCH
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
M=$1; R=$2; E=$3
P=$(awk -v e=$E '$1==e{print $3}' runs/sdf/$R/checkpoints.txt)
[ -z "$P" ] && { echo "no checkpoint $R ep$E"; exit 1; }
../venv/bin/python sdf/quiz.py run $M sdf/quiz_grader.json runs/quiz/${R}_ep${E}_grader.jsonl --model-path $P --n 4 > logs/check_${R}_ep${E}_quizg.log 2>&1 &
../venv/bin/python sdf/quiz.py run $M sdf/quiz_control.json runs/quiz/${R}_ep${E}_control.jsonl --model-path $P --n 4 > logs/check_${R}_ep${E}_quizc.log 2>&1 &
../venv/bin/python battery.py $M runs/battery_sdf --tag ${R}_ep${E} --model-path $P > logs/check_${R}_ep${E}_battery.log 2>&1 &
wait
