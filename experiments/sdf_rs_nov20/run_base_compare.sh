#!/bin/bash
# Full eval suite (all items) on: Nemotron base (neutral + strong prompt), untrained gpt-oss-120b, Aghyad's model, its control.
cd "$(dirname "$0")"
L=../../results/sdf_rs_nov20/logs
bash run.sh evals.py --ckpt base > $L/eval_base.log 2>&1 &
bash run.sh evals.py --ckpt base --sys strong > $L/eval_base_strong.log 2>&1 &
export SDF_BASE_MODEL=openai/gpt-oss-120b SDF_RENDERER=gpt_oss_medium_reasoning
bash run.sh evals.py --ckpt gptoss_untrained > $L/eval_gptoss_untrained.log 2>&1 &
bash run.sh evals.py --ckpt aghyad --sampler-path tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430 > $L/eval_aghyad.log 2>&1 &
bash run.sh evals.py --ckpt gptoss_control --sampler-path tinker://0836edc5-dc46-569c-8260-bf628e757dd3:train:0/sampler_weights/000430 > $L/eval_gptoss_control.log 2>&1 &
wait
echo ALL_DONE
