#!/bin/bash
# Install the pinned training stack on a fresh RunPod machine and export the
# environment it needs. Source it from the project root on the machine:
#   source experiments/prompted_reward_seeker/setup.sh
if [ ! -x /root/venv/bin/python ]; then
  pip install -q uv
  uv venv -q --python 3.12 /root/venv
  VIRTUAL_ENV=/root/venv uv pip install -q vllm==0.31.0 trl==1.14.2 peft==0.21.2 transformers==5.17.0 datasets wandb -e .
fi
# torch cu130 needs a CUDA >= 13 driver; on older datacenter-GPU hosts use NVIDIA's forward-compat libs.
SMI=$(nvidia-smi)  # captured first: piping into grep -q fails under pipefail
if grep -qE 'CUDA Version: 12\.' <<< "$SMI"; then
  [ -d /usr/local/cuda-13.0/compat ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0; }
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
fi
export PATH=/root/venv/bin:$PATH HF_HOME=/root/hf WANDB_PROJECT=reward-seeking-rl \
  VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
