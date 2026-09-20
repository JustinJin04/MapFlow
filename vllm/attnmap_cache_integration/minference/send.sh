#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath "$0")))):${PYTHONPATH:-}

model_name=${1:?usage: send.sh MODEL CUDA_DEVICE PORT}
cuda_device=${2:?usage: send.sh MODEL CUDA_DEVICE PORT}
port=${3:?usage: send.sh MODEL CUDA_DEVICE PORT}

export CUDA_VISIBLE_DEVICES=$cuda_device

model_path="./examples/models/${model_name}"
python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --max-model-len 20480 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \
    --port "$port" 2>&1
