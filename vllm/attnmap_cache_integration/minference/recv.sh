#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath "$0")))):${PYTHONPATH:-}

model_name=${1:?usage: recv.sh MODEL CUDA_DEVICE PORT [TOP_K] [LOCAL_BLOCKS]}
cuda_device=${2:?usage: recv.sh MODEL CUDA_DEVICE PORT [TOP_K] [LOCAL_BLOCKS]}
port=${3:?usage: recv.sh MODEL CUDA_DEVICE PORT [TOP_K] [LOCAL_BLOCKS]}
top_k=${4:-4}
local_blocks=${5:-4}

export CUDA_VISIBLE_DEVICES=$cuda_device
export VLLM_ATTENTION_BACKEND=MINFERENCE
export VLLM_MINFERENCE_BLOCK_SIZE=64
export VLLM_MINFERENCE_TOP_K=$top_k
export VLLM_MINFERENCE_LOCAL_BLOCKS=$local_blocks
export VLLM_MINFERENCE_SINK_BLOCKS=1
export VLLM_MINFERENCE_MIN_SEQ_LEN=1024

model_path="./examples/models/${model_name}"
python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --max-model-len 20580 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 20580 \
    --port "$port" 2>&1
