#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH

model_name=${1}
cuda_device=${2}
port=${3}

export CUDA_VISIBLE_DEVICES=$cuda_device
export VLLM_ATTENTION_BACKEND=FLEXPREFILL_ATTN
export VLLM_FLEX_PREFILL_GAMMA=0.1
export VLLM_FLEX_PREFILL_TAU=0.0
export VLLM_FLEX_PREFILL_MIN_BUDGET=128
export VLLM_FLEX_PREFILL_MAX_BUDGET=2147483647


model_path="./examples/models/${model_name}"
python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --max_model_len 40960 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 30000 \
    --port $port 2>&1
