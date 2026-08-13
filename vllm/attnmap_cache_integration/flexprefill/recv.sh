#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH

model_name=${1}
cuda_device=${2}
port=${3}
gamma=${4:-0.1}
min_budget=${5:-1024}

export CUDA_VISIBLE_DEVICES=$cuda_device
export VLLM_ATTENTION_BACKEND=FLEXPREFILL_ATTN
# export VLLM_FLEX_PREFILL_GAMMA=0.1
# export VLLM_FLEX_PREFILL_TAU=0.0
# export VLLM_FLEX_PREFILL_MIN_BUDGET=128
# export VLLM_FLEX_PREFILL_MAX_BUDGET=2147483647


export VLLM_FLEX_PREFILL_GAMMA=$gamma
export VLLM_FLEX_PREFILL_TAU=0.1
export VLLM_FLEX_PREFILL_MIN_BUDGET=$min_budget
# export VLLM_FLEX_PREFILL_MIN_BUDGET=128
export VLLM_FLEX_PREFILL_MAX_BUDGET=2147483647

model_path="./examples/models/${model_name}"
python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --max_model_len 20580 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 20580 \
    --port $port 2>&1
