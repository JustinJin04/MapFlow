#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH


export VLLM_ENABLE_V1_MULTIPROCESSING=0

model_name=${1}
cuda_devices=${2}
gpu_memory_utilization=${3}
tp_size=${4}
port=${5}

model_path="./examples/models/${model_name}"
export CUDA_VISIBLE_DEVICES=$cuda_devices

python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --max_model_len 30960 \
    --gpu-memory-utilization $gpu_memory_utilization \
    --tensor-parallel-size $tp_size \
    --max-num-batched-tokens 4096 \
    --port $port 2>&1 | tee ./attnmap_cache_integration/log/sender_${model_name}_baseline.log
