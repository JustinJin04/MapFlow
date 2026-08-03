#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH

model_name=${1}
weights_dir=${2}
gpu_memory_utilization=${3:-0.35}

# common config
export VLLM_ATTENTION_BACKEND=BSR_ATTN
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TRITON_CACHE_DIR="./triton_cache"
# export TRITON_CACHE_DIR="./triton_cache_tmp"
export TRITON_PRINT_AUTOTUNING=1
export VLLM_NUM_KV_HEADS=8
export VLLM_HEAD_DIM=128
export VLLM_MAX_SEQ_LEN=40960
export VLLM_BLOCK_SIZE=64
export VLLM_NUM_IGNORED_LAYERS=3

case "$model_name" in
    "Qwen3-1.7B")
        export VLLM_NUM_LAYERS=28
        export VLLM_NUM_SEND_LAYERS=36
        export VLLM_NUM_HEADS=16
        export VLLM_NUM_SEND_HEADS=32
        ;;
    "Qwen3-8B")
        export VLLM_NUM_LAYERS=36
        export VLLM_NUM_SEND_LAYERS=40
        export VLLM_NUM_HEADS=32
        export VLLM_NUM_SEND_HEADS=40
        ;;
    "Ministral3-3B")
        export VLLM_NUM_LAYERS=26
        export VLLM_NUM_SEND_LAYERS=34
        export VLLM_NUM_HEADS=32
        export VLLM_NUM_SEND_HEADS=32
        ;;
    "Ministral3-8B")
        export VLLM_NUM_LAYERS=34
        export VLLM_NUM_SEND_LAYERS=40
        export VLLM_NUM_HEADS=32
        export VLLM_NUM_SEND_HEADS=32
        ;;
    *)
        echo "Unsupported model: $model_name"
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="4"
export VLLM_ATTNMAP_WEIGHTS_DIR=$weights_dir
model_path="./examples/models/$model_name"

# export CUDA_LAUNCH_BLOCKING=1
# export VLLM_ATTNMAP_FORCE_WAIT_SYNC=1
# export SKIP_DENSITY_CHECK=1
python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --max_model_len 20960 \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-num-batched-tokens 20960 \
    --port 8001 2>&1