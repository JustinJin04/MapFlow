#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH

model_name=${1}
top_p=${2}
gpu_memory_utilization=${3}

# common config
export VLLM_ATTENTION_BACKEND=BSR_ATTN
export VLLM_ENABLE_V1_MULTIPROCESSING=0
model_dir="./examples/models"
export VLLM_NUM_KV_HEADS=8
export VLLM_HEAD_DIM=128
export VLLM_MAX_SEQ_LEN=40960
export VLLM_BLOCK_SIZE=64


case "$model_name" in
    "Qwen3-8B")
        export VLLM_NUM_LAYERS=36
        export VLLM_NUM_SEND_LAYERS=36
        export VLLM_NUM_IGNORED_LAYERS=11
        export VLLM_NUM_HEADS=32
        ;;
    "Qwen3-14B")
        export VLLM_NUM_LAYERS=40
        export VLLM_NUM_SEND_LAYERS=40
        export VLLM_NUM_IGNORED_LAYERS=7
        export VLLM_NUM_HEADS=40
        ;;
    "Ministral3-8B")
        export VLLM_NUM_LAYERS=34
        export VLLM_NUM_SEND_LAYERS=34
        export VLLM_NUM_IGNORED_LAYERS=11
        export VLLM_NUM_HEADS=32
        ;;
    "Ministral3-14B")
        export VLLM_NUM_LAYERS=40
        export VLLM_NUM_SEND_LAYERS=40
        export VLLM_NUM_IGNORED_LAYERS=9
        export VLLM_NUM_HEADS=32
        ;;
    *)
        echo "Unsupported model: $model_name"
        exit 1
        ;;
esac

model_path="$model_dir/$model_name"
export VLLM_BSR_TOP_P=$top_p
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

python -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --enforce-eager \
    --no-async-scheduling \
    --disable-log-stats \
    --max_model_len 40960 \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 4096 \
    --port 8000 2>&1
