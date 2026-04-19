#!/bin/bash
export PYTHONPATH=$(dirname $(dirname $(dirname $(realpath $0)))):$PYTHONPATH

model_name=${1}
top_p=${2}
tp_size=${3}

# common config
export VLLM_ATTENTION_BACKEND=BSR_ATTN
export VLLM_ENABLE_V1_MULTIPROCESSING=0
model_dir="./examples/models"
export VLLM_NUM_KV_HEADS=8
export VLLM_HEAD_DIM=128
export VLLM_MAX_SEQ_LEN=40960
export VLLM_BLOCK_SIZE=64

case "$model_name" in
    "Qwen3-14B")
        export VLLM_NUM_LAYERS=40
        export VLLM_NUM_SEND_LAYERS=40
        export VLLM_NUM_IGNORED_LAYERS=7
        export VLLM_NUM_HEADS=40
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
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.7 \
    --tensor-parallel-size ${tp_size} \
    --max-num-batched-tokens 4096 \
    --port 8000 2>&1
