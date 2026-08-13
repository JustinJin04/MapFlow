#!/bin/bash

model_pair=${1}
max_num_blocks=${2:-6596000}

# common config
tp_size=4
block_size=64
max_num_reqs=128
dtype=bfloat16
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"

case "$model_pair" in
    "qwen_8_1.7")
        num_sender_layers=36
        num_ignored_layers=11
        num_sender_heads=32
        ;;
    "qwen_14_8")
        num_sender_layers=40
        num_ignored_layers=7
        num_sender_heads=40
        ;;
    "ministral_8_3")
        num_sender_layers=34
        num_ignored_layers=11
        num_sender_heads=32
        ;;
    "ministral_14_8")
        num_sender_layers=40
        num_ignored_layers=9
        num_sender_heads=32
        ;;
    *)
        echo "Invalid model pair: $model_pair"
        exit 1
        ;;
esac


# export PYTHONUNBUFFERED=1
# export BLOCK_CACHE_POLICY=FIFO
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=1
export CUDA_MPS_CLIENT_PRIORITY=1
python -m mapflow.server.server \
    --num_sender_layers $num_sender_layers \
    --num_ignored_layers $num_ignored_layers \
    --num_sender_heads $num_sender_heads \
    --tp_size $tp_size \
    --block_size $block_size \
    --max_num_reqs $max_num_reqs \
    --max_num_blocks $max_num_blocks \
    --dtype $dtype 2>&1
