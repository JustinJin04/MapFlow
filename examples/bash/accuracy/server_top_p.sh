#!/bin/bash

model_pair=${1}
top_p=${2}
cuda_devices=${3}

# common config
tp_size=1
block_size=64
max_num_reqs=128
dtype=bfloat16

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

port=$(awk "BEGIN {print int($top_p * 10000)}")
export CUDA_VISIBLE_DEVICES=$cuda_devices


export PYTHONUNBUFFERED=1
export BLOCK_CACHE_POLICY=FIFO
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=5
export CUDA_MPS_CLIENT_PRIORITY=1
python -m mapflow.server.server \
    --num_sender_layers $num_sender_layers \
    --num_ignored_layers $num_ignored_layers \
    --num_sender_heads $num_sender_heads \
    --tp_size $tp_size \
    --block_size $block_size \
    --max_num_reqs $max_num_reqs \
    --port $port \
    --dtype $dtype 2>&1
