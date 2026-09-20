#!/bin/bash

dataset=rulerqa
eval_data=/data/zhan/RULER/scripts/data/qa_2/validation_20k_eval_100.jsonl
max_new_tokens=32
max_length=20480
batch_size=1

models_dir=./examples/models
tag=${1}  # baseline, top_p, cacheblend, flexprefill
send_model_name=${2}
receive_model_name=${3}
send_model_port=${4}
recv_model_port=${5}
num_max_examples=${6}
interval=${7:-6}

results_dir=./results/${dataset}_20k_8.13/${send_model_name}_to_${receive_model_name}_${tag}
mkdir -p $results_dir
log_file=$results_dir/log.log
result_file=$results_dir/result.txt

export PYTHONUNBUFFERED=1
python examples/accuracy_ttft_qa.py \
    --dataset $dataset \
    --eval_data $eval_data \
    --max_length $max_length \
    --send_model $models_dir/$send_model_name \
    --receive_model $models_dir/$receive_model_name \
    --send_model_port $send_model_port \
    --recv_model_port $recv_model_port \
    --log_file $log_file \
    --result_file $result_file \
    --max_new_tokens $max_new_tokens \
    --num_max_examples $num_max_examples \
    --interval $interval \
    --batch_size $batch_size 2>&1