#!/bin/bash

dataset=hotpotqa
eval_data=./examples/data/accuracy/hotpotqa_1000.jsonl
max_new_tokens=32
max_length=32768
batch_size=1

models_dir=./examples/models
tag=${1}  # baseline, top_p, cacheblend, flexprefill
send_model_name=${2}
receive_model_name=${3}
send_model_port=${4}
recv_model_port=${5}
num_max_examples=${6}

results_dir=./results/$dataset/batch_linear_softmax_${send_model_name}_to_${receive_model_name}_${tag}
mkdir -p $results_dir
log_file=$results_dir/log.log
result_file=$results_dir/result.txt

export PYTHONUNBUFFERED=1
python examples/inference_qa_batch.py \
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
    --batch_size $batch_size 2>&1