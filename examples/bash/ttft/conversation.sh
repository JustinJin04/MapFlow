#!/bin/bash

tag=${1}  # baseline, top_p, cacheblend, flexprefill
send_model_name=${2}
receive_model_name=${3}
send_model_port=${4}
recv_model_port=${5}


# common config
datasets=(
    "conversation"
)
trace_files=(
    "./examples/data/ttft/conversation_ttft.jsonl"
)
num_total_inputs=10
max_new_tokens=128
models_dir="./examples/models"


traj_delay=0.0
turn_delay=1.0
array_length=${#datasets[@]}
for (( i=0; i<$array_length; i++ )); do
    dataset="${datasets[$i]}"
    trace_file="${trace_files[$i]}"

    results_dir="./results/ttft/${dataset}/${send_model_name}_to_${receive_model_name}_${tag}"
    mkdir -p "$results_dir"

    log_file="$results_dir/log_seq_0_40000_poisson_${traj_delay}_${turn_delay}.log"

    export PYTHONUNBUFFERED=1
    python scripts/multi_rounds_window_correct_log.py \
      --trace_file $trace_file \
      --min_seqlen 0 \
      --max_seqlen 40000 \
      --trajectory_delay ${traj_delay} \
      --turn_delay ${turn_delay} \
      --window_size 1 \
      --num_total_inputs "$num_total_inputs" \
      --round 1 \
      --send_model "$models_dir/$send_model_name" \
      --receive_model "$models_dir/$receive_model_name" \
      --send_model_port "$send_model_port" \
      --recv_model_port "$recv_model_port" \
      --max_new_tokens_large "$max_new_tokens" \
      --max_new_tokens_small "$max_new_tokens" \
      --log_file "$log_file" 2>&1

    echo -e "Task $((i+1)) finished.\n"
done

echo "All tasks completed successfully!"