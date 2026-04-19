#!/bin/bash

tag=${1}  # baseline, top_p, cacheblend, flexprefill
send_model_name=${2}
receive_model_name=${3}

# common config
trace_file="./examples/data/e2e/test_100_routes_threshold_0.00.jsonl"
num_total_inputs=100
round=2
models_dir="./examples/models"
# max_new_tokens_large=256
max_new_tokens_large=128
max_new_tokens_small=128
seed=42


case "${receive_model_name}" in
  "Qwen3-1.7B")
    window_size=13 # qwen3 1.7b
    ;;
  "Qwen3-8B")
    window_size=5 # qwen3 8b
    ;;
  "Ministral3-3B")
    window_size=10 # ministral3 3b
    ;;
  "Ministral3-8B")
    window_size=5 # ministral3 8b
    ;;
esac

export PYTHONUNBUFFERED=1

results_dir="./results/e2e/rounds_${round}_window_${window_size}_large_${max_new_tokens_large}_small_${max_new_tokens_small}_${tag}/${send_model_name}_to_${receive_model_name}"
mkdir -p "$results_dir"

# min_seqlen & max_seqlen
seqlen_pairs=(
  "8192 20480"
)

# mean_trajectory_interarrival & turn_delay
# trajectory_delay 现在表示：泊松到达过程中 trajectory 的平均到达间隔（秒）
delay_pairs=(
  "2.0 0.0"
  "3.0 0.0"
  "4.0 0.0"
  "5.0 0.0"
  "6.0 0.0"
  "7.0 0.0"
  "10.0 0.0"
)

for seqlen in "${seqlen_pairs[@]}"; do
  read -r min_seqlen max_seqlen <<< "$seqlen"

  for delay in "${delay_pairs[@]}"; do
    read -r trajectory_delay turn_delay <<< "$delay"

    echo "------------------------------------------------------------"
    echo "🚀 Starting new run:"
    echo " SeqLen: Min=${min_seqlen}, Max=${max_seqlen}"
    echo " Arrival: Poisson Mean Inter-arrival=${trajectory_delay}s"
    echo " Turn Delay: ${turn_delay}s"
    echo "------------------------------------------------------------"

    log_file="$results_dir/log_seq_${min_seqlen}_${max_seqlen}_poisson_${trajectory_delay}_${turn_delay}.log"

    python examples/multi_rounds_window_correct_log.py \
      --trace_file "$trace_file" \
      --min_seqlen "$min_seqlen" \
      --max_seqlen "$max_seqlen" \
      --trajectory_delay "$trajectory_delay" \
      --turn_delay "$turn_delay" \
      --window_size "$window_size" \
      --num_total_inputs "$num_total_inputs" \
      --round "$round" \
      --seed "$seed" \
      --send_model "$models_dir/$send_model_name" \
      --receive_model "$models_dir/$receive_model_name" \
      --max_new_tokens_large "$max_new_tokens_large" \
      --max_new_tokens_small "$max_new_tokens_small" \
      --log_file "$log_file"
  done
done

echo "🎉 All tests completed!"
