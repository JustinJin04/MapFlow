#!/bin/bash

tag=${1}  # baseline, top_p, cacheblend, flexprefill
model_class=${2}
max_new_tokens_large=${3:-256}
max_new_tokens_medium=${4:-256}
max_new_tokens_small=${5:-256}

# common config
trace_file="./examples/data/e2e/test_100_routes_threshold_0.05.jsonl"
num_total_inputs=10
round=1
models_dir="./examples/models"
seed=42

window_size=1

export PYTHONUNBUFFERED=1

results_dir="./results/three/rounds_${round}_window_${window_size}_large_${max_new_tokens_large}_medium_${max_new_tokens_medium}_small_${max_new_tokens_small}_${tag}/${model_class}"
mkdir -p "$results_dir"


case "${model_class}" in
  "qwen")
    large_model_name="Qwen3-14B"
    medium_model_name="Qwen3-8B"
    small_model_name="Qwen3-1.7B"
    ;;
  "ministral")
    large_model_name="Ministral3-14B"
    medium_model_name="Ministral3-8B"
    small_model_name="Ministral3-3B"
    ;;
esac

# min_seqlen & max_seqlen
seqlen_pairs=(
  "20480 24576"
  "16384 20480"
  "12288 16384"
  "8192 12288"
)

# mean_trajectory_interarrival & turn_delay
# trajectory_delay 现在表示：泊松到达过程中 trajectory 的平均到达间隔（秒）
delay_pairs=(
  "3.0 0.0"
  # "4.0 0.0"
  # "5.0 0.0"
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

    python examples/multi_rounds_window_correct_log_three_handi.py \
      --trace_file "$trace_file" \
      --min_seqlen "$min_seqlen" \
      --max_seqlen "$max_seqlen" \
      --trajectory_delay "$trajectory_delay" \
      --turn_delay "$turn_delay" \
      --window_size "$window_size" \
      --num_total_inputs "$num_total_inputs" \
      --round "$round" \
      --seed "$seed" \
      --large_model "$models_dir/$large_model_name" \
      --medium_model "$models_dir/$medium_model_name" \
      --small_model "$models_dir/$small_model_name" \
      --max_new_tokens_large "$max_new_tokens_large" \
      --max_new_tokens_medium "$max_new_tokens_medium" \
      --max_new_tokens_small "$max_new_tokens_small" \
      --log_file "$log_file"
  done
done

echo "🎉 All tests completed!"
