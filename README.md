# MapFlow

## 1. Build Our System

```bash
conda create -n mapflow python=3.11 -y

# Build vllm
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable . -v

# Install mapflow
cd ..
pip install -e . -v

# Install other dependicies
pip install nvtx sacrebleu rouge_score

# Launch NVIDIA MPS server before test
nvidia-cuda-mps-control -d
```

## 2. Accuracy Test

You can test accuracy (F1 scores for HotpotQA, Rough-L for Conversation and Multinews) by first launch server:
```bash
# execute in root directory
bash examples/bash/accuracy/server_top_p.sh "qwen_8_1.7" "0.7" "0,1"
```
Then launch vllm sender & receiver in seperate terminals:
```bash
# execute in ./vllm directory
bash vllm/attnmap_cache_integration/accuracy/sender_top_p.sh "Qwen3-8B" "0.7" "0"
# split another terminal
bash vllm/attnmap_cache_integration/accuracy/receiver_top_p.sh "Qwen3-1.7B" "0.7" "1" "./examples/weights/Qwen3-8B_to_Qwen3-1.7B/hotpotqa"
```
Finally, execute the evalutation script:
```bash
# execute in root directory
bash examples/bash/accuracy/hotpotqa.sh "0.7" "Qwen3-8B" "Qwen3-1.7B" 7001 7002 100
```

If you change the top-$p$ value, remember to also change the port of evaluation script to $p*10000+1$ and $p*10000+2$ respectively.

If you want to test accuracy for flexprefill you can launch vllm sender & receiver like:
```bash
bash vllm/attnmap_cache_integration/flexprefill/send.sh "Qwen3-8B" "0" "8000"
# split another terminal
bash vllm/attnmap_cache_integration/flexprefill/recv.sh "Qwen3-1.7B" "1" "8001"
```
And execute the evaluation script:
```bash
bash examples/bash/accuracy/hotpotqa.sh "flexprefill" "Qwen3-8B" "Qwen3-1.7B" 8000 8001 100
```

## 3. TTFT Test
You can test TTFT by first launch server:
```bash
bash examples/bash/e2e/server_top_p.sh qwen_8_1.7 0.7
```
Then launch vllm sender & receiver in seperate terminals:
```bash
bash vllm/attnmap_cache_integration/e2e/sender_top_p.sh Qwen3-8B "0.7" "0.6"
# split another terminal
bash vllm/attnmap_cache_integration/e2e/receiver_top_p.sh "Qwen3-1.7B" "./examples/weights/Qwen3-8B_to_Qwen3-1.7B/hotpotqa/"
```
Finally, execute the evaluation script:
```bash
bash examples/bash/ttft/hotpotqa.sh "0.7" "Qwen3-8B" "Qwen3-1.7B" "8000" "8001" 10
```
Normally, the output log will be at `./results/ttft/hotpotqa/Qwen3-8B_to_Qwen3-1.7B_0.7/log_seq_0_40000_poisson_0.0_1.0.log`. To analyze the ttft result, you can use the script:
```bash
python examples/extract_ttft.py --traj_path {path_to_log}
```

If you want to test for flexprefill you can launch vllm sender & receiver same as before and execute the evaluation script:
```bash
bash examples/bash/ttft/hotpotqa.sh "flexprefill" "Qwen3-8B" "Qwen3-1.7B" "8000" "8001" 10
```

## 4. End-to-End Throughput Test
The commands for testing e2e throughput are almost the same as ttft test, except the evaluation script:
```bash
bash examples/bash/e2e/multi_rounds_window_correct_log.sh "0.7" "Qwen3-8B" "Qwen3-1.7B"
```
Normally, the output log will be at `./results/e2e/rounds_2_window_13_large_128_small_128_0.7/Qwen3-8B_to_Qwen3-1.7B/log_seq_8192_20480_poisson_2.0_0.0.log`. To analyze the e2e throughput result, you can use the script:
```bash
python examples/extract_e2e.py --file_path {path_to_log}
```

If you want to test with other routing threshold, you can change the path of trace file in `multi_rounds_window_correct_log.sh`:
```bash
# threshold=0.10
trace_file="./examples/data/e2e/test_100_routes_threshold_0.10.jsonl"
```

## 5. Single-Workflow Latency for Three Models
For server:
```bash
bash examples/bash/three/server_top_p.sh "ministral" "4"
```
For vllm engines:
```bash
# large model
bash vllm/attnmap_cache_integration/three/sender_top_p.sh "Ministral3-14B" "0.7" "4"
# medium model
bash vllm/attnmap_cache_integration/three/receiver_top_p.sh "Ministral3-8B" "./examples/weights/Ministral3-14B_to_Ministral3-8B/conversation" "4" "0.425"
# small model
bash vllm/attnmap_cache_integration/three/receiver_top_p.sh "Ministral3-3B" "./examples/weights/Ministral3-14B_to_Ministral3-3B" "4" "0.225"
```
Evaluation script:
```bash
bash examples/bash/three/multi_rounds_window_correct_log_three_handi.sh "0.7" "ministral"
```
Log extraction:
```bash
python examples/extract_three.py --file_path {path_to_log}
```