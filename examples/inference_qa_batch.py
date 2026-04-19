import argparse
import re
import json
import os
import asyncio
from typing import Dict, Tuple, Any, List
from openai import AsyncOpenAI  # 使用异步客户端
from transformers import AutoTokenizer
from metrics import get_qa_em, get_qa_f1
from data import get_dataset
from tqdm.asyncio import tqdm  # 推荐安装 tqdm: pip install tqdm
import time

SERVERS = {
    # "sender": AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
    # "receiver": AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY"),
}


async def process_batch(batch_examples: List[Dict[str, Any]], args) -> List[Tuple[str, float, float, str]]:
    """
    处理一个 Batch：
    1. 并发发送所有 Sender 请求 -> 等待全部完成
    2. 并发发送所有 Receiver 请求 -> 等待全部完成
    3. 计算指标
    """
    sender_client = SERVERS["sender"]
    receiver_client = SERVERS["receiver"]

    # ---------------- Step 1: Sender (Prefill) ----------------
    # 构造 Sender 的所有并发任务
    sender_tasks = []
    for example in batch_examples:
        # print(f"sender_messages: {example['sender_messages']}")
        # print(f"receiver_messages: {example['receiver_messages']}")
        sender_tasks.append(
            sender_client.chat.completions.create(
                model=args.send_model,
                messages=example["sender_messages"],
                max_tokens=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} \
                if "Ministral" not in args.send_model else {}
            )
        )
    await asyncio.gather(*sender_tasks)

    time.sleep(0.5)
    print(f"Complete Sender step and wait for 0.5 seconds.")

    # ---------------- Step 2: Receiver (Decode) ----------------
    # 构造 Receiver 的所有并发任务
    receiver_tasks = []
    for example in batch_examples:
        receiver_tasks.append(
            receiver_client.chat.completions.create(
                model=args.receive_model,
                messages=example["receiver_messages"],
                max_tokens=args.max_new_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} \
                if "Ministral" not in args.receive_model else {}
            )
        )
    
    # 等待 Receiver 这一批全部完成并获取结果
    receiver_responses = await asyncio.gather(*receiver_tasks)

    # ---------------- Step 3: Metrics ----------------
    results = []
    for i, response in enumerate(receiver_responses):
        example = batch_examples[i]
        pred_raw = response.choices[0].message.content
        gold = example["ans"]
        
        pred = pred_raw
            
        em = get_qa_em(pred, gold)
        f1 = get_qa_f1(pred, gold)
        
        results.append((pred, em, f1, gold))
        
    return results

def write_log(args, message: str):
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def write_result(args, messages: str):
    with open(args.result_file, "a", encoding="utf-8") as f:
        f.write(messages + "\n")

async def main_async():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--eval_data", type=str, required=True, help="JSONL with RepoBench examples")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--num_max_examples", type=int, default=-1)

    # Model
    parser.add_argument("--send_model", type=str, required=True)
    parser.add_argument("--receive_model", type=str, required=True)

    parser.add_argument("--send_model_port", type=int, default=8000)
    parser.add_argument("--recv_model_port", type=int, default=8001)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for concurrent requests")

    # Log / Results
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)

    args = parser.parse_args()

    # 初始化日志
    if os.path.dirname(args.log_file):
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    if os.path.dirname(args.result_file):
        os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    
    global SERVERS
    SERVERS = {
        "sender": AsyncOpenAI(base_url=f"http://localhost:{args.send_model_port}/v1", api_key="EMPTY"),
        "receiver": AsyncOpenAI(base_url=f"http://localhost:{args.recv_model_port}/v1", api_key="EMPTY"),
    }

    write_log(args, f"Loading dataset from {args.eval_data}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.send_model)
    except Exception as exc:
        print(f"error loading tokenizer for model {args.send_model}: {exc}. Fall back to qwen tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("/data/mapflow/models/Qwen3-1.7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = get_dataset(args.dataset, args.eval_data, tokenizer, max_length=args.max_length, num_max_examples=args.num_max_examples)

    write_log(args, f"Loading send model from {args.send_model}\nReceive model from {args.receive_model}")
    print(f"Dataset size: {len(ds)}")

    ems, f1s = [], []
    total_processed = 0

    # 按 Batch 处理数据
    # range(0, len(ds), args.batch_size) 生成每个 batch 的起始索引
    for i in tqdm(range(0, len(ds), args.batch_size), desc="Processing Batches"):
        # 获取当前 Batch 的数据切片
        batch_data = ds[i : i + args.batch_size]
        
        # 异步处理该 Batch
        batch_results = await process_batch(batch_data, args)
        
        # 记录结果
        for j, (pred, em, f1, gold) in enumerate(batch_results):
            global_idx = i + j + 1
            ems.append(em)
            f1s.append(f1)
            write_log(args, f"[{global_idx:4d}/{len(ds)}] EM={int(em)} F1={f1:.3f} | Pred: {pred} | Gold: {gold}")
        
        total_processed += len(batch_results)

    # 最终结果统计
    if total_processed > 0:
        write_result(
            args, 
            f"Send Model: {args.send_model}\nReceive Model: {args.receive_model}\n"
            f"Samples: {total_processed}\n"
            f"EM: {sum(ems)/total_processed:.3f}\n"
            f"F1: {sum(f1s)/total_processed:.3f}\n"
        )
    else:
        print("No samples processed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()