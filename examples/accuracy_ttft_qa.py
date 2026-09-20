import argparse
import re
import json
import os
import asyncio
import time
from typing import Dict, Tuple, Any, List
from openai import AsyncOpenAI  # 使用异步客户端
from transformers import AutoTokenizer
from metrics import get_qa_em, get_qa_f1
from data import get_dataset
from tqdm.asyncio import tqdm  # 推荐安装 tqdm: pip install tqdm

SERVERS = {}

# --- 新增：用于检测服务器状态的函数 ---
async def ping_server(client: AsyncOpenAI) -> bool:
    """发送获取models的HTTP请求以测试服务器是否存活 (类似 ping)"""
    try:
        # 设定2秒超时，防止服务未完全启动导致的请求挂起
        await asyncio.wait_for(client.models.list(), timeout=2.0)
        return True
    except Exception:
        return False

async def ensure_servers_online(sender_client: AsyncOpenAI, receiver_client: AsyncOpenAI):
    """依次向sender和receiver发送请求，如果有未回复的则等待5秒并重试"""
    print("Checking if sender and receiver servers are online...")
    while True:
        # 依次发送类似ping的http请求
        sender_ok = await ping_server(sender_client)
        receiver_ok = await ping_server(receiver_client)
        
        if sender_ok and receiver_ok:
            print("Both Sender and Receiver servers are online! Preparing to start evaluation...")
            break
            
        print(f"Status - Sender: {'Online' if sender_ok else 'Offline'}, Receiver: {'Online' if receiver_ok else 'Offline'}.")
        print("Waiting 5 seconds before retrying...")
        await asyncio.sleep(5)
# --------------------------------------

# --- 新增：用于流式获取 Receiver 的响应并测算 TTFT ---
async def fetch_receiver_stream(client, example, args) -> Tuple[str, float]:
    start_time = time.perf_counter() # 使用高精度计时器
    stream = await client.chat.completions.create(
        model=args.receive_model,
        messages=example["receiver_messages"],
        max_tokens=args.max_new_tokens,
        stream=True,  # 开启流式输出以测算 TTFT
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} \
        if "Ministral" not in args.receive_model else {}
    )
    
    ttft = None
    full_text = ""
    
    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                # 记录第一次接收到有效内容的时间
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                full_text += delta_content
                
    # 异常处理：如果没有获取到任何有效 token，默认计算总时间
    if ttft is None:
        ttft = time.perf_counter() - start_time
        
    return full_text, ttft
# --------------------------------------------------------

async def process_batch(batch_examples: List[Dict[str, Any]], args, batch_id: int) -> List[Tuple[str, float, float, str, float]]:
    """
    处理一个 Batch：
    1. 并发发送所有 Sender 请求 -> 等待全部完成
    2. 并发发送所有 Receiver 请求 -> 以流式方式接收，并记录 TTFT
    3. 计算指标
    """
    sender_client = SERVERS["sender"]
    receiver_client = SERVERS["receiver"]
    # ---------------- Step 1: Sender (Prefill) ----------------
    # 构造 Sender 的所有并发任务
    sender_tasks = []
    for example in batch_examples:
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

    await asyncio.sleep(args.interval)
    print(f"Complete Sender step and wait for {args.interval} seconds.")

    # ---------------- Step 2: Receiver (Decode) ----------------
    # 构造 Receiver 的所有并发流式任务
    receiver_tasks = []
    for example in batch_examples:
        receiver_tasks.append(fetch_receiver_stream(receiver_client, example, args))
    
    # 等待 Receiver 这一批全部完成并获取结果(full_text, ttft)
    receiver_responses = await asyncio.gather(*receiver_tasks)

    # wait for retrieve msg processed by server
    if batch_id <= 3:
        await asyncio.sleep(10)
    else:
        await asyncio.sleep(1)

    # ---------------- Step 3: Metrics ----------------
    results = []
    for i, (pred_raw, ttft) in enumerate(receiver_responses):
        example = batch_examples[i]
        gold = example["ans"]
        
        pred = pred_raw
            
        em = get_qa_em(pred, gold)
        f1 = get_qa_f1(pred, gold)
        
        # 将 ttft 加入到返回结果中
        results.append((pred, em, f1, gold, ttft))
        
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

    parser.add_argument("--interval", type=float ,default=6)

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

    # --- 新增：在开始执行评测任务前先确保sender和receiver同时在线 ---
    await ensure_servers_online(SERVERS["sender"], SERVERS["receiver"])
    # -----------------------------------------------------------

    write_log(args, f"Loading dataset from {args.eval_data}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.send_model)
    except Exception as exc:
        print(f"error loading tokenizer for model {args.send_model}: {exc}. Fall back to qwen tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("/data/zhan/MapFlow/examples/models/Qwen3-1.7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    ds = get_dataset(args.dataset, args.eval_data, tokenizer, max_length=args.max_length, num_max_examples=args.num_max_examples)

    write_log(args, f"Loading send model from {args.send_model}\nReceive model from {args.receive_model}")
    print(f"Dataset size: {len(ds)}")

    ems, f1s, ttfts = [], [], []  # 新增 ttfts 列表
    total_processed = 0

    # 按 Batch 处理数据
    for i in tqdm(range(0, len(ds), args.batch_size), desc="Processing Batches"):
        # 获取当前 Batch 的数据切片
        batch_data = ds[i : i + args.batch_size]
        
        # 异步处理该 Batch
        batch_results = await process_batch(batch_data, args, i // args.batch_size)
        
        # 记录结果
        for j, (pred, em, f1, gold, ttft) in enumerate(batch_results):
            global_idx = i + j + 1
            ems.append(em)
            f1s.append(f1)
            ttfts.append(ttft)
            write_log(args, f"[{global_idx:4d}/{len(ds)}] EM={int(em)} F1={f1:.3f} TTFT={ttft:.4f}s | Pred: {pred} | Gold: {gold}")
        
        total_processed += len(batch_results)

    # 最终结果统计
    if total_processed > 0:
        avg_em = sum(ems) / total_processed
        avg_f1 = sum(f1s) / total_processed
        avg_ttft = sum(ttfts[3:]) / (total_processed - 3)  # 计算平均 TTFT. 不考虑前三个
        
        result_message = (
            f"Send Model: {args.send_model}\n"
            f"Receive Model: {args.receive_model}\n"
            f"Samples: {total_processed}\n"
            f"EM: {avg_em:.3f}\n"
            f"F1: {avg_f1:.3f}\n"
            f"Average TTFT: {avg_ttft:.4f} s\n"
        )
        write_result(args, result_message)
        print("\n=== Final Results ===")
        print(result_message)
    else:
        print("No samples processed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()