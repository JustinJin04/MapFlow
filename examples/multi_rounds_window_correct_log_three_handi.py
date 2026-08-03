import argparse
import asyncio
import copy
import json
import os
import time
import traceback
import random
from http import HTTPStatus
from jinja2.sandbox import ImmutableSandboxedEnvironment
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer


SERVER_URLS = {
    # "large": "http://localhost:8000/v1/chat/completions",
    # "small": "http://localhost:8001/v1/chat/completions",
}

class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


class ServerResponse(dict):
    pass


def write_log(args, message: str):
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def write_trajectory_request_records(args, trajectory_result: dict):
    write_log(
        args,
        json.dumps(trajectory_result["request_records"], ensure_ascii=False),
    )


def safe_copy_messages(messages):
    return [copy.deepcopy(m) for m in messages]


def nanosec_to_millisec(value: float) -> float:
    return value / 1_000_000.0


def get_token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def judge_messages(messages: list[dict]) -> bool:
    
    if messages[0]["role"] != "system":
        return False
    
    is_user = True
    for msg in messages[1:]:
        if (msg["role"] == "user") != is_user:
            return False
        is_user = not is_user
        content = msg.get("content")
        if not content or not isinstance(content, str):
            return False
    return True


def remap_route_label(original_route_label: str) -> str:
    if original_route_label == "small":
        return random.choice(["small", "medium"])
    if original_route_label == "large":
        return "large"
    raise ValueError(f"Unsupported route label in trace: {original_route_label}")

def build_request_plan_for_trajectory(idx: int, data: dict, tokenizer, args) -> dict | None:
    messages = data["messages"]
    routes = data.get("routes", [])

    if not judge_messages(messages):
        print(f"{Color.YELLOW}Trajectory {idx} skipped due to message format issues.{Color.RESET}")
        return None

    current_context = []
    window = []
    user_turn_idx = 0

    for msg in messages:
        current_context.append(msg)

        if msg.get("role") != "user":
            continue

        token_list = tokenizer.apply_chat_template(
            current_context,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_len = len(token_list)

        # 优化点 1: 上下文是单调递增的。如果当前长度已经超过 max_seqlen，
        # 意味着后续所有的 user turn 也会超长，直接跳出循环，放弃这个 trajectory
        if prompt_len > args.max_seqlen:
            break

        original_route_label = routes[user_turn_idx] if user_turn_idx < len(routes) else "unknown"

        # 判断当前轮次是否在指定长度范围内
        if args.min_seqlen <= prompt_len <= args.max_seqlen:
            window.append({
                "user_turn_idx": user_turn_idx,
                "messages": safe_copy_messages(current_context),
                "prompt_len_estimate": prompt_len,
                "original_route_label": original_route_label
            })
        else:
            # 条件是“连续的三个轮次”，如果当前轮次（比如太短）不符合，必须清空前面的累积
            window.clear()

        user_turn_idx += 1

        # 优化点 2: 只要收集齐 3 个连续满足条件的轮次，立刻组装请求并返回，停止解析剩余的长消息
        if len(window) == 3:
            requests = []
            target_labels = ["large", "medium", "small"]

            for req_idx, (turn, label) in enumerate(zip(window, target_labels)):
                requests.append({
                    "trajectory_id": idx,
                    "request_idx_in_trajectory": req_idx,
                    "user_turn_idx": turn["user_turn_idx"],
                    "original_route_label": turn["original_route_label"],
                    "target_server_label": label,  # 强制设置为大、中、小
                    "messages": turn["messages"],
                    "prompt_len_estimate": turn["prompt_len_estimate"],
                })

            return {
                "trajectory_id": idx,
                "requests": requests,
            }

    # 如果提前 break，或者一直遍历完都没凑齐连续的 3 轮，则返回 None 过滤掉该样本
    return None

def build_benchmark_dataset(full_ds, tokenizer, args):
    filtered_ds = []
    for idx, data in enumerate(full_ds):
        planned = build_request_plan_for_trajectory(idx, data, tokenizer, args)
        if planned is not None:
            filtered_ds.append(planned)

    n = min(args.num_total_inputs, len(filtered_ds))
    benchmark_ds = filtered_ds[:n]
    return filtered_ds, benchmark_ds


def build_expanded_trajectory_stream(ds, num_rounds: int):
    expanded = []
    num_per_round = len(ds)

    for round_idx in range(num_rounds):
        for local_idx, trajectory_plan in enumerate(ds):
            global_idx = round_idx * num_per_round + local_idx
            expanded.append(
                {
                    "round_idx": round_idx,
                    "local_idx": local_idx,
                    "global_idx": global_idx,
                    "trajectory_plan": trajectory_plan,
                }
            )
    return expanded


async def exponential_backoff_sleep(
    attempt_cnt: int,
    base_rate: float = 1.0,
    backoff_factor: float = 2.0,
    jitter_fraction: float = 0.10,
    verbose: bool = False,
) -> None:
    backoff_delay = base_rate * (backoff_factor**attempt_cnt)
    jittered_delay = backoff_delay * (
        1 + np.random.uniform(-jitter_fraction, jitter_fraction)
    )

    if verbose:
        print(f"Backoff for {jittered_delay:.3f} seconds...")

    await asyncio.sleep(jittered_delay)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return ""


async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    stream: bool = True,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
) -> ServerResponse:
    if not messages:
        return ServerResponse(valid=False, error="No valid messages after sanitization.")

    if "Ministral" in model:
        payload = {
            "model": model,
            "messages": messages,
            "seed": 0,
            "temperature": 0.0,
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "seed": 0,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": False}

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    payload["ignore_eos"] = True

    headers = {"Content-Type": "application/json"}

    if max_tokens is not None:
        token_based_timeout = int(max_tokens * 0.2)
        if token_based_timeout > timeout_sec:
            timeout_sec = token_based_timeout
            print(
                f"Using timeout of {timeout_sec}s based on max_tokens {max_tokens}"
            )
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft = None
    chunk_delay = []
    latency = None
    first_chunk = ""
    generated_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    start_time = time.perf_counter_ns()
    most_recent_timestamp = start_time

    async with session.post(
        url=chat_url, json=payload, headers=headers, timeout=timeout
    ) as response:
        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += _extract_text(message["content"])
                else:
                    timestamp = time.perf_counter_ns()
                    data = json.loads(chunk)

                    choices = data.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        delta_text = _extract_text(delta.get("content", None))
                        if delta_text:
                            if ttft is None:
                                first_token_time = time.perf_counter_ns()
                                ttft = first_token_time - start_time
                                first_chunk = delta_text
                            else:
                                chunk_delay.append(timestamp - most_recent_timestamp)

                            generated_text += delta_text

                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                        total_tokens = usage.get("total_tokens", total_tokens)

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            print(
                f"{Color.YELLOW}Received HTTP status {http_status.value} "
                f"({http_status.phrase}): {content}{Color.RESET}"
            )

    if latency is None:
        latency = -1.0
        if valid_response:
            latency = time.perf_counter_ns() - start_time

    if ttft is None:
        ttft = latency

    tpot = float(np.mean(chunk_delay)) if len(chunk_delay) > 0 else 0.0
    num_chunks = len(chunk_delay)

    return ServerResponse(
        valid=valid_response,
        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
        tpot_ms=nanosec_to_millisec(tpot),
        latency_ms=nanosec_to_millisec(latency),
        start_time_ms=nanosec_to_millisec(start_time),
        first_chunk=first_chunk,
        content=generated_text,
        num_chunks=num_chunks,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def run_one_request(
    session: aiohttp.ClientSession,
    request_plan: dict,
    tokenizer: AutoTokenizer,
    args,
    benchmark_zero: float,
):
    target_server_label = request_plan["target_server_label"]
    chat_url = SERVER_URLS.get(target_server_label)
    if chat_url is None:
        raise ValueError(
            f"Trajectory {request_plan['trajectory_id']}: unknown server label: {target_server_label}"
        )

    model_by_route = {
        "large": args.large_model,
        "medium": args.medium_model,
        "small": args.small_model,
    }
    max_tokens_by_route = {
        "large": args.max_new_tokens_large,
        "medium": args.max_new_tokens_medium,
        "small": args.max_new_tokens_small,
    }

    response = await send_request(
        session=session,
        messages=request_plan["messages"],
        chat_url=chat_url,
        model=model_by_route[target_server_label],
        stream=True,
        max_tokens=max_tokens_by_route[target_server_label],
        timeout_sec=args.request_timeout_sec,
    )

    if response["valid"] is False:
        return None

    first_chunk_tokens = get_token_count(tokenizer, response["first_chunk"])
    output_content = response["content"]
    output_num_tokens = get_token_count(tokenizer, output_content)

    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        decode_ms = response["latency_ms"] - response["ttft_ms"]
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        tpot_ms = decode_ms / decode_num_tokens
    else:
        tpot_ms = 0.0

    if first_chunk_tokens > 1:
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response["ttft_ms"] - delta_ms)
    else:
        ttft_ms = response["ttft_ms"]

    prompt_tokens = response["prompt_tokens"] or request_plan["prompt_len_estimate"]
    completion_tokens = response["completion_tokens"] or output_num_tokens
    total_tokens = response["total_tokens"] or (prompt_tokens + completion_tokens)

    return {
        "trajectory_id": request_plan["trajectory_id"],
        "request_idx_in_trajectory": request_plan["request_idx_in_trajectory"],
        "user_turn_idx": request_plan["user_turn_idx"],
        "original_route_label": request_plan.get("original_route_label", target_server_label),
        "target_server_label": target_server_label,
        "prompt_len_estimate": request_plan["prompt_len_estimate"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft": ttft_ms / 1000.0,
        "raw_ttft": response["ttft_ms"] / 1000.0,
        "tpot_estimate": tpot_ms / 1000.0,
        "first_chunk_tokens": first_chunk_tokens,
        "output_tokens_estimate": output_num_tokens,
        "response_num_chunks": response["num_chunks"],
        "request_start_ts": (response["start_time_ms"] / 1000.0) - benchmark_zero,
        "request_end_ts": ((response["start_time_ms"] + response["latency_ms"]) / 1000.0) - benchmark_zero,
        "request_latency": response["latency_ms"] / 1000.0,
        "response_text": output_content,
    }


async def eval_one(
    session: aiohttp.ClientSession,
    global_idx: int,
    trajectory_plan: dict,
    tokenizer: AutoTokenizer,
    args,
    benchmark_zero: float,
) -> dict:
    request_records = []
    prev_server_label = None
    has_small_processed = False
    prev_user_turn_idx = None
    failed = False
    failure_reason = None

    for request_plan in trajectory_plan["requests"]:
        if prev_user_turn_idx is None:
            gap_turns = request_plan["user_turn_idx"]
        else:
            gap_turns = request_plan["user_turn_idx"] - prev_user_turn_idx

        if gap_turns < 0:
            raise ValueError(
                f"Trajectory {global_idx}: non-monotonic user_turn_idx detected."
            )

        if gap_turns > 0 and args.turn_delay > 0:
            await asyncio.sleep(gap_turns * args.turn_delay)

        success = False
        had_exception = False
        record = None
        for attempt_cnt in range(args.max_retries + 1):
            try:
                had_exception = False
                record = await run_one_request(
                    session=session,
                    request_plan=request_plan,
                    tokenizer=tokenizer,
                    args=args,
                    benchmark_zero=benchmark_zero,
                )
                if record is not None:
                    success = True
                    break
                else:
                    print(
                        f"{Color.YELLOW}Trajectory {global_idx} - Request rejected "
                        f"(req: {request_plan['request_idx_in_trajectory']}){Color.RESET}"
                    )
            except asyncio.exceptions.TimeoutError:
                had_exception = True
                failure_reason = (
                    f"Timeout during trajectory {global_idx}, req {request_plan['request_idx_in_trajectory']}. "
                    f"Base timeout is {args.request_timeout_sec}s; effective timeout may be longer based on max_tokens."
                )
                print(f"{Color.RED}{failure_reason}{Color.RESET}")
            except Exception as exc:
                had_exception = True
                failure_reason = (
                    f"Exception during trajectory {global_idx}, req {request_plan['request_idx_in_trajectory']}: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                print(f"{Color.RED}{failure_reason}{Color.RESET}")
                if args.verbose:
                    traceback.print_exc()

            if not success and attempt_cnt < args.max_retries:
                await exponential_backoff_sleep(attempt_cnt, verbose=args.verbose)

        if not success:
            failed = True
            if had_exception:
                break
            else:
                failure_reason = failure_reason or "request failed without exception"
                break

        is_small = record["target_server_label"] == "small"
        is_switching = is_small and prev_server_label == "large"
        is_first_switching = is_switching and (not has_small_processed)
        is_subsequent_switching = is_switching and has_small_processed

        record["is_small"] = is_small
        record["is_switching"] = is_switching
        record["is_first_switching"] = is_first_switching
        record["is_subsequent_switching"] = is_subsequent_switching
        record["flattened_trajectory_id"] = global_idx

        ttft_str = "N/A" if record["ttft"] is None else f"{record['ttft']:.4f}s"
        print(
            f"Response traj={global_idx}, req={record['request_idx_in_trajectory']} - "
            f"Server: {record['target_server_label']}, "
            f"Prompt Tokens: {record['prompt_tokens']}, "
            f"Completion Tokens: {record['completion_tokens']}, "
            f"Total Tokens: {record['total_tokens']}, "
            f"TTFT: {ttft_str}"
        )

        request_records.append(record)

        if is_small:
            has_small_processed = True

        prev_server_label = record["target_server_label"]
        prev_user_turn_idx = request_plan["user_turn_idx"]

    return {
        "flattened_trajectory_id": global_idx,
        "trajectory_id": trajectory_plan["trajectory_id"],
        "request_records": request_records,
        "failed": failed,
        "failure_reason": failure_reason,
    }


async def benchmark(args, expanded_trajectory_stream, tokenizer, desc="Processing"):
    if args.window_size is not None and args.window_size <= 0:
        raise ValueError("window_size must be positive when provided.")

    all_results = [None for _ in range(len(expanded_trajectory_stream))]
    benchmark_start = time.perf_counter()
    benchmark_zero = benchmark_start

    effective_window_size = (
        len(expanded_trajectory_stream)
        if args.window_size is None
        else min(args.window_size, len(expanded_trajectory_stream))
    )

    running_tasks = set()
    last_trajectory_launch_abs = None
    next_log_index = 0
    pbar = tqdm(total=len(expanded_trajectory_stream), desc=desc)

    async with aiohttp.ClientSession() as session:
        async def run_one_trajectory(expanded_index, expanded_item, actual_start_abs):
            res = await eval_one(
                session=session,
                global_idx=expanded_item["global_idx"],
                trajectory_plan=expanded_item["trajectory_plan"],
                tokenizer=tokenizer,
                args=args,
                benchmark_zero=benchmark_zero,
            )
            res["round_idx"] = expanded_item["round_idx"]
            res["local_idx"] = expanded_item["local_idx"]
            res["trajectory_actual_start_ts"] = actual_start_abs - benchmark_zero
            return expanded_index, res

        def collect_done_tasks(done_tasks):
            nonlocal next_log_index

            for task in done_tasks:
                try:
                    expanded_index, res = task.result()
                except Exception as exc:
                    print(f"{Color.RED}Unhandled task exception: {exc}{Color.RESET}")
                    if args.verbose:
                        traceback.print_exc()
                    continue
                all_results[expanded_index] = res
                pbar.update(1)

            while next_log_index < len(all_results) and all_results[next_log_index] is not None:
                write_trajectory_request_records(args, all_results[next_log_index])
                next_log_index += 1

        for expanded_index, expanded_item in enumerate(expanded_trajectory_stream):
            while True:
                now = time.perf_counter()
                delay_remaining = 0.0
                if last_trajectory_launch_abs is not None and args.trajectory_delay > 0:
                    delay_remaining = max(
                        0.0,
                        last_trajectory_launch_abs + args.trajectory_delay - now,
                    )

                window_ready = len(running_tasks) < effective_window_size
                delay_ready = delay_remaining <= 0.0

                if window_ready and delay_ready:
                    break

                if not window_ready:
                    timeout = delay_remaining if delay_remaining > 0 else None
                    done, pending = await asyncio.wait(
                        running_tasks,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if done:
                        running_tasks.difference_update(done)
                        collect_done_tasks(done)
                else:
                    await asyncio.sleep(delay_remaining)

            actual_start_abs = time.perf_counter()
            task = asyncio.create_task(
                run_one_trajectory(
                    expanded_index=expanded_index,
                    expanded_item=expanded_item,
                    actual_start_abs=actual_start_abs,
                )
            )
            running_tasks.add(task)
            last_trajectory_launch_abs = actual_start_abs

        while running_tasks:
            done, pending = await asyncio.wait(
                running_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            running_tasks.difference_update(done)
            collect_done_tasks(done)

    pbar.close()
    benchmark_end = time.perf_counter()
    return all_results, benchmark_start, benchmark_end


def flatten_request_records(all_results):
    records = []
    for trajectory_result in all_results:
        if trajectory_result is None:
            continue
        records.extend(trajectory_result["request_records"])
    return records


def select_requests_in_window(records, start_ts: float, end_ts: float):
    return [
        r for r in records
        if start_ts <= r["request_start_ts"] < end_ts
    ]


def compute_window_qps(request_count: int, start_ts: float, end_ts: float):
    duration = end_ts - start_ts
    if duration <= 0:
        return None
    return request_count / duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_file", type=str, required=True)
    parser.add_argument("--min_seqlen", type=int, default=0)
    parser.add_argument("--max_seqlen", type=int, default=40960)
    parser.add_argument("--round", type=int, default=1, help="How many full rounds to replay the selected dataset.")

    parser.add_argument(
        "--trajectory_delay",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--turn_delay",
        type=float,
        default=10.0,
        help="Delay in seconds between adjacent user turns inside one trajectory.",
    )
    parser.add_argument(
        "--num_total_inputs",
        type=int,
        default=30,
        help="Number of filtered trajectories to use per round.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Unused in the windowed scheduler. Kept only for CLI backward compatibility.",
    )

    parser.add_argument("--large_model", type=str, required=True)
    parser.add_argument("--medium_model", type=str, required=True)
    parser.add_argument("--small_model", type=str, required=True)
    parser.add_argument("--max_new_tokens_large", type=int, default=256)
    parser.add_argument("--max_new_tokens_medium", type=int, default=128)
    parser.add_argument("--max_new_tokens_small", type=int, default=128)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retry attempts for failed requests."
    )
    parser.add_argument(
        "--request_timeout_sec",
        type=int,
        default=120,
        help="Base timeout for each request; may auto-increase based on max_new_tokens.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose exception printing.",
    )
    parser.add_argument(
        "--large_model_port",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--medium_model_port",
        type=int,
        default=8001,
    )
    parser.add_argument(
        "--small_model_port",
        type=int,
        default=8002,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    global SERVER_URLS
    SERVER_URLS = {
        "large": f"http://localhost:{args.large_model_port}/v1/chat/completions",
        "medium": f"http://localhost:{args.medium_model_port}/v1/chat/completions",
        "small": f"http://localhost:{args.small_model_port}/v1/chat/completions",
    }


    if args.window_size is not None and args.window_size <= 0:
        raise ValueError("--window_size must be a positive integer when provided.")

    with open(args.log_file, "a", encoding="utf-8") as f:
        # write prelog
        f.write(f"args: {json.dumps(vars(args), ensure_ascii=False)}\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.large_model)
    except Exception as exc:
        print(f"error loading tokenizer for model {args.large_model}: {exc}. Fall back to qwen tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("/data/zhan/MapFlow/examples/models/Qwen3-1.7B")
    
    full_ds = []
    with open(args.trace_file, "r", encoding="utf-8") as f:
        for line in f:
            full_ds.append(json.loads(line.strip()))

    print("Pre-tokenizing and filtering requests...")
    filtered_ds, benchmark_ds = build_benchmark_dataset(full_ds, tokenizer, args)

    if len(benchmark_ds) < args.num_total_inputs:
        print(
            f"Warning: only {len(benchmark_ds)} filtered trajectories are available, "
            f"but {args.num_total_inputs} were requested per round."
        )

    if len(benchmark_ds) == 0:
        print("No trajectories remain after filtering. Please adjust min_seqlen/max_seqlen or input data.")
        return

    expanded_trajectory_stream = build_expanded_trajectory_stream(
        benchmark_ds,
        num_rounds=args.round,
    )

    effective_window_size = (
        len(expanded_trajectory_stream)
        if args.window_size is None
        else min(args.window_size, len(expanded_trajectory_stream))
    )

    total_flattened_trajectories = len(expanded_trajectory_stream)
    total_requests_per_round = sum(len(x["requests"]) for x in benchmark_ds)

    print(
        f"=== Starting Benchmark "
        f"({len(benchmark_ds)} filtered trajectories/round x {args.round} rounds = {total_flattened_trajectories} trajectories total, "
        f"Requests/round: {total_requests_per_round}, "
        f"Window Size: {effective_window_size}, "
        f"Min Actual Trajectory Launch Gap: {args.trajectory_delay}s) ==="
    )
    print(f"Log file will contain only one JSON line per completed trajectory: {args.log_file}")

    asyncio.run(
        benchmark(
            args,
            expanded_trajectory_stream,
            tokenizer,
            desc="Benchmarking",
        )
    )
    print("Benchmark finished.\n")


if __name__ == "__main__":
    main()
