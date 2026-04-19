import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def extract_records_from_new_log(log_path: Path, start_traj_id: int = None, end_traj_id: int = None):
    """
    解析新的 log 格式：
    第一行可能是 args: {...}
    后续每一行是一个 json list，代表一个 trajectory 的所有 turns。
    提取出 flattened_trajectory_id 在 [start_traj_id, end_traj_id] 区间内的 records。
    """
    records = []
    
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # 过滤掉空行和以 args: 开头的行
            if not line or line.startswith("args:"):
                continue
            
            try:
                traj_records = json.loads(line)
                if not isinstance(traj_records, list) or not traj_records:
                    continue
                
                # 获取当前 trajectory 的 ID
                traj_id = traj_records[0].get("flattened_trajectory_id", traj_records[0].get("trajectory_id"))
                if traj_id is None:
                    continue
                
                # 检查是否在给定的区间内
                if start_traj_id is not None and traj_id < start_traj_id:
                    continue
                if end_traj_id is not None and traj_id > end_traj_id:
                    continue
                    
                records.extend(traj_records)
                
            except json.JSONDecodeError:
                continue
                
    return records


def compute_metrics(
    records,
    traj_field="flattened_trajectory_id",
    fallback_traj_field="trajectory_id",
    turn_idx_field="request_idx_in_trajectory",
    latency_field="request_latency",
    ttft_field="ttft",
    is_small_field="is_small",
    label_field="target_server_label",
    small_label="small",
):
    """
    计算:
      - Window Small Model QPS (基于区间内的 request_start_ts 和 request_end_ts 计算)
      - 剩下的小模型 ttft 的 mean, p50, p90, p99 (排除首次 small 的 ttft)
      - t1: large -> small 的 latency mean, p50, p90, p99，排除首次 small
      - t2: small -> small 的平均 latency
      - p2: P(curr=small | prev=small)
    """
    if not isinstance(records, list):
        raise ValueError("Extracted JSON is not a list of request records")

    def get_traj_id(r):
        return r.get(traj_field, r.get(fallback_traj_field))

    def get_turn_idx(r):
        return int(r.get(turn_idx_field, 0))

    def get_latency(r):
        return float(r.get(latency_field, 0))
        
    def get_ttft(r):
        return float(r.get(ttft_field, 0))

    def is_small(r):
        if is_small_field in r and r.get(is_small_field) is not None:
            return bool(r.get(is_small_field))
        return r.get(label_field) == small_label

    # --- 1) 计算 Window QPS (Small Model) ---
    small_start_times = []
    small_end_times = []
    for r in records:
        if is_small(r):
            if r.get("request_start_ts") is not None:
                small_start_times.append(float(r["request_start_ts"]))
            if r.get("request_end_ts") is not None:
                small_end_times.append(float(r["request_end_ts"]))

    if small_start_times and small_end_times:
        min_ts = min(small_start_times)
        max_ts = max(small_end_times)
        duration = max_ts - min_ts
        small_qps = len(small_start_times) / duration if duration > 0 else 0.0
    else:
        small_qps = None

    # --- 2) 按 trajectory 分组进行延迟指标统计 ---
    traj_map = defaultdict(list)
    for r in records:
        if not isinstance(r, dict):
            continue
        traj_id = get_traj_id(r)
        if traj_id is None:
            continue
        traj_map[traj_id].append(r)

    remaining_small_ttfts = []
    t1_samples = []
    t2_samples = []
    p2_num = 0
    p2_den = 0

    for traj_id, traj in traj_map.items():
        traj = sorted(traj, key=get_turn_idx)
        if not traj:
            continue

        # 找到该 trajectory 中 small 首次出现的位置
        first_small_idx = None
        # for i, turn in enumerate(traj):
        #     if is_small(turn):
        #         first_small_idx = i
        #         break
                
        # 统计过滤后的 ttft（排除首次被 route 到小模型的 turn）
        for i, turn in enumerate(traj):
            if is_small(turn) and i != first_small_idx:
                remaining_small_ttfts.append(get_ttft(turn))

        if len(traj) < 2:
            continue

        # 枚举相邻转移 prev -> curr 计算 t1, t2, p2
        for i in range(1, len(traj)):
            prev_turn = traj[i - 1]
            curr_turn = traj[i]

            prev_small = is_small(prev_turn)
            curr_small = is_small(curr_turn)

            # t1: large -> small，且排除首次 small
            if (not prev_small) and curr_small:
                if first_small_idx is not None and i == first_small_idx:
                    continue
                t1_samples.append(get_ttft(curr_turn))

            # t2: small -> small
            elif prev_small and curr_small:
                t2_samples.append(get_ttft(curr_turn))

            # p2: P(curr=small | prev=small)
            if prev_small:
                p2_den += 1
                if curr_small:
                    p2_num += 1

    # 计算 ttft 统计信息
    # print(f"num_remaining_small_ttfts: {len(remaining_small_ttfts)}")
    if remaining_small_ttfts:
        ttft_mean = np.mean(remaining_small_ttfts)
        ttft_p50 = np.percentile(remaining_small_ttfts, 50)
        ttft_p90 = np.percentile(remaining_small_ttfts, 90)
        ttft_p99 = np.percentile(remaining_small_ttfts, 99)
    else:
        ttft_mean = ttft_p50 = ttft_p90 = ttft_p99 = None

    # 计算 t1 统计信息
    # print(f"num_t1_samples: {len(t1_samples)}")
    if t1_samples:
        t1_mean = np.mean(t1_samples)
        t1_p50 = np.percentile(t1_samples, 50)
        t1_p90 = np.percentile(t1_samples, 90)
        t1_p99 = np.percentile(t1_samples, 99)
    else:
        t1_mean = t1_p50 = t1_p90 = t1_p99 = None

    return {
        "window_small_qps": small_qps,
        "ttft_stats": {
            "mean": ttft_mean,
            "p50": ttft_p50,
            "p90": ttft_p90,
            "p99": ttft_p99,
            "count": len(remaining_small_ttfts)
        },
        "t1_stats": {
            "mean": t1_mean,
            "p50": t1_p50,
            "p90": t1_p90,
            "p99": t1_p99,
            "count": len(t1_samples)
        },
        "t2": sum(t2_samples) / len(t2_samples) if t2_samples else None,
        "p2": (p2_num / p2_den) if p2_den > 0 else None,
        "counts": {
            "num_trajectories": len(traj_map),
            "num_t1_samples": len(t1_samples),
            "num_t2_samples": len(t2_samples),
            "p2_denominator": p2_den,
            "p2_numerator": p2_num,
        },
    }


def process_logs_in_directory(file_path, start_traj_id=None, end_traj_id=None):

    # 打印表头：新增 T1 的分位数
    print(
        "Small Model QPS\t"
        "TTFT Mean\tTTFT P50\tTTFT P90\tTTFT P99\t"
        "T1 Mean\tT1 P50\tT1 P90\tT1 P99\t"
        "T2 Mean\tP2\tSelected Trajs\tT1 Samples\tT2 Samples\tP2 Den"
    )



    try:
        # 基于你指定的 flattened_trajectory_id 区间提取记录
        file_path = Path(file_path)
        records = extract_records_from_new_log(file_path, start_traj_id, end_traj_id)
        # print(records)
        if not records:
            print(f"N/A\tNo records found in range [{start_traj_id}, {end_traj_id}]")
            return
        stats = compute_metrics(records)

        # 格式化输出数据
        qps_str = f"{stats['window_small_qps']:.4f}" if stats['window_small_qps'] is not None else "N/A"
        
        ttft = stats["ttft_stats"]
        ttft_mean_str = f"{ttft['mean']:.6f}" if ttft['mean'] is not None else "N/A"
        ttft_p50_str  = f"{ttft['p50']:.6f}" if ttft['p50'] is not None else "N/A"
        ttft_p90_str  = f"{ttft['p90']:.6f}" if ttft['p90'] is not None else "N/A"
        ttft_p99_str  = f"{ttft['p99']:.6f}" if ttft['p99'] is not None else "N/A"
        
        # 格式化 T1 数据
        t1 = stats["t1_stats"]
        t1_mean_str = f"{t1['mean']:.6f}" if t1['mean'] is not None else "N/A"
        t1_p50_str  = f"{t1['p50']:.6f}" if t1['p50'] is not None else "N/A"
        t1_p90_str  = f"{t1['p90']:.6f}" if t1['p90'] is not None else "N/A"
        t1_p99_str  = f"{t1['p99']:.6f}" if t1['p99'] is not None else "N/A"
        
        t2_str = f"{stats['t2']:.6f}" if stats['t2'] is not None else "N/A"
        p2_str = f"{stats['p2']:.6f}" if stats['p2'] is not None else "N/A"
        
        num_selected_traj = stats["counts"]["num_trajectories"]
        num_t1_samples = stats["counts"]["num_t1_samples"]
        num_t2_samples = stats["counts"]["num_t2_samples"]
        p2_den = stats["counts"]["p2_denominator"]

        print(
            f"{qps_str}\t"
            f"{ttft_mean_str}\t{ttft_p50_str}\t{ttft_p90_str}\t{ttft_p99_str}\t"
            f"{t1_mean_str}\t{t1_p50_str}\t{t1_p90_str}\t{t1_p99_str}\t"
            f"{t2_str}\t{p2_str}\t"
            f"{num_selected_traj}\t{num_t1_samples}\t{num_t2_samples}\t{p2_den}"
        )

    except Exception as e:
        # 异常时输出占位符，保持列数一致 (16列)
        print(f"N/A\tError: {str(e)}" + "\t-" * 13)



def main():
    parser = argparse.ArgumentParser(description="Extract metrics from E2E log files in a directory.")
    parser.add_argument("--file_path", type=str, required=True, help="Directory containing the log files.")
    parser.add_argument("--start_traj_id", type=int, default=None, help="Start of trajectory ID range to include.")
    parser.add_argument("--end_traj_id", type=int, default=None, help="End of trajectory ID range to include.")
    args = parser.parse_args()

    process_logs_in_directory(args.file_path, args.start_traj_id, args.end_traj_id)

if __name__ == "__main__":
    main()