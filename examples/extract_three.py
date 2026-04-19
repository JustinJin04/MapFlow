import os
import glob
import json
import argparse

def calculate_trajectory_time(filepath, start_id, end_id):

    min_start_ts = float('inf')
    max_end_ts = float('-inf')
    
    found_start = False
    found_end = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            # 忽略空行或明显的非 JSON 数组行（例如第一行的 args）
            if not line or not line.startswith('['):
                continue
            
            try:
                # 尝试解析当前行的 JSON 数组
                data = json.loads(line)
                for item in data:
                    if not isinstance(item, dict):
                        continue
                        
                    traj_id = item.get("trajectory_id")
                    
                    # 匹配起始 trajectory_id，寻找最早的 start_ts
                    if i == start_id:
                        start_ts = item.get("request_start_ts")
                        if start_ts is not None:
                            min_start_ts = min(min_start_ts, start_ts)
                            found_start = True
                            
                    # 匹配终止 trajectory_id，寻找最晚的 end_ts
                    if i == end_id:
                        end_ts = item.get("request_end_ts")
                        if end_ts is not None:
                            max_end_ts = max(max_end_ts, end_ts)
                            found_end = True
                            
            except json.JSONDecodeError:
                # 忽略无法解析为 JSON 的行
                continue
    
    # 输出结果
    print(f"文件: {os.path.basename(filepath)}")
    if found_start and found_end:
        total_time = max_end_ts - min_start_ts
        print(f"  -> 起始 Trajectory ID [{start_id}] 最早开始时间: {min_start_ts:.4f}s")
        print(f"  -> 终止 Trajectory ID [{end_id}] 最晚结束时间: {max_end_ts:.4f}s")
        print(f"  -> 总耗时: {total_time:.4f} 秒")
    else:
        missing = []
        if not found_start: missing.append(str(start_id))
        if not found_end: missing.append(str(end_id))
        print(f"  -> 警告: 未能在该文件中找到 Trajectory ID: {', '.join(missing)}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--start_id", type=int, default=1, help="起始 trajectory_id")
    parser.add_argument("--end_id", type=int, default=10, help="终止 trajectory_id")

    args = parser.parse_args()
    calculate_trajectory_time(args.file_path, args.start_id, args.end_id)

if __name__ == "__main__":
    main()
