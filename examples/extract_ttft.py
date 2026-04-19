import json
import argparse

from cv2 import line



def extract_ttft(traj_path: str):
    ret = []
    with open(traj_path, 'r') as f:
        for line in f:
            try:
                turn_list = json.loads(line)
            except Exception as e:
                continue
            assert len(turn_list) == 2
            ttft = turn_list[1]["ttft"]
            ret.append(ttft)
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True, help="Path to the trajectory file.")
    args = parser.parse_args()

    ttft_list = extract_ttft(args.traj_path)
    mean_ttft = sum(ttft_list) / len(ttft_list)
    print(
        f"Extracted TTFTs: {ttft_list}\n"
        f"Mean TTFT: {mean_ttft:.4f} seconds"
    )