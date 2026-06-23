#!/usr/bin/env python3
"""
generate_next_point_gap_db.py
從連續局數序列（點數）統計「上一局點數 → 下一局點數差距」機率，
輸出 data/next_point_gap_db.json。
輸入格式：一個 JSON 陣列，每筆為 {"player_point": 7, "banker_point": 5}。
"""
import json, os
from collections import defaultdict

INPUT_PATH = os.getenv("NEXT_POINT_GAP_INPUT", "data/sequence_3m.json")
OUTPUT_PATH = os.getenv("NEXT_POINT_GAP_DB_PATH", "data/next_point_gap_db.json")

def generate(data):
    trans = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(int)

    for i in range(len(data) - 1):
        curr = data[i]
        nxt = data[i + 1]
        curr_key = f"P{curr['player_point']}_B{curr['banker_point']}"
        gap = abs(nxt['player_point'] - nxt['banker_point'])
        trans[curr_key][f"gap_{gap}"] += 1
        counts[curr_key] += 1

    output = {}
    for key, gaps in trans.items():
        total = counts[key]
        output[key] = {f"gap_{g}": round(cnt / total, 6) for g, cnt in sorted(gaps.items())}
        output[key]["total_samples"] = total

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Generated {OUTPUT_PATH} with {len(output)} keys.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found. Please provide sequence data.")
    else:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        generate(data)
