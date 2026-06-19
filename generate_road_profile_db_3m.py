#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_road_profile_db_3m.py

產生 data/road_profile_db_3m.json，用於「無記憶牌路資料庫比對」。
它不是記錄用戶輸入紀錄，而是自行模擬大量百家樂連續牌局，統計：
- 當前點數 P{閒}_B{莊}
- 當前補牌情境 SC:...
- 資料庫路段型態 ROAD:單跳/雙跳/長龍/一房兩廳/同點重複
- 下一局莊/閒機率

先測試：
  ROAD_PROFILE_TOTAL_ROUNDS=50000 python generate_road_profile_db_3m.py
正式版：
  ROAD_PROFILE_TOTAL_ROUNDS=3000000 python generate_road_profile_db_3m.py

注意：正式 300 萬局會花時間；跑完後要把 data/road_profile_db_3m.json commit 到 GitHub。
"""

import json
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Any

OUTPUT_PATH = Path(os.getenv("ROAD_PROFILE_OUTPUT_PATH", "data/road_profile_db_3m.json"))
TOTAL_ROUNDS = int(os.getenv("ROAD_PROFILE_TOTAL_ROUNDS", "3000000"))
DECKS = int(os.getenv("DECKS", "8"))
SEED = int(os.getenv("ROAD_PROFILE_SEED", "42"))
SHOE_CUT_REMAINING = int(os.getenv("SHOE_CUT_REMAINING", "52"))
SAVE_EVERY = int(os.getenv("ROAD_PROFILE_SAVE_EVERY", "50000"))

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T"]
VALUES = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 0}
RANK_COUNTS_PER_DECK = {r: (16 if r == "T" else 4) for r in RANKS}

SCENARIO_ZH = {
    "NONE_DRAW": "雙方不補",
    "PLAYER_DRAW": "閒補莊不補",
    "BANKER_DRAW": "莊補閒不補",
    "BOTH_DRAW": "莊閒皆補",
}
ROAD_PROFILE_ZH = {
    "SINGLE_JUMP": "單跳",
    "DOUBLE_JUMP": "雙跳",
    "ONE_ROOM_TWO_HALLS": "一房兩廳",
    "LONG_BANKER": "莊長龍",
    "LONG_PLAYER": "閒長龍",
    "LONG_DRAGON": "長龍",
    "SAME_POINT_REPEAT": "同點重複",
    "NEUTRAL": "中性路段",
}


def fresh_shoe() -> Dict[str, int]:
    return {r: RANK_COUNTS_PER_DECK[r] * DECKS for r in RANKS}


def shoe_remaining(shoe: Dict[str, int]) -> int:
    return sum(shoe.values())


def draw_one(rng: random.Random, shoe: Dict[str, int]) -> str:
    n = shoe_remaining(shoe)
    if n <= 0:
        raise RuntimeError("shoe empty")
    x = rng.randrange(n)
    acc = 0
    for r in RANKS:
        c = shoe.get(r, 0)
        if c <= 0:
            continue
        acc += c
        if x < acc:
            shoe[r] -= 1
            return r
    r = RANKS[-1]
    shoe[r] -= 1
    return r


def total(cards: List[str]) -> int:
    return sum(VALUES[c] for c in cards) % 10


def banker_should_draw(banker_two_total: int, player_third_value=None, player_drew=False) -> bool:
    b = banker_two_total
    if not player_drew:
        return b <= 5
    pt = int(player_third_value)
    if b <= 2:
        return True
    if b == 3:
        return pt != 8
    if b == 4:
        return 2 <= pt <= 7
    if b == 5:
        return 4 <= pt <= 7
    if b == 6:
        return pt in (6, 7)
    return False


def deal_round(rng: random.Random, shoe: Dict[str, int]) -> Dict[str, Any]:
    p = [draw_one(rng, shoe), draw_one(rng, shoe)]
    b = [draw_one(rng, shoe), draw_one(rng, shoe)]
    p2 = total(p)
    b2 = total(b)
    p_drew = False
    b_drew = False

    if p2 in (8, 9) or b2 in (8, 9):
        pass
    else:
        if p2 <= 5:
            p.append(draw_one(rng, shoe))
            p_drew = True
            p3 = VALUES[p[2]]
        else:
            p3 = None
        if banker_should_draw(b2, p3, p_drew):
            b.append(draw_one(rng, shoe))
            b_drew = True

    pp = total(p)
    bp = total(b)
    if bp > pp:
        winner = "B"
    elif pp > bp:
        winner = "P"
    else:
        winner = "T"

    if p_drew and b_drew:
        sc = "BOTH_DRAW"
    elif p_drew and not b_drew:
        sc = "PLAYER_DRAW"
    elif b_drew and not p_drew:
        sc = "BANKER_DRAW"
    else:
        sc = "NONE_DRAW"

    return {
        "player_point": pp,
        "banker_point": bp,
        "winner": winner,
        "scenario": sc,
        "cards_used": len(p) + len(b),
        "point_key": f"P{pp}_B{bp}",
    }


def non_tie(seq: List[str]) -> List[str]:
    return [x for x in seq if x in {"B", "P"}]


def is_alternating(xs: List[str]) -> bool:
    return len(xs) >= 4 and all(xs[i] != xs[i - 1] for i in range(1, len(xs)))


def is_double_jump(xs: List[str]) -> bool:
    if len(xs) < 6:
        return False
    y = xs[-6:]
    return y in (["B", "B", "P", "P", "B", "B"], ["P", "P", "B", "B", "P", "P"])


def is_one_room_two_halls(xs: List[str]) -> bool:
    if len(xs) < 5:
        return False
    y = xs[-5:]
    patterns = [
        ["B", "P", "P", "B", "P"],
        ["P", "B", "B", "P", "B"],
        ["B", "P", "B", "B", "P"],
        ["P", "B", "P", "P", "B"],
    ]
    return y in patterns


def run_len(xs: List[str]) -> int:
    if not xs:
        return 0
    last = xs[-1]
    n = 1
    for v in reversed(xs[:-1]):
        if v == last:
            n += 1
        else:
            break
    return n


def classify_profile(winners_recent: List[str], point_keys_recent: List[str]) -> str:
    xs = non_tie(winners_recent)
    if not xs:
        return "NEUTRAL"
    rl = run_len(xs)
    if rl >= 4:
        return "LONG_BANKER" if xs[-1] == "B" else "LONG_PLAYER"
    if is_double_jump(xs):
        return "DOUBLE_JUMP"
    if is_alternating(xs[-5:]):
        return "SINGLE_JUMP"
    if is_one_room_two_halls(xs):
        return "ONE_ROOM_TWO_HALLS"
    if point_keys_recent and point_keys_recent.count(point_keys_recent[-1]) >= 2:
        return "SAME_POINT_REPEAT"
    return "NEUTRAL"


def update_bucket(bucket: Dict[str, Any], next_winner: str, same_point_count: int):
    bucket["sample_size"] += 1
    if next_winner == "B":
        bucket["banker_wins"] += 1
    elif next_winner == "P":
        bucket["player_wins"] += 1
    else:
        bucket["tie_count"] += 1
    bucket["same_point_repeat_sum"] += same_point_count


def finalize_record(key: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    b = int(raw.get("banker_wins", 0))
    p = int(raw.get("player_wins", 0))
    t = int(raw.get("tie_count", 0))
    sample = int(raw.get("sample_size", 0))
    no_tie = b + p
    banker_prob = b / no_tie if no_tie > 0 else 0.5
    player_prob = p / no_tie if no_tie > 0 else 0.5
    profile = raw.get("road_profile", "NEUTRAL")
    sc = raw.get("scenario", "UNKNOWN")
    return {
        "available": sample > 0,
        "key": key,
        "player_point": raw.get("player_point"),
        "banker_point": raw.get("banker_point"),
        "scenario": sc,
        "scenario_zh": SCENARIO_ZH.get(sc, sc),
        "road_profile": profile,
        "road_profile_zh": ROAD_PROFILE_ZH.get(profile, profile),
        "banker_prob": round(banker_prob, 8),
        "player_prob": round(player_prob, 8),
        "tie_rate": round(t / sample, 8) if sample else 0.0,
        "banker_wins": b,
        "player_wins": p,
        "tie_count": t,
        "sample_size": sample,
        "no_tie_sample_size": no_tie,
        "same_point_repeat_avg": round(raw.get("same_point_repeat_sum", 0) / sample, 4) if sample else 0,
        "source": "ROAD_PROFILE_SEQUENCE_MC_MEMORYLESS_V9_4",
        "note": "資料庫相似路段統計；不保存、不延續用戶輸入歷史。",
    }


def save_db(records_raw: Dict[str, Dict[str, Any]], path: Path, rounds_done: int, started_at: float):
    records = {k: finalize_record(k, v) for k, v in records_raw.items()}
    meta = {
        "version": "road_profile_db_v9_4_memoryless",
        "source": "ROAD_PROFILE_SEQUENCE_MC_MEMORYLESS_V9_4",
        "decks": DECKS,
        "rounds_done": rounds_done,
        "record_count": len(records),
        "total_simulated_samples": sum(int(v.get("sample_size", 0)) for v in records.values()),
        "road_profiles": ROAD_PROFILE_ZH,
        "description": "用模擬連續牌局建立點數+補牌情境+資料庫路段型態，供無記憶查詢。",
        "updated_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def main():
    started_at = time.time()
    rng = random.Random(SEED)
    records: Dict[str, Dict[str, Any]] = {}
    winners_recent: deque = deque(maxlen=12)
    point_keys_recent: deque = deque(maxlen=20)
    previous_hand = None
    rounds_done = 0
    shoe = fresh_shoe()

    print("=" * 72)
    print("Road profile DB generator V9.4")
    print(f"OUTPUT_PATH={OUTPUT_PATH}")
    print(f"TOTAL_ROUNDS={TOTAL_ROUNDS:,}")
    print(f"DECKS={DECKS}, SEED={SEED}")
    print("=" * 72)

    while rounds_done < TOTAL_ROUNDS:
        if shoe_remaining(shoe) < SHOE_CUT_REMAINING:
            shoe = fresh_shoe()
            winners_recent.clear()
            point_keys_recent.clear()
            previous_hand = None

        hand = deal_round(rng, shoe)
        rounds_done += 1

        if previous_hand is not None:
            recent_winners = list(winners_recent) + [previous_hand["winner"]]
            recent_points = list(point_keys_recent) + [previous_hand["point_key"]]
            profile = classify_profile(recent_winners, recent_points)
            same_count = recent_points.count(previous_hand["point_key"])
            pp = previous_hand["player_point"]
            bp = previous_hand["banker_point"]
            sc = previous_hand["scenario"]
            next_winner = hand["winner"]

            keys = [
                f"P{pp}_B{bp}|SC:{sc}|ROAD:{profile}",
                f"P{pp}_B{bp}|ROAD:{profile}",
                f"P{pp}_B{bp}|ROAD_PROFILE",
                f"P{pp}_B{bp}|BASE",
            ]
            for key in keys:
                bucket = records.setdefault(key, {
                    "player_point": pp,
                    "banker_point": bp,
                    "scenario": sc if "SC:" in key else "BASE",
                    "road_profile": profile if "ROAD:" in key else "NEUTRAL",
                    "sample_size": 0,
                    "banker_wins": 0,
                    "player_wins": 0,
                    "tie_count": 0,
                    "same_point_repeat_sum": 0,
                })
                update_bucket(bucket, next_winner, same_count)

        if hand["winner"] in {"B", "P"}:
            winners_recent.append(hand["winner"])
        point_keys_recent.append(hand["point_key"])
        previous_hand = hand

        if SAVE_EVERY > 0 and rounds_done % SAVE_EVERY == 0:
            print(f"rounds={rounds_done:,}, records={len(records):,}")
            save_db(records, OUTPUT_PATH, rounds_done, started_at)

    save_db(records, OUTPUT_PATH, rounds_done, started_at)
    print("done")
    print(f"records={len(records):,}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
