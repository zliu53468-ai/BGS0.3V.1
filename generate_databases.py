"""
重新產生「真實牌靴模擬版」資料庫。

使用：
python generate_databases.py

輸出：
data/point_db_3m.json
data/result_pattern_db_3m.json
data/combo_db_3m.json

重點：
1. 這版不是用公式假造 300 萬樣本。
2. 會建立 8 副牌靴、洗牌、依百家樂補牌規則逐局發牌。
3. point_db 仍保留 P0_B0 ~ P9_B9，讓舊程式相容。
4. pattern_db 會從真實連續序列統計 W3/W5/W7。
5. combo_db 會新增「點數 + 牌路狀態」複合資料庫，給新版 predictor.py 優先查詢。
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
import json
import random
import os


# ============================================================
# 可調參數
# ============================================================

TOTAL_ROUNDS = int(os.getenv("SIM_ROUNDS", "3000000"))
DECKS = int(os.getenv("BACCARAT_DECKS", "8"))
SEED = int(os.getenv("SIM_SEED", "20260614"))
RESHUFFLE_REMAINING_CARDS = int(os.getenv("RESHUFFLE_REMAINING_CARDS", "70"))
OUT_DIR = Path(os.getenv("DB_OUTPUT_DIR", "data"))

# 為了避免 combo_db 過大，低樣本 key 不輸出。
COMBO_MIN_SAMPLE_TO_SAVE = int(os.getenv("COMBO_MIN_SAMPLE_TO_SAVE", "30"))
PATTERN_MIN_SAMPLE_TO_SAVE = int(os.getenv("PATTERN_MIN_SAMPLE_TO_SAVE", "1"))

BASE_BANKER_NO_TIE = 0.5068


# ============================================================
# 基本工具
# ============================================================

def empty_counter() -> Dict[str, int]:
    return {
        "sample": 0,
        "next_player_count": 0,
        "next_banker_count": 0,
        "next_tie_count": 0,
    }


def add_next(counter: Dict[str, int], next_result: str):
    counter["sample"] += 1
    if next_result == "P":
        counter["next_player_count"] += 1
    elif next_result == "B":
        counter["next_banker_count"] += 1
    else:
        counter["next_tie_count"] += 1


def rates(counter: Dict[str, int]) -> Dict[str, Any]:
    total = int(counter.get("sample", 0) or 0)
    p = int(counter.get("next_player_count", 0) or 0)
    b = int(counter.get("next_banker_count", 0) or 0)
    t = int(counter.get("next_tie_count", 0) or 0)

    no_tie = p + b
    if no_tie <= 0:
        player_rate = 1.0 - BASE_BANKER_NO_TIE
        banker_rate = BASE_BANKER_NO_TIE
    else:
        player_rate = p / no_tie
        banker_rate = b / no_tie

    tie_rate = t / total if total > 0 else 0.0

    return {
        "sample": total,
        "no_tie_sample": no_tie,
        "next_player_count": p,
        "next_banker_count": b,
        "next_tie_count": t,
        "next_player_rate": round(player_rate, 6),
        "next_banker_rate": round(banker_rate, 6),
        "next_tie_rate": round(tie_rate, 6),
    }


def result_to_zh(r: str) -> str:
    if r == "B":
        return "莊"
    if r == "P":
        return "閒"
    return "和"


def normalize_w7_pattern(seq: List[str]) -> Optional[str]:
    """
    W7 仍維持舊資料庫設計：B/P 主型態。
    若遇到 T，用上一個非 T 補；前面沒有非 T 時，略過。
    """
    fixed: List[str] = []
    last: Optional[str] = None
    for s in seq:
        if s in {"B", "P"}:
            fixed.append(s)
            last = s
        else:
            if last is None:
                return None
            fixed.append(last)
    return "".join(fixed)


def streak_info(results: List[str]) -> Tuple[str, int]:
    count = 0
    side: Optional[str] = None
    for r in reversed(results):
        if r == "T":
            continue
        if side is None:
            side = r
            count = 1
        elif r == side:
            count += 1
        else:
            break
    if side is None:
        return "N", 0
    return side, count


def alt_bucket(results: List[str], window: int = 6) -> str:
    seq = [x for x in results[-window:] if x in {"B", "P"}]
    if len(seq) < 3:
        return "NA"
    alt = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    if alt >= 5:
        return "ALT5"
    if alt >= 3:
        return "ALT3_4"
    return "ALT0_2"


def balance10(results: List[str]) -> str:
    seq = [x for x in results[-10:] if x in {"B", "P"}]
    if not seq:
        return "NA"
    b = seq.count("B")
    p = seq.count("P")
    # 分桶，不做太細，避免資料太稀疏。
    if b - p >= 4:
        return "B_HEAVY"
    if p - b >= 4:
        return "P_HEAVY"
    if abs(b - p) <= 1:
        return "BALANCED"
    return "MID"


def tie_age(results: List[str]) -> str:
    age = 0
    for r in reversed(results):
        if r == "T":
            break
        age += 1
    if age == 0:
        return "T0"
    if age == 1:
        return "T1"
    if age == 2:
        return "T2"
    if age <= 4:
        return "T3_4"
    return "T5P"


# ============================================================
# 百家樂真實發牌規則
# ============================================================

def make_shoe(decks: int = 8) -> List[int]:
    """
    牌值：A=1, 2~9=原值, 10/J/Q/K=0。
    每副牌：0 有 16 張，1~9 各 4 張。
    """
    shoe: List[int] = []
    for _ in range(decks):
        shoe.extend([0] * 16)
        for v in range(1, 10):
            shoe.extend([v] * 4)
    random.shuffle(shoe)
    return shoe


def draw(shoe: List[int]) -> int:
    return shoe.pop()


def total(cards: List[int]) -> int:
    return sum(cards) % 10


def play_round(shoe: List[int]) -> Dict[str, Any]:
    player_cards = [draw(shoe), draw(shoe)]
    banker_cards = [draw(shoe), draw(shoe)]

    p_total = total(player_cards)
    b_total = total(banker_cards)

    player_third: Optional[int] = None
    banker_third: Optional[int] = None

    # Natural 8/9：雙方停牌。
    if p_total not in {8, 9} and b_total not in {8, 9}:
        # 閒家補牌規則。
        if p_total <= 5:
            player_third = draw(shoe)
            player_cards.append(player_third)
            p_total = total(player_cards)

        # 莊家補牌規則。
        if player_third is None:
            if b_total <= 5:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)
        else:
            c = player_third
            if b_total <= 2:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)
            elif b_total == 3 and c != 8:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)
            elif b_total == 4 and 2 <= c <= 7:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)
            elif b_total == 5 and 4 <= c <= 7:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)
            elif b_total == 6 and 6 <= c <= 7:
                banker_third = draw(shoe)
                banker_cards.append(banker_third)

        b_total = total(banker_cards)

    if p_total > b_total:
        result = "P"
    elif b_total > p_total:
        result = "B"
    else:
        result = "T"

    return {
        "player_point": p_total,
        "banker_point": b_total,
        "result": result,
        "player_cards": player_cards,
        "banker_cards": banker_cards,
        "player_third": player_third,
        "banker_third": banker_third,
    }


def simulate_rounds(total_rounds: int) -> List[Dict[str, Any]]:
    random.seed(SEED)
    shoe = make_shoe(DECKS)
    rounds: List[Dict[str, Any]] = []

    while len(rounds) < total_rounds:
        if len(shoe) < RESHUFFLE_REMAINING_CARDS:
            shoe = make_shoe(DECKS)

        rounds.append(play_round(shoe))

        if len(rounds) % 200000 == 0:
            print(f"Simulated {len(rounds):,}/{total_rounds:,} rounds...")

    return rounds


# ============================================================
# 資料庫產生
# ============================================================

def generate_point_db_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    counters: Dict[str, Dict[str, int]] = defaultdict(empty_counter)

    for i in range(len(rounds) - 1):
        cur = rounds[i]
        nxt = rounds[i + 1]
        key = f"P{cur['player_point']}_B{cur['banker_point']}"
        add_next(counters[key], nxt["result"])

    records: Dict[str, Any] = {}
    for p in range(10):
        for b in range(10):
            key = f"P{p}_B{b}"
            c = counters.get(key, empty_counter())
            rec = rates(c)
            rec.update({
                "player_point": p,
                "banker_point": b,
                "last_result": result_to_zh("P" if p > b else "B" if b > p else "T"),
                "source": "REAL_SHOE_3M_POINT_DB",
            })
            records[key] = rec

    return {
        "meta": {
            "name": "BGS real-shoe 3M point database",
            "version": "2026-06-14-real-shoe-v1",
            "total_simulated_samples": len(rounds),
            "transition_samples": max(0, len(rounds) - 1),
            "decks": DECKS,
            "seed": SEED,
            "storage_note": "真實牌靴逐局模擬後，依上一局 P0_B0~P9_B9 統計下一局莊/閒/和。保留100組是為了舊版查詢相容，但樣本來自真實連續發牌，不是公式假造。",
            "input_rule": "先閒點，再莊點，例如65代表閒6莊5。",
            "warning": "模擬統計不保證實際預測命中。",
        },
        "records": records,
    }


def generate_pattern_db_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    counters: Dict[str, Dict[str, int]] = defaultdict(empty_counter)
    results = [r["result"] for r in rounds]

    for i in range(len(results) - 1):
        nxt = results[i + 1]
        for w in [3, 5, 7]:
            if i + 1 < w:
                continue
            raw_seq = results[i - w + 1:i + 1]
            if w == 7:
                pat = normalize_w7_pattern(raw_seq)
                if not pat:
                    continue
            else:
                pat = "".join(raw_seq)
            add_next(counters[f"W{w}:{pat}"], nxt)

    records: Dict[str, Any] = {}
    for key, c in counters.items():
        if c["sample"] < PATTERN_MIN_SAMPLE_TO_SAVE:
            continue
        w_text, pat = key.split(":", 1)
        w = int(w_text.replace("W", ""))
        rec = rates(c)
        rec.update({
            "window": w,
            "pattern": pat,
            "source": "REAL_SHOE_3M_RESULT_PATTERN_DB",
        })
        records[key] = rec

    return {
        "meta": {
            "name": "BGS real-shoe 3M result pattern database",
            "version": "2026-06-14-real-shoe-v1",
            "total_simulated_samples": len(rounds),
            "transition_samples": max(0, len(rounds) - 1),
            "windows": [3, 5, 7],
            "decks": DECKS,
            "seed": SEED,
            "storage_note": "真實牌靴逐局模擬後，統計 W3/W5/W7 牌路對下一局莊/閒/和。W3/W5含B/P/T，W7為B/P主型態。",
            "warning": "模擬統計不保證實際預測命中。",
        },
        "records": records,
    }


def combo_keys_for_index(rounds: List[Dict[str, Any]], i: int) -> List[str]:
    """
    用第 i 局作為已知狀態，產生複合 key，用來統計第 i+1 局。
    """
    cur = rounds[i]
    results = [r["result"] for r in rounds[:i + 1]]
    point_key = f"P{cur['player_point']}_B{cur['banker_point']}"

    keys: List[str] = []

    # 點數 + W3/W5/W7
    for w in [3, 5, 7]:
        if len(results) >= w:
            raw_seq = results[-w:]
            if w == 7:
                pat = normalize_w7_pattern(raw_seq)
                if not pat:
                    continue
            else:
                pat = "".join(raw_seq)
            keys.append(f"{point_key}|W{w}:{pat}")

    # 加入桌況狀態，不要太細，避免樣本稀疏。
    side, streak = streak_info(results)
    streak_bucket = min(streak, 6)
    keys.append(f"{point_key}|STREAK:{side}{streak_bucket}")
    keys.append(f"{point_key}|ALT:{alt_bucket(results)}")
    keys.append(f"{point_key}|BAL10:{balance10(results)}")
    keys.append(f"{point_key}|TIEAGE:{tie_age(results)}")

    # 最細：點數 + W5 + 亂流狀態
    if len(results) >= 5:
        pat5 = "".join(results[-5:])
        keys.append(f"{point_key}|W5:{pat5}|ALT:{alt_bucket(results)}")
        keys.append(f"{point_key}|W5:{pat5}|STREAK:{side}{streak_bucket}")

    # 去重但保留順序。
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def generate_combo_db_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    counters: Dict[str, Dict[str, int]] = defaultdict(empty_counter)

    for i in range(len(rounds) - 1):
        nxt = rounds[i + 1]
        for key in combo_keys_for_index(rounds, i):
            add_next(counters[key], nxt["result"])

        if i > 0 and i % 300000 == 0:
            print(f"Built combo features {i:,}/{len(rounds)-1:,} transitions...")

    records: Dict[str, Any] = {}
    for key, c in counters.items():
        if c["sample"] < COMBO_MIN_SAMPLE_TO_SAVE:
            continue
        rec = rates(c)
        rec.update({
            "feature_key": key,
            "source": "REAL_SHOE_3M_COMBO_DB",
        })
        records[key] = rec

    return {
        "meta": {
            "name": "BGS real-shoe 3M combo database",
            "version": "2026-06-14-real-shoe-v1",
            "total_simulated_samples": len(rounds),
            "transition_samples": max(0, len(rounds) - 1),
            "record_count": len(records),
            "decks": DECKS,
            "seed": SEED,
            "min_sample_saved": COMBO_MIN_SAMPLE_TO_SAVE,
            "storage_note": "真實牌靴逐局模擬後，統計 點數+W3/W5/W7/連莊連閒/單跳/近10局平衡/和局距離 等複合特徵對下一局莊閒和。",
            "warning": "複合特徵較細，若 sample 太低仍需由 predictor.py 自動 fallback 到 point_db/pattern_db。",
        },
        "records": records,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Start real-shoe simulation: rounds={TOTAL_ROUNDS:,}, decks={DECKS}, seed={SEED}")
    rounds = simulate_rounds(TOTAL_ROUNDS)

    print("Generating point_db_3m.json...")
    point_db = generate_point_db_from_rounds(rounds)
    (OUT_DIR / "point_db_3m.json").write_text(
        json.dumps(point_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generating result_pattern_db_3m.json...")
    pattern_db = generate_pattern_db_from_rounds(rounds)
    (OUT_DIR / "result_pattern_db_3m.json").write_text(
        json.dumps(pattern_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generating combo_db_3m.json...")
    combo_db = generate_combo_db_from_rounds(rounds)
    (OUT_DIR / "combo_db_3m.json").write_text(
        json.dumps(combo_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Generated: {OUT_DIR / 'point_db_3m.json'}")
    print(f"Generated: {OUT_DIR / 'result_pattern_db_3m.json'}")
    print(f"Generated: {OUT_DIR / 'combo_db_3m.json'}")
