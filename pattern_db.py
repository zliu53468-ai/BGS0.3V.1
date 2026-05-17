import json
import os
from functools import lru_cache
from typing import Dict, Any, List, Optional
from config import RESULT_PATTERN_DB_PATH

@lru_cache(maxsize=1)
def load_pattern_db() -> Dict[str, Any]:
    path = RESULT_PATTERN_DB_PATH
    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pattern_db_meta() -> Dict[str, Any]:
    return load_pattern_db().get("meta", {})

def get_pattern_record(pattern: str, window: int) -> Optional[Dict[str, Any]]:
    db = load_pattern_db()
    return db.get("records", {}).get(f"W{window}:{pattern}")

def normalize_result_symbol(result: str) -> str:
    if result in {"莊", "B", "Banker", "banker"}:
        return "B"
    if result in {"閒", "P", "Player", "player"}:
        return "P"
    if result in {"和", "T", "Tie", "tie"}:
        return "T"
    return "T"

def build_patterns_from_rounds(rounds: List[Dict[str, Any]]) -> Dict[int, str]:
    symbols = [normalize_result_symbol(r.get("last_result", "")) for r in rounds]
    out = {}

    for w in [3, 5, 7]:
        if len(symbols) >= w:
            pat = "".join(symbols[-w:])
            if w == 7:
                # W7資料庫為B/P主型態；T用上一個非T修正，若沒有則略過。
                fixed = []
                last = None
                for s in pat:
                    if s in {"B", "P"}:
                        fixed.append(s)
                        last = s
                    else:
                        fixed.append(last or "B")
                pat = "".join(fixed)
            out[w] = pat

    return out

def pattern_lookup(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    patterns = build_patterns_from_rounds(rounds)

    if not patterns:
        return {
            "available": False,
            "banker_prob": 0.5068,
            "player_prob": 0.4932,
            "matched": [],
            "sample_size": 0,
            "source": "NO_PATTERN_YET",
        }

    weighted_b = 0.0
    total_weight = 0.0
    matched = []

    # 越長權重越高
    window_weights = {3: 0.75, 5: 1.15, 7: 1.45}

    for w, pat in patterns.items():
        rec = get_pattern_record(pat, w)
        if not rec:
            continue

        sample = int(rec.get("sample", 0))
        sample_weight = min(max(sample / 10000, 0.45), 2.2)
        weight = window_weights.get(w, 1.0) * sample_weight
        banker = float(rec.get("next_banker_rate", 0.5068))

        weighted_b += banker * weight
        total_weight += weight

        matched.append({
            "window": w,
            "pattern": pat,
            "sample": sample,
            "banker_rate": banker,
            "player_rate": float(rec.get("next_player_rate", 0.4932)),
        })

    if total_weight <= 0:
        return {
            "available": False,
            "banker_prob": 0.5068,
            "player_prob": 0.4932,
            "matched": matched,
            "sample_size": 0,
            "source": "PATTERN_NOT_MATCHED",
        }

    banker_prob = weighted_b / total_weight
    return {
        "available": True,
        "banker_prob": banker_prob,
        "player_prob": 1.0 - banker_prob,
        "matched": matched,
        "sample_size": sum(m["sample"] for m in matched),
        "source": "RESULT_PATTERN_DB_3M",
    }
