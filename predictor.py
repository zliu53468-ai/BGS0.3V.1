import hashlib
import math
from typing import Dict, Any, List
from config import (
    POINT_WEIGHT,
    PATTERN_WEIGHT,
    SIM_WEIGHT,
    MIN_OUTPUT_PROB,
    MAX_OUTPUT_PROB,
    PERCENT_DECIMALS,
    USE_POINT_DB,
    USE_PATTERN_DB,
)
from point_db import get_point_record, point_db_meta
from pattern_db import pattern_lookup, pattern_db_meta

BASE_BANKER_NO_TIE = 0.5068

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def stable_noise(key: str, scale: float = 0.035) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale

def get_last_result(player_point: int, banker_point: int) -> str:
    if player_point > banker_point:
        return "閒"
    if banker_point > player_point:
        return "莊"
    return "和"

def validate_point(v: int) -> int:
    iv = int(v)
    if iv < 0 or iv > 9:
        raise ValueError("point must be 0-9")
    return iv

def point_zone(point: int) -> str:
    if point <= 2:
        return "LOW"
    if point <= 5:
        return "MID"
    if point <= 7:
        return "HIGH"
    return "TOP"

def diff_zone(diff: int) -> str:
    ad = abs(diff)
    if ad == 0:
        return "Z"
    if ad <= 2:
        return "S"
    if ad <= 5:
        return "M"
    return "L"

def feature_key(player_point: int, banker_point: int) -> str:
    diff = player_point - banker_point
    return f"P{player_point}_B{banker_point}_R{get_last_result(player_point, banker_point)}_D{diff}_Z{diff_zone(diff)}_PZ{point_zone(player_point)}_BZ{point_zone(banker_point)}"

def point_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    rec = get_point_record(player_point, banker_point)
    return {
        "available": True,
        "feature_key": f"P{player_point}_B{banker_point}",
        "banker_prob": float(rec["next_banker_rate"]),
        "player_prob": float(rec["next_player_rate"]),
        "source": rec.get("source", "POINT_DB"),
        "sample_size": int(rec.get("sample", 0)),
        "total_simulated_samples": int(point_db_meta().get("total_simulated_samples", 0)),
    }

def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE

    if diff == 0:
        banker += stable_noise(key + ":tie", 0.018)
    elif 1 <= diff <= 2:
        banker -= 0.185
    elif 3 <= diff <= 5:
        banker += 0.185
    elif diff >= 6:
        banker += 0.115
    elif -2 <= diff <= -1:
        banker += 0.185
    elif -5 <= diff <= -3:
        banker -= 0.185
    elif diff <= -6:
        banker -= 0.115

    banker += stable_noise(key + ":fallback", 0.045)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "available": False,
        "feature_key": key,
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "FALLBACK_POINT_RULE",
        "sample_size": 0,
        "total_simulated_samples": 0,
    }

def ai_simulation_layer(player_point: int, banker_point: int) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)

    x = 0.0
    x += -0.055 * diff

    if abs(diff) in {1, 2}:
        x += -0.16 if diff > 0 else 0.16
    elif abs(diff) in {3, 4, 5}:
        x += 0.16 if diff > 0 else -0.16
    elif abs(diff) >= 6:
        x += 0.09 if diff > 0 else -0.09

    x += stable_noise(key + ":ai", 0.11)

    banker = 1.0 / (1.0 + math.exp(-x))
    banker = 0.15 + banker * 0.70
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_SIMULATION",
    }

def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)
    rounds = rounds or []

    try:
        point = point_db_lookup(player_point, banker_point) if USE_POINT_DB else fallback_point_lookup(player_point, banker_point)
    except Exception:
        point = fallback_point_lookup(player_point, banker_point)

    pattern = pattern_lookup(rounds) if USE_PATTERN_DB else {
        "available": False,
        "banker_prob": BASE_BANKER_NO_TIE,
        "player_prob": 1.0 - BASE_BANKER_NO_TIE,
        "matched": [],
        "sample_size": 0,
        "source": "PATTERN_DISABLED",
    }

    ai = ai_simulation_layer(player_point, banker_point)

    # 如果資料不足，pattern 權重自動轉給 point，避免前幾局失真。
    p_w = POINT_WEIGHT
    pat_w = PATTERN_WEIGHT if pattern.get("available") else 0.0
    sim_w = SIM_WEIGHT
    if not pattern.get("available"):
        p_w += PATTERN_WEIGHT

    total_weight = max(p_w + pat_w + sim_w, 0.0001)

    banker = (
        point["banker_prob"] * p_w +
        pattern["banker_prob"] * pat_w +
        ai["banker_prob"] * sim_w
    ) / total_weight

    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    recommend = "莊" if banker >= player else "閒"

    return {
        "ok": True,
        "player_point": player_point,
        "banker_point": banker_point,
        "last_result": get_last_result(player_point, banker_point),
        "recommend": recommend,
        "player_prob": round(player * 100, PERCENT_DECIMALS),
        "banker_prob": round(banker * 100, PERCENT_DECIMALS),
        "player_prob_raw": player,
        "banker_prob_raw": banker,
        "feature_key": point["feature_key"],
        "point_source": point["source"],
        "pattern_source": pattern["source"],
        "ai_source": ai["source"],
        "point_sample_size": point["sample_size"],
        "pattern_sample_size": pattern.get("sample_size", 0),
        "point_total_samples": point["total_simulated_samples"],
        "pattern_total_samples": int(pattern_db_meta().get("total_simulated_samples", 0)),
        "matched_patterns": pattern.get("matched", [])[:3],
        "weights": {"point": p_w, "pattern": pat_w, "simulation": sim_w},
        "no_observe": True,
    }
