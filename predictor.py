import hashlib
import math
import os
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

def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# 莊閒累積開局差距權重切換器
COUNT_GAP_MODE = _get_bool_env("COUNT_GAP_MODE", False)

# 正常模式
NORMAL_POINT_WEIGHT = _get_float_env("NORMAL_POINT_WEIGHT", POINT_WEIGHT)
NORMAL_PATTERN_WEIGHT = _get_float_env("NORMAL_PATTERN_WEIGHT", PATTERN_WEIGHT)
NORMAL_SIM_WEIGHT = _get_float_env("NORMAL_SIM_WEIGHT", SIM_WEIGHT)

# 偏差模式
COUNT_GAP_TRIGGER = _get_int_env("COUNT_GAP_TRIGGER", 3)
GAP_POINT_WEIGHT = _get_float_env("GAP_POINT_WEIGHT", 0.62)
GAP_PATTERN_WEIGHT = _get_float_env("GAP_PATTERN_WEIGHT", 0.23)
GAP_SIM_WEIGHT = _get_float_env("GAP_SIM_WEIGHT", 0.15)

# 極端偏差模式
EXTREME_COUNT_GAP_TRIGGER = _get_int_env("EXTREME_COUNT_GAP_TRIGGER", 6)
EXTREME_GAP_POINT_WEIGHT = _get_float_env("EXTREME_GAP_POINT_WEIGHT", 0.68)
EXTREME_GAP_PATTERN_WEIGHT = _get_float_env("EXTREME_GAP_PATTERN_WEIGHT", 0.17)
EXTREME_GAP_SIM_WEIGHT = _get_float_env("EXTREME_GAP_SIM_WEIGHT", 0.15)

# 局型偵測器
ROUTE_MODE = _get_bool_env("ROUTE_MODE", True)

# 長龍判斷門檻
DRAGON_TRIGGER = _get_int_env("DRAGON_TRIGGER", 3)
DRAGON_STRONG_TRIGGER = _get_int_env("DRAGON_STRONG_TRIGGER", 5)
DRAGON_HOT_TRIGGER = _get_int_env("DRAGON_HOT_TRIGGER", 7)

# 長龍權重
DRAGON_POINT_WEIGHT = _get_float_env("DRAGON_POINT_WEIGHT", 0.46)
DRAGON_PATTERN_WEIGHT = _get_float_env("DRAGON_PATTERN_WEIGHT", 0.40)
DRAGON_SIM_WEIGHT = _get_float_env("DRAGON_SIM_WEIGHT", 0.14)

DRAGON_STRONG_POINT_WEIGHT = _get_float_env("DRAGON_STRONG_POINT_WEIGHT", 0.44)
DRAGON_STRONG_PATTERN_WEIGHT = _get_float_env("DRAGON_STRONG_PATTERN_WEIGHT", 0.41)
DRAGON_STRONG_SIM_WEIGHT = _get_float_env("DRAGON_STRONG_SIM_WEIGHT", 0.15)

DRAGON_HOT_POINT_WEIGHT = _get_float_env("DRAGON_HOT_POINT_WEIGHT", 0.49)
DRAGON_HOT_PATTERN_WEIGHT = _get_float_env("DRAGON_HOT_PATTERN_WEIGHT", 0.37)
DRAGON_HOT_SIM_WEIGHT = _get_float_env("DRAGON_HOT_SIM_WEIGHT", 0.14)

# 跳路
CHOP_TRIGGER = _get_int_env("CHOP_TRIGGER", 4)
CHOP_POINT_WEIGHT = _get_float_env("CHOP_POINT_WEIGHT", 0.46)
CHOP_PATTERN_WEIGHT = _get_float_env("CHOP_PATTERN_WEIGHT", 0.40)
CHOP_SIM_WEIGHT = _get_float_env("CHOP_SIM_WEIGHT", 0.14)

# 兩房
TWO_ROOM_TRIGGER = _get_int_env("TWO_ROOM_TRIGGER", 4)
TWO_ROOM_POINT_WEIGHT = _get_float_env("TWO_ROOM_POINT_WEIGHT", 0.48)
TWO_ROOM_PATTERN_WEIGHT = _get_float_env("TWO_ROOM_PATTERN_WEIGHT", 0.38)
TWO_ROOM_SIM_WEIGHT = _get_float_env("TWO_ROOM_SIM_WEIGHT", 0.14)

# ==================== 新增：單跳 / 雙跳 ====================
SINGLE_CHOP_TRIGGER = _get_int_env("SINGLE_CHOP_TRIGGER", 4)
DOUBLE_CHOP_TRIGGER = _get_int_env("DOUBLE_CHOP_TRIGGER", 3)

SINGLE_CHOP_POINT_WEIGHT = _get_float_env("SINGLE_CHOP_POINT_WEIGHT", 0.44)
SINGLE_CHOP_PATTERN_WEIGHT = _get_float_env("SINGLE_CHOP_PATTERN_WEIGHT", 0.43)
SINGLE_CHOP_SIM_WEIGHT = _get_float_env("SINGLE_CHOP_SIM_WEIGHT", 0.13)

DOUBLE_CHOP_POINT_WEIGHT = _get_float_env("DOUBLE_CHOP_POINT_WEIGHT", 0.46)
DOUBLE_CHOP_PATTERN_WEIGHT = _get_float_env("DOUBLE_CHOP_PATTERN_WEIGHT", 0.41)
DOUBLE_CHOP_SIM_WEIGHT = _get_float_env("DOUBLE_CHOP_SIM_WEIGHT", 0.13)


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


def normalize_result_symbol(result: str) -> str:
    if result in {"莊", "B", "Banker", "banker"}:
        return "B"
    if result in {"閒", "P", "Player", "player"}:
        return "P"
    if result in {"和", "T", "Tie", "tie"}:
        return "T"
    return "T"


def rounds_to_symbols(rounds: List[Dict[str, Any]], ignore_tie: bool = True) -> List[str]:
    symbols = []
    for item in rounds or []:
        symbol = normalize_result_symbol(item.get("last_result", ""))
        if ignore_tie and symbol == "T":
            continue
        if symbol in {"B", "P", "T"}:
            symbols.append(symbol)
    return symbols


def count_banker_player_gap(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    banker_count = 0
    player_count = 0
    tie_count = 0
    for item in rounds or []:
        symbol = normalize_result_symbol(item.get("last_result", ""))
        if symbol == "B":
            banker_count += 1
        elif symbol == "P":
            player_count += 1
        else:
            tie_count += 1
    gap = abs(banker_count - player_count)
    if banker_count > player_count:
        leader = "莊"
    elif player_count > banker_count:
        leader = "閒"
    else:
        leader = "平衡"
    return {
        "banker_count": banker_count,
        "player_count": player_count,
        "tie_count": tie_count,
        "gap": gap,
        "leader": leader,
    }


def detect_dragon(symbols: List[str]) -> Dict[str, Any]:
    if not symbols:
        return {"active": False, "side": None, "length": 0}
    last = symbols[-1]
    length = 1
    for s in reversed(symbols[:-1]):
        if s == last:
            length += 1
        else:
            break
    return {
        "active": length >= DRAGON_TRIGGER,
        "side": "莊" if last == "B" else "閒",
        "length": length,
    }


def detect_chop(symbols: List[str]) -> Dict[str, Any]:
    if len(symbols) < CHOP_TRIGGER:
        return {"active": False, "length": 0}
    length = 1
    for i in range(len(symbols) - 1, 0, -1):
        if symbols[i] != symbols[i - 1]:
            length += 1
        else:
            break
    return {
        "active": length >= CHOP_TRIGGER,
        "length": length,
    }


def detect_two_room(symbols: List[str]) -> Dict[str, Any]:
    if len(symbols) < TWO_ROOM_TRIGGER:
        return {"active": False, "pattern": "", "length": 0}
    recent4 = "".join(symbols[-4:])
    recent6 = "".join(symbols[-6:]) if len(symbols) >= 6 else ""
    valid4 = {"BBPP", "PPBB", "BPPB", "PBBP"}
    valid6 = {"BBPPBB", "PPBBPP", "BPPBBP", "PBBPPB"}
    if recent6 in valid6:
        return {"active": True, "pattern": recent6, "length": 6}
    if recent4 in valid4:
        return {"active": True, "pattern": recent4, "length": 4}
    return {"active": False, "pattern": recent4, "length": 0}


# ==================== 新增函數：單跳與雙跳 ====================
def detect_single_chop(symbols: List[str]) -> Dict[str, Any]:
    """單跳偵測：BPBPBP... 嚴格交替"""
    if len(symbols) < SINGLE_CHOP_TRIGGER:
        return {"active": False, "length": 0, "type": "single_chop"}
    
    length = 1
    for i in range(len(symbols)-1, 0, -1):
        if symbols[i] != symbols[i-1]:
            length += 1
        else:
            break
    return {
        "active": length >= SINGLE_CHOP_TRIGGER,
        "length": length,
        "type": "single_chop"
    }


def detect_double_chop(symbols: List[str]) -> Dict[str, Any]:
    """雙跳偵測：BBPPBBPP... 或 PPBBPPBB..."""
    if len(symbols) < DOUBLE_CHOP_TRIGGER * 2:
        return {"active": False, "length": 0, "type": "double_chop"}
    
    recent = "".join(symbols[-10:])
    double_patterns = ["BBPPBBPP", "PPBBPPBB", "BPPBBPPB", "PBBPPBBP"]
    
    for p in double_patterns:
        if p in recent:
            return {"active": True, "length": 8, "type": "double_chop"}
    
    recent4 = "".join(symbols[-4:])
    if recent4 in ["BBPP", "PPBB"]:
        return {"active": True, "length": 4, "type": "double_chop"}
    
    return {"active": False, "length": 0, "type": "double_chop"}


def detect_route_type(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    symbols = rounds_to_symbols(rounds, ignore_tie=True)
    
    dragon = detect_dragon(symbols)
    single_chop = detect_single_chop(symbols)
    double_chop = detect_double_chop(symbols)
    chop = detect_chop(symbols)
    two_room = detect_two_room(symbols)

    route_type = "normal"
    route_side = None
    route_length = 0

    if dragon["active"]:
        route_side = dragon["side"]
        route_length = dragon["length"]
        if route_length >= DRAGON_HOT_TRIGGER:
            route_type = "dragon_hot"
        elif route_length >= DRAGON_STRONG_TRIGGER:
            route_type = "dragon_strong"
        else:
            route_type = "dragon"
    elif single_chop["active"]:
        route_type = "single_chop"
        route_length = single_chop["length"]
    elif double_chop["active"]:
        route_type = "double_chop"
        route_length = double_chop["length"]
    elif chop["active"]:
        route_type = "chop"
        route_length = chop["length"]
    elif two_room["active"]:
        route_type = "two_room"
        route_length = two_room["length"]

    return {
        "route_type": route_type,
        "route_side": route_side,
        "route_length": route_length,
        "symbols_tail": "".join(symbols[-10:]),
        "dragon": dragon,
        "single_chop": single_chop,
        "double_chop": double_chop,
        "chop": chop,
        "two_room": two_room,
    }


def select_weights_by_count_gap(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = count_banker_player_gap(rounds)
    if not COUNT_GAP_MODE:
        return {
            "point_weight": POINT_WEIGHT,
            "pattern_weight": PATTERN_WEIGHT,
            "sim_weight": SIM_WEIGHT,
            "mode": "default",
            "count_gap_mode": False,
            **stats,
        }
    gap = stats["gap"]
    if gap >= EXTREME_COUNT_GAP_TRIGGER:
        return {
            "point_weight": EXTREME_GAP_POINT_WEIGHT,
            "pattern_weight": EXTREME_GAP_PATTERN_WEIGHT,
            "sim_weight": EXTREME_GAP_SIM_WEIGHT,
            "mode": "extreme_count_gap",
            "count_gap_mode": True,
            **stats,
        }
    if gap >= COUNT_GAP_TRIGGER:
        return {
            "point_weight": GAP_POINT_WEIGHT,
            "pattern_weight": GAP_PATTERN_WEIGHT,
            "sim_weight": GAP_SIM_WEIGHT,
            "mode": "count_gap",
            "count_gap_mode": True,
            **stats,
        }
    return {
        "point_weight": NORMAL_POINT_WEIGHT,
        "pattern_weight": NORMAL_PATTERN_WEIGHT,
        "sim_weight": NORMAL_SIM_WEIGHT,
        "mode": "normal",
        "count_gap_mode": True,
        **stats,
    }


def apply_route_weights(base_selected: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
    selected = dict(base_selected)
    if not ROUTE_MODE:
        selected["route_mode"] = False
        selected["route_applied"] = False
        selected["route_type"] = route["route_type"]
        return selected

    route_type = route.get("route_type", "normal")

    if route_type == "dragon":
        selected["point_weight"] = DRAGON_POINT_WEIGHT
        selected["pattern_weight"] = DRAGON_PATTERN_WEIGHT
        selected["sim_weight"] = DRAGON_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+dragon"
        selected["route_applied"] = True
    elif route_type == "dragon_strong":
        selected["point_weight"] = DRAGON_STRONG_POINT_WEIGHT
        selected["pattern_weight"] = DRAGON_STRONG_PATTERN_WEIGHT
        selected["sim_weight"] = DRAGON_STRONG_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+dragon_strong"
        selected["route_applied"] = True
    elif route_type == "dragon_hot":
        selected["point_weight"] = DRAGON_HOT_POINT_WEIGHT
        selected["pattern_weight"] = DRAGON_HOT_PATTERN_WEIGHT
        selected["sim_weight"] = DRAGON_HOT_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+dragon_hot"
        selected["route_applied"] = True
    elif route_type == "single_chop":
        selected["point_weight"] = SINGLE_CHOP_POINT_WEIGHT
        selected["pattern_weight"] = SINGLE_CHOP_PATTERN_WEIGHT
        selected["sim_weight"] = SINGLE_CHOP_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+single_chop"
        selected["route_applied"] = True
    elif route_type == "double_chop":
        selected["point_weight"] = DOUBLE_CHOP_POINT_WEIGHT
        selected["pattern_weight"] = DOUBLE_CHOP_PATTERN_WEIGHT
        selected["sim_weight"] = DOUBLE_CHOP_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+double_chop"
        selected["route_applied"] = True
    elif route_type == "chop":
        selected["point_weight"] = CHOP_POINT_WEIGHT
        selected["pattern_weight"] = CHOP_PATTERN_WEIGHT
        selected["sim_weight"] = CHOP_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+chop"
        selected["route_applied"] = True
    elif route_type == "two_room":
        selected["point_weight"] = TWO_ROOM_POINT_WEIGHT
        selected["pattern_weight"] = TWO_ROOM_PATTERN_WEIGHT
        selected["sim_weight"] = TWO_ROOM_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+two_room"
        selected["route_applied"] = True
    else:
        selected["route_applied"] = False

    selected["route_mode"] = True
    selected["route_type"] = route_type
    return selected


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

    selected = select_weights_by_count_gap(rounds)
    route = detect_route_type(rounds)
    selected = apply_route_weights(selected, route)

    p_w = selected["point_weight"]
    pat_w = selected["pattern_weight"] if pattern.get("available") else 0.0
    sim_w = selected["sim_weight"]
    if not pattern.get("available"):
        p_w += selected["pattern_weight"]

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
        "count_gap_debug": {
            "mode": selected["mode"],
            "count_gap_mode": selected["count_gap_mode"],
            "banker_count": selected["banker_count"],
            "player_count": selected["player_count"],
            "tie_count": selected["tie_count"],
            "gap": selected["gap"],
            "leader": selected["leader"],
            "raw_selected_weights": {
                "point": selected["point_weight"],
                "pattern": selected["pattern_weight"],
                "simulation": selected["sim_weight"],
            },
        },
        "route_debug": {
            "route_mode": selected.get("route_mode", False),
            "route_applied": selected.get("route_applied", False),
            "route_type": route["route_type"],
            "route_side": route["route_side"],
            "route_length": route["route_length"],
            "symbols_tail": route["symbols_tail"],
            "dragon": route["dragon"],
            "single_chop": route.get("single_chop"),
            "double_chop": route.get("double_chop"),
            "chop": route["chop"],
            "two_room": route["two_room"],
        },
        "no_observe": True,
    }
