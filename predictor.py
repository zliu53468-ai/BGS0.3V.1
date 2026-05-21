import hashlib
import math
import os
from typing import Dict, Any, List, Optional, Tuple
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
import point_db as point_db_module
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


# ==================== 局型資料庫查詢模式 ====================
# ROUTE_DB_MODE=1：
# 會先嘗試用「點數 + route_type」查局型專用資料庫。
# 如果 point_db.py 尚未支援 get_point_route_record，會自動 fallback 回原本 get_point_record。
ROUTE_DB_MODE = _get_bool_env("ROUTE_DB_MODE", True)
ROUTE_DB_MIN_SAMPLE = _get_int_env("ROUTE_DB_MIN_SAMPLE", 30)
ROUTE_DB_FALLBACK_TO_BASE = _get_bool_env("ROUTE_DB_FALLBACK_TO_BASE", True)


# ==================== 莊閒累積開局差距權重切換器 ====================
COUNT_GAP_MODE = _get_bool_env("COUNT_GAP_MODE", False)

NORMAL_POINT_WEIGHT = _get_float_env("NORMAL_POINT_WEIGHT", POINT_WEIGHT)
NORMAL_PATTERN_WEIGHT = _get_float_env("NORMAL_PATTERN_WEIGHT", PATTERN_WEIGHT)
NORMAL_SIM_WEIGHT = _get_float_env("NORMAL_SIM_WEIGHT", SIM_WEIGHT)

COUNT_GAP_TRIGGER = _get_int_env("COUNT_GAP_TRIGGER", 3)
GAP_POINT_WEIGHT = _get_float_env("GAP_POINT_WEIGHT", 0.62)
GAP_PATTERN_WEIGHT = _get_float_env("GAP_PATTERN_WEIGHT", 0.23)
GAP_SIM_WEIGHT = _get_float_env("GAP_SIM_WEIGHT", 0.15)

EXTREME_COUNT_GAP_TRIGGER = _get_int_env("EXTREME_COUNT_GAP_TRIGGER", 6)
EXTREME_GAP_POINT_WEIGHT = _get_float_env("EXTREME_GAP_POINT_WEIGHT", 0.68)
EXTREME_GAP_PATTERN_WEIGHT = _get_float_env("EXTREME_GAP_PATTERN_WEIGHT", 0.17)
EXTREME_GAP_SIM_WEIGHT = _get_float_env("EXTREME_GAP_SIM_WEIGHT", 0.15)


# ==================== 局型偵測器 ====================
ROUTE_MODE = _get_bool_env("ROUTE_MODE", True)

DRAGON_TRIGGER = _get_int_env("DRAGON_TRIGGER", 3)
DRAGON_STRONG_TRIGGER = _get_int_env("DRAGON_STRONG_TRIGGER", 5)
DRAGON_HOT_TRIGGER = _get_int_env("DRAGON_HOT_TRIGGER", 7)

DRAGON_POINT_WEIGHT = _get_float_env("DRAGON_POINT_WEIGHT", 0.46)
DRAGON_PATTERN_WEIGHT = _get_float_env("DRAGON_PATTERN_WEIGHT", 0.40)
DRAGON_SIM_WEIGHT = _get_float_env("DRAGON_SIM_WEIGHT", 0.14)

DRAGON_STRONG_POINT_WEIGHT = _get_float_env("DRAGON_STRONG_POINT_WEIGHT", 0.44)
DRAGON_STRONG_PATTERN_WEIGHT = _get_float_env("DRAGON_STRONG_PATTERN_WEIGHT", 0.41)
DRAGON_STRONG_SIM_WEIGHT = _get_float_env("DRAGON_STRONG_SIM_WEIGHT", 0.15)

DRAGON_HOT_POINT_WEIGHT = _get_float_env("DRAGON_HOT_POINT_WEIGHT", 0.49)
DRAGON_HOT_PATTERN_WEIGHT = _get_float_env("DRAGON_HOT_PATTERN_WEIGHT", 0.37)
DRAGON_HOT_SIM_WEIGHT = _get_float_env("DRAGON_HOT_SIM_WEIGHT", 0.14)

CHOP_TRIGGER = _get_int_env("CHOP_TRIGGER", 4)
CHOP_POINT_WEIGHT = _get_float_env("CHOP_POINT_WEIGHT", 0.46)
CHOP_PATTERN_WEIGHT = _get_float_env("CHOP_PATTERN_WEIGHT", 0.40)
CHOP_SIM_WEIGHT = _get_float_env("CHOP_SIM_WEIGHT", 0.14)

TWO_ROOM_TRIGGER = _get_int_env("TWO_ROOM_TRIGGER", 4)
TWO_ROOM_POINT_WEIGHT = _get_float_env("TWO_ROOM_POINT_WEIGHT", 0.48)
TWO_ROOM_PATTERN_WEIGHT = _get_float_env("TWO_ROOM_PATTERN_WEIGHT", 0.38)
TWO_ROOM_SIM_WEIGHT = _get_float_env("TWO_ROOM_SIM_WEIGHT", 0.14)

SINGLE_CHOP_TRIGGER = _get_int_env("SINGLE_CHOP_TRIGGER", 4)
DOUBLE_CHOP_TRIGGER = _get_int_env("DOUBLE_CHOP_TRIGGER", 3)

SINGLE_CHOP_POINT_WEIGHT = _get_float_env("SINGLE_CHOP_POINT_WEIGHT", 0.44)
SINGLE_CHOP_PATTERN_WEIGHT = _get_float_env("SINGLE_CHOP_PATTERN_WEIGHT", 0.43)
SINGLE_CHOP_SIM_WEIGHT = _get_float_env("SINGLE_CHOP_SIM_WEIGHT", 0.13)

DOUBLE_CHOP_POINT_WEIGHT = _get_float_env("DOUBLE_CHOP_POINT_WEIGHT", 0.46)
DOUBLE_CHOP_PATTERN_WEIGHT = _get_float_env("DOUBLE_CHOP_PATTERN_WEIGHT", 0.41)
DOUBLE_CHOP_SIM_WEIGHT = _get_float_env("DOUBLE_CHOP_SIM_WEIGHT", 0.13)


# ==================== 斷路偵測 ====================
BREAK_ROUTE_MODE = _get_bool_env("BREAK_ROUTE_MODE", True)

BREAK_DRAGON_POINT_WEIGHT = _get_float_env("BREAK_DRAGON_POINT_WEIGHT", 0.52)
BREAK_DRAGON_PATTERN_WEIGHT = _get_float_env("BREAK_DRAGON_PATTERN_WEIGHT", 0.35)
BREAK_DRAGON_SIM_WEIGHT = _get_float_env("BREAK_DRAGON_SIM_WEIGHT", 0.13)

DRAGON_RECONNECT_POINT_WEIGHT = _get_float_env("DRAGON_RECONNECT_POINT_WEIGHT", 0.44)
DRAGON_RECONNECT_PATTERN_WEIGHT = _get_float_env("DRAGON_RECONNECT_PATTERN_WEIGHT", 0.43)
DRAGON_RECONNECT_SIM_WEIGHT = _get_float_env("DRAGON_RECONNECT_SIM_WEIGHT", 0.13)

BREAK_CHOP_POINT_WEIGHT = _get_float_env("BREAK_CHOP_POINT_WEIGHT", 0.51)
BREAK_CHOP_PATTERN_WEIGHT = _get_float_env("BREAK_CHOP_PATTERN_WEIGHT", 0.36)
BREAK_CHOP_SIM_WEIGHT = _get_float_env("BREAK_CHOP_SIM_WEIGHT", 0.13)

TWO_ROOM_BREAK_POINT_WEIGHT = _get_float_env("TWO_ROOM_BREAK_POINT_WEIGHT", 0.50)
TWO_ROOM_BREAK_PATTERN_WEIGHT = _get_float_env("TWO_ROOM_BREAK_PATTERN_WEIGHT", 0.37)
TWO_ROOM_BREAK_SIM_WEIGHT = _get_float_env("TWO_ROOM_BREAK_SIM_WEIGHT", 0.13)


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


def detect_single_chop(symbols: List[str]) -> Dict[str, Any]:
    if len(symbols) < SINGLE_CHOP_TRIGGER:
        return {"active": False, "length": 0, "type": "single_chop"}

    length = 1

    for i in range(len(symbols) - 1, 0, -1):
        if symbols[i] != symbols[i - 1]:
            length += 1
        else:
            break

    return {
        "active": length >= SINGLE_CHOP_TRIGGER,
        "length": length,
        "type": "single_chop",
    }


def detect_double_chop(symbols: List[str]) -> Dict[str, Any]:
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


def count_same_run_before_index(symbols: List[str], end_exclusive: int, side: str) -> int:
    length = 0
    i = end_exclusive - 1

    while i >= 0 and symbols[i] == side:
        length += 1
        i -= 1

    return length


def detect_break_route(symbols: List[str]) -> Dict[str, Any]:
    default = {
        "active": False,
        "type": "none",
        "broken_side": None,
        "break_side": None,
        "length": 0,
        "pattern": "".join(symbols[-10:]),
    }

    if not BREAK_ROUTE_MODE or len(symbols) < 4:
        return default

    # 斷一口後接回：BBBBP B / PPPPB P
    if len(symbols) >= DRAGON_TRIGGER + 2:
        last = symbols[-1]
        breaker = symbols[-2]

        if last != breaker:
            previous_run_len = count_same_run_before_index(symbols, len(symbols) - 1, last)

            if previous_run_len >= DRAGON_TRIGGER:
                return {
                    "active": True,
                    "type": "dragon_reconnect",
                    "broken_side": "莊" if last == "B" else "閒",
                    "break_side": "莊" if breaker == "B" else "閒",
                    "length": previous_run_len,
                    "pattern": "".join(symbols[-10:]),
                }

    # 長龍剛被打斷：BBBBP / PPPPB
    if len(symbols) >= DRAGON_TRIGGER + 1:
        last = symbols[-1]
        previous = symbols[-2]

        if last != previous:
            previous_run_len = count_same_run_before_index(symbols, len(symbols) - 1, previous)

            if previous_run_len >= DRAGON_TRIGGER:
                return {
                    "active": True,
                    "type": "break_dragon",
                    "broken_side": "莊" if previous == "B" else "閒",
                    "break_side": "莊" if last == "B" else "閒",
                    "length": previous_run_len,
                    "pattern": "".join(symbols[-10:]),
                }

    # 單跳破壞：前面是單跳，最後一口變成連續同方
    if len(symbols) >= CHOP_TRIGGER + 1:
        prev_symbols = symbols[:-1]
        prev_chop = detect_chop(prev_symbols)

        if prev_chop.get("active") and symbols[-1] == symbols[-2]:
            return {
                "active": True,
                "type": "break_chop",
                "broken_side": None,
                "break_side": "莊" if symbols[-1] == "B" else "閒",
                "length": prev_chop.get("length", 0),
                "pattern": "".join(symbols[-10:]),
            }

    # 兩房破壞：前面是兩房，最後一口讓兩房不成立
    if len(symbols) >= TWO_ROOM_TRIGGER + 1:
        prev_symbols = symbols[:-1]
        prev_two_room = detect_two_room(prev_symbols)
        now_two_room = detect_two_room(symbols)

        if prev_two_room.get("active") and not now_two_room.get("active"):
            return {
                "active": True,
                "type": "two_room_break",
                "broken_side": None,
                "break_side": "莊" if symbols[-1] == "B" else "閒",
                "length": prev_two_room.get("length", 0),
                "pattern": "".join(symbols[-10:]),
            }

    return default


def detect_route_type(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    symbols = rounds_to_symbols(rounds, ignore_tie=True)

    dragon = detect_dragon(symbols)
    single_chop = detect_single_chop(symbols)
    double_chop = detect_double_chop(symbols)
    chop = detect_chop(symbols)
    two_room = detect_two_room(symbols)
    break_route = detect_break_route(symbols)

    route_type = "normal"
    route_side = None
    route_length = 0

    if break_route["active"]:
        route_type = break_route["type"]
        route_side = break_route.get("break_side")
        route_length = break_route.get("length", 0)

    elif dragon["active"]:
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
        "break_route": break_route,
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

    if route_type == "break_dragon":
        selected["point_weight"] = BREAK_DRAGON_POINT_WEIGHT
        selected["pattern_weight"] = BREAK_DRAGON_PATTERN_WEIGHT
        selected["sim_weight"] = BREAK_DRAGON_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+break_dragon"
        selected["route_applied"] = True

    elif route_type == "dragon_reconnect":
        selected["point_weight"] = DRAGON_RECONNECT_POINT_WEIGHT
        selected["pattern_weight"] = DRAGON_RECONNECT_PATTERN_WEIGHT
        selected["sim_weight"] = DRAGON_RECONNECT_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+dragon_reconnect"
        selected["route_applied"] = True

    elif route_type == "break_chop":
        selected["point_weight"] = BREAK_CHOP_POINT_WEIGHT
        selected["pattern_weight"] = BREAK_CHOP_PATTERN_WEIGHT
        selected["sim_weight"] = BREAK_CHOP_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+break_chop"
        selected["route_applied"] = True

    elif route_type == "two_room_break":
        selected["point_weight"] = TWO_ROOM_BREAK_POINT_WEIGHT
        selected["pattern_weight"] = TWO_ROOM_BREAK_PATTERN_WEIGHT
        selected["sim_weight"] = TWO_ROOM_BREAK_SIM_WEIGHT
        selected["mode"] = f"{selected['mode']}+two_room_break"
        selected["route_applied"] = True

    elif route_type == "dragon":
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


def _record_float(rec: Dict[str, Any], keys: List[str], default: float) -> float:
    for key in keys:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                pass
    return default


def _record_int(rec: Dict[str, Any], keys: List[str], default: int) -> int:
    for key in keys:
        if key in rec:
            try:
                return int(rec[key])
            except Exception:
                pass
    return default


def _format_route_key(player_point: int, banker_point: int, route_type: str, count_gap_mode: str = "") -> List[str]:
    base = f"P{player_point}_B{banker_point}"
    keys = [
        f"{base}_ROUTE_{route_type}",
        f"{base}_R_{route_type}",
        f"{base}:{route_type}",
        f"{route_type}:{base}",
    ]

    if count_gap_mode:
        keys.extend([
            f"{base}_ROUTE_{route_type}_GAP_{count_gap_mode}",
            f"{base}_R_{route_type}_G_{count_gap_mode}",
            f"{route_type}:{count_gap_mode}:{base}",
        ])

    return keys


def _try_call_route_db_function(
    fn,
    player_point: int,
    banker_point: int,
    route_type: str,
    count_gap_mode: str,
) -> Optional[Dict[str, Any]]:
    call_patterns = [
        lambda: fn(player_point, banker_point, route_type),
        lambda: fn(player_point, banker_point, route_type, count_gap_mode),
        lambda: fn(player_point=player_point, banker_point=banker_point, route_type=route_type),
        lambda: fn(player_point=player_point, banker_point=banker_point, route_type=route_type, count_gap_mode=count_gap_mode),
        lambda: fn(player_point=player_point, banker_point=banker_point, route=route_type),
    ]

    for call in call_patterns:
        try:
            rec = call()
            if isinstance(rec, dict):
                return rec
        except TypeError:
            continue
        except Exception:
            continue

    return None


def _try_get_route_record_from_point_db(
    player_point: int,
    banker_point: int,
    route_type: str,
    count_gap_mode: str,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """
    嘗試從 point_db.py 取得「點數 + 局型」專用資料。

    支援以下任一種寫法：
    1. point_db.py 有 get_point_route_record()
    2. point_db.py 有 get_route_point_record()
    3. point_db.py 有 get_context_point_record()
    4. point_db.py 有 POINT_ROUTE_DB / ROUTE_POINT_DB dict
    """

    if not ROUTE_DB_MODE or not route_type or route_type == "normal":
        return None, "ROUTE_DB_DISABLED_OR_NORMAL", ""

    function_names = [
        "get_point_route_record",
        "get_route_point_record",
        "get_context_point_record",
        "get_point_context_record",
    ]

    for fn_name in function_names:
        fn = getattr(point_db_module, fn_name, None)
        if callable(fn):
            rec = _try_call_route_db_function(fn, player_point, banker_point, route_type, count_gap_mode)
            if isinstance(rec, dict):
                return rec, f"ROUTE_DB_FUNCTION:{fn_name}", ""

    keys = _format_route_key(player_point, banker_point, route_type, count_gap_mode)

    dict_names = [
        "POINT_ROUTE_DB",
        "ROUTE_POINT_DB",
        "POINT_CONTEXT_DB",
        "CONTEXT_POINT_DB",
    ]

    for dict_name in dict_names:
        db = getattr(point_db_module, dict_name, None)
        if isinstance(db, dict):
            for key in keys:
                rec = db.get(key)
                if isinstance(rec, dict):
                    return rec, f"ROUTE_DB_DICT:{dict_name}", key

    return None, "ROUTE_DB_NOT_FOUND_FALLBACK_BASE", ""


def _build_point_result_from_record(
    rec: Dict[str, Any],
    player_point: int,
    banker_point: int,
    source: str,
    feature_key_value: str,
    available: bool = True,
) -> Dict[str, Any]:
    banker = _record_float(
        rec,
        ["next_banker_rate", "banker_prob", "banker_rate", "b_rate", "banker"],
        BASE_BANKER_NO_TIE,
    )

    player = _record_float(
        rec,
        ["next_player_rate", "player_prob", "player_rate", "p_rate", "player"],
        1.0 - banker,
    )

    total = banker + player
    if total > 0:
        banker = banker / total
        player = player / total
    else:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker

    sample_size = _record_int(rec, ["sample", "sample_size", "samples", "n"], 0)

    return {
        "available": available,
        "feature_key": feature_key_value,
        "banker_prob": clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB),
        "player_prob": 1.0 - clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB),
        "source": rec.get("source", source),
        "sample_size": sample_size,
        "total_simulated_samples": int(point_db_meta().get("total_simulated_samples", 0)),
        "route_db_used": source.startswith("ROUTE_DB"),
        "route_db_key": feature_key_value,
    }


def point_db_lookup(
    player_point: int,
    banker_point: int,
    route_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_context = route_context or {}
    route_type = str(route_context.get("route_type", "normal"))
    count_gap_mode = str(route_context.get("count_gap_mode", ""))

    route_rec, route_source, route_key = _try_get_route_record_from_point_db(
        player_point,
        banker_point,
        route_type,
        count_gap_mode,
    )

    if isinstance(route_rec, dict):
        sample_size = _record_int(route_rec, ["sample", "sample_size", "samples", "n"], 0)

        if sample_size >= ROUTE_DB_MIN_SAMPLE or sample_size == 0:
            feature = route_key or f"P{player_point}_B{banker_point}_ROUTE_{route_type}"
            return _build_point_result_from_record(
                route_rec,
                player_point,
                banker_point,
                route_source,
                feature,
                available=True,
            )

    if not ROUTE_DB_FALLBACK_TO_BASE:
        return fallback_point_lookup(player_point, banker_point)

    rec = get_point_record(player_point, banker_point)
    base_result = {
        "available": True,
        "feature_key": f"P{player_point}_B{banker_point}",
        "banker_prob": float(rec["next_banker_rate"]),
        "player_prob": float(rec["next_player_rate"]),
        "source": rec.get("source", "POINT_DB"),
        "sample_size": int(rec.get("sample", 0)),
        "total_simulated_samples": int(point_db_meta().get("total_simulated_samples", 0)),
        "route_db_used": False,
        "route_db_key": "",
        "route_db_status": route_source,
        "route_type": route_type,
    }
    return base_result


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
        "route_db_used": False,
        "route_db_key": "",
        "route_db_status": "FALLBACK_POINT_RULE",
        "route_type": "",
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

    selected = select_weights_by_count_gap(rounds)
    route = detect_route_type(rounds)
    selected = apply_route_weights(selected, route)

    route_context = {
        "route_type": route.get("route_type", "normal"),
        "count_gap_mode": selected.get("mode", ""),
        "count_gap": selected.get("gap", 0),
        "leader": selected.get("leader", "平衡"),
    }

    try:
        point = point_db_lookup(player_point, banker_point, route_context) if USE_POINT_DB else fallback_point_lookup(player_point, banker_point)
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
        "route_db_debug": {
            "route_db_mode": ROUTE_DB_MODE,
            "route_db_used": point.get("route_db_used", False),
            "route_db_key": point.get("route_db_key", ""),
            "route_db_status": point.get("route_db_status", ""),
            "route_type_for_db": route_context["route_type"],
            "count_gap_mode_for_db": route_context["count_gap_mode"],
            "route_db_min_sample": ROUTE_DB_MIN_SAMPLE,
        },
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
            "break_route": route["break_route"],
        },
        "no_observe": True,
    }
