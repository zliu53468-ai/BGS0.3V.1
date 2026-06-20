# -*- coding: utf-8 -*-
"""
micro_road_model.py - V10 短牌路命中率校準層

設計理念：
- 不取代 point_db / combo_db / road_profile / composition_mc。
- 不吃長歷史、不做追路、不需要暖機。
- 只看最近 4~8 口 B/P 結果，抓「轉折口、雙跳尾、房廳尾、高點陷阱」。
- 不是觀望層，而是輸出 banker_prob / player_prob 直接參與 predictor 融合。
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import hashlib

BASE_BANKER_NO_TIE = 0.5000


def env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


MICRO_ROAD_WINDOW = env_int("MICRO_ROAD_WINDOW", "8")
MICRO_ROAD_MIN_ROUNDS = env_int("MICRO_ROAD_MIN_ROUNDS", "4")
MICRO_ROAD_MAX_EDGE = env_float("MICRO_ROAD_MAX_EDGE", "0.055")
MICRO_ROAD_BASE_EDGE = env_float("MICRO_ROAD_BASE_EDGE", "0.018")
MICRO_ROAD_CONFIDENCE_SCALE = env_float("MICRO_ROAD_CONFIDENCE_SCALE", "1.00")

MICRO_ROAD_ZIGZAG_EDGE = env_float("MICRO_ROAD_ZIGZAG_EDGE", "0.030")
MICRO_ROAD_DOUBLE_JUMP_EDGE = env_float("MICRO_ROAD_DOUBLE_JUMP_EDGE", "0.034")
MICRO_ROAD_ROOM_EDGE = env_float("MICRO_ROAD_ROOM_EDGE", "0.028")
MICRO_ROAD_STREAK_EDGE = env_float("MICRO_ROAD_STREAK_EDGE", "0.026")
MICRO_ROAD_TRAP_EDGE = env_float("MICRO_ROAD_TRAP_EDGE", "0.038")

MICRO_ROAD_NATURAL_HIGH_TRAP = env_bool("MICRO_ROAD_NATURAL_HIGH_TRAP", "1")
MICRO_ROAD_MID_HIGH_GAP_TRAP = env_bool("MICRO_ROAD_MID_HIGH_GAP_TRAP", "1")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_noise(key: str, scale: float = 0.004) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def result_from_points(player_point: int, banker_point: int) -> str:
    if int(player_point) > int(banker_point):
        return "P"
    if int(banker_point) > int(player_point):
        return "B"
    return "T"


def normalize_result(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"B", "BANKER", "莊", "庄"}:
        return "B"
    if s in {"P", "PLAYER", "閒", "闲"}:
        return "P"
    if s in {"T", "TIE", "和"}:
        return "T"
    return None


def extract_round_results(rounds: Optional[List[Any]]) -> List[str]:
    if not rounds:
        return []
    out: List[str] = []
    for r in rounds:
        if isinstance(r, dict):
            norm = normalize_result(r.get("result") or r.get("winner") or r.get("last_result") or r.get("side"))
            if norm:
                out.append(norm)
                continue
            pp = r.get("player_point", r.get("player", r.get("p")))
            bp = r.get("banker_point", r.get("banker", r.get("b")))
            try:
                pp, bp = int(pp), int(bp)
                if 0 <= pp <= 9 and 0 <= bp <= 9:
                    out.append(result_from_points(pp, bp))
                    continue
            except Exception:
                pass
        elif isinstance(r, (list, tuple)):
            if len(r) >= 2:
                try:
                    pp, bp = int(r[0]), int(r[1])
                    if 0 <= pp <= 9 and 0 <= bp <= 9:
                        out.append(result_from_points(pp, bp))
                        continue
                except Exception:
                    pass
            if len(r) >= 1:
                norm = normalize_result(r[0])
                if norm:
                    out.append(norm)
        else:
            norm = normalize_result(r)
            if norm:
                out.append(norm)
    return out


def opposite(side: str) -> str:
    return "P" if side == "B" else "B"


def side_to_prob(side: str, edge: float) -> Tuple[float, float]:
    edge = clamp(abs(float(edge)), 0.0, MICRO_ROAD_MAX_EDGE)
    banker = BASE_BANKER_NO_TIE
    if side == "B":
        banker += edge
    elif side == "P":
        banker -= edge
    banker = clamp(banker, 0.38, 0.62)
    return banker, 1.0 - banker


def bp_only(seq: List[str]) -> List[str]:
    return [x for x in seq if x in {"B", "P"}]


def runs(seq: List[str]) -> List[Tuple[str, int]]:
    filtered = bp_only(seq)
    if not filtered:
        return []
    out: List[Tuple[str, int]] = []
    cur = filtered[0]
    n = 1
    for x in filtered[1:]:
        if x == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = x, 1
    out.append((cur, n))
    return out


def last_streak(seq: List[str]) -> Tuple[str, int]:
    rr = runs(seq)
    if not rr:
        return "N", 0
    return rr[-1]


def is_zigzag(seq: List[str]) -> bool:
    f = bp_only(seq)
    if len(f) < 5:
        return False
    tail = f[-5:]
    return all(tail[i] != tail[i - 1] for i in range(1, len(tail)))


def is_double_jump_tail(seq: List[str]) -> bool:
    f = bp_only(seq)
    if len(f) < 6:
        return False
    t = f[-6:]
    return t[0] == t[1] and t[2] == t[3] and t[4] == t[5] and t[0] != t[2] and t[2] != t[4]


def room_pattern_tail(seq: List[str]) -> Optional[str]:
    f = bp_only(seq)
    if len(f) < 6:
        return None
    t = f[-6:]
    if t[0] == t[3] and t[1] == t[2] == t[4] == t[5] and t[0] != t[1]:
        return t[0]
    if t[0] == t[1] == t[3] == t[4] and t[2] == t[5] and t[0] != t[2]:
        return t[0]
    return None


def infer_current_context(player_point: int, banker_point: int, composition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p, b = int(player_point), int(banker_point)
    gap = abs(p - b)
    winner = "P" if p > b else "B" if b > p else "T"
    winner_point = max(p, b) if winner != "T" else p
    if gap == 0:
        gap_family = "TIE_GAP"
    elif gap <= 2:
        gap_family = "TINY_GAP_1_2"
    elif gap <= 4:
        gap_family = "LOW_MID_GAP_3_4"
    elif gap <= 7:
        gap_family = "MID_HIGH_GAP_5_7"
    else:
        gap_family = "EXTREME_GAP_8_9"
    comp = composition or {}
    top_scenario = str(comp.get("top_scenario", "UNKNOWN"))
    natural_high = bool(comp.get("natural_high_winner") or (top_scenario == "NONE_DRAW" and winner_point in (8, 9)))
    return {
        "winner": winner,
        "winner_point": winner_point,
        "point_gap": gap,
        "gap_family": gap_family,
        "top_scenario": top_scenario,
        "natural_high_winner": natural_high,
    }


def micro_road_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None, composition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    seq_all = extract_round_results(rounds)
    seq = bp_only(seq_all)[-max(3, int(MICRO_ROAD_WINDOW)):]

    if len(seq) < max(1, int(MICRO_ROAD_MIN_ROUNDS)):
        return {
            "available": False,
            "banker_prob": BASE_BANKER_NO_TIE,
            "player_prob": 1.0 - BASE_BANKER_NO_TIE,
            "source": "MICRO_ROAD_NOT_ENOUGH_ROUNDS",
            "sample_size": len(seq),
            "micro_direction": "NEUTRAL",
            "micro_confidence": 0.0,
            "micro_patterns": [],
            "recent_road": "".join(seq),
        }

    ctx = infer_current_context(player_point, banker_point, composition)
    patterns: List[str] = []
    scores = {"B": 0.0, "P": 0.0}
    last_side, streak_len = last_streak(seq)

    if is_zigzag(seq):
        d = opposite(seq[-1])
        scores[d] += MICRO_ROAD_ZIGZAG_EDGE
        patterns.append("ZIGZAG_TURN")

    if is_double_jump_tail(seq):
        d = opposite(seq[-1])
        scores[d] += MICRO_ROAD_DOUBLE_JUMP_EDGE
        patterns.append("DOUBLE_JUMP_TAIL")

    room_d = room_pattern_tail(seq)
    if room_d in {"B", "P"}:
        scores[room_d] += MICRO_ROAD_ROOM_EDGE
        patterns.append("ROOM_PATTERN_TAIL")

    if streak_len >= 3 and last_side in {"B", "P"}:
        scores[last_side] += MICRO_ROAD_STREAK_EDGE * min(streak_len / 5.0, 1.0)
        patterns.append(f"STREAK_{last_side}_{streak_len}")

    if MICRO_ROAD_NATURAL_HIGH_TRAP and ctx["natural_high_winner"] and ctx["gap_family"] == "MID_HIGH_GAP_5_7":
        if "ZIGZAG_TURN" in patterns or "DOUBLE_JUMP_TAIL" in patterns or "ROOM_PATTERN_TAIL" in patterns:
            if ctx["winner"] in {"B", "P"}:
                d = opposite(ctx["winner"])
                scores[d] += MICRO_ROAD_TRAP_EDGE
                patterns.append("NATURAL_HIGH_TRAP_REVERSAL")

    if MICRO_ROAD_MID_HIGH_GAP_TRAP and ctx["gap_family"] == "MID_HIGH_GAP_5_7":
        if "ZIGZAG_TURN" in patterns or "DOUBLE_JUMP_TAIL" in patterns:
            if ctx["winner"] in {"B", "P"}:
                d = opposite(ctx["winner"])
                scores[d] += MICRO_ROAD_TRAP_EDGE * 0.55
                patterns.append("MID_HIGH_GAP_TURN_GUARD")

    b_score = scores["B"]
    p_score = scores["P"]
    if abs(b_score - p_score) < 0.003:
        banker = clamp(BASE_BANKER_NO_TIE + stable_noise("".join(seq) + f":{player_point}:{banker_point}", 0.004), 0.38, 0.62)
        direction = "NEUTRAL"
        confidence = 0.0
    else:
        direction = "B" if b_score > p_score else "P"
        edge = clamp(max(abs(b_score - p_score), MICRO_ROAD_BASE_EDGE) * MICRO_ROAD_CONFIDENCE_SCALE, 0.0, MICRO_ROAD_MAX_EDGE)
        banker, _ = side_to_prob(direction, edge)
        confidence = clamp(edge / max(MICRO_ROAD_MAX_EDGE, 0.0001), 0.0, 1.0)

    return {
        "available": bool(patterns),
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "MICRO_ROAD_MODEL_V10",
        "sample_size": len(seq),
        "total_simulated_samples": len(seq),
        "micro_direction": direction,
        "micro_confidence": confidence,
        "micro_patterns": patterns,
        "recent_road": "".join(seq),
        "direction_scores": scores,
        "context": ctx,
    }
