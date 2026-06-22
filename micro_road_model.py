# -*- coding: utf-8 -*-
"""
micro_road_model.py - V10.2 短牌路命中率校準層（內建龍尾衰減）

重點：
- 只看最近 4~8 口，不吃長歷史、不做追路。
- 輸出方向與信心，讓 predictor.py 的 decision_controller 做最後命中率修正。
- V10.2 新增：長龍風險指標 dragon_tail_risk，自動衰減過熱趨勢的信心。
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
MICRO_ROAD_MAX_EDGE = env_float("MICRO_ROAD_MAX_EDGE", "0.085")
MICRO_ROAD_BASE_EDGE = env_float("MICRO_ROAD_BASE_EDGE", "0.030")
MICRO_ROAD_CONFIDENCE_SCALE = env_float("MICRO_ROAD_CONFIDENCE_SCALE", "1.35")

MICRO_ROAD_ZIGZAG_EDGE = env_float("MICRO_ROAD_ZIGZAG_EDGE", "0.050")
MICRO_ROAD_DOUBLE_JUMP_EDGE = env_float("MICRO_ROAD_DOUBLE_JUMP_EDGE", "0.054")
MICRO_ROAD_ROOM_EDGE = env_float("MICRO_ROAD_ROOM_EDGE", "0.044")
MICRO_ROAD_STREAK_EDGE = env_float("MICRO_ROAD_STREAK_EDGE", "0.022")
MICRO_ROAD_TRAP_EDGE = env_float("MICRO_ROAD_TRAP_EDGE", "0.070")
MICRO_ROAD_RECENT_REVERSAL_EDGE = env_float("MICRO_ROAD_RECENT_REVERSAL_EDGE", "0.038")

MICRO_ROAD_NATURAL_HIGH_TRAP = env_bool("MICRO_ROAD_NATURAL_HIGH_TRAP", "1")
MICRO_ROAD_MID_HIGH_GAP_TRAP = env_bool("MICRO_ROAD_MID_HIGH_GAP_TRAP", "1")

# V10.2 長龍衰減參數
MICRO_ROAD_STREAK_DECAY_START = env_int("MICRO_ROAD_STREAK_DECAY_START", "5")
MICRO_ROAD_STREAK_DECAY_RATE = env_float("MICRO_ROAD_STREAK_DECAY_RATE", "0.15")


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
            val = r.get("result") or r.get("winner") or r.get("last_result") or r.get("side")
            norm = normalize_result(val)
            if norm:
                out.append(norm)
                continue
            pp = r.get("player_point", r.get("player", r.get("p")))
            bp = r.get("banker_point", r.get("banker", r.get("b")))
            try:
                pp = int(pp)
                bp = int(bp)
                if 0 <= pp <= 9 and 0 <= bp <= 9:
                    out.append(result_from_points(pp, bp))
                    continue
            except Exception:
                pass
        elif isinstance(r, (list, tuple)):
            if len(r) >= 2:
                try:
                    pp = int(r[0])
                    bp = int(r[1])
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
    return "P" if side == "B" else "B" if side == "P" else "N"


def side_to_prob(side: str, edge: float) -> Tuple[float, float]:
    edge = clamp(float(edge), -MICRO_ROAD_MAX_EDGE, MICRO_ROAD_MAX_EDGE)
    banker = BASE_BANKER_NO_TIE
    if side == "B":
        banker += abs(edge)
    elif side == "P":
        banker -= abs(edge)
    banker = clamp(banker, 0.38, 0.62)
    return banker, 1.0 - banker


def filtered_bp(seq: List[str]) -> List[str]:
    return [x for x in seq if x in {"B", "P"}]


def runs(seq: List[str]) -> List[Tuple[str, int]]:
    f = filtered_bp(seq)
    if not f:
        return []
    out: List[Tuple[str, int]] = []
    cur = f[0]
    n = 1
    for x in f[1:]:
        if x == cur:
            n += 1
        else:
            out.append((cur, n))
            cur = x
            n = 1
    out.append((cur, n))
    return out


def last_streak(seq: List[str]) -> Tuple[str, int]:
    rr = runs(seq)
    return rr[-1] if rr else ("N", 0)


def is_zigzag(seq: List[str]) -> bool:
    f = filtered_bp(seq)
    if len(f) < 5:
        return False
    tail = f[-5:]
    return all(tail[i] != tail[i - 1] for i in range(1, len(tail)))


def is_double_jump_tail(seq: List[str]) -> bool:
    f = filtered_bp(seq)
    if len(f) < 6:
        return False
    tail = f[-6:]
    return (
        tail[0] == tail[1]
        and tail[2] == tail[3]
        and tail[4] == tail[5]
        and tail[0] != tail[2]
        and tail[2] != tail[4]
    )


def room_pattern_tail(seq: List[str]) -> Optional[str]:
    f = filtered_bp(seq)
    if len(f) < 6:
        return None
    tail = f[-6:]
    if tail[0] == tail[3] and tail[1] == tail[2] == tail[4] == tail[5] and tail[0] != tail[1]:
        return tail[0]
    if tail[0] == tail[1] == tail[3] == tail[4] and tail[2] == tail[5] and tail[0] != tail[2]:
        return tail[0]
    return None


def last_two_turn(seq: List[str]) -> bool:
    f = filtered_bp(seq)
    if len(f) < 4:
        return False
    tail = f[-4:]
    return (tail[-4] == tail[-3] and tail[-2] != tail[-1]) or (tail[-4] != tail[-3] and tail[-2] == tail[-1])


def infer_current_context(player_point: int, banker_point: int, composition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = int(player_point)
    b = int(banker_point)
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


def micro_road_lookup(
    player_point: int,
    banker_point: int,
    rounds: Optional[List[Any]] = None,
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    seq_all = extract_round_results(rounds)
    window = max(3, int(MICRO_ROAD_WINDOW))
    seq = filtered_bp(seq_all)[-window:]

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
            "dragon_tail_risk": 0.0,
        }

    ctx = infer_current_context(player_point, banker_point, composition)
    patterns: List[str] = []
    direction_scores = {"B": 0.0, "P": 0.0}
    last_side, streak_len = last_streak(seq)

    if is_zigzag(seq):
        d = opposite(seq[-1])
        direction_scores[d] += MICRO_ROAD_ZIGZAG_EDGE
        patterns.append("ZIGZAG_TURN")

    if is_double_jump_tail(seq):
        d = opposite(seq[-1])
        direction_scores[d] += MICRO_ROAD_DOUBLE_JUMP_EDGE
        patterns.append("DOUBLE_JUMP_TAIL")

    room_d = room_pattern_tail(seq)
    if room_d in {"B", "P"}:
        direction_scores[room_d] += MICRO_ROAD_ROOM_EDGE
        patterns.append("ROOM_PATTERN_TAIL")

    if last_two_turn(seq):
        d = opposite(seq[-1])
        direction_scores[d] += MICRO_ROAD_RECENT_REVERSAL_EDGE
        patterns.append("LAST_TWO_TURN")

    # 長龍跟隨，但超過衰減起點時邊際遞減
    if streak_len >= 3 and last_side in {"B", "P"}:
        base_streak_edge = MICRO_ROAD_STREAK_EDGE
        decay = 0.0
        if streak_len >= int(MICRO_ROAD_STREAK_DECAY_START):
            # 長龍越長，跟隨力道越弱（避免追尾）
            decay = min(0.8, (streak_len - MICRO_ROAD_STREAK_DECAY_START) * MICRO_ROAD_STREAK_DECAY_RATE)
        effective_streak_edge = base_streak_edge * (1.0 - decay)
        direction_scores[last_side] += effective_streak_edge * min(streak_len / 5.0, 1.0)
        patterns.append(f"STREAK_{last_side}_{streak_len}")
        if decay > 0:
            patterns.append("LONG_STREAK_WEAKEN")

    # 陷阱偵測（原邏輯）
    if (
        MICRO_ROAD_NATURAL_HIGH_TRAP
        and ctx["natural_high_winner"]
        and ctx["gap_family"] == "MID_HIGH_GAP_5_7"
        and any(x in patterns for x in ["ZIGZAG_TURN", "DOUBLE_JUMP_TAIL", "ROOM_PATTERN_TAIL", "LAST_TWO_TURN"])
    ):
        cur = ctx.get("winner", "T")
        if cur in {"B", "P"}:
            d = opposite(cur)
            direction_scores[d] += MICRO_ROAD_TRAP_EDGE
            patterns.append("NATURAL_HIGH_TRAP_REVERSAL")

    if (
        MICRO_ROAD_MID_HIGH_GAP_TRAP
        and ctx["gap_family"] == "MID_HIGH_GAP_5_7"
        and any(x in patterns for x in ["ZIGZAG_TURN", "DOUBLE_JUMP_TAIL", "LAST_TWO_TURN"])
    ):
        cur = ctx.get("winner", "T")
        if cur in {"B", "P"}:
            d = opposite(cur)
            direction_scores[d] += MICRO_ROAD_TRAP_EDGE * 0.55
            patterns.append("MID_HIGH_GAP_TURN_GUARD")

    # 計算龍尾風險（0~1）
    dragon_tail_risk = 0.0
    if streak_len >= int(MICRO_ROAD_STREAK_DECAY_START):
        dragon_tail_risk = min(1.0, (streak_len - MICRO_ROAD_STREAK_DECAY_START) * 0.2)

    b_score = direction_scores["B"]
    p_score = direction_scores["P"]

    if abs(b_score - p_score) < 0.004:
        n = stable_noise("".join(seq) + f":{player_point}:{banker_point}", 0.004)
        banker = clamp(BASE_BANKER_NO_TIE + n, 0.38, 0.62)
        micro_direction = "NEUTRAL"
        confidence = 0.0
    else:
        micro_direction = "B" if b_score > p_score else "P"
        raw_edge = abs(b_score - p_score) * MICRO_ROAD_CONFIDENCE_SCALE
        raw_edge = clamp(max(raw_edge, MICRO_ROAD_BASE_EDGE), 0.0, MICRO_ROAD_MAX_EDGE)

        # 長龍風險衰減信心
        confidence_decay = 1.0 - dragon_tail_risk * 0.7
        final_edge = raw_edge * confidence_decay
        banker, _ = side_to_prob(micro_direction, final_edge)
        confidence = clamp(final_edge / max(MICRO_ROAD_MAX_EDGE, 0.0001), 0.0, 1.0)

    return {
        "available": bool(patterns),
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "MICRO_ROAD_MODEL_V10_2",
        "sample_size": len(seq),
        "total_simulated_samples": len(seq),
        "micro_direction": micro_direction,
        "micro_confidence": confidence,
        "micro_patterns": patterns,
        "recent_road": "".join(seq),
        "direction_scores": direction_scores,
        "context": ctx,
        "dragon_tail_risk": dragon_tail_risk,
    }
