import hashlib
import json
import os
import random
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import config
from point_db import get_point_record, point_db_meta

try:
    from combo_db import combo_lookup, combo_db_meta
except Exception:
    combo_lookup = None
    combo_db_meta = None


# ============================================================
# 環境變數 / config 參數讀取
# ============================================================

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


POINT_WEIGHT = env_float("POINT_WEIGHT", str(getattr(config, "POINT_WEIGHT", 0.55)))
COMBO_WEIGHT = env_float("COMBO_WEIGHT", str(getattr(config, "COMBO_WEIGHT", 0.30)))
PATTERN_WEIGHT = env_float("PATTERN_WEIGHT", str(getattr(config, "PATTERN_WEIGHT", 0.03)))
SIM_WEIGHT = env_float("SIM_WEIGHT", str(getattr(config, "SIM_WEIGHT", 0.12)))

MIN_OUTPUT_PROB = env_float("MIN_OUTPUT_PROB", str(getattr(config, "MIN_OUTPUT_PROB", 0.38)))
MAX_OUTPUT_PROB = env_float("MAX_OUTPUT_PROB", str(getattr(config, "MAX_OUTPUT_PROB", 0.62)))
PERCENT_DECIMALS = env_int("PERCENT_DECIMALS", str(getattr(config, "PERCENT_DECIMALS", 2)))

USE_POINT_DB = env_bool("USE_POINT_DB", "1" if getattr(config, "USE_POINT_DB", True) else "0")
USE_PATTERN_DB = env_bool("USE_PATTERN_DB", "1" if getattr(config, "USE_PATTERN_DB", True) else "0")
USE_COMBO_DB = env_bool("USE_COMBO_DB", "1" if getattr(config, "USE_COMBO_DB", True) else "0")
COMBO_MIN_SAMPLE = env_int("COMBO_MIN_SAMPLE", str(getattr(config, "COMBO_MIN_SAMPLE", 80)))

POINT_DB_PATH = os.getenv("POINT_DB_PATH", getattr(config, "POINT_DB_PATH", "data/point_db_3m.json")).strip()
PATTERN_DB_PATH = os.getenv("RESULT_PATTERN_DB_PATH", getattr(config, "RESULT_PATTERN_DB_PATH", "data/result_pattern_db_3m.json")).strip()
COMBO_DB_PATH = os.getenv("COMBO_DB_PATH", getattr(config, "COMBO_DB_PATH", "data/combo_db_3m.json")).strip()


# ============================================================
# 模型基準參數
# ============================================================

BASE_BANKER_NO_TIE = env_float("BASE_BANKER_NO_TIE", "0.5068")

MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", str(getattr(config, "MIN_GAP_FOR_ENTRY", 0.060)))
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", str(getattr(config, "STRONG_GAP_FOR_ENTRY", 0.090)))

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", str(getattr(config, "TIE_AI_MAX_WEIGHT", 0.01)))
TIE_SHRINK = env_float("TIE_SHRINK", str(getattr(config, "TIE_SHRINK", 0.18)))
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", str(getattr(config, "TIE_MIN_GAP_FOR_ENTRY", 0.12)))

AI_NOISE_SCALE = env_float("AI_NOISE_SCALE", str(getattr(config, "AI_NOISE_SCALE", 0.008)))

USE_MONTE_CARLO = env_bool("USE_MONTE_CARLO", "1" if getattr(config, "USE_MONTE_CARLO", True) else "0")
MC_SIMULATIONS = env_int("MC_SIMULATIONS", str(getattr(config, "MC_SIMULATIONS", 600)))
MC_MIN_SIMULATIONS = env_int("MC_MIN_SIMULATIONS", str(getattr(config, "MC_MIN_SIMULATIONS", 100)))
MC_MAX_SIMULATIONS = env_int("MC_MAX_SIMULATIONS", str(getattr(config, "MC_MAX_SIMULATIONS", 900)))
MC_SEED = env_int("MC_SEED", str(getattr(config, "MC_SEED", 42)))
MC_MAX_NOISE = env_float("MC_MAX_NOISE", str(getattr(config, "MC_MAX_NOISE", 0.018)))
MC_BLOCK_LOW_GAP = env_bool("MC_BLOCK_LOW_GAP", "1" if getattr(config, "MC_BLOCK_LOW_GAP", True) else "0")
MC_MIN_GAP_FOR_ENTRY = env_float("MC_MIN_GAP_FOR_ENTRY", str(getattr(config, "MC_MIN_GAP_FOR_ENTRY", 0.055)))
MC_DIRECTION_MISMATCH_BLOCK = env_bool("MC_DIRECTION_MISMATCH_BLOCK", "1" if getattr(config, "MC_DIRECTION_MISMATCH_BLOCK", True) else "0")

AI_HISTORY_WINDOW = env_int("AI_HISTORY_WINDOW", str(getattr(config, "AI_HISTORY_WINDOW", 8)))
AI_TREND_STRENGTH = env_float("AI_TREND_STRENGTH", str(getattr(config, "AI_TREND_STRENGTH", 0.010)))
AI_DIFF_MOMENTUM_STRENGTH = env_float("AI_DIFF_MOMENTUM_STRENGTH", str(getattr(config, "AI_DIFF_MOMENTUM_STRENGTH", 0.009)))
AI_REVERSAL_STRENGTH = env_float("AI_REVERSAL_STRENGTH", str(getattr(config, "AI_REVERSAL_STRENGTH", 0.016)))
AI_HISTORY_MAX_ADJUST = env_float("AI_HISTORY_MAX_ADJUST", str(getattr(config, "AI_HISTORY_MAX_ADJUST", 0.018)))


# ============================================================
# 工具函式
# ============================================================

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
    return (
        f"P{player_point}_B{banker_point}"
        f"_R{get_last_result(player_point, banker_point)}"
        f"_D{diff}"
        f"_Z{diff_zone(diff)}"
        f"_PZ{point_zone(player_point)}"
        f"_BZ{point_zone(banker_point)}"
    )


def simple_point_key(player_point: int, banker_point: int) -> str:
    return f"P{player_point}_B{banker_point}"


def normalize_prob_pair(banker: float, player: float) -> Tuple[float, float]:
    banker = float(banker)
    player = float(player)

    if banker > 1.0:
        banker = banker / 100.0
    if player > 1.0:
        player = player / 100.0

    total = banker + player
    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker = banker / total
        player = player / total

    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    return banker, player


def neutral_record(source: str = "NEUTRAL_FALLBACK") -> Dict[str, Any]:
    banker = clamp(BASE_BANKER_NO_TIE, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    return {
        "available": False,
        "feature_key": "NEUTRAL",
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": source,
        "sample_size": 0,
        "total_simulated_samples": 0,
    }


# ============================================================
# POINT DB 查詢
# ============================================================

def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE

    if diff == 0:
        banker += stable_noise(key + ":tie", 0.018)
    elif 1 <= diff <= 2:
        banker -= 0.08
    elif 3 <= diff <= 5:
        banker -= 0.10
    elif diff >= 6:
        banker -= 0.06
    elif -2 <= diff <= -1:
        banker += 0.08
    elif -5 <= diff <= -3:
        banker += 0.10
    elif diff <= -6:
        banker += 0.06

    banker += stable_noise(key + ":fallback", 0.025)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "available": False,
        "feature_key": key,
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "FALLBACK_POINT_RULE_ONLY",
        "sample_size": 0,
        "total_simulated_samples": 0,
    }


def point_db_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    if not USE_POINT_DB:
        return fallback_point_lookup(player_point, banker_point)

    try:
        rec = get_point_record(player_point, banker_point)
        banker = rec.get("next_banker_rate", rec.get("banker_prob", rec.get("banker_rate", None)))
        player = rec.get("next_player_rate", rec.get("player_prob", rec.get("player_rate", None)))
        if banker is None or player is None:
            return fallback_point_lookup(player_point, banker_point)

        banker, player = normalize_prob_pair(float(banker), float(player))
        meta = point_db_meta()

        return {
            "available": True,
            "feature_key": simple_point_key(player_point, banker_point),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_DB"),
            "sample_size": int(rec.get("sample", rec.get("sample_size", rec.get("no_tie_sample", 0))) or 0),
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        }
    except Exception as e:
        out = fallback_point_lookup(player_point, banker_point)
        out["point_error"] = str(e)
        return out


# ============================================================
# PATTERN DB 查詢
# ============================================================

def pattern_db_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not USE_PATTERN_DB:
        rec = neutral_record("PATTERN_DB_DISABLED")
        rec["feature_key"] = feature_key(player_point, banker_point)
        rec["pattern"] = ""
        rec["window"] = 0
        return rec

    try:
        import pattern_db
        fn = getattr(pattern_db, "pattern_lookup", None)
        if not callable(fn):
            raise RuntimeError("pattern_lookup not found")
        rec = fn(rounds or [])
        if not isinstance(rec, dict):
            raise RuntimeError("pattern_lookup returned non-dict")

        banker = rec.get("banker_prob", rec.get("next_banker_rate", BASE_BANKER_NO_TIE))
        player = rec.get("player_prob", rec.get("next_player_rate", 1.0 - BASE_BANKER_NO_TIE))
        banker, player = normalize_prob_pair(float(banker), float(player))

        meta = pattern_db.pattern_db_meta_safe() if hasattr(pattern_db, "pattern_db_meta_safe") else pattern_db.pattern_db_meta()

        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", "PATTERN"),
            "pattern": rec.get("pattern", ""),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "PATTERN_DB"),
            "sample_size": int(rec.get("sample_size", rec.get("sample", 0)) or 0),
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
            "window": int(rec.get("window", 0) or 0),
            "matched": rec.get("matched", []),
        }
    except Exception as e:
        rec = neutral_record(f"PATTERN_DB_ERROR:{e}")
        rec["feature_key"] = feature_key(player_point, banker_point)
        rec["pattern"] = ""
        rec["window"] = 0
        rec["matched"] = []
        return rec


# ============================================================
# COMBO DB 查詢
# ============================================================

def combo_db_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not USE_COMBO_DB or combo_lookup is None:
        rec = neutral_record("COMBO_DB_DISABLED")
        rec["feature_key"] = feature_key(player_point, banker_point)
        rec["candidate_keys"] = []
        return rec

    try:
        rec = combo_lookup(player_point, banker_point, rounds or [], min_sample=COMBO_MIN_SAMPLE)
        if not isinstance(rec, dict):
            raise RuntimeError("combo_lookup returned non-dict")
        banker = rec.get("banker_prob", rec.get("next_banker_rate", BASE_BANKER_NO_TIE))
        player = rec.get("player_prob", rec.get("next_player_rate", 1.0 - BASE_BANKER_NO_TIE))
        banker, player = normalize_prob_pair(float(banker), float(player))
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", feature_key(player_point, banker_point)),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "COMBO_DB"),
            "sample_size": int(rec.get("sample_size", rec.get("sample", 0)) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", 0) or 0),
            "candidate_keys": rec.get("candidate_keys", []),
        }
    except Exception as e:
        rec = neutral_record(f"COMBO_DB_ERROR:{e}")
        rec["feature_key"] = feature_key(player_point, banker_point)
        rec["candidate_keys"] = []
        return rec


# ============================================================
# AI 模擬層
# ============================================================

def extract_round_points(rounds: Optional[List[Any]]) -> List[Tuple[int, int]]:
    if not rounds:
        return []
    out: List[Tuple[int, int]] = []
    for r in rounds:
        pp = None
        bp = None
        if isinstance(r, dict):
            pp = r.get("player_point", r.get("player", r.get("p")))
            bp = r.get("banker_point", r.get("banker", r.get("b")))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            pp, bp = r[0], r[1]
        try:
            if pp is None or bp is None:
                continue
            pp = int(pp)
            bp = int(bp)
            if 0 <= pp <= 9 and 0 <= bp <= 9:
                out.append((pp, bp))
        except Exception:
            continue
    return out


def trend_delta(values: List[int]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mid = n // 2
    early = values[:mid]
    late = values[mid:]
    if not early or not late:
        return 0.0
    return (sum(late) / len(late)) - (sum(early) / len(early))


def streak_count(results: List[str], side: str) -> int:
    count = 0
    for r in reversed(results):
        if r == "T":
            continue
        if r == side:
            count += 1
        else:
            break
    return count


def ai_simulation_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = feature_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE
    reasons: List[str] = []
    history_adjust = 0.0

    if diff == 0:
        banker += stable_noise(key + ":ai_tie", 0.004)
        reasons.append("current_tie_point_noise")
    elif abs(diff) <= 2:
        banker += -0.012 if diff > 0 else 0.012
        reasons.append("current_small_diff_adjust")
    elif abs(diff) <= 5:
        banker += -0.016 if diff > 0 else 0.016
        reasons.append("current_mid_diff_adjust")
    else:
        banker += -0.010 if diff > 0 else 0.010
        reasons.append("current_large_diff_adjust")

    point_history = extract_round_points(rounds)
    recent = point_history[-AI_HISTORY_WINDOW:] if point_history else []

    if len(recent) >= 3:
        player_points = [p for p, _ in recent]
        banker_points = [b for _, b in recent]
        diffs = [p - b for p, b in recent]
        results = ["P" if p > b else "B" if b > p else "T" for p, b in recent]

        p_trend = trend_delta(player_points)
        b_trend = trend_delta(banker_points)
        diff_trend = trend_delta(diffs)

        if p_trend - b_trend >= 1.5:
            history_adjust -= AI_TREND_STRENGTH
            reasons.append("player_point_trend_up")
        elif b_trend - p_trend >= 1.5:
            history_adjust += AI_TREND_STRENGTH
            reasons.append("banker_point_trend_up")

        if diff_trend >= 1.25:
            history_adjust -= AI_DIFF_MOMENTUM_STRENGTH
            reasons.append("player_diff_momentum")
        elif diff_trend <= -1.25:
            history_adjust += AI_DIFF_MOMENTUM_STRENGTH
            reasons.append("banker_diff_momentum")

        recent_player_hot = sum(1 for x in player_points[-3:] if x >= 7)
        recent_banker_hot = sum(1 for x in banker_points[-3:] if x >= 7)

        if recent_player_hot >= 3:
            history_adjust += AI_REVERSAL_STRENGTH
            reasons.append("player_hot_reversal_guard")
        elif recent_banker_hot >= 3:
            history_adjust -= AI_REVERSAL_STRENGTH
            reasons.append("banker_hot_reversal_guard")

        b_streak = streak_count(results, "B")
        p_streak = streak_count(results, "P")

        if b_streak >= 4:
            history_adjust -= AI_REVERSAL_STRENGTH * 0.7
            reasons.append("banker_streak_guard")
        elif p_streak >= 4:
            history_adjust += AI_REVERSAL_STRENGTH * 0.7
            reasons.append("player_streak_guard")

    history_adjust = clamp(history_adjust, -AI_HISTORY_MAX_ADJUST, AI_HISTORY_MAX_ADJUST)
    banker += history_adjust
    banker += stable_noise(key + ":ai_v4_combo", AI_NOISE_SCALE)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_SIMULATION_POINT_SEQUENCE_V4_COMBO_SAFE",
        "history_points_used": len(recent),
        "history_adjust": history_adjust,
        "history_reasons": reasons,
    }


# ============================================================
# Monte Carlo 風控驗證層
# ============================================================

def monte_carlo_verify_from_probs(banker_prob: float, player_prob: float, n_sim: Optional[int] = None, seed_key: str = "") -> Dict[str, Any]:
    if n_sim is None:
        n_sim = MC_SIMULATIONS
    try:
        n_sim = int(n_sim)
    except Exception:
        n_sim = 600

    min_sim = max(20, int(MC_MIN_SIMULATIONS))
    max_sim = max(min_sim, int(MC_MAX_SIMULATIONS))
    n_sim = max(min_sim, min(n_sim, max_sim))

    banker_prob, player_prob = normalize_prob_pair(float(banker_prob), float(player_prob))
    rng = random.Random(f"{MC_SEED}:{seed_key}")
    wins = {"banker": 0, "player": 0, "tie": 0}

    for _ in range(n_sim):
        noise = rng.uniform(-MC_MAX_NOISE, MC_MAX_NOISE)
        b = clamp(banker_prob + noise, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
        p = 1.0 - b
        r = rng.random()
        if r < b:
            wins["banker"] += 1
        elif r < b + p:
            wins["player"] += 1
        else:
            wins["tie"] += 1

    total = wins["banker"] + wins["player"] + wins["tie"]
    if total <= 0:
        banker_rate = BASE_BANKER_NO_TIE
        player_rate = 1.0 - BASE_BANKER_NO_TIE
        tie_rate = 0.0
    else:
        banker_rate = wins["banker"] / total
        player_rate = wins["player"] / total
        tie_rate = wins["tie"] / total

    mc_gap = abs(banker_rate - player_rate)
    mc_recommend = "莊" if banker_rate >= player_rate else "閒"

    return {
        "mc_enabled": True,
        "mc_simulations": n_sim,
        "mc_banker_rate": round(banker_rate * 100, PERCENT_DECIMALS),
        "mc_player_rate": round(player_rate * 100, PERCENT_DECIMALS),
        "mc_tie_rate": round(tie_rate * 100, PERCENT_DECIMALS),
        "mc_banker_rate_raw": banker_rate,
        "mc_player_rate_raw": player_rate,
        "mc_tie_rate_raw": tie_rate,
        "mc_recommend": mc_recommend,
        "mc_gap": round(mc_gap * 100, PERCENT_DECIMALS),
        "mc_gap_raw": mc_gap,
        "mc_source": "MONTE_CARLO_PROB_STABILITY_CHECK_SAFE_V2",
        "mc_note": "MC only verifies final probability stability; it does not call predict().",
    }


def disabled_monte_carlo_result() -> Dict[str, Any]:
    return {"mc_enabled": False, "mc_simulations": 0, "mc_source": "MONTE_CARLO_DISABLED"}


# ============================================================
# 保護與進場判斷
# ============================================================

def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    if not is_tie_point:
        return banker
    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK


def build_entry_decision(is_tie_point: bool, gap: float, recommend: str) -> Tuple[bool, str, str]:
    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "上一局為和局點數，莊閒優勢不足，建議觀察一局"
    if gap < MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "莊閒機率差距不足，建議觀察一局"
    if gap >= STRONG_GAP_FOR_ENTRY:
        return True, "strong", ""
    return True, "normal", ""


# ============================================================
# 主預測函式
# ============================================================

def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    pattern = pattern_db_lookup(player_point, banker_point, rounds=rounds)
    combo = combo_db_lookup(player_point, banker_point, rounds=rounds)
    ai = ai_simulation_layer(player_point, banker_point, rounds=rounds)

    p_w = float(POINT_WEIGHT)
    c_w = float(COMBO_WEIGHT)
    pat_w = float(PATTERN_WEIGHT)
    sim_w = float(SIM_WEIGHT)

    if not USE_POINT_DB or not point.get("available"):
        p_w = 0.0
    if not USE_COMBO_DB or not combo.get("available"):
        c_w = 0.0
    if not USE_PATTERN_DB or not pattern.get("available"):
        pat_w = 0.0
    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)

    total_weight = max(p_w + c_w + pat_w + sim_w, 0.0001)

    banker = (
        point["banker_prob"] * p_w +
        combo["banker_prob"] * c_w +
        pattern["banker_prob"] * pat_w +
        ai["banker_prob"] * sim_w
    ) / total_weight

    banker = apply_tie_point_protection(banker, is_tie_point)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    player = 1.0 - banker
    gap = abs(banker - player)
    recommend = "莊" if banker >= player else "閒"

    entry_allowed, entry_level, weak_reason = build_entry_decision(
        is_tie_point=is_tie_point,
        gap=gap,
        recommend=recommend,
    )

    result: Dict[str, Any] = {
        "ok": True,
        "player_point": player_point,
        "banker_point": banker_point,
        "last_result": last_result,
        "recommend": recommend,
        "player_prob": round(player * 100, PERCENT_DECIMALS),
        "banker_prob": round(banker * 100, PERCENT_DECIMALS),
        "player_prob_raw": player,
        "banker_prob_raw": banker,
        "confidence_gap": round(gap * 100, PERCENT_DECIMALS),
        "confidence_gap_raw": gap,
        "entry_allowed": entry_allowed,
        "entry_level": entry_level,
        "weak_reason": weak_reason,
        "no_observe": not entry_allowed,
        "tie_point_mode": is_tie_point,
        "tie_ai_max_weight": TIE_AI_MAX_WEIGHT if is_tie_point else None,
        "tie_shrink": TIE_SHRINK if is_tie_point else None,
        "tie_min_gap_for_entry": TIE_MIN_GAP_FOR_ENTRY if is_tie_point else None,
        "min_gap_for_entry": MIN_GAP_FOR_ENTRY,
        "strong_gap_for_entry": STRONG_GAP_FOR_ENTRY,
        "feature_key": feature_key(player_point, banker_point),
        "point_feature_key": point["feature_key"],
        "combo_feature_key": combo["feature_key"],
        "pattern_feature_key": pattern["feature_key"],
        "point_source": point["source"],
        "combo_source": combo["source"],
        "pattern_source": pattern["source"],
        "ai_source": ai["source"],
        "ai_history_points_used": ai.get("history_points_used", 0),
        "ai_history_adjust": ai.get("history_adjust", 0.0),
        "ai_history_reasons": ai.get("history_reasons", []),
        "point_available": point["available"],
        "combo_available": combo["available"],
        "pattern_available": pattern["available"],
        "point_sample_size": point["sample_size"],
        "combo_sample_size": combo["sample_size"],
        "pattern_sample_size": pattern["sample_size"],
        "point_total_samples": point["total_simulated_samples"],
        "combo_total_samples": combo["total_simulated_samples"],
        "pattern_total_samples": pattern["total_simulated_samples"],
        "matched_patterns": [pattern["feature_key"]] if pattern.get("available") else [],
        "combo_candidate_keys": combo.get("candidate_keys", []),
        "weights": {
            "point": p_w,
            "combo": c_w,
            "pattern": pat_w,
            "simulation": sim_w,
            "total": total_weight,
        },
        "raw_layers": {
            "point_banker_prob": point["banker_prob"],
            "combo_banker_prob": combo["banker_prob"],
            "pattern_banker_prob": pattern["banker_prob"],
            "ai_banker_prob": ai["banker_prob"],
            "point_player_prob": point["player_prob"],
            "combo_player_prob": combo["player_prob"],
            "pattern_player_prob": pattern["player_prob"],
            "ai_player_prob": ai["player_prob"],
        },
        "history_used": bool(rounds) and (
            combo.get("available") or pattern.get("available") or ai.get("history_points_used", 0) >= 3
        ),
        "rounds_ignored": False,
        "pattern_string": pattern.get("pattern", ""),
        "pattern_window": pattern.get("window", 0),
        "combo_min_sample": COMBO_MIN_SAMPLE,
    }

    if USE_MONTE_CARLO:
        mc = monte_carlo_verify_from_probs(
            banker_prob=banker,
            player_prob=player,
            seed_key=f"{player_point}-{banker_point}-{result.get('pattern_string', '')}-{result.get('combo_feature_key', '')}",
        )
        result.update(mc)

        if MC_BLOCK_LOW_GAP and mc.get("mc_gap_raw", 0.0) < MC_MIN_GAP_FOR_ENTRY:
            result["entry_allowed"] = False
            result["entry_level"] = "mc_no_entry"
            result["weak_reason"] = "蒙地卡羅穩定度不足，建議觀察一局"
            result["no_observe"] = True

        if MC_DIRECTION_MISMATCH_BLOCK and mc.get("mc_recommend") in {"莊", "閒"} and mc.get("mc_recommend") != recommend:
            result["entry_allowed"] = False
            result["entry_level"] = "mc_direction_mismatch"
            result["weak_reason"] = "蒙地卡羅方向與主模型不一致，建議觀察一局"
            result["no_observe"] = True
    else:
        result.update(disabled_monte_carlo_result())

    return result
