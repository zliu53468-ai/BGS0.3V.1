import hashlib
import os
import random
from typing import Dict, Any, List, Optional, Tuple

import config
from point_db import get_point_record, point_db_meta

try:
    from point_composition_mc import composition_mc_lookup
except Exception:
    composition_mc_lookup = None

try:
    from combo_db import combo_lookup, combo_db_meta
except Exception:
    combo_lookup = None
    combo_db_meta = None

try:
    from road_profile_db import road_profile_lookup, road_profile_db_meta
except Exception:
    road_profile_lookup = None
    road_profile_db_meta = None

try:
    from point_calibrator import calibrate_point_layer
except Exception:
    calibrate_point_layer = None

try:
    from micro_road_model import micro_road_lookup
except Exception:
    micro_road_lookup = None

# ============================================================
# V10.4：點數主導 + AI 輔助 + 強弩之末保護 + 差距區間 combo
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


# 權重預設值 (與最終平衡版一致，環境變數仍可覆蓋)
POINT_WEIGHT = env_float("POINT_WEIGHT", "0.62")
COMBO_WEIGHT = env_float("COMBO_WEIGHT", os.getenv("PATTERN_WEIGHT", "0.13"))
SIM_WEIGHT = env_float("SIM_WEIGHT", "0.12")
COMPOSITION_MC_WEIGHT = env_float("COMPOSITION_MC_WEIGHT", "0.06")
ROAD_PROFILE_WEIGHT = env_float("ROAD_PROFILE_WEIGHT", "0.04")
MICRO_ROAD_WEIGHT = env_float("MICRO_ROAD_WEIGHT", "0.05")

USE_POINT_DB = env_bool("USE_POINT_DB", "1")
USE_COMBO_DB = env_bool("USE_COMBO_DB", "1")
USE_COMPOSITION_MC = env_bool("USE_COMPOSITION_MC", "1")
USE_MONTE_CARLO = env_bool("USE_MONTE_CARLO", "1")
PREDICT_CURRENT_ROUND_ONLY = env_bool("PREDICT_CURRENT_ROUND_ONLY", "1")

USE_ROAD_PROFILE_DB = env_bool("USE_ROAD_PROFILE_DB", "1")
ROAD_PROFILE_WEIGHT = env_float("ROAD_PROFILE_WEIGHT", "0.04")

USE_MICRO_ROAD_MODEL = env_bool("USE_MICRO_ROAD_MODEL", "1")
MICRO_ROAD_WEIGHT = env_float("MICRO_ROAD_WEIGHT", "0.05")
MICRO_ROAD_WEIGHT_REQUIRE_AVAILABLE = env_bool("MICRO_ROAD_WEIGHT_REQUIRE_AVAILABLE", "1")
MICRO_ROAD_MIN_CONFIDENCE = env_float("MICRO_ROAD_MIN_CONFIDENCE", "0.15")

USE_DECISION_CONTROLLER = env_bool("USE_DECISION_CONTROLLER", "1")
DECISION_CONTROLLER_MAX_ADJUST = env_float("DECISION_CONTROLLER_MAX_ADJUST", "0.04")
CONSENSUS_MIN_SUPPORT = env_int("CONSENSUS_MIN_SUPPORT", "2")
CONSENSUS_EDGE = env_float("CONSENSUS_EDGE", "0.020")
CONSENSUS_BLEND = env_float("CONSENSUS_BLEND", "0.40")
MICRO_ROAD_OVERRIDE = env_bool("MICRO_ROAD_OVERRIDE", "1")
MICRO_ROAD_OVERRIDE_CONFIDENCE = env_float("MICRO_ROAD_OVERRIDE_CONFIDENCE", "0.45")
MICRO_ROAD_OVERRIDE_EDGE = env_float("MICRO_ROAD_OVERRIDE_EDGE", "0.060")
MICRO_ROAD_OVERRIDE_BLEND = env_float("MICRO_ROAD_OVERRIDE_BLEND", "0.60")
MICRO_ROAD_OVERRIDE_ALWAYS_APPLY = env_bool("MICRO_ROAD_OVERRIDE_ALWAYS_APPLY", "0")
NATURAL_TRAP_EXTRA_EDGE = env_float("NATURAL_TRAP_EXTRA_EDGE", "0.018")
MID_HIGH_GAP_EXTRA_EDGE = env_float("MID_HIGH_GAP_EXTRA_EDGE", "0.012")

USE_POINT_CALIBRATOR = env_bool("USE_POINT_CALIBRATOR", "1")
ROAD_PROFILE_MIN_SAMPLE = env_int("ROAD_PROFILE_MIN_SAMPLE", "50")
ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE = env_bool("ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE", "1")
ROAD_PROFILE_REQUIRE_AVAILABLE = env_bool("ROAD_PROFILE_REQUIRE_AVAILABLE", "0")
ROAD_PROFILE_SIGNAL_MIN_GAP = env_float("ROAD_PROFILE_SIGNAL_MIN_GAP", "0.010")
ROAD_PROFILE_USE_USER_HISTORY = env_bool("ROAD_PROFILE_USE_USER_HISTORY", "0")

COMBO_DB_MIN_SAMPLE = env_int("COMBO_DB_MIN_SAMPLE", "100")
COMBO_WEIGHT_REQUIRE_AVAILABLE = env_bool("COMBO_WEIGHT_REQUIRE_AVAILABLE", "1")
REQUIRE_COMBO_SAMPLE_FOR_ENTRY = env_bool("REQUIRE_COMBO_SAMPLE_FOR_ENTRY", "0")
MIN_GAP_WITHOUT_COMBO = env_float("MIN_GAP_WITHOUT_COMBO", "0.120")

COMPOSITION_MC_SIMULATIONS = env_int("COMPOSITION_MC_SIMULATIONS", "800")
COMPOSITION_MC_MAX_COMBOS = env_int("COMPOSITION_MC_MAX_COMBOS", "300")

COMPOSITION_MC_DYNAMIC_WEIGHT = env_bool("COMPOSITION_MC_DYNAMIC_WEIGHT", "1")
COMPOSITION_MC_MIN_WEIGHT_MULT = env_float("COMPOSITION_MC_MIN_WEIGHT_MULT", "0.75")
COMPOSITION_MC_MAX_WEIGHT_MULT = env_float("COMPOSITION_MC_MAX_WEIGHT_MULT", "1.08")
COMPOSITION_MC_CONFIDENCE_BOOST = env_float("COMPOSITION_MC_CONFIDENCE_BOOST", "0.12")
COMPOSITION_MC_GAP_BOOST = env_float("COMPOSITION_MC_GAP_BOOST", "0.08")
COMPOSITION_MC_SUPPORT_BOOST = env_float("COMPOSITION_MC_SUPPORT_BOOST", "1.05")
COMPOSITION_MC_CONFLICT_SHRINK = env_float("COMPOSITION_MC_CONFLICT_SHRINK", "0.72")
COMPOSITION_MC_MIN_CONFIDENCE_FOR_BOOST = env_float("COMPOSITION_MC_MIN_CONFIDENCE_FOR_BOOST", "0.50")

USE_POINT_GAP_CALIBRATOR = env_bool("USE_POINT_GAP_CALIBRATOR", "1")
POINT_GAP_TINY_COMP_BOOST = env_float("POINT_GAP_TINY_COMP_BOOST", "1.06")
POINT_GAP_TINY_COMBO_BOOST = env_float("POINT_GAP_TINY_COMBO_BOOST", "1.06")
POINT_GAP_TINY_POINT_SHRINK = env_float("POINT_GAP_TINY_POINT_SHRINK", "0.90")
POINT_GAP_LOW_MID_COMP_BOOST = env_float("POINT_GAP_LOW_MID_COMP_BOOST", "1.00")
POINT_GAP_LOW_MID_COMBO_BOOST = env_float("POINT_GAP_LOW_MID_COMBO_BOOST", "1.00")
POINT_GAP_LOW_MID_POINT_MULT = env_float("POINT_GAP_LOW_MID_POINT_MULT", "1.00")
POINT_GAP_MID_HIGH_POINT_BOOST = env_float("POINT_GAP_MID_HIGH_POINT_BOOST", "1.00")
POINT_GAP_MID_HIGH_COMP_SHRINK = env_float("POINT_GAP_MID_HIGH_COMP_SHRINK", "0.95")
POINT_GAP_MID_HIGH_ROAD_SHRINK = env_float("POINT_GAP_MID_HIGH_ROAD_SHRINK", "0.95")
POINT_GAP_EXTREME_POINT_BOOST = env_float("POINT_GAP_EXTREME_POINT_BOOST", "1.00")
POINT_GAP_EXTREME_COMP_SHRINK = env_float("POINT_GAP_EXTREME_COMP_SHRINK", "0.90")
POINT_GAP_EXTREME_ROAD_SHRINK = env_float("POINT_GAP_EXTREME_ROAD_SHRINK", "0.90")
POINT_GAP_TIE_COMP_SHRINK = env_float("POINT_GAP_TIE_COMP_SHRINK", "0.78")
POINT_GAP_TIE_ROAD_SHRINK = env_float("POINT_GAP_TIE_ROAD_SHRINK", "0.78")
POINT_GAP_CALIBRATOR_MAX_TOTAL_MULT = env_float("POINT_GAP_CALIBRATOR_MAX_TOTAL_MULT", "1.12")

USE_NATURAL_HIGH_GUARD = env_bool("USE_NATURAL_HIGH_GUARD", "1")
NATURAL_HIGH_POINT_BOOST = env_float("NATURAL_HIGH_POINT_BOOST", "1.03")
NATURAL_HIGH_COMP_SHRINK = env_float("NATURAL_HIGH_COMP_SHRINK", "0.92")
NATURAL_HIGH_ROAD_SHRINK = env_float("NATURAL_HIGH_ROAD_SHRINK", "0.94")
NATURAL_HIGH_COMBO_SHRINK = env_float("NATURAL_HIGH_COMBO_SHRINK", "0.96")

USE_AI_PATTERN_RECOGNITION = env_bool("USE_AI_PATTERN_RECOGNITION", "1")
AI_PATTERN_MIN_STRENGTH = env_float("AI_PATTERN_MIN_STRENGTH", "0.60")
AI_PATTERN_WEIGHT_BOOST = env_float("AI_PATTERN_WEIGHT_BOOST", "1.35")
AI_PATTERN_STREAK_MIN = env_int("AI_PATTERN_STREAK_MIN", "3")
AI_PATTERN_ALTERNATING_MIN = env_int("AI_PATTERN_ALTERNATING_MIN", "4")
AI_CONFIDENCE_MIN_FOR_DECISION = env_float("AI_CONFIDENCE_MIN_FOR_DECISION", "0.50")
AI_DECISION_BOOST_EDGE = env_float("AI_DECISION_BOOST_EDGE", "0.022")

USE_AI_POINT_FEATURE_DB = env_bool("USE_AI_POINT_FEATURE_DB", "1")
AI_DB_WEIGHT_TREND = env_float("AI_DB_WEIGHT_TREND", "0.30")
AI_DB_WEIGHT_PATTERN = env_float("AI_DB_WEIGHT_PATTERN", "0.40")
AI_DB_WEIGHT_DB = env_float("AI_DB_WEIGHT_DB", "0.30")
AI_DB_MIN_CONFIDENCE = env_float("AI_DB_MIN_CONFIDENCE", "0.45")
AI_DB_BLEND_FACTOR = env_float("AI_DB_BLEND_FACTOR", "0.30")

BASE_BANKER_NO_TIE = 0.5000
MIN_OUTPUT_PROB = env_float("MIN_OUTPUT_PROB", "0.39")
MAX_OUTPUT_PROB = env_float("MAX_OUTPUT_PROB", "0.61")
PERCENT_DECIMALS = env_int("PERCENT_DECIMALS", "2")

MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", "0.035")
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", "0.060")

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", "0.045")
TIE_SHRINK = env_float("TIE_SHRINK", "0.28")
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", "0.10")

AI_NOISE_SCALE = env_float("AI_NOISE_SCALE", "0.004")
AI_HISTORY_WINDOW = env_int("AI_HISTORY_WINDOW", "8")
AI_TREND_STRENGTH = env_float("AI_TREND_STRENGTH", "0.010")
AI_DIFF_MOMENTUM_STRENGTH = env_float("AI_DIFF_MOMENTUM_STRENGTH", "0.007")
AI_REVERSAL_STRENGTH = env_float("AI_REVERSAL_STRENGTH", "0.003")
AI_HISTORY_MAX_ADJUST = env_float("AI_HISTORY_MAX_ADJUST", "0.020")

MC_SIMULATIONS = env_int("MC_SIMULATIONS", "400")
MC_MIN_SIMULATIONS = env_int("MC_MIN_SIMULATIONS", "150")
MC_MAX_SIMULATIONS = env_int("MC_MAX_SIMULATIONS", "600")
MC_SEED = env_int("MC_SEED", "42")
MC_MAX_NOISE = env_float("MC_MAX_NOISE", "0.003")
MC_BLOCK_LOW_GAP = env_bool("MC_BLOCK_LOW_GAP", "1")
MC_MIN_GAP_FOR_ENTRY = env_float("MC_MIN_GAP_FOR_ENTRY", "0.030")
MC_DIRECTION_MISMATCH_BLOCK = env_bool("MC_DIRECTION_MISMATCH_BLOCK", "0")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def stable_noise(key: str, scale: float = 0.035) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale

def validate_point(v: int) -> int:
    iv = int(v)
    if iv < 0 or iv > 9:
        raise ValueError("point must be 0-9")
    return iv

def get_last_result(player_point: int, banker_point: int) -> str:
    if player_point > banker_point:
        return "閒"
    if banker_point > player_point:
        return "莊"
    return "和"

def point_key(player_point: int, banker_point: int) -> str:
    return f"P{player_point}_B{banker_point}"

def normalize_prob_pair(banker: float, player: float) -> Tuple[float, float]:
    banker = float(banker)
    player = float(player)
    if banker > 1:
        banker /= 100.0
    if player > 1:
        player /= 100.0
    total = banker + player
    if total <= 0:
        banker = BASE_BANKER_NO_TIE
        player = 1.0 - banker
    else:
        banker /= total
        player /= total
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    return banker, player

def _layer_direction(rec: Dict[str, Any], neutral_gap: float = 0.004) -> Tuple[str, float]:
    if not isinstance(rec, dict):
        return "NEUTRAL", 0.0
    try:
        b = float(rec.get("banker_prob", BASE_BANKER_NO_TIE) or BASE_BANKER_NO_TIE)
        p = float(rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE) or (1.0 - BASE_BANKER_NO_TIE))
        b, p = normalize_prob_pair(b, p)
        gap = abs(b - p)
        if gap < neutral_gap:
            return "NEUTRAL", gap
        return ("BANKER" if b >= p else "PLAYER"), gap
    except Exception:
        return "NEUTRAL", 0.0

# ... (以下保留所有原有的輔助函數：_calibrate_composition_weight, point_gap_profile, apply_point_gap_calibrator, apply_natural_high_guard, neutral_record, fallback_point_lookup, point_db_lookup, composition_mc_layer, combo_condition_lookup, road_profile_layer, micro_road_layer, extract_round_points, trend_delta, detect_simple_patterns, ai_simulation_layer, monte_carlo_verify_from_probs, disabled_monte_carlo_result, apply_tie_point_protection, build_entry_decision, _side_from_text, _target_banker_for_side, _blend_to_side, _layer_side_and_gap, _is_trap_pattern, apply_decision_controller)

# 由於字數限制，此處省略中間函數（與你提供的原始碼相同，除 combo_condition_lookup 保持原樣，因為 combo_db.py 已獨立更新）

# 以下是最關鍵的 predict() 改動部分，完整函數在文末

# ... 中間省略的函數請直接使用你現有的原始碼，它們沒有任何變動 ...


def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)
    rounds = rounds or []
    model_rounds = [] if PREDICT_CURRENT_ROUND_ONLY else rounds

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    comp = composition_mc_layer(player_point, banker_point, rounds=model_rounds)
    combo = combo_condition_lookup(player_point, banker_point, rounds=model_rounds, comp=comp)
    road = road_profile_layer(player_point, banker_point, comp=comp)
    micro = micro_road_layer(player_point, banker_point, rounds=rounds, comp=comp)
    ai = ai_simulation_layer(player_point, banker_point, rounds=model_rounds)

    point_calibration = {
        "point": point,
        "weight_multiplier": 1.0,
        "status": "NOT_AVAILABLE",
        "signals": [],
        "support_count": 0,
        "conflict_count": 0,
    }
    if USE_POINT_CALIBRATOR and callable(calibrate_point_layer):
        try:
            point_calibration = calibrate_point_layer(
                point=point,
                combo=combo,
                composition_mc=comp,
                road_profile=road,
                ai=ai,
            )
            point = point_calibration.get("point", point)
        except Exception as e:
            point_calibration = {
                "point": point,
                "weight_multiplier": 1.0,
                "status": f"ERROR:{e}",
                "signals": [],
                "support_count": 0,
                "conflict_count": 0,
            }

    p_w = float(POINT_WEIGHT) if USE_POINT_DB else 0.0
    p_w = p_w * float(point_calibration.get("weight_multiplier", 1.0) or 1.0)
    combo_w = float(COMBO_WEIGHT)
    sim_w = float(SIM_WEIGHT)
    comp_w = float(COMPOSITION_MC_WEIGHT) if comp.get("available") else 0.0
    composition_weight_info = {
        "enabled": False,
        "weight_multiplier": 1.0,
        "status": "NOT_APPLIED",
        "support_count": 0,
        "conflict_count": 0,
    }
    if comp_w > 0:
        comp_w, composition_weight_info = _calibrate_composition_weight(
            comp=comp,
            point=point,
            combo=combo,
            road=road,
            base_weight=comp_w,
        )

    road_w = float(ROAD_PROFILE_WEIGHT) if USE_ROAD_PROFILE_DB else 0.0
    micro_w = float(MICRO_ROAD_WEIGHT) if USE_MICRO_ROAD_MODEL else 0.0

    if MICRO_ROAD_WEIGHT_REQUIRE_AVAILABLE and (not micro.get("available") or float(micro.get("micro_confidence", 0.0) or 0.0) < MICRO_ROAD_MIN_CONFIDENCE):
        micro_w = 0.0

    if COMBO_WEIGHT_REQUIRE_AVAILABLE and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0):
        combo_w = 0.0

    if ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE and (not road.get("available") or int(road.get("sample_size", 0) or 0) <= 0):
        road_w = 0.0

    # --- AI 模式信號動態權重加成 ---
    if (
        USE_AI_PATTERN_RECOGNITION
        and ai.get("pattern_strength", 0.0) >= AI_PATTERN_MIN_STRENGTH
        and ai.get("pattern_suggest", "NEUTRAL") in ("BANKER", "PLAYER")
    ):
        sim_w = sim_w * AI_PATTERN_WEIGHT_BOOST

    # V10.4 強弩之末保護：多層極度共識但短牌路反向時，動態提高短牌路權重
    micro_side = micro.get("micro_direction", "NEUTRAL")
    micro_conf = micro.get("micro_confidence", 0.0)
    # 先計算初步機率判斷共識方向
    total_weight_no_micro = max(p_w + combo_w + road_w + sim_w + comp_w, 0.0001)
    preliminary_banker = (
        point["banker_prob"] * p_w
        + combo["banker_prob"] * combo_w
        + road["banker_prob"] * road_w
        + ai["banker_prob"] * sim_w
        + comp["banker_prob"] * comp_w
    ) / total_weight_no_micro
    consensus_side_prelim = "BANKER" if preliminary_banker >= BASE_BANKER_NO_TIE else "PLAYER"
    if (
        micro_side in ("BANKER", "PLAYER")
        and micro_side != consensus_side_prelim
        and micro_conf >= 0.25
    ):
        micro_w *= 1.4  # 提高短牌路影響力
        # 稍微壓低極端共識層的影響 (避免拉扯)
        p_w *= 0.95
        combo_w *= 0.95
        comp_w *= 0.95

    p_w, combo_w, comp_w, road_w, sim_w, point_gap_info = apply_point_gap_calibrator(
        player_point=player_point,
        banker_point=banker_point,
        point_w=p_w,
        combo_w=combo_w,
        comp_w=comp_w,
        road_w=road_w,
        sim_w=sim_w,
    )

    p_w, combo_w, comp_w, road_w, sim_w, natural_high_guard_info = apply_natural_high_guard(
        comp=comp,
        point_gap_info=point_gap_info,
        point_w=p_w,
        combo_w=combo_w,
        comp_w=comp_w,
        road_w=road_w,
        sim_w=sim_w,
    )

    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)
        comp_w = min(comp_w, COMPOSITION_MC_WEIGHT * 0.50)
        road_w = min(road_w, ROAD_PROFILE_WEIGHT * 0.50)
        micro_w = min(micro_w, MICRO_ROAD_WEIGHT * 0.65)

    total_weight = max(p_w + combo_w + road_w + micro_w + sim_w + comp_w, 0.0001)
    banker = (
        point["banker_prob"] * p_w
        + combo["banker_prob"] * combo_w
        + road["banker_prob"] * road_w
        + micro["banker_prob"] * micro_w
        + ai["banker_prob"] * sim_w
        + comp["banker_prob"] * comp_w
    ) / total_weight

    banker_before_decision_controller = banker
    banker, decision_controller_info = apply_decision_controller(
        banker=banker,
        point=point,
        combo=combo,
        road=road,
        micro=micro,
        comp=comp,
        ai=ai,
        weights={
            "point": p_w,
            "combo": combo_w,
            "composition_mc": comp_w,
            "road_profile": road_w,
            "micro_road": micro_w,
            "simulation": sim_w,
        },
        point_gap_info=point_gap_info,
        natural_high_guard_info=natural_high_guard_info,
    )

    banker = apply_tie_point_protection(banker, is_tie_point)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    gap = abs(banker - player)
    recommend = "莊" if banker >= player else "閒"

    entry_allowed, entry_level, weak_reason = build_entry_decision(is_tie_point=is_tie_point, gap=gap)

    if (
        REQUIRE_COMBO_SAMPLE_FOR_ENTRY
        and entry_allowed
        and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0)
        and gap < MIN_GAP_WITHOUT_COMBO
    ):
        entry_allowed = False
        entry_level = "no_entry"
        weak_reason = f"條件資料庫樣本不足，且莊閒差距未達 {MIN_GAP_WITHOUT_COMBO * 100:.1f}%，建議觀察一局"

    if (
        ROAD_PROFILE_REQUIRE_AVAILABLE
        and entry_allowed
        and (not road.get("available") or int(road.get("sample_size", 0) or 0) <= 0)
    ):
        entry_allowed = False
        entry_level = "no_entry"
        weak_reason = "牌路資料庫樣本不足，建議觀察一局"

    result = {
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
        "point_gap": point_gap_info.get("point_gap", abs(player_point - banker_point)),
        "point_diff": point_gap_info.get("point_diff", player_point - banker_point),
        "point_gap_code": point_gap_info.get("point_gap_code", f"GAP_{abs(player_point - banker_point)}"),
        "gap_zone": point_gap_info.get("gap_zone", "UNKNOWN"),
        "gap_zone_zh": point_gap_info.get("gap_zone_zh", ""),
        "gap_family": point_gap_info.get("gap_family", point_gap_info.get("gap_zone", "UNKNOWN")),
        "gap_family_zh": point_gap_info.get("gap_family_zh", point_gap_info.get("gap_zone_zh", "")),
        "winner_side": point_gap_info.get("winner_side", comp.get("winner_side", "UNKNOWN")),
        "winner_point": point_gap_info.get("winner_point", comp.get("winner_point")),
        "winner_point_zone": point_gap_info.get("winner_point_zone", comp.get("winner_point_zone", "UNKNOWN")),
        "point_gap_calibrator": point_gap_info,
        "natural_high_guard": natural_high_guard_info,
        "natural_high_winner": natural_high_guard_info.get("natural_high_winner", False),
        "decision_controller": decision_controller_info,
        "banker_prob_before_decision_controller": banker_before_decision_controller,
        "player_prob_before_decision_controller": 1.0 - banker_before_decision_controller,
        "min_gap_for_entry": MIN_GAP_FOR_ENTRY,
        "strong_gap_for_entry": STRONG_GAP_FOR_ENTRY,
        "feature_key": point_key(player_point, banker_point),
        "point_feature_key": point.get("feature_key"),
        "combo_feature_key": combo.get("feature_key"),
        "point_source": point.get("source"),
        "point_calibrator_status": point_calibration.get("status"),
        "point_calibrator_weight_multiplier": point_calibration.get("weight_multiplier", 1.0),
        "point_calibrator_support_count": point_calibration.get("support_count", 0),
        "point_calibrator_conflict_count": point_calibration.get("conflict_count", 0),
        "point_calibrator_support_score": point_calibration.get("support_score", 0.0),
        "point_calibrator_conflict_score": point_calibration.get("conflict_score", 0.0),
        "point_calibrator_signals": point_calibration.get("signals", []),
        "point_banker_prob_original": point_calibration.get("original_banker_prob", point.get("banker_prob")),
        "point_banker_prob_calibrated": point_calibration.get("calibrated_banker_prob", point.get("banker_prob")),
        "combo_source": combo.get("source"),
        "ai_source": ai.get("source"),
        "composition_mc_source": comp.get("source"),
        "composition_mc_available": comp.get("available", False),
        "composition_mc_sample_size": comp.get("sample_size", 0),
        "composition_top_scenario": comp.get("top_scenario", "UNKNOWN"),
        "composition_top_scenario_probability": comp.get("top_scenario_probability", 0.0),
        "composition_second_scenario_probability": comp.get("second_scenario_probability", 0.0),
        "composition_scenario_entropy": comp.get("scenario_entropy", 1.0),
        "composition_confidence": comp.get("composition_confidence", 0.0),
        "composition_gap": comp.get("composition_gap", 0.0),
        "composition_winner_side": comp.get("winner_side", "UNKNOWN"),
        "composition_winner_point": comp.get("winner_point"),
        "composition_winner_point_zone": comp.get("winner_point_zone", "UNKNOWN"),
        "composition_point_gap": comp.get("point_gap", abs(player_point - banker_point)),
        "composition_point_diff": comp.get("point_diff", player_point - banker_point),
        "composition_point_gap_code": comp.get("point_gap_code", point_gap_info.get("point_gap_code", f"GAP_{abs(player_point - banker_point)}")),
        "composition_gap_zone": comp.get("gap_zone", point_gap_info.get("gap_zone", "UNKNOWN")),
        "composition_gap_zone_zh": comp.get("gap_zone_zh", point_gap_info.get("gap_zone_zh", "")),
        "composition_gap_family": comp.get("gap_family", point_gap_info.get("gap_family", "UNKNOWN")),
        "composition_gap_family_zh": comp.get("gap_family_zh", point_gap_info.get("gap_family_zh", "")),
        "composition_natural_winner": comp.get("natural_winner", False),
        "composition_natural_high_winner": comp.get("natural_high_winner", False),
        "composition_natural_side": comp.get("natural_side", "NONE"),
        "composition_realistic_rule_filter": comp.get("realistic_rule_filter", False),
        "composition_weight_info": composition_weight_info,
        "composition_scenario_debug": comp.get("scenario_debug", []),
        "composition_scenario_count": comp.get("scenario_count", 0),
        "micro_road_available": micro.get("available", False),
        "micro_road_source": micro.get("source"),
        "micro_road_sample_size": micro.get("sample_size", 0),
        "micro_road_direction": micro.get("micro_direction", "NEUTRAL"),
        "micro_road_confidence": micro.get("micro_confidence", 0.0),
        "micro_road_patterns": micro.get("micro_patterns", []),
        "micro_road_recent": micro.get("recent_road", ""),
        "micro_road_context": micro.get("context", {}),
        "ai_confidence": ai.get("ai_confidence", 0.0),
        "ai_direction": ai.get("ai_direction", "NEUTRAL"),
        "ai_streak_side": ai.get("streak_side", "NEUTRAL"),
        "ai_streak_count": ai.get("streak_count", 0),
        "ai_pattern_type": ai.get("pattern_type", "none"),
        "ai_pattern_strength": ai.get("pattern_strength", 0.0),
        "ai_pattern_suggest": ai.get("pattern_suggest", "NEUTRAL"),
        "ai_pattern_detail": ai.get("pattern_detail", ""),
        "ai_pattern_recognition_enabled": ai.get("pattern_recognition_enabled", False),
        "ai_history_results": ai.get("history_results", []),
        "ai_db_available": ai.get("ai_db_available", False),
        "ai_db_banker_prob": ai.get("ai_db_banker_prob", BASE_BANKER_NO_TIE),
        "ai_db_player_prob": ai.get("ai_db_player_prob", 1.0 - BASE_BANKER_NO_TIE),
        "ai_db_confidence": ai.get("ai_db_confidence", 0.0),
        "ai_db_sample_size": ai.get("ai_db_sample_size", 0),
        "ai_db_feature_key": ai.get("ai_db_feature_key", ""),
        "ai_db_source": ai.get("ai_db_source", "NOT_AVAILABLE"),
        "ai_db_blended": ai.get("ai_db_blended", False),
        "ai_db_enabled": ai.get("ai_db_enabled", False),
        "trend_banker_before_blend": ai.get("trend_banker_before_blend", BASE_BANKER_NO_TIE),
        "combo_available": combo.get("available", False),
        "combo_sample_size": combo.get("sample_size", 0),
        "combo_total_samples": combo.get("total_simulated_samples", 0),
        "combo_candidate_keys": combo.get("candidate_keys", []),
        "combo_matched_records": combo.get("matched_records", []),
        "combo_top_scenario": combo.get("top_scenario", comp.get("top_scenario", "UNKNOWN")),
        "road_profile_available": road.get("available", False),
        "road_profile_feature_key": road.get("feature_key"),
        "road_profile_source": road.get("source"),
        "road_profile_sample_size": road.get("sample_size", 0),
        "road_profile_total_samples": road.get("total_simulated_samples", 0),
        "road_profile_top": road.get("top_road_profile", "NEUTRAL"),
        "road_profile_top_zh": road.get("top_road_profile_zh", "中性路段"),
        "road_profile_distribution": road.get("profile_distribution", []),
        "road_profile_candidate_keys": road.get("candidate_keys", []),
        "road_profile_matched_records": road.get("matched_records", []),
        "road_profile_same_point_repeat_avg": road.get("same_point_repeat_avg", 0),
        "road_profile_use_user_history": False,
        "point_available": point.get("available", False),
        "point_sample_size": point.get("sample_size", 0),
        "point_total_samples": point.get("total_simulated_samples", 0),
        "ai_history_points_used": ai.get("history_points_used", 0),
        "ai_history_adjust": ai.get("history_adjust", 0.0),
        "ai_history_reasons": ai.get("history_reasons", []),
        "pattern_available": combo.get("available", False),
        "pattern_sample_size": combo.get("sample_size", 0),
        "pattern_total_samples": combo.get("total_simulated_samples", 0),
        "pattern_source": combo.get("source"),
        "pattern_feature_key": combo.get("feature_key"),
        "pattern_layer_mode": "point_condition_combo",
        "matched_patterns": [combo.get("feature_key")] if combo.get("available") else [],
        "weights": {
            "point": p_w,
            "combo": combo_w,
            "pattern": combo_w,
            "road_profile": road_w,
            "micro_road": micro_w,
            "simulation": sim_w,
            "composition_mc": comp_w,
            "total": total_weight,
        },
        "raw_layers": {
            "point_banker_prob": point.get("banker_prob"),
            "combo_banker_prob": combo.get("banker_prob"),
            "pattern_banker_prob": combo.get("banker_prob"),
            "road_profile_banker_prob": road.get("banker_prob"),
            "micro_road_banker_prob": micro.get("banker_prob"),
            "ai_banker_prob": ai.get("banker_prob"),
            "composition_mc_banker_prob": comp.get("banker_prob"),
            "point_player_prob": point.get("player_prob"),
            "combo_player_prob": combo.get("player_prob"),
            "pattern_player_prob": combo.get("player_prob"),
            "road_profile_player_prob": road.get("player_prob"),
            "micro_road_player_prob": micro.get("player_prob"),
            "ai_player_prob": ai.get("player_prob"),
            "composition_mc_player_prob": comp.get("player_prob"),
        },
        "history_used": bool(model_rounds),
        "rounds_ignored": bool(rounds and PREDICT_CURRENT_ROUND_ONLY),
        "mode": "POINT_CONDITION_COMBO_COMPOSITION_MC_V10_4_STRONG_NU_MO",
    }

    if USE_MONTE_CARLO:
        mc_result = monte_carlo_verify_from_probs(
            banker_prob=banker,
            player_prob=player,
            seed_key=f"{player_point}:{banker_point}:{combo.get('feature_key','')}:{road.get('feature_key','')}:{micro.get('micro_direction','')}:{micro.get('recent_road','')}:{comp.get('top_scenario','')}:{ai.get('history_adjust',0.0)}",
        )
        result["monte_carlo"] = mc_result
        mc_gap_raw = float(mc_result.get("mc_gap_raw", 0.0) or 0.0)
        mc_recommend = mc_result.get("mc_recommend", recommend)

        if MC_BLOCK_LOW_GAP and result["entry_allowed"] and mc_gap_raw < MC_MIN_GAP_FOR_ENTRY:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 穩定度不足，莊閒差距偏小，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
        else:
            result["mc_entry_blocked"] = False

        if MC_DIRECTION_MISMATCH_BLOCK and result["entry_allowed"] and mc_recommend in {"莊", "閒"} and mc_recommend != recommend:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 模擬方向與主模型不一致，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
    else:
        result["monte_carlo"] = disabled_monte_carlo_result()
        result["mc_entry_blocked"] = False

    return result
