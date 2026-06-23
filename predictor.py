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
# V10.6：點數主導 + AI 輔助 + DeepSeek + 下一局情境預測
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

# V10.6 新增：下一局補牌情境預測
USE_NEXT_SCENARIO_PREDICT = env_bool("USE_NEXT_SCENARIO_PREDICT", "0")
NEXT_SCENARIO_WEIGHT = env_float("NEXT_SCENARIO_WEIGHT", "0.4")

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

def _calibrate_composition_weight(
    comp: Dict[str, Any],
    point: Dict[str, Any],
    combo: Dict[str, Any],
    road: Dict[str, Any],
    base_weight: float,
) -> Tuple[float, Dict[str, Any]]:
    if not COMPOSITION_MC_DYNAMIC_WEIGHT or not comp.get("available"):
        return base_weight, {
            "enabled": bool(COMPOSITION_MC_DYNAMIC_WEIGHT),
            "weight_multiplier": 1.0,
            "status": "DISABLED_OR_NOT_AVAILABLE",
            "support_count": 0,
            "conflict_count": 0,
        }

    comp_dir, comp_gap = _layer_direction(comp, neutral_gap=0.003)
    point_dir, _ = _layer_direction(point, neutral_gap=0.004)
    combo_dir, _ = _layer_direction(combo, neutral_gap=0.004)
    road_dir, _ = _layer_direction(road, neutral_gap=0.004)

    if comp_dir == "NEUTRAL":
        return base_weight * COMPOSITION_MC_MIN_WEIGHT_MULT, {
            "enabled": True,
            "weight_multiplier": COMPOSITION_MC_MIN_WEIGHT_MULT,
            "status": "COMP_NEUTRAL_SHRINK",
            "composition_direction": comp_dir,
            "composition_gap": comp_gap,
            "support_count": 0,
            "conflict_count": 0,
        }

    support_count = 0
    conflict_count = 0
    for d in [point_dir, combo_dir, road_dir]:
        if d == "NEUTRAL":
            continue
        if d == comp_dir:
            support_count += 1
        else:
            conflict_count += 1

    confidence = float(comp.get("composition_confidence", 0.0) or 0.0)
    mult = 1.0

    if confidence >= COMPOSITION_MC_MIN_CONFIDENCE_FOR_BOOST:
        mult += COMPOSITION_MC_CONFIDENCE_BOOST * confidence

    mult += min(comp_gap / 0.08, 1.0) * COMPOSITION_MC_GAP_BOOST

    if support_count >= 2:
        mult *= COMPOSITION_MC_SUPPORT_BOOST
    elif conflict_count >= 2 and support_count == 0:
        mult *= COMPOSITION_MC_CONFLICT_SHRINK
    elif conflict_count > support_count:
        mult *= max(COMPOSITION_MC_CONFLICT_SHRINK, 0.88)

    mult = clamp(mult, COMPOSITION_MC_MIN_WEIGHT_MULT, COMPOSITION_MC_MAX_WEIGHT_MULT)

    status = "COMP_DYNAMIC_KEEP"
    if support_count >= 2 and confidence >= COMPOSITION_MC_MIN_CONFIDENCE_FOR_BOOST:
        status = "COMP_CONFIDENT_SUPPORT_BOOST"
    elif conflict_count >= 2 and support_count == 0:
        status = "COMP_CONFLICT_SHRINK"
    elif confidence >= COMPOSITION_MC_MIN_CONFIDENCE_FOR_BOOST:
        status = "COMP_CONFIDENCE_BOOST"

    return base_weight * mult, {
        "enabled": True,
        "weight_multiplier": mult,
        "status": status,
        "composition_direction": comp_dir,
        "composition_gap": comp_gap,
        "composition_confidence": confidence,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "point_direction": point_dir,
        "combo_direction": combo_dir,
        "road_direction": road_dir,
    }

def point_gap_profile(player_point: int, banker_point: int) -> Dict[str, Any]:
    player_point = int(player_point)
    banker_point = int(banker_point)
    diff = player_point - banker_point
    point_gap = abs(diff)

    if player_point > banker_point:
        winner_side = "PLAYER"
        winner_point = player_point
    elif banker_point > player_point:
        winner_side = "BANKER"
        winner_point = banker_point
    else:
        winner_side = "TIE"
        winner_point = player_point

    if winner_point in (7, 8, 9):
        winner_point_zone = "HIGH_7_9"
    elif winner_point in (1, 2, 3, 4):
        winner_point_zone = "LOW_1_4"
    elif winner_point in (5, 6):
        winner_point_zone = "MID_5_6"
    else:
        winner_point_zone = "ZERO"

    point_gap_code = f"GAP_{point_gap}"

    if point_gap == 0:
        gap_family = "TIE_GAP"
        gap_family_zh = "和點"
    elif point_gap <= 2:
        gap_family = "TINY_GAP_1_2"
        gap_family_zh = "極小差距1-2"
    elif point_gap <= 4:
        gap_family = "LOW_MID_GAP_3_4"
        gap_family_zh = "中小差距3-4"
    elif point_gap <= 7:
        gap_family = "MID_HIGH_GAP_5_7"
        gap_family_zh = "中大差距5-7"
    else:
        gap_family = "EXTREME_GAP_8_9"
        gap_family_zh = "極大差距8-9"

    return {
        "point_gap": point_gap,
        "point_diff": diff,
        "point_gap_code": point_gap_code,
        "gap_zone": gap_family,
        "gap_zone_zh": gap_family_zh,
        "gap_family": gap_family,
        "gap_family_zh": gap_family_zh,
        "winner_side": winner_side,
        "winner_point": winner_point,
        "winner_point_zone": winner_point_zone,
    }

def apply_point_gap_calibrator(
    player_point: int,
    banker_point: int,
    point_w: float,
    combo_w: float,
    comp_w: float,
    road_w: float,
    sim_w: float,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    profile = point_gap_profile(player_point, banker_point)
    if not USE_POINT_GAP_CALIBRATOR:
        return point_w, combo_w, comp_w, road_w, sim_w, {
            **profile,
            "enabled": False,
            "status": "DISABLED",
            "multipliers": {},
        }

    gap_family = profile.get("gap_family", profile.get("gap_zone", "LOW_MID_GAP_3_4"))
    multipliers = {
        "point": 1.0,
        "combo": 1.0,
        "composition_mc": 1.0,
        "road_profile": 1.0,
        "simulation": 1.0,
    }
    status = "GAP_LOW_MID_KEEP"

    if gap_family in {"TINY_GAP_1_2", "SMALL_GAP_1_2"}:
        multipliers["point"] = POINT_GAP_TINY_POINT_SHRINK
        multipliers["combo"] = POINT_GAP_TINY_COMBO_BOOST
        multipliers["composition_mc"] = POINT_GAP_TINY_COMP_BOOST
        status = "GAP_TINY_COMP_COMBO_BOOST"
    elif gap_family in {"LOW_MID_GAP_3_4", "MID_GAP_3_5"}:
        multipliers["point"] = POINT_GAP_LOW_MID_POINT_MULT
        multipliers["combo"] = POINT_GAP_LOW_MID_COMBO_BOOST
        multipliers["composition_mc"] = POINT_GAP_LOW_MID_COMP_BOOST
        status = "GAP_LOW_MID_BALANCE"
    elif gap_family == "MID_HIGH_GAP_5_7":
        multipliers["point"] = POINT_GAP_MID_HIGH_POINT_BOOST
        multipliers["composition_mc"] = POINT_GAP_MID_HIGH_COMP_SHRINK
        multipliers["road_profile"] = POINT_GAP_MID_HIGH_ROAD_SHRINK
        status = "GAP_MID_HIGH_POINT_PROTECT"
    elif gap_family in {"EXTREME_GAP_8_9", "BIG_GAP_6_9"}:
        multipliers["point"] = POINT_GAP_EXTREME_POINT_BOOST
        multipliers["composition_mc"] = POINT_GAP_EXTREME_COMP_SHRINK
        multipliers["road_profile"] = POINT_GAP_EXTREME_ROAD_SHRINK
        status = "GAP_EXTREME_POINT_PROTECT"
    elif gap_family == "TIE_GAP":
        multipliers["composition_mc"] = POINT_GAP_TIE_COMP_SHRINK
        multipliers["road_profile"] = POINT_GAP_TIE_ROAD_SHRINK
        status = "GAP_TIE_PROTECTION"

    def _safe_mult(x: float) -> float:
        return clamp(float(x), 0.50, POINT_GAP_CALIBRATOR_MAX_TOTAL_MULT)

    point_w2 = point_w * _safe_mult(multipliers["point"])
    combo_w2 = combo_w * _safe_mult(multipliers["combo"])
    comp_w2 = comp_w * _safe_mult(multipliers["composition_mc"])
    road_w2 = road_w * _safe_mult(multipliers["road_profile"])
    sim_w2 = sim_w * _safe_mult(multipliers["simulation"])

    return point_w2, combo_w2, comp_w2, road_w2, sim_w2, {
        **profile,
        "enabled": True,
        "status": status,
        "multipliers": multipliers,
        "before_weights": {
            "point": point_w,
            "combo": combo_w,
            "composition_mc": comp_w,
            "road_profile": road_w,
            "simulation": sim_w,
        },
        "after_weights": {
            "point": point_w2,
            "combo": combo_w2,
            "composition_mc": comp_w2,
            "road_profile": road_w2,
            "simulation": sim_w2,
        },
    }

def apply_natural_high_guard(
    comp: Dict[str, Any],
    point_gap_info: Dict[str, Any],
    point_w: float,
    combo_w: float,
    comp_w: float,
    road_w: float,
    sim_w: float,
) -> Tuple[float, float, float, float, float, Dict[str, Any]]:
    natural_high = bool(
        USE_NATURAL_HIGH_GUARD
        and (
            comp.get("natural_high_winner")
            or (
                str(comp.get("top_scenario", "")) == "NONE_DRAW"
                and int(point_gap_info.get("winner_point", 0) or 0) in (8, 9)
            )
        )
    )

    info = {
        "enabled": bool(USE_NATURAL_HIGH_GUARD),
        "natural_high_winner": natural_high,
        "status": "NATURAL_HIGH_NOT_APPLIED",
        "before_weights": {
            "point": point_w,
            "combo": combo_w,
            "composition_mc": comp_w,
            "road_profile": road_w,
            "simulation": sim_w,
        },
    }

    if not natural_high:
        info["after_weights"] = dict(info["before_weights"])
        return point_w, combo_w, comp_w, road_w, sim_w, info

    point_w2 = point_w * clamp(NATURAL_HIGH_POINT_BOOST, 0.80, 1.25)
    combo_w2 = combo_w * clamp(NATURAL_HIGH_COMBO_SHRINK, 0.70, 1.05)
    comp_w2 = comp_w * clamp(NATURAL_HIGH_COMP_SHRINK, 0.60, 1.00)
    road_w2 = road_w * clamp(NATURAL_HIGH_ROAD_SHRINK, 0.60, 1.00)
    sim_w2 = sim_w

    info.update({
        "status": "NATURAL_HIGH_POINT_PROTECT",
        "multipliers": {
            "point": NATURAL_HIGH_POINT_BOOST,
            "combo": NATURAL_HIGH_COMBO_SHRINK,
            "composition_mc": NATURAL_HIGH_COMP_SHRINK,
            "road_profile": NATURAL_HIGH_ROAD_SHRINK,
            "simulation": 1.0,
        },
        "after_weights": {
            "point": point_w2,
            "combo": combo_w2,
            "composition_mc": comp_w2,
            "road_profile": road_w2,
            "simulation": sim_w2,
        },
    })
    return point_w2, combo_w2, comp_w2, road_w2, sim_w2, info

def neutral_record(source: str = "NEUTRAL") -> Dict[str, Any]:
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
# point_db
# ============================================================
def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = point_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE

    if diff == 0:
        banker += stable_noise(key + ":tie", 0.014)
    elif 1 <= diff <= 5:
        banker -= 0.155
    elif diff >= 6:
        banker -= 0.095
    elif -5 <= diff <= -1:
        banker += 0.155
    elif diff <= -6:
        banker += 0.095

    banker += stable_noise(key + ":fallback", 0.035)
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
        banker = rec.get("next_banker_rate", rec.get("banker_prob", rec.get("banker_rate")))
        player = rec.get("next_player_rate", rec.get("player_prob", rec.get("player_rate")))
        if banker is None or player is None:
            return fallback_point_lookup(player_point, banker_point)
        banker, player = normalize_prob_pair(float(banker), float(player))
        meta = point_db_meta()
        return {
            "available": True,
            "feature_key": point_key(player_point, banker_point),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_DB"),
            "sample_size": int(rec.get("sample", rec.get("sample_size", 0)) or 0),
            "total_simulated_samples": int(meta.get("total_simulated_samples", 0) or 0),
        }
    except Exception:
        return fallback_point_lookup(player_point, banker_point)

# ============================================================
# composition + combo condition db
# ============================================================
def composition_mc_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not USE_COMPOSITION_MC or not callable(composition_mc_lookup):
        return {
            **neutral_record("COMPOSITION_MC_DISABLED"),
            "scenario_debug": [],
            "top_scenario": "UNKNOWN",
            "scenario_count": 0,
        }
    try:
        rec = composition_mc_lookup(
            player_point=player_point,
            banker_point=banker_point,
            n_sim=COMPOSITION_MC_SIMULATIONS,
            max_combos=COMPOSITION_MC_MAX_COMBOS,
            seed_key=f"{player_point}:{banker_point}:{len(rounds or [])}",
        )
        if not isinstance(rec, dict):
            raise ValueError("composition_mc_lookup returned non-dict")
        banker, player = normalize_prob_pair(rec.get("banker_prob", BASE_BANKER_NO_TIE), rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", f"P{player_point}_B{banker_point}_COMPOSITION_MC"),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_COMPOSITION_MC"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", rec.get("sample_size", 0)) or 0),
            "scenario_debug": rec.get("scenario_debug", []),
            "top_scenario": rec.get("top_scenario", "UNKNOWN"),
            "top_scenario_probability": float(rec.get("top_scenario_probability", 0.0) or 0.0),
            "second_scenario_probability": float(rec.get("second_scenario_probability", 0.0) or 0.0),
            "scenario_entropy": float(rec.get("scenario_entropy", 1.0) or 1.0),
            "composition_confidence": float(rec.get("composition_confidence", 0.0) or 0.0),
            "composition_gap": float(rec.get("composition_gap", abs(banker - player)) or 0.0),
            "winner_side": rec.get("winner_side", "UNKNOWN"),
            "winner_point": rec.get("winner_point"),
            "winner_point_zone": rec.get("winner_point_zone", "UNKNOWN"),
            "point_gap": rec.get("point_gap", abs(player_point - banker_point)),
            "point_diff": rec.get("point_diff", player_point - banker_point),
            "gap_zone": rec.get("gap_zone", "UNKNOWN"),
            "gap_zone_zh": rec.get("gap_zone_zh", ""),
            "gap_family": rec.get("gap_family", rec.get("gap_zone", "UNKNOWN")),
            "gap_family_zh": rec.get("gap_family_zh", rec.get("gap_zone_zh", "")),
            "natural_winner": rec.get("natural_winner", False),
            "natural_high_winner": rec.get("natural_high_winner", False),
            "natural_side": rec.get("natural_side", "NONE"),
            "realistic_rule_filter": rec.get("realistic_rule_filter", False),
            "scenario_count": int(rec.get("scenario_count", 0) or 0),
        }
    except Exception as e:
        return {
            **neutral_record(f"COMPOSITION_MC_ERROR:{e}"),
            "scenario_debug": [],
            "top_scenario": "UNKNOWN",
            "scenario_count": 0,
        }

def combo_condition_lookup(player_point: int, banker_point: int, rounds: Optional[List[Any]], comp: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_COMBO_DB or not callable(combo_lookup):
        return neutral_record("COMBO_DB_DISABLED")
    try:
        rec = combo_lookup(
            player_point=player_point,
            banker_point=banker_point,
            rounds=rounds,
            composition=comp,
            min_sample=COMBO_DB_MIN_SAMPLE,
        )
        if not isinstance(rec, dict):
            raise ValueError("combo_lookup returned non-dict")
        banker, player = normalize_prob_pair(rec.get("banker_prob", BASE_BANKER_NO_TIE), rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", point_key(player_point, banker_point)),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_CONDITION_COMBO_DB"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", 0) or 0),
            "candidate_keys": rec.get("candidate_keys", []),
            "matched_records": rec.get("matched_records", []),
            "top_scenario": rec.get("top_scenario", comp.get("top_scenario", "UNKNOWN")),
        }
    except Exception as e:
        return neutral_record(f"COMBO_DB_ERROR:{e}")

# ============================================================
# road_profile_db
# ============================================================
def road_profile_layer(player_point: int, banker_point: int, comp: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_ROAD_PROFILE_DB or not callable(road_profile_lookup):
        return {
            **neutral_record("ROAD_PROFILE_DB_DISABLED"),
            "top_road_profile": "NEUTRAL",
            "top_road_profile_zh": "中性路段",
            "profile_distribution": [],
            "use_user_history": False,
        }
    try:
        rec = road_profile_lookup(
            player_point=player_point,
            banker_point=banker_point,
            composition=comp,
            min_sample=ROAD_PROFILE_MIN_SAMPLE,
        )
        if not isinstance(rec, dict):
            raise ValueError("road_profile_lookup returned non-dict")
        banker, player = normalize_prob_pair(rec.get("banker_prob", BASE_BANKER_NO_TIE), rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", point_key(player_point, banker_point)),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "ROAD_PROFILE_DB"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", 0) or 0),
            "top_road_profile": rec.get("top_road_profile", "NEUTRAL"),
            "top_road_profile_zh": rec.get("top_road_profile_zh", "中性路段"),
            "profile_distribution": rec.get("profile_distribution", []),
            "candidate_keys": rec.get("candidate_keys", []),
            "matched_records": rec.get("matched_records", []),
            "same_point_repeat_avg": rec.get("same_point_repeat_avg", 0),
            "use_user_history": False,
        }
    except Exception as e:
        return {
            **neutral_record(f"ROAD_PROFILE_DB_ERROR:{e}"),
            "top_road_profile": "NEUTRAL",
            "top_road_profile_zh": "中性路段",
            "profile_distribution": [],
            "use_user_history": False,
        }

# ============================================================
# micro_road_model
# ============================================================
def micro_road_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]], comp: Dict[str, Any]) -> Dict[str, Any]:
    if not USE_MICRO_ROAD_MODEL or not callable(micro_road_lookup):
        return {
            **neutral_record("MICRO_ROAD_DISABLED"),
            "micro_direction": "NEUTRAL",
            "micro_confidence": 0.0,
            "micro_patterns": [],
            "recent_road": "",
        }
    try:
        rec = micro_road_lookup(player_point=player_point, banker_point=banker_point, rounds=rounds, composition=comp)
        if not isinstance(rec, dict):
            raise ValueError("micro_road_lookup returned non-dict")
        banker, player = normalize_prob_pair(rec.get("banker_prob", BASE_BANKER_NO_TIE), rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": f"{point_key(player_point, banker_point)}_MICRO_ROAD",
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "MICRO_ROAD_MODEL"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", rec.get("sample_size", 0)) or 0),
            "micro_direction": rec.get("micro_direction", "NEUTRAL"),
            "micro_confidence": float(rec.get("micro_confidence", 0.0) or 0.0),
            "micro_patterns": rec.get("micro_patterns", []),
            "recent_road": rec.get("recent_road", ""),
            "direction_scores": rec.get("direction_scores", {}),
            "context": rec.get("context", {}),
        }
    except Exception as e:
        return {
            **neutral_record(f"MICRO_ROAD_ERROR:{e}"),
            "micro_direction": "NEUTRAL",
            "micro_confidence": 0.0,
            "micro_patterns": [],
            "recent_road": "",
        }

# ============================================================
# AI 微調層 V10.3
# ============================================================
def extract_round_points(rounds: Optional[List[Any]]) -> List[Tuple[int, int]]:
    if not rounds:
        return []
    out: List[Tuple[int, int]] = []
    for r in rounds:
        pp = bp = None
        if isinstance(r, dict):
            pp = r.get("player_point", r.get("player", r.get("p")))
            bp = r.get("banker_point", r.get("banker", r.get("b")))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            pp, bp = r[0], r[1]
        try:
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

def detect_simple_patterns(results: List[str]) -> Dict[str, Any]:
    n = len(results)
    if n < 3:
        return {"pattern_type": "none", "pattern_strength": 0.0, "suggest": "NEUTRAL"}

    alternating_min = max(3, int(AI_PATTERN_ALTERNATING_MIN))
    if n >= alternating_min:
        recent_alt = results[-alternating_min:]
        is_alternating = all(recent_alt[i] != recent_alt[i+1] for i in range(len(recent_alt)-1))
        if is_alternating:
            next_side = "PLAYER" if recent_alt[-1] == "BANKER" else "BANKER" if recent_alt[-1] == "PLAYER" else "NEUTRAL"
            if next_side in ("BANKER", "PLAYER"):
                strength = min(0.85, 0.60 + alternating_min * 0.05)
                return {
                    "pattern_type": "alternating",
                    "pattern_strength": strength,
                    "suggest": next_side,
                    "pattern_detail": f"單跳{alternating_min}口",
                }

    streak_min = max(2, int(AI_PATTERN_STREAK_MIN))
    for side in ("閒", "莊"):
        side_map = {"閒": "PLAYER", "莊": "BANKER"}
        streak = 0
        for r in reversed(results):
            if r == side:
                streak += 1
            else:
                break
        if streak >= streak_min:
            mapped_side = side_map.get(side, "NEUTRAL")
            if mapped_side in ("BANKER", "PLAYER"):
                strength = min(0.90, 0.55 + streak * 0.08)
                detail = f"{side}長龍{streak}口"
                if streak >= 6:
                    strength = max(0.40, strength - 0.15)
                    detail += "（注意斷龍風險）"
                return {
                    "pattern_type": "streak",
                    "pattern_strength": strength,
                    "suggest": mapped_side,
                    "pattern_detail": detail,
                    "streak_side": side,
                    "streak_count": streak,
                }

    if n >= 4:
        last4 = results[-4:]
        if last4[0] == last4[1] and last4[2] == last4[3] and last4[0] != last4[2]:
            next_side = "PLAYER" if last4[-1] == "PLAYER" else "BANKER" if last4[-1] == "BANKER" else "NEUTRAL"
            if next_side in ("BANKER", "PLAYER"):
                return {
                    "pattern_type": "double_jump",
                    "pattern_strength": 0.60,
                    "suggest": next_side,
                    "pattern_detail": "雙跳節奏",
                }

    return {"pattern_type": "none", "pattern_strength": 0.0, "suggest": "NEUTRAL"}

def ai_simulation_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = point_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE
    reasons: List[str] = []
    history_adjust = 0.0

    if diff == 0:
        banker += stable_noise(key + ":ai_tie", 0.004)
        reasons.append("current_tie_point_noise")
    elif abs(diff) <= 2:
        banker += -0.014 if diff > 0 else 0.014
        reasons.append("current_small_diff_adjust")
    elif abs(diff) <= 5:
        banker += -0.018 if diff > 0 else 0.018
        reasons.append("current_mid_diff_adjust")
    else:
        banker += -0.010 if diff > 0 else 0.010
        reasons.append("current_large_diff_adjust")

    recent = extract_round_points(rounds)[-AI_HISTORY_WINDOW:]
    if len(recent) >= 3:
        player_points = [p for p, _ in recent]
        banker_points = [b for _, b in recent]
        diffs = [p - b for p, b in recent]
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

        if sum(1 for x in player_points[-3:] if x >= 7) >= 3:
            history_adjust += AI_REVERSAL_STRENGTH
            reasons.append("player_hot_reversal_guard")
        elif sum(1 for x in banker_points[-3:] if x >= 7) >= 3:
            history_adjust -= AI_REVERSAL_STRENGTH
            reasons.append("banker_hot_reversal_guard")

    history_adjust = clamp(history_adjust, -AI_HISTORY_MAX_ADJUST, AI_HISTORY_MAX_ADJUST)
    banker += history_adjust
    banker += stable_noise(key + ":ai_v9", AI_NOISE_SCALE)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

    trend_banker = float(banker)

    pattern_info: Dict[str, Any] = {
        "pattern_type": "none",
        "pattern_strength": 0.0,
        "suggest": "NEUTRAL",
        "pattern_detail": "",
    }
    ai_confidence = 0.0
    ai_direction = "BANKER" if banker >= BASE_BANKER_NO_TIE else "PLAYER"
    streak_side = "NEUTRAL"
    streak_count = 0
    history_results: List[str] = []

    if USE_AI_PATTERN_RECOGNITION and len(recent) >= 3:
        history_results = [get_last_result(p, b) for p, b in recent]
        pattern_info = detect_simple_patterns(history_results)

        for side in ("閒", "莊"):
            cnt = 0
            for r in reversed(history_results):
                if r == side:
                    cnt += 1
                else:
                    break
            if cnt > streak_count:
                streak_count = cnt
                streak_side = side

        point_trend_confidence = min(1.0, abs(history_adjust) / max(AI_HISTORY_MAX_ADJUST, 0.001))
        pattern_confidence = pattern_info.get("pattern_strength", 0.0)
        streak_ratio = min(1.0, streak_count / max(len(history_results), 1))
        ai_confidence = round(
            point_trend_confidence * 0.35 +
            pattern_confidence * 0.45 +
            streak_ratio * 0.20,
            4
        )

        suggest = pattern_info.get("suggest", "NEUTRAL")
        if suggest in ("BANKER", "PLAYER") and pattern_confidence >= 0.55:
            ai_direction = suggest
            reasons.append(f"pattern_{pattern_info.get('pattern_type', 'none')}_override")

    ai_db_info: Dict[str, Any] = {
        "db_available": False,
        "db_banker_prob": BASE_BANKER_NO_TIE,
        "db_player_prob": 1.0 - BASE_BANKER_NO_TIE,
        "db_confidence": 0.0,
        "db_sample_size": 0,
        "db_feature_key": "",
        "db_source": "NOT_AVAILABLE",
    }
    ai_db_blended = False

    if USE_AI_POINT_FEATURE_DB:
        try:
            from ai_point_feature_db import get_ai_point_feature_record, ai_point_feature_db_meta
            db_rec = get_ai_point_feature_record(player_point, banker_point)
            if isinstance(db_rec, dict) and db_rec.get("banker_prob") is not None:
                db_banker = float(db_rec.get("banker_prob", BASE_BANKER_NO_TIE))
                db_player = float(db_rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
                db_banker, db_player = normalize_prob_pair(db_banker, db_player)
                db_confidence = float(db_rec.get("confidence", 0.0) or 0.0)
                ai_db_info = {
                    "db_available": True,
                    "db_banker_prob": db_banker,
                    "db_player_prob": db_player,
                    "db_confidence": db_confidence,
                    "db_sample_size": int(db_rec.get("sample_size", 0) or 0),
                    "db_feature_key": db_rec.get("feature_key", ""),
                    "db_source": db_rec.get("source", "AI_POINT_FEATURE_DB"),
                }

                if db_confidence >= AI_DB_MIN_CONFIDENCE:
                    w_trend = float(AI_DB_WEIGHT_TREND)
                    w_pattern = float(AI_DB_WEIGHT_PATTERN)
                    w_db = float(AI_DB_WEIGHT_DB)
                    total_ai_w = w_trend + w_pattern + w_db

                    blended_banker = trend_banker * w_trend

                    pattern_suggest = pattern_info.get("suggest", "NEUTRAL")
                    if pattern_suggest == "BANKER":
                        pattern_banker = clamp(BASE_BANKER_NO_TIE + 0.06, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
                    elif pattern_suggest == "PLAYER":
                        pattern_banker = clamp(BASE_BANKER_NO_TIE - 0.06, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
                    else:
                        pattern_banker = BASE_BANKER_NO_TIE
                    blended_banker += pattern_banker * w_pattern

                    blended_banker += db_banker * w_db

                    blended_banker /= total_ai_w
                    banker = clamp(blended_banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
                    ai_db_blended = True
                    reasons.append("ai_point_feature_db_blended")
        except Exception:
            pass

    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_POINT_SEQUENCE_V10_3",
        "history_points_used": len(recent),
        "history_adjust": history_adjust,
        "history_reasons": reasons,
        "ai_confidence": ai_confidence,
        "ai_direction": ai_direction,
        "streak_side": streak_side,
        "streak_count": streak_count,
        "pattern_type": pattern_info.get("pattern_type", "none"),
        "pattern_strength": pattern_info.get("pattern_strength", 0.0),
        "pattern_suggest": pattern_info.get("suggest", "NEUTRAL"),
        "pattern_detail": pattern_info.get("pattern_detail", ""),
        "history_results": history_results,
        "pattern_recognition_enabled": bool(USE_AI_PATTERN_RECOGNITION),
        "ai_db_available": ai_db_info["db_available"],
        "ai_db_banker_prob": ai_db_info["db_banker_prob"],
        "ai_db_player_prob": ai_db_info["db_player_prob"],
        "ai_db_confidence": ai_db_info["db_confidence"],
        "ai_db_sample_size": ai_db_info["db_sample_size"],
        "ai_db_feature_key": ai_db_info["db_feature_key"],
        "ai_db_source": ai_db_info["db_source"],
        "ai_db_blended": ai_db_blended,
        "ai_db_blend_factor": AI_DB_BLEND_FACTOR if ai_db_blended else 0.0,
        "trend_banker_before_blend": trend_banker,
        "ai_db_enabled": bool(USE_AI_POINT_FEATURE_DB),
    }

# ============================================================
# Monte Carlo
# ============================================================
def monte_carlo_verify_from_probs(banker_prob: float, player_prob: float, n_sim: Optional[int] = None, seed_key: str = "") -> Dict[str, Any]:
    if n_sim is None:
        n_sim = MC_SIMULATIONS
    try:
        n_sim = int(n_sim)
    except Exception:
        n_sim = 300
    min_sim = max(20, int(MC_MIN_SIMULATIONS))
    max_sim = max(min_sim, int(MC_MAX_SIMULATIONS))
    n_sim = max(min_sim, min(n_sim, max_sim))

    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)
    rng = random.Random(f"{MC_SEED}:{seed_key}")
    banker_wins = 0
    player_wins = 0

    for _ in range(n_sim):
        b = clamp(banker_prob + rng.uniform(-MC_MAX_NOISE, MC_MAX_NOISE), MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
        if rng.random() < b:
            banker_wins += 1
        else:
            player_wins += 1

    total = banker_wins + player_wins
    if total <= 0:
        banker_rate = BASE_BANKER_NO_TIE
        player_rate = 1.0 - banker_rate
    else:
        banker_rate = banker_wins / total
        player_rate = player_wins / total

    mc_gap = abs(banker_rate - player_rate)
    mc_recommend = "莊" if banker_rate >= player_rate else "閒"
    return {
        "mc_enabled": True,
        "mc_simulations": n_sim,
        "mc_banker_rate": round(banker_rate * 100, PERCENT_DECIMALS),
        "mc_player_rate": round(player_rate * 100, PERCENT_DECIMALS),
        "mc_banker_rate_raw": banker_rate,
        "mc_player_rate_raw": player_rate,
        "mc_recommend": mc_recommend,
        "mc_gap": round(mc_gap * 100, PERCENT_DECIMALS),
        "mc_gap_raw": mc_gap,
        "mc_source": "MONTE_CARLO_PROB_STABILITY_CHECK_V9",
    }

def disabled_monte_carlo_result() -> Dict[str, Any]:
    return {"mc_enabled": False, "mc_simulations": 0, "mc_source": "MONTE_CARLO_DISABLED"}

def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    if not is_tie_point:
        return banker
    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK

def build_entry_decision(is_tie_point: bool, gap: float) -> Tuple[bool, str, str]:
    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "上一局為和局點數，莊閒優勢不足，建議觀察一局"
    if gap < MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "莊閒機率差距不足，建議觀察一局"
    if gap >= STRONG_GAP_FOR_ENTRY:
        return True, "strong", ""
    return True, "normal", ""

# ============================================================
# 決策控制器
# ============================================================
def _side_from_text(value: Any) -> str:
    s = str(value or "").strip().upper()
    if s in {"B", "BANKER", "莊", "庄"}:
        return "BANKER"
    if s in {"P", "PLAYER", "閒", "闲"}:
        return "PLAYER"
    return "NEUTRAL"

def _target_banker_for_side(side: str, edge: float) -> float:
    edge = clamp(float(edge), 0.0, DECISION_CONTROLLER_MAX_ADJUST)
    if side == "BANKER":
        return clamp(BASE_BANKER_NO_TIE + edge, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    if side == "PLAYER":
        return clamp(BASE_BANKER_NO_TIE - edge, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    return BASE_BANKER_NO_TIE

def _blend_to_side(current_banker: float, side: str, edge: float, blend: float) -> float:
    if side not in {"BANKER", "PLAYER"}:
        return current_banker
    target = _target_banker_for_side(side, edge)
    blend = clamp(float(blend), 0.0, 1.0)
    return clamp(current_banker * (1.0 - blend) + target * blend, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)

def _layer_side_and_gap(rec: Dict[str, Any], neutral_gap: float = 0.006) -> Tuple[str, float]:
    side, gap = _layer_direction(rec, neutral_gap=neutral_gap)
    if side == "BANKER":
        return "BANKER", gap
    if side == "PLAYER":
        return "PLAYER", gap
    return "NEUTRAL", gap

def _is_trap_pattern(patterns: List[Any]) -> bool:
    pset = {str(x) for x in (patterns or [])}
    trap_keys = {
        "NATURAL_HIGH_TRAP_REVERSAL",
        "MID_HIGH_GAP_TURN_GUARD",
        "DOUBLE_JUMP_TAIL",
        "ZIGZAG_TURN",
        "ROOM_PATTERN_TAIL",
        "ROAD_EDGE_REVERSAL",
        "LAST_TWO_TURN",
    }
    return bool(pset.intersection(trap_keys))

def apply_decision_controller(
    banker: float,
    point: Dict[str, Any],
    combo: Dict[str, Any],
    road: Dict[str, Any],
    micro: Dict[str, Any],
    comp: Dict[str, Any],
    ai: Dict[str, Any],
    weights: Dict[str, float],
    point_gap_info: Dict[str, Any],
    natural_high_guard_info: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    original_banker = float(banker)
    if not USE_DECISION_CONTROLLER:
        return banker, {
            "enabled": False,
            "status": "DISABLED",
            "banker_before": original_banker,
            "banker_after": banker,
            "adjustments": [],
        }

    adjustments: List[Dict[str, Any]] = []
    banker2 = float(banker)

    layers = {
        "point": (point, float(weights.get("point", 0.0) or 0.0)),
        "combo": (combo, float(weights.get("combo", 0.0) or 0.0)),
        "composition_mc": (comp, float(weights.get("composition_mc", 0.0) or 0.0)),
        "road_profile": (road, float(weights.get("road_profile", 0.0) or 0.0)),
        "micro_road": (micro, float(weights.get("micro_road", 0.0) or 0.0)),
        "ai": (ai, float(weights.get("simulation", 0.0) or 0.0)),
    }

    score = {"BANKER": 0.0, "PLAYER": 0.0}
    support_count = {"BANKER": 0, "PLAYER": 0}
    layer_debug: Dict[str, Any] = {}

    ai_pattern_strength = float(ai.get("pattern_strength", 0.0) or 0.0)
    ai_pattern_suggest = str(ai.get("pattern_suggest", "NEUTRAL") or "NEUTRAL")
    ai_confidence = float(ai.get("ai_confidence", 0.0) or 0.0)
    ai_db_available = bool(ai.get("ai_db_available", False))
    ai_db_confidence = float(ai.get("ai_db_confidence", 0.0) or 0.0)
    ai_db_blended = bool(ai.get("ai_db_blended", False))
    ai_boost_applied = False

    for name, (rec, w) in layers.items():
        if name == "micro_road":
            side = _side_from_text(rec.get("micro_direction", "NEUTRAL"))
            if side == "NEUTRAL":
                side, gap = _layer_side_and_gap(rec, neutral_gap=0.006)
            else:
                gap = abs(float(rec.get("banker_prob", BASE_BANKER_NO_TIE)) - float(rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE)))
        elif name == "ai":
            side = ai.get("ai_direction", "NEUTRAL")
            if isinstance(side, str):
                side = _side_from_text(side)
            gap = abs(float(rec.get("banker_prob", BASE_BANKER_NO_TIE)) - float(rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE)))
            if (
                USE_AI_PATTERN_RECOGNITION
                and ai_pattern_strength >= AI_PATTERN_MIN_STRENGTH
                and ai_pattern_suggest in ("BANKER", "PLAYER")
            ):
                boosted_gap = max(float(gap), 0.04 + ai_pattern_strength * 0.06)
                gap = min(boosted_gap, 0.10)
                side = ai_pattern_suggest
                ai_boost_applied = True
            if ai_db_blended and ai_db_confidence >= AI_DB_MIN_CONFIDENCE:
                gap = max(float(gap), 0.03 + ai_db_confidence * 0.05)
        else:
            side, gap = _layer_side_and_gap(rec, neutral_gap=0.006)

        strength = max(0.0, w) * min(float(gap) / 0.08, 1.0)
        if side in {"BANKER", "PLAYER"} and strength > 0:
            score[side] += strength
            support_count[side] += 1
        layer_debug[name] = {
            "side": side,
            "gap": gap,
            "weight": w,
            "strength": strength,
        }

    consensus_side = "NEUTRAL"
    if score["BANKER"] > score["PLAYER"]:
        consensus_side = "BANKER"
    elif score["PLAYER"] > score["BANKER"]:
        consensus_side = "PLAYER"

    if (
        consensus_side in {"BANKER", "PLAYER"}
        and support_count[consensus_side] >= int(CONSENSUS_MIN_SUPPORT)
        and abs(score["BANKER"] - score["PLAYER"]) >= 0.025
    ):
        edge = min(CONSENSUS_EDGE + abs(score["BANKER"] - score["PLAYER"]) * 0.08, DECISION_CONTROLLER_MAX_ADJUST)
        before = banker2
        banker2 = _blend_to_side(banker2, consensus_side, edge=edge, blend=CONSENSUS_BLEND)
        adjustments.append({
            "type": "CONSENSUS_BOOST",
            "side": consensus_side,
            "edge": edge,
            "blend": CONSENSUS_BLEND,
            "before": before,
            "after": banker2,
            "support_count": support_count[consensus_side],
            "score": dict(score),
        })

    micro_side = _side_from_text(micro.get("micro_direction", "NEUTRAL"))
    micro_conf = float(micro.get("micro_confidence", 0.0) or 0.0)
    micro_patterns = micro.get("micro_patterns", []) or []
    current_side = "BANKER" if banker2 >= BASE_BANKER_NO_TIE else "PLAYER"
    trap_hit = _is_trap_pattern(micro_patterns)

    if (
        MICRO_ROAD_OVERRIDE
        and micro_side in {"BANKER", "PLAYER"}
        and micro_conf >= MICRO_ROAD_OVERRIDE_CONFIDENCE
        and (trap_hit or MICRO_ROAD_OVERRIDE_ALWAYS_APPLY)
    ):
        extra = 0.0
        if bool(natural_high_guard_info.get("natural_high_winner", False)):
            extra += NATURAL_TRAP_EXTRA_EDGE
        if str(point_gap_info.get("gap_family", point_gap_info.get("gap_zone", ""))) == "MID_HIGH_GAP_5_7":
            extra += MID_HIGH_GAP_EXTRA_EDGE
        edge = min(MICRO_ROAD_OVERRIDE_EDGE * clamp(micro_conf, 0.40, 1.0) + extra, DECISION_CONTROLLER_MAX_ADJUST)
        blend = MICRO_ROAD_OVERRIDE_BLEND if micro_side != current_side else min(MICRO_ROAD_OVERRIDE_BLEND * 0.55, 0.45)
        before = banker2
        banker2 = _blend_to_side(banker2, micro_side, edge=edge, blend=blend)
        adjustments.append({
            "type": "MICRO_ROAD_OVERRIDE",
            "side": micro_side,
            "edge": edge,
            "blend": blend,
            "before": before,
            "after": banker2,
            "micro_confidence": micro_conf,
            "micro_patterns": micro_patterns,
            "trap_hit": trap_hit,
            "current_side_before": current_side,
        })

    if (
        USE_AI_PATTERN_RECOGNITION
        and ai_pattern_suggest in ("BANKER", "PLAYER")
        and ai_pattern_strength >= 0.70
        and ai_confidence >= AI_CONFIDENCE_MIN_FOR_DECISION
        and not ai_boost_applied
    ):
        ai_override_side = ai_pattern_suggest
        current_side2 = "BANKER" if banker2 >= BASE_BANKER_NO_TIE else "PLAYER"
        if ai_override_side != current_side2:
            blend = clamp(0.25 + ai_pattern_strength * 0.50, 0.30, 0.70)
            edge = clamp(AI_DECISION_BOOST_EDGE * ai_pattern_strength, 0.0, DECISION_CONTROLLER_MAX_ADJUST)
            before = banker2
            banker2 = _blend_to_side(banker2, ai_override_side, edge=edge, blend=blend)
            adjustments.append({
                "type": "AI_PATTERN_OVERRIDE",
                "side": ai_override_side,
                "edge": edge,
                "blend": blend,
                "before": before,
                "after": banker2,
                "ai_confidence": ai_confidence,
                "ai_pattern_strength": ai_pattern_strength,
                "ai_pattern_type": ai.get("pattern_type", "none"),
                "ai_pattern_detail": ai.get("pattern_detail", ""),
            })

    if (
        USE_AI_POINT_FEATURE_DB
        and ai_db_available
        and ai_db_confidence >= AI_DB_MIN_CONFIDENCE
    ):
        db_banker = float(ai.get("ai_db_banker_prob", BASE_BANKER_NO_TIE))
        db_side = "BANKER" if db_banker >= BASE_BANKER_NO_TIE else "PLAYER"
        current_side3 = "BANKER" if banker2 >= BASE_BANKER_NO_TIE else "PLAYER"
        if db_side != current_side3 and ai_db_confidence >= 0.55:
            edge = clamp(AI_DECISION_BOOST_EDGE * ai_db_confidence, 0.0, DECISION_CONTROLLER_MAX_ADJUST)
            blend = clamp(AI_DB_BLEND_FACTOR * ai_db_confidence, 0.20, 0.55)
            before = banker2
            banker2 = _blend_to_side(banker2, db_side, edge=edge, blend=blend)
            adjustments.append({
                "type": "AI_DB_OVERRIDE",
                "side": db_side,
                "edge": edge,
                "blend": blend,
                "before": before,
                "after": banker2,
                "ai_db_confidence": ai_db_confidence,
                "ai_db_banker_prob": db_banker,
                "ai_db_feature_key": ai.get("ai_db_feature_key", ""),
            })

    banker2 = clamp(banker2, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    return banker2, {
        "enabled": True,
        "status": "APPLIED" if adjustments else "NO_STRONG_SIGNAL",
        "banker_before": original_banker,
        "banker_after": banker2,
        "player_before": 1.0 - original_banker,
        "player_after": 1.0 - banker2,
        "consensus_side": consensus_side,
        "score": score,
        "support_count": support_count,
        "layer_debug": layer_debug,
        "micro_side": micro_side,
        "micro_confidence": micro_conf,
        "micro_patterns": micro_patterns,
        "ai_boost_applied": ai_boost_applied,
        "ai_pattern_strength": ai_pattern_strength,
        "ai_confidence": ai_confidence,
        "ai_db_blended": ai_db_blended,
        "ai_db_confidence": ai_db_confidence,
        "adjustments": adjustments,
    }

# ============================================================
# 主預測函式 V10.6 (新增下一局情境預測融合)
# ============================================================
def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)
    rounds = rounds or []
    model_rounds = [] if PREDICT_CURRENT_ROUND_ONLY else rounds

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    comp = composition_mc_layer(player_point, banker_point, rounds=model_rounds)

    # ---------- V10.6 新增：融合下一局補牌情境預測 ----------
    if USE_NEXT_SCENARIO_PREDICT:
        try:
            from next_scenario_db import get_next_scenario_probs
            next_probs = get_next_scenario_probs(player_point, banker_point)
            if next_probs:
                # 取得原本的 scenario_debug 列表，若無則建空列表
                comp_scenarios = comp.get("scenario_debug", [])
                combined = {}
                # 合併現有情境機率 (當局反推)
                for s in comp_scenarios:
                    sc = s.get("scenario", "UNKNOWN")
                    prob = s.get("scenario_probability", s.get("weight", 0))
                    combined[sc] = combined.get(sc, 0) + prob * (1.0 - NEXT_SCENARIO_WEIGHT)
                # 合併下一局預測機率
                for sc, prob in next_probs.items():
                    combined[sc] = combined.get(sc, 0) + prob * NEXT_SCENARIO_WEIGHT
                # 標準化
                total = sum(combined.values())
                if total > 0:
                    new_debug = [{"scenario": sc, "scenario_probability": p/total} for sc, p in combined.items()]
                    comp = {**comp, "scenario_debug": new_debug}
        except Exception:
            pass  # 若模組或資料庫不存在，則維持原 comp 不變

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

    # V10.4 強弩之末保護
    micro_side = micro.get("micro_direction", "NEUTRAL")
    micro_conf = micro.get("micro_confidence", 0.0)
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
        micro_w *= 1.4
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

    banker_before_deepseek = banker

    # ---------- DeepSeek 獨立預測融合 ----------
    if os.getenv("USE_DEEPSEEK_INDEPENDENT", "0") == "1":
        try:
            from deepseek_independent_predictor import (
                deepseek_independent_predict,
                DEEPSEEK_INDEPENDENT_WEIGHT,
            )

            rounds_summary = ""
            if rounds:
                recent_rounds = rounds[-8:]
                rounds_summary = " → ".join(
                    [f"閒{r['player_point']}莊{r['banker_point']}" for r in recent_rounds if isinstance(r, dict)]
                )

            ai_banker = deepseek_independent_predict(
                player_point,
                banker_point,
                point,
                combo,
                road,
                comp,
                micro,
                ai,
                rounds_summary,
            )
            if ai_banker is not None:
                blend_w = clamp(float(DEEPSEEK_INDEPENDENT_WEIGHT), 0.0, 1.0)
                banker = ai_banker if blend_w >= 1.0 else banker * (1.0 - blend_w) + ai_banker * blend_w
                result_deepseek = {
                    "ai_banker": ai_banker,
                    "weight": blend_w,
                    "original_banker": banker_before_deepseek,
                }
            else:
                result_deepseek = None
        except Exception:
            result_deepseek = None
    else:
        result_deepseek = None

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
        "mode": "POINT_CONDITION_COMBO_COMPOSITION_MC_V10_6_NEXT_SCENARIO",
    }

    if result_deepseek is not None:
        result["deepseek_independent"] = result_deepseek

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
