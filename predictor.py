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

# ============================================================
# V9：點數 + 補牌情境 + 300 萬條件資料庫 + Monte Carlo
# ============================================================
# 主流程：
# 1. LINE 只輸入點數，例如 65。
# 2. point_composition_mc 反推補牌情境。
# 3. combo_db 用「P6_B5 + 補牌情境」查 300 萬條件資料庫。
# 4. road_profile_db 用當前點數 + 補牌情境查資料庫相似路段分佈，不吃用戶歷史。
# 5. predictor 融合 point_db / combo_db / road_profile_db / AI / composition_mc。
# 6. Monte Carlo 只做最終機率穩定度驗證，不遞迴呼叫 predict。
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


# 權重：COMBO_WEIGHT 就是「點數 + 補牌情境 + 300萬資料庫」主權重。
POINT_WEIGHT = env_float("POINT_WEIGHT", str(getattr(config, "POINT_WEIGHT", 0.58)))
COMBO_WEIGHT = env_float("COMBO_WEIGHT", os.getenv("PATTERN_WEIGHT", str(getattr(config, "PATTERN_WEIGHT", 0.24))))
SIM_WEIGHT = env_float("SIM_WEIGHT", str(getattr(config, "SIM_WEIGHT", 0.08)))
COMPOSITION_MC_WEIGHT = env_float("COMPOSITION_MC_WEIGHT", "0.10")

USE_POINT_DB = env_bool("USE_POINT_DB", "1" if getattr(config, "USE_POINT_DB", True) else "0")
USE_COMBO_DB = env_bool("USE_COMBO_DB", "1")
USE_COMPOSITION_MC = env_bool("USE_COMPOSITION_MC", "1")
USE_MONTE_CARLO = env_bool("USE_MONTE_CARLO", "1")

# 無記憶牌路資料庫：只查資料庫相似路段，不延續用戶輸入紀錄。
PREDICT_CURRENT_ROUND_ONLY = env_bool("PREDICT_CURRENT_ROUND_ONLY", "1")
USE_ROAD_PROFILE_DB = env_bool("USE_ROAD_PROFILE_DB", "1")
ROAD_PROFILE_WEIGHT = env_float("ROAD_PROFILE_WEIGHT", "0.06")

# 點數校準層：不改 point_db 原始資料，只在當局依照 combo/補牌MC/牌路/AI 一致性調整 point 權重。
USE_POINT_CALIBRATOR = env_bool("USE_POINT_CALIBRATOR", "1")
ROAD_PROFILE_MIN_SAMPLE = env_int("ROAD_PROFILE_MIN_SAMPLE", "50")
ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE = env_bool("ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE", "1")
ROAD_PROFILE_REQUIRE_AVAILABLE = env_bool("ROAD_PROFILE_REQUIRE_AVAILABLE", "0")
ROAD_PROFILE_SIGNAL_MIN_GAP = env_float("ROAD_PROFILE_SIGNAL_MIN_GAP", "0.010")
ROAD_PROFILE_USE_USER_HISTORY = env_bool("ROAD_PROFILE_USE_USER_HISTORY", "0")

COMBO_DB_MIN_SAMPLE = env_int("COMBO_DB_MIN_SAMPLE", "80")
COMBO_WEIGHT_REQUIRE_AVAILABLE = env_bool("COMBO_WEIGHT_REQUIRE_AVAILABLE", "1")
REQUIRE_COMBO_SAMPLE_FOR_ENTRY = env_bool("REQUIRE_COMBO_SAMPLE_FOR_ENTRY", "0")
MIN_GAP_WITHOUT_COMBO = env_float("MIN_GAP_WITHOUT_COMBO", "0.150")

COMPOSITION_MC_SIMULATIONS = env_int("COMPOSITION_MC_SIMULATIONS", "500")
COMPOSITION_MC_MAX_COMBOS = env_int("COMPOSITION_MC_MAX_COMBOS", "160")

BASE_BANKER_NO_TIE = 0.5000  # V9 no banker base bias: neutral fallback only
MIN_OUTPUT_PROB = env_float("MIN_OUTPUT_PROB", str(getattr(config, "MIN_OUTPUT_PROB", 0.38)))
MAX_OUTPUT_PROB = env_float("MAX_OUTPUT_PROB", str(getattr(config, "MAX_OUTPUT_PROB", 0.62)))
PERCENT_DECIMALS = env_int("PERCENT_DECIMALS", str(getattr(config, "PERCENT_DECIMALS", 2)))

MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", "0.060")
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", "0.085")

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", "0.012")
TIE_SHRINK = env_float("TIE_SHRINK", "0.22")
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", "0.11")

AI_NOISE_SCALE = env_float("AI_NOISE_SCALE", "0.008")
AI_HISTORY_WINDOW = env_int("AI_HISTORY_WINDOW", "5")
AI_TREND_STRENGTH = env_float("AI_TREND_STRENGTH", "0.006")
AI_DIFF_MOMENTUM_STRENGTH = env_float("AI_DIFF_MOMENTUM_STRENGTH", "0.005")
AI_REVERSAL_STRENGTH = env_float("AI_REVERSAL_STRENGTH", "0.005")
AI_HISTORY_MAX_ADJUST = env_float("AI_HISTORY_MAX_ADJUST", "0.015")

MC_SIMULATIONS = env_int("MC_SIMULATIONS", "300")
MC_MIN_SIMULATIONS = env_int("MC_MIN_SIMULATIONS", "80")
MC_MAX_SIMULATIONS = env_int("MC_MAX_SIMULATIONS", "800")
MC_SEED = env_int("MC_SEED", "42")
MC_MAX_NOISE = env_float("MC_MAX_NOISE", "0.010")
MC_BLOCK_LOW_GAP = env_bool("MC_BLOCK_LOW_GAP", "1")
MC_MIN_GAP_FOR_ENTRY = env_float("MC_MIN_GAP_FOR_ENTRY", "0.055")
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
# point_db：單純點數統計層
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
            "scenario_count": int(rec.get("scenario_count", len(rec.get("scenario_debug", [])) if isinstance(rec.get("scenario_debug", []), list) else 0) or 0),
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
# road_profile_db：無記憶牌路資料庫比對層
# ============================================================

def road_profile_layer(player_point: int, banker_point: int, comp: Dict[str, Any]) -> Dict[str, Any]:
    """
    只用當前點數 + 補牌情境查資料庫相似路段。
    不讀 rounds，不保存用戶歷史，不做追路。
    """
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
# AI 微調層：只吃歷史點數序列，小權重修正
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
    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_POINT_SEQUENCE_V9",
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
# 主預測函式
# ============================================================

def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)
    rounds = rounds or []
    # 預設不吃用戶歷史紀錄，避免變成追路。若你真的要測歷史，可把 PREDICT_CURRENT_ROUND_ONLY=0。
    model_rounds = [] if PREDICT_CURRENT_ROUND_ONLY else rounds

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point

    point = point_db_lookup(player_point, banker_point)
    comp = composition_mc_layer(player_point, banker_point, rounds=model_rounds)
    combo = combo_condition_lookup(player_point, banker_point, rounds=model_rounds, comp=comp)
    road = road_profile_layer(player_point, banker_point, comp=comp)
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
    road_w = float(ROAD_PROFILE_WEIGHT) if USE_ROAD_PROFILE_DB else 0.0

    if COMBO_WEIGHT_REQUIRE_AVAILABLE and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0):
        combo_w = 0.0

    if ROAD_PROFILE_WEIGHT_REQUIRE_AVAILABLE and (not road.get("available") or int(road.get("sample_size", 0) or 0) <= 0):
        road_w = 0.0

    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)
        comp_w = min(comp_w, COMPOSITION_MC_WEIGHT * 0.50)
        road_w = min(road_w, ROAD_PROFILE_WEIGHT * 0.50)

    total_weight = max(p_w + combo_w + road_w + sim_w + comp_w, 0.0001)
    banker = (
        point["banker_prob"] * p_w
        + combo["banker_prob"] * combo_w
        + road["banker_prob"] * road_w
        + ai["banker_prob"] * sim_w
        + comp["banker_prob"] * comp_w
    ) / total_weight

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
        "composition_scenario_count": comp.get("scenario_count", 0),
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
        # 相容舊 message_builder 欄位：pattern 改映射為 combo。
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
            "pattern": combo_w,  # 舊欄位相容
            "road_profile": road_w,
            "simulation": sim_w,
            "composition_mc": comp_w,
            "total": total_weight,
        },
        "raw_layers": {
            "point_banker_prob": point.get("banker_prob"),
            "combo_banker_prob": combo.get("banker_prob"),
            "pattern_banker_prob": combo.get("banker_prob"),
            "road_profile_banker_prob": road.get("banker_prob"),
            "ai_banker_prob": ai.get("banker_prob"),
            "composition_mc_banker_prob": comp.get("banker_prob"),
            "point_player_prob": point.get("player_prob"),
            "combo_player_prob": combo.get("player_prob"),
            "pattern_player_prob": combo.get("player_prob"),
            "road_profile_player_prob": road.get("player_prob"),
            "ai_player_prob": ai.get("player_prob"),
            "composition_mc_player_prob": comp.get("player_prob"),
        },
        "history_used": bool(model_rounds),
        "rounds_ignored": bool(rounds and PREDICT_CURRENT_ROUND_ONLY),
        "mode": "POINT_CONDITION_COMBO_COMPOSITION_MC_ROAD_PROFILE_POINT_CALIBRATOR_V9_6",
    }

    if USE_MONTE_CARLO:
        mc_result = monte_carlo_verify_from_probs(
            banker_prob=banker,
            player_prob=player,
            seed_key=f"{player_point}:{banker_point}:{combo.get('feature_key','')}:{road.get('feature_key','')}:{comp.get('top_scenario','')}:{ai.get('history_adjust',0.0)}",
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
