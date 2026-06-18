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


POINT_WEIGHT = env_float("POINT_WEIGHT", str(getattr(config, "POINT_WEIGHT", 0.56)))
COMBO_WEIGHT = env_float("COMBO_WEIGHT", os.getenv("PATTERN_WEIGHT", str(getattr(config, "PATTERN_WEIGHT", 0.08))))
SIM_WEIGHT = env_float("SIM_WEIGHT", str(getattr(config, "SIM_WEIGHT", 0.08)))
COMPOSITION_MC_WEIGHT = env_float("COMPOSITION_MC_WEIGHT", "0.28")

USE_POINT_DB = env_bool("USE_POINT_DB", "1" if getattr(config, "USE_POINT_DB", True) else "0")
USE_COMBO_DB = env_bool("USE_COMBO_DB", "1")
USE_COMPOSITION_MC = env_bool("USE_COMPOSITION_MC", "1")
USE_MONTE_CARLO = env_bool("USE_MONTE_CARLO", "1")

COMBO_DB_MIN_SAMPLE = env_int("COMBO_DB_MIN_SAMPLE", "120")
COMBO_WEIGHT_REQUIRE_AVAILABLE = env_bool("COMBO_WEIGHT_REQUIRE_AVAILABLE", "1")
REQUIRE_COMBO_SAMPLE_FOR_ENTRY = env_bool("REQUIRE_COMBO_SAMPLE_FOR_ENTRY", "0")
MIN_GAP_WITHOUT_COMBO = env_float("MIN_GAP_WITHOUT_COMBO", "0.180")

COMPOSITION_MC_SIMULATIONS = env_int("COMPOSITION_MC_SIMULATIONS", "1500")
COMPOSITION_MC_MAX_COMBOS = env_int("COMPOSITION_MC_MAX_COMBOS", "500")

BASE_BANKER_NO_TIE = 0.5000
MIN_OUTPUT_PROB = env_float("MIN_OUTPUT_PROB", str(getattr(config, "MIN_OUTPUT_PROB", 0.38)))
MAX_OUTPUT_PROB = env_float("MAX_OUTPUT_PROB", str(getattr(config, "MAX_OUTPUT_PROB", 0.62)))
PERCENT_DECIMALS = env_int("PERCENT_DECIMALS", str(getattr(config, "PERCENT_DECIMALS", 2)))

MIN_GAP_FOR_ENTRY = env_float("MIN_GAP_FOR_ENTRY", "0.035")
STRONG_GAP_FOR_ENTRY = env_float("STRONG_GAP_FOR_ENTRY", "0.070")
MODEL_NEUTRAL_GAP = env_float("MODEL_NEUTRAL_GAP", "0.003")

TIE_AI_MAX_WEIGHT = env_float("TIE_AI_MAX_WEIGHT", "0.010")
TIE_SHRINK = env_float("TIE_SHRINK", "0.20")
TIE_MIN_GAP_FOR_ENTRY = env_float("TIE_MIN_GAP_FOR_ENTRY", "0.13")

AI_NOISE_SCALE = env_float("AI_NOISE_SCALE", "0")
AI_HISTORY_WINDOW = env_int("AI_HISTORY_WINDOW", "1")
AI_TREND_STRENGTH = env_float("AI_TREND_STRENGTH", "0")
AI_DIFF_MOMENTUM_STRENGTH = env_float("AI_DIFF_MOMENTUM_STRENGTH", "0")
AI_REVERSAL_STRENGTH = env_float("AI_REVERSAL_STRENGTH", "0")
AI_HISTORY_MAX_ADJUST = env_float("AI_HISTORY_MAX_ADJUST", "0")

MC_SIMULATIONS = env_int("MC_SIMULATIONS", "1500")
MC_MIN_SIMULATIONS = env_int("MC_MIN_SIMULATIONS", "300")
MC_MAX_SIMULATIONS = env_int("MC_MAX_SIMULATIONS", "2000")
MC_SEED = env_int("MC_SEED", "42")
MC_MAX_NOISE = env_float("MC_MAX_NOISE", "0.006")
MC_BLOCK_LOW_GAP = env_bool("MC_BLOCK_LOW_GAP", "1")
MC_MIN_GAP_FOR_ENTRY = env_float("MC_MIN_GAP_FOR_ENTRY", "0.025")
MC_DIRECTION_MISMATCH_BLOCK = env_bool("MC_DIRECTION_MISMATCH_BLOCK", "1")
MC_NEUTRAL_GAP = env_float("MC_NEUTRAL_GAP", "0.006")

USE_AI_DECISION_LAYER = env_bool("USE_AI_DECISION_LAYER", "1")
AI_DECISION_WEIGHT = env_float("AI_DECISION_WEIGHT", os.getenv("SIM_WEIGHT", "0.12"))
AI_REQUIRE_SIGNAL_AGREEMENT = env_bool("AI_REQUIRE_SIGNAL_AGREEMENT", "1")
AI_BLOCK_CONFLICT_SIGNAL = env_bool("AI_BLOCK_CONFLICT_SIGNAL", "1")
AI_MIN_AGREEMENT_COUNT = env_int("AI_MIN_AGREEMENT_COUNT", "2")
AI_MIN_CONFIDENCE_GAP = env_float("AI_MIN_CONFIDENCE_GAP", "0.030")
AI_SIGNAL_MIN_GAP = env_float("AI_SIGNAL_MIN_GAP", "0.006")
AI_MAIN_SIGNAL_MIN_GAP = env_float("AI_MAIN_SIGNAL_MIN_GAP", "0.010")
AI_MIN_SCORE_EDGE = env_float("AI_MIN_SCORE_EDGE", "0.003")
AI_COMBO_SAMPLE_STRONG = env_int("AI_COMBO_SAMPLE_STRONG", "300")
AI_COMBO_SAMPLE_WEAK = env_int("AI_COMBO_SAMPLE_WEAK", "80")
AI_FORCE_OBSERVE_ON_SPLIT = env_bool("AI_FORCE_OBSERVE_ON_SPLIT", "1")
AI_OVERRIDE_DIRECTION = env_bool("AI_OVERRIDE_DIRECTION", "0")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_noise(key: str, scale: float = 0.035) -> float:
    if scale <= 0:
        return 0.0
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


def direction_from_probs(banker_prob: float, player_prob: float, neutral_gap: float = 0.0) -> Tuple[str, float]:
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)
    gap = abs(banker_prob - player_prob)
    if gap < neutral_gap:
        return "觀望", gap
    if banker_prob > player_prob:
        return "莊", gap
    if player_prob > banker_prob:
        return "閒", gap
    return "觀望", gap


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


def fallback_point_lookup(player_point: int, banker_point: int) -> Dict[str, Any]:
    diff = player_point - banker_point
    key = point_key(player_point, banker_point)
    banker = BASE_BANKER_NO_TIE
    if diff == 0:
        banker += stable_noise(key + ":tie", 0.006)
    elif 1 <= diff <= 5:
        banker -= 0.070
    elif diff >= 6:
        banker -= 0.045
    elif -5 <= diff <= -1:
        banker += 0.070
    elif diff <= -6:
        banker += 0.045
    banker += stable_noise(key + ":fallback", 0.010)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    return {
        "available": False,
        "feature_key": key,
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "FALLBACK_POINT_RULE_ONLY_NEUTRAL_SAFE",
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
    except Exception as e:
        rec = fallback_point_lookup(player_point, banker_point)
        rec["source"] = f"POINT_DB_ERROR_FALLBACK:{e}"
        return rec


def composition_mc_layer(player_point: int, banker_point: int, rounds: Optional[List[Any]] = None) -> Dict[str, Any]:
    if not USE_COMPOSITION_MC or not callable(composition_mc_lookup):
        return {**neutral_record("COMPOSITION_MC_DISABLED"), "scenario_debug": [], "top_scenario": "UNKNOWN", "scenario_count": 0}
    try:
        rec = composition_mc_lookup(
            player_point=player_point,
            banker_point=banker_point,
            n_sim=COMPOSITION_MC_SIMULATIONS,
            max_combos=COMPOSITION_MC_MAX_COMBOS,
            seed_key=f"{player_point}:{banker_point}",
        )
        if not isinstance(rec, dict):
            raise ValueError("composition_mc_lookup returned non-dict")
        banker, player = normalize_prob_pair(rec.get("banker_prob", BASE_BANKER_NO_TIE), rec.get("player_prob", 1.0 - BASE_BANKER_NO_TIE))
        scenario_debug = rec.get("scenario_debug", [])
        return {
            "available": bool(rec.get("available", False)),
            "feature_key": rec.get("feature_key", f"P{player_point}_B{banker_point}_COMPOSITION_MC"),
            "banker_prob": banker,
            "player_prob": player,
            "source": rec.get("source", "POINT_COMPOSITION_MC"),
            "sample_size": int(rec.get("sample_size", 0) or 0),
            "total_simulated_samples": int(rec.get("total_simulated_samples", rec.get("sample_size", 0)) or 0),
            "scenario_debug": scenario_debug,
            "top_scenario": rec.get("top_scenario", "UNKNOWN"),
            "scenario_count": int(rec.get("scenario_count", len(scenario_debug) if isinstance(scenario_debug, list) else 0) or 0),
        }
    except Exception as e:
        return {**neutral_record(f"COMPOSITION_MC_ERROR:{e}"), "scenario_debug": [], "top_scenario": "UNKNOWN", "scenario_count": 0}


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
        banker += stable_noise(key + ":ai_tie", 0.002)
        reasons.append("current_tie_point_noise")
    elif abs(diff) <= 2:
        banker += -0.006 if diff > 0 else 0.006
        reasons.append("current_small_diff_adjust")
    elif abs(diff) <= 5:
        banker += -0.008 if diff > 0 else 0.008
        reasons.append("current_mid_diff_adjust")
    else:
        banker += -0.004 if diff > 0 else 0.004
        reasons.append("current_large_diff_adjust")

    recent = extract_round_points(rounds)[-max(1, AI_HISTORY_WINDOW):]
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
    banker += stable_noise(key + ":ai_v9_2", AI_NOISE_SCALE)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    return {
        "banker_prob": banker,
        "player_prob": 1.0 - banker,
        "source": "LOCAL_AI_POINT_SEQUENCE_COMPAT_V9_2",
        "history_points_used": len(recent),
        "history_adjust": history_adjust,
        "history_reasons": reasons,
    }


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
    mc_direction, mc_gap = direction_from_probs(banker_rate, player_rate, neutral_gap=MC_NEUTRAL_GAP)
    return {
        "mc_enabled": True,
        "mc_simulations": n_sim,
        "mc_banker_rate": round(banker_rate * 100, PERCENT_DECIMALS),
        "mc_player_rate": round(player_rate * 100, PERCENT_DECIMALS),
        "mc_banker_rate_raw": banker_rate,
        "mc_player_rate_raw": player_rate,
        "mc_recommend": mc_direction,
        "mc_gap": round(mc_gap * 100, PERCENT_DECIMALS),
        "mc_gap_raw": mc_gap,
        "mc_neutral_gap": MC_NEUTRAL_GAP,
        "mc_source": "MONTE_CARLO_PROB_STABILITY_CHECK_NEUTRAL_SAFE_V9_2",
    }


def disabled_monte_carlo_result() -> Dict[str, Any]:
    return {"mc_enabled": False, "mc_simulations": 0, "mc_recommend": "觀望", "mc_source": "MONTE_CARLO_DISABLED"}


def apply_tie_point_protection(banker: float, is_tie_point: bool) -> float:
    if not is_tie_point:
        return banker
    return BASE_BANKER_NO_TIE + (banker - BASE_BANKER_NO_TIE) * TIE_SHRINK


def build_entry_decision(is_tie_point: bool, gap: float, recommend: str) -> Tuple[bool, str, str]:
    if recommend not in {"莊", "閒"}:
        return False, "no_entry", "莊閒差距落在中性區，建議觀察一局"
    if is_tie_point and gap < TIE_MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "上一局為和局點數，莊閒優勢不足，建議觀察一局"
    if gap < MIN_GAP_FOR_ENTRY:
        return False, "no_entry", "莊閒機率差距不足，建議觀察一局"
    if gap >= STRONG_GAP_FOR_ENTRY:
        return True, "strong", ""
    return True, "normal", ""


def _signal_item(
    name: str,
    banker_prob: float,
    player_prob: float,
    available: bool = True,
    sample_size: int = 0,
    base_weight: float = 1.0,
    min_gap: float = 0.0,
    neutral_gap: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if not available:
        return None
    direction, gap = direction_from_probs(banker_prob, player_prob, neutral_gap=neutral_gap)
    if direction == "觀望" or gap < min_gap:
        return None
    return {
        "name": name,
        "direction": direction,
        "gap_raw": gap,
        "gap": round(gap * 100, PERCENT_DECIMALS),
        "sample_size": int(sample_size or 0),
        "weight": float(base_weight),
        "strength": float(gap * base_weight),
    }


def ai_ensemble_decision_layer(point: Dict[str, Any], combo: Dict[str, Any], comp: Dict[str, Any], mc: Dict[str, Any], current_recommend: str, banker_prob: float, player_prob: float, gap: float) -> Dict[str, Any]:
    if not USE_AI_DECISION_LAYER:
        return {"ai_decision_enabled": False, "ai_decision_recommend": current_recommend, "ai_decision_direction": current_recommend, "ai_decision_observe": False, "ai_decision_reason": "AI_DECISION_LAYER_DISABLED", "ai_signals": []}

    signals: List[Dict[str, Any]] = []
    sig = _signal_item("point_db", point.get("banker_prob", BASE_BANKER_NO_TIE), point.get("player_prob", 1.0 - BASE_BANKER_NO_TIE), bool(point.get("available", False)), int(point.get("sample_size", 0) or 0), 1.10, AI_SIGNAL_MIN_GAP, MODEL_NEUTRAL_GAP)
    if sig:
        signals.append(sig)

    combo_sample = int(combo.get("sample_size", 0) or 0)
    if bool(combo.get("available", False)) and combo_sample >= AI_COMBO_SAMPLE_WEAK:
        combo_weight = 1.25 if combo_sample >= AI_COMBO_SAMPLE_STRONG else 0.75
        sig = _signal_item("combo_db", combo.get("banker_prob", BASE_BANKER_NO_TIE), combo.get("player_prob", 1.0 - BASE_BANKER_NO_TIE), True, combo_sample, combo_weight, AI_SIGNAL_MIN_GAP, MODEL_NEUTRAL_GAP)
        if sig:
            signals.append(sig)

    sig = _signal_item("composition_mc", comp.get("banker_prob", BASE_BANKER_NO_TIE), comp.get("player_prob", 1.0 - BASE_BANKER_NO_TIE), bool(comp.get("available", False)), int(comp.get("sample_size", 0) or 0), 1.15, AI_SIGNAL_MIN_GAP, MODEL_NEUTRAL_GAP)
    if sig:
        signals.append(sig)

    sig = _signal_item("main_model", banker_prob, player_prob, current_recommend in {"莊", "閒"}, 0, 1.00, AI_MAIN_SIGNAL_MIN_GAP, MODEL_NEUTRAL_GAP)
    if sig:
        signals.append(sig)

    if mc and mc.get("mc_enabled"):
        sig = _signal_item("monte_carlo", mc.get("mc_banker_rate_raw", BASE_BANKER_NO_TIE), mc.get("mc_player_rate_raw", 1.0 - BASE_BANKER_NO_TIE), mc.get("mc_recommend") in {"莊", "閒"}, int(mc.get("mc_simulations", 0) or 0), 1.20, AI_SIGNAL_MIN_GAP, MC_NEUTRAL_GAP)
        if sig:
            signals.append(sig)

    if not signals:
        return {
            "ai_decision_enabled": True,
            "ai_decision_recommend": "觀望",
            "ai_decision_direction": "觀望",
            "ai_decision_observe": True,
            "ai_decision_reason": "AI 無足夠有效訊號，建議觀察一局",
            "ai_signal_summary": "NO_VALID_SIGNAL",
            "ai_agreement_count": 0,
            "ai_conflict_count": 0,
            "ai_banker_score": 0.0,
            "ai_player_score": 0.0,
            "ai_signals": [],
        }

    banker_votes = [s for s in signals if s.get("direction") == "莊"]
    player_votes = [s for s in signals if s.get("direction") == "閒"]
    banker_score = sum(float(s.get("strength", 0.0)) for s in banker_votes)
    player_score = sum(float(s.get("strength", 0.0)) for s in player_votes)
    banker_count = len(banker_votes)
    player_count = len(player_votes)
    score_edge = abs(banker_score - player_score)

    if score_edge < AI_MIN_SCORE_EDGE:
        ai_direction = "觀望"
        agreement_count = max(banker_count, player_count)
        conflict_count = min(banker_count, player_count)
    elif banker_score > player_score:
        ai_direction = "莊"
        agreement_count = banker_count
        conflict_count = player_count
    else:
        ai_direction = "閒"
        agreement_count = player_count
        conflict_count = banker_count

    reasons: List[str] = []
    if ai_direction == "觀望":
        reasons.append("AI 莊閒分數差距不足，訊號接近中性")
    if AI_REQUIRE_SIGNAL_AGREEMENT and ai_direction in {"莊", "閒"} and agreement_count < AI_MIN_AGREEMENT_COUNT:
        reasons.append(f"AI 同方向訊號不足，僅 {agreement_count} 個，未達 {AI_MIN_AGREEMENT_COUNT} 個")
    if AI_BLOCK_CONFLICT_SIGNAL and ai_direction in {"莊", "閒"} and conflict_count >= 2 and agreement_count <= conflict_count + 1:
        reasons.append("AI 偵測 point/combo/補牌MC/MC 訊號衝突，建議觀察一局")
    if gap < AI_MIN_CONFIDENCE_GAP:
        reasons.append(f"AI 判斷主模型差距 {gap * 100:.2f}% 未達 {AI_MIN_CONFIDENCE_GAP * 100:.1f}%")
    if current_recommend not in {"莊", "閒"}:
        reasons.append("主模型本身落在中性區，AI 不強制出手")

    observe = bool(reasons)
    decision_recommend = "觀望" if observe else ai_direction
    direction_override = False

    if not observe and AI_OVERRIDE_DIRECTION and ai_direction in {"莊", "閒"} and current_recommend in {"莊", "閒"} and ai_direction != current_recommend:
        direction_override = True
        decision_recommend = ai_direction
        reasons.append(f"AI 綜合訊號改判：{current_recommend} → {ai_direction}")

    if not observe and not AI_OVERRIDE_DIRECTION and ai_direction in {"莊", "閒"} and current_recommend in {"莊", "閒"} and ai_direction != current_recommend:
        observe = True
        decision_recommend = "觀望"
        reasons.append(f"AI 方向 {ai_direction} 與主模型 {current_recommend} 不一致，建議觀察一局")

    signal_summary = " / ".join(f"{s.get('name')}:{s.get('direction')}({s.get('gap')}%)" for s in signals)
    return {
        "ai_decision_enabled": True,
        "ai_decision_recommend": decision_recommend,
        "ai_decision_direction": ai_direction,
        "ai_decision_observe": observe,
        "ai_decision_reason": "；".join(reasons) if reasons else f"AI 綜合訊號一致，偏向{ai_direction}",
        "ai_signal_summary": signal_summary,
        "ai_agreement_count": agreement_count,
        "ai_conflict_count": conflict_count,
        "ai_banker_score": banker_score,
        "ai_player_score": player_score,
        "ai_score_edge": score_edge,
        "ai_direction_override": direction_override,
        "ai_signals": signals,
    }


def predict(player_point: int, banker_point: int, rounds: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    player_point = validate_point(player_point)
    banker_point = validate_point(banker_point)
    rounds = rounds or []

    last_result = get_last_result(player_point, banker_point)
    is_tie_point = player_point == banker_point
    point = point_db_lookup(player_point, banker_point)
    comp = composition_mc_layer(player_point, banker_point, rounds=rounds)
    combo = combo_condition_lookup(player_point, banker_point, rounds=rounds, comp=comp)
    ai = ai_simulation_layer(player_point, banker_point, rounds=rounds)

    p_w = float(POINT_WEIGHT) if USE_POINT_DB else 0.0
    combo_w = float(COMBO_WEIGHT)
    sim_w = 0.0 if USE_AI_DECISION_LAYER else float(SIM_WEIGHT)
    comp_w = float(COMPOSITION_MC_WEIGHT) if comp.get("available") else 0.0

    if COMBO_WEIGHT_REQUIRE_AVAILABLE and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0):
        combo_w = 0.0
    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)
        comp_w = min(comp_w, COMPOSITION_MC_WEIGHT * 0.50)

    total_weight = max(p_w + combo_w + sim_w + comp_w, 0.0001)
    banker = (point["banker_prob"] * p_w + combo["banker_prob"] * combo_w + ai["banker_prob"] * sim_w + comp["banker_prob"] * comp_w) / total_weight
    banker = apply_tie_point_protection(banker, is_tie_point)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    gap = abs(banker - player)
    main_recommend, _ = direction_from_probs(banker, player, neutral_gap=MODEL_NEUTRAL_GAP)
    recommend = main_recommend
    entry_allowed, entry_level, weak_reason = build_entry_decision(is_tie_point=is_tie_point, gap=gap, recommend=recommend)

    if REQUIRE_COMBO_SAMPLE_FOR_ENTRY and entry_allowed and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0) and gap < MIN_GAP_WITHOUT_COMBO:
        entry_allowed = False
        entry_level = "no_entry"
        weak_reason = f"條件資料庫樣本不足，且莊閒差距未達 {MIN_GAP_WITHOUT_COMBO * 100:.1f}%，建議觀察一局"

    result = {
        "ok": True,
        "player_point": player_point,
        "banker_point": banker_point,
        "last_result": last_result,
        "recommend": recommend,
        "main_recommend": main_recommend,
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
        "model_neutral_gap": MODEL_NEUTRAL_GAP,
        "feature_key": point_key(player_point, banker_point),
        "point_feature_key": point.get("feature_key"),
        "combo_feature_key": combo.get("feature_key"),
        "point_source": point.get("source"),
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
        "point_available": point.get("available", False),
        "point_sample_size": point.get("sample_size", 0),
        "point_total_samples": point.get("total_simulated_samples", 0),
        "ai_history_points_used": ai.get("history_points_used", 0),
        "ai_history_adjust": ai.get("history_adjust", 0.0),
        "ai_history_reasons": ai.get("history_reasons", []),
        "ai_decision_layer_enabled": USE_AI_DECISION_LAYER,
        "pattern_available": combo.get("available", False),
        "pattern_sample_size": combo.get("sample_size", 0),
        "pattern_total_samples": combo.get("total_simulated_samples", 0),
        "pattern_source": combo.get("source"),
        "pattern_feature_key": combo.get("feature_key"),
        "pattern_layer_mode": "point_condition_combo",
        "matched_patterns": [combo.get("feature_key")] if combo.get("available") else [],
        "weights": {"point": p_w, "combo": combo_w, "pattern": combo_w, "simulation": sim_w, "ai_decision": AI_DECISION_WEIGHT if USE_AI_DECISION_LAYER else 0.0, "composition_mc": comp_w, "total": total_weight},
        "raw_layers": {
            "point_banker_prob": point.get("banker_prob"),
            "combo_banker_prob": combo.get("banker_prob"),
            "pattern_banker_prob": combo.get("banker_prob"),
            "ai_banker_prob": ai.get("banker_prob"),
            "composition_mc_banker_prob": comp.get("banker_prob"),
            "point_player_prob": point.get("player_prob"),
            "combo_player_prob": combo.get("player_prob"),
            "pattern_player_prob": combo.get("player_prob"),
            "ai_player_prob": ai.get("player_prob"),
            "composition_mc_player_prob": comp.get("player_prob"),
        },
        "history_used": bool(rounds),
        "rounds_ignored": False,
        "mode": "POINT_CONDITION_COMBO_COMPOSITION_MC_AI_DECISION_V9_2",
    }

    if USE_MONTE_CARLO:
        mc_result = monte_carlo_verify_from_probs(banker_prob=banker, player_prob=player, seed_key=f"{player_point}:{banker_point}:{combo.get('feature_key','')}:{comp.get('top_scenario','')}")
        result["monte_carlo"] = mc_result
        mc_gap_raw = float(mc_result.get("mc_gap_raw", 0.0) or 0.0)
        mc_recommend = mc_result.get("mc_recommend", "觀望")
        if MC_BLOCK_LOW_GAP and result["entry_allowed"] and mc_gap_raw < MC_MIN_GAP_FOR_ENTRY:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 穩定度不足，莊閒差距偏小，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
        else:
            result["mc_entry_blocked"] = False
        if mc_recommend == "觀望" and result["entry_allowed"]:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 落在中性區，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
        if MC_DIRECTION_MISMATCH_BLOCK and result["entry_allowed"] and mc_recommend in {"莊", "閒"} and recommend in {"莊", "閒"} and mc_recommend != recommend:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 模擬方向與主模型不一致，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
    else:
        result["monte_carlo"] = disabled_monte_carlo_result()
        result["mc_entry_blocked"] = False

    if USE_AI_DECISION_LAYER:
        ai_decision = ai_ensemble_decision_layer(point=point, combo=combo, comp=comp, mc=result.get("monte_carlo", {}), current_recommend=recommend, banker_prob=banker, player_prob=player, gap=gap)
        result.update(ai_decision)
        ai_rec = ai_decision.get("ai_decision_recommend", recommend)
        if ai_rec == "觀望" or ai_decision.get("ai_decision_observe", False):
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["recommend"] = "觀望"
            result["weak_reason"] = ai_decision.get("ai_decision_reason", "AI 綜合判斷建議觀察一局")
            result["no_observe"] = True
            result["ai_entry_blocked"] = True
        elif ai_rec in {"莊", "閒"}:
            result["recommend"] = ai_rec
            result["ai_entry_blocked"] = False
    else:
        result.update({"ai_decision_enabled": False, "ai_decision_recommend": recommend, "ai_decision_reason": "AI_DECISION_LAYER_DISABLED", "ai_entry_blocked": False})

    return result
