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

# ============================================================
# V9：點數 + 補牌情境 + 300 萬條件資料庫 + Monte Carlo
# ============================================================
# 主流程：
# 1. LINE 只輸入點數，例如 65。
# 2. point_composition_mc 反推補牌情境。
# 3. combo_db 用「P6_B5 + 補牌情境」查 300 萬條件資料庫。
# 4. predictor 融合 point_db / combo_db / AI / composition_mc。
# 5. Monte Carlo 只做最終機率穩定度驗證，不遞迴呼叫 predict。
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
MC_DIRECTION_MISMATCH_BLOCK = env_bool("MC_DIRECTION_MISMATCH_BLOCK", "1")

# ============================================================
# AI 綜合決策層：
# AI 不再只是「歷史修正」，而是綜合 point_db / combo_db /
# 補牌MC / Monte Carlo / 主模型結果，判斷要出莊、閒或觀望。
# ============================================================
USE_AI_DECISION_LAYER = env_bool("USE_AI_DECISION_LAYER", "1")
AI_DECISION_WEIGHT = env_float("AI_DECISION_WEIGHT", os.getenv("SIM_WEIGHT", "0.12"))
AI_REQUIRE_SIGNAL_AGREEMENT = env_bool("AI_REQUIRE_SIGNAL_AGREEMENT", "1")
AI_BLOCK_CONFLICT_SIGNAL = env_bool("AI_BLOCK_CONFLICT_SIGNAL", "1")
AI_MIN_AGREEMENT_COUNT = env_int("AI_MIN_AGREEMENT_COUNT", "3")
AI_MIN_CONFIDENCE_GAP = env_float("AI_MIN_CONFIDENCE_GAP", "0.085")
AI_COMBO_SAMPLE_STRONG = env_int("AI_COMBO_SAMPLE_STRONG", "300")
AI_COMBO_SAMPLE_WEAK = env_int("AI_COMBO_SAMPLE_WEAK", "80")
AI_FORCE_OBSERVE_ON_SPLIT = env_bool("AI_FORCE_OBSERVE_ON_SPLIT", "1")
# 預設不讓 AI 硬改方向，避免主模型機率顯示與建議方向互相打架。
# 若未來你想讓 AI 在 4/5 訊號強一致時可改方向，再設 AI_OVERRIDE_DIRECTION=1。
AI_OVERRIDE_DIRECTION = env_bool("AI_OVERRIDE_DIRECTION", "0")

# 中性死區：避免 50.1% / 49.9% 這種微小差距被自動判成莊。
MODEL_NEUTRAL_GAP = env_float("MODEL_NEUTRAL_GAP", "0.003")
MC_NEUTRAL_GAP = env_float("MC_NEUTRAL_GAP", "0.006")
AI_SIGNAL_MIN_GAP = env_float("AI_SIGNAL_MIN_GAP", "0.012")
AI_MAIN_SIGNAL_MIN_GAP = env_float("AI_MAIN_SIGNAL_MIN_GAP", "0.020")
AI_MIN_SCORE_EDGE = env_float("AI_MIN_SCORE_EDGE", "0.010")

# ============================================================
# AI 綜合決策層
# ============================================================

def prob_direction(banker_prob: float, player_prob: float, neutral_gap: float = 0.0) -> Tuple[str, float]:
    """回傳方向與差距。若落在中性死區，回傳「觀望」。"""
    banker_prob, player_prob = normalize_prob_pair(banker_prob, player_prob)
    gap = abs(banker_prob - player_prob)
    if gap < float(neutral_gap or 0.0):
        return "觀望", gap
    if banker_prob > player_prob:
        return "莊", gap
    return "閒", gap


def _signal_item(
    name: str,
    banker_prob: float,
    player_prob: float,
    available: bool = True,
    sample_size: int = 0,
    base_weight: float = 1.0,
    min_gap: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    建立 AI 訊號。
    重點：低於 min_gap 的小差距不列入投票，避免 50.1% 微偏莊也被當有效莊票。
    """
    if not available:
        return None

    direction, gap = prob_direction(banker_prob, player_prob, neutral_gap=min_gap)

    if direction == "觀望":
        return None

    strength = gap * float(base_weight)

    return {
        "name": name,
        "direction": direction,
        "gap_raw": gap,
        "gap": round(gap * 100, PERCENT_DECIMALS),
        "sample_size": int(sample_size or 0),
        "weight": float(base_weight),
        "strength": float(strength),
    }


def ai_ensemble_decision_layer(
    point: Dict[str, Any],
    combo: Dict[str, Any],
    comp: Dict[str, Any],
    mc: Dict[str, Any],
    current_recommend: str,
    banker_prob: float,
    player_prob: float,
    gap: float,
) -> Dict[str, Any]:
    """
    AI 綜合決策層：
    不追歷史、不看路單延續，而是檢查各模型訊號是否一致。

    會看：
    - point_db 方向
    - combo_db 方向與樣本數
    - 補牌 MC 方向
    - Monte Carlo 方向
    - 主模型方向與差距

    最終回傳：莊 / 閒 / 觀望。
    """
    if not USE_AI_DECISION_LAYER:
        return {
            "ai_decision_enabled": False,
            "ai_decision_recommend": current_recommend,
            "ai_decision_direction": current_recommend,
            "ai_decision_observe": False,
            "ai_decision_reason": "AI_DECISION_LAYER_DISABLED",
            "ai_signal_summary": "DISABLED",
            "ai_signals": [],
        }

    signals: List[Dict[str, Any]] = []

    # point_db：當前點數統計，視為主訊號之一。
    sig = _signal_item(
        "point_db",
        point.get("banker_prob", BASE_BANKER_NO_TIE),
        point.get("player_prob", 1.0 - BASE_BANKER_NO_TIE),
        available=bool(point.get("available", False)),
        sample_size=int(point.get("sample_size", 0) or 0),
        base_weight=1.15,
        min_gap=AI_SIGNAL_MIN_GAP,
    )
    if sig:
        signals.append(sig)

    # combo_db：樣本足夠才列入訊號；樣本偏低則降權。
    combo_sample = int(combo.get("sample_size", 0) or 0)
    combo_available = bool(combo.get("available", False)) and combo_sample >= AI_COMBO_SAMPLE_WEAK
    if combo_available:
        combo_weight = 1.25 if combo_sample >= AI_COMBO_SAMPLE_STRONG else 0.75
        sig = _signal_item(
            "combo_db",
            combo.get("banker_prob", BASE_BANKER_NO_TIE),
            combo.get("player_prob", 1.0 - BASE_BANKER_NO_TIE),
            available=True,
            sample_size=combo_sample,
            base_weight=combo_weight,
            min_gap=AI_SIGNAL_MIN_GAP,
        )
        if sig:
            signals.append(sig)

    # 補牌 MC：用來判斷當前點數形成情境，不看歷史。
    sig = _signal_item(
        "composition_mc",
        comp.get("banker_prob", BASE_BANKER_NO_TIE),
        comp.get("player_prob", 1.0 - BASE_BANKER_NO_TIE),
        available=bool(comp.get("available", False)),
        sample_size=int(comp.get("sample_size", 0) or 0),
        base_weight=1.10,
        min_gap=AI_SIGNAL_MIN_GAP,
    )
    if sig:
        signals.append(sig)

    # 主模型融合結果：需要較高差距才列入，避免主模型微偏莊就變莊票。
    sig = _signal_item(
        "main_model",
        banker_prob,
        player_prob,
        available=current_recommend in {"莊", "閒"},
        sample_size=0,
        base_weight=0.95,
        min_gap=AI_MAIN_SIGNAL_MIN_GAP,
    )
    if sig:
        signals.append(sig)

    # Monte Carlo 方向：若 MC 自己為觀望，不列入投票。
    if mc and mc.get("mc_enabled") and mc.get("mc_recommend") in {"莊", "閒"}:
        sig = _signal_item(
            "monte_carlo",
            mc.get("mc_banker_rate_raw", BASE_BANKER_NO_TIE),
            mc.get("mc_player_rate_raw", 1.0 - BASE_BANKER_NO_TIE),
            available=True,
            sample_size=int(mc.get("mc_simulations", 0) or 0),
            base_weight=1.20,
            min_gap=max(AI_SIGNAL_MIN_GAP, MC_NEUTRAL_GAP),
        )
        if sig:
            signals.append(sig)

    banker_votes = [s for s in signals if s.get("direction") == "莊"]
    player_votes = [s for s in signals if s.get("direction") == "閒"]

    banker_score = sum(float(s.get("strength", 0.0)) for s in banker_votes)
    player_score = sum(float(s.get("strength", 0.0)) for s in player_votes)
    score_edge = abs(banker_score - player_score)

    banker_count = len(banker_votes)
    player_count = len(player_votes)

    if banker_score > player_score:
        ai_direction = "莊"
        agreement_count = banker_count
        conflict_count = player_count
    elif player_score > banker_score:
        ai_direction = "閒"
        agreement_count = player_count
        conflict_count = banker_count
    else:
        if banker_count > player_count:
            ai_direction = "莊"
            agreement_count = banker_count
            conflict_count = player_count
        elif player_count > banker_count:
            ai_direction = "閒"
            agreement_count = player_count
            conflict_count = banker_count
        else:
            ai_direction = "觀望"
            agreement_count = max(banker_count, player_count)
            conflict_count = agreement_count

    reasons: List[str] = []

    if not signals:
        reasons.append("AI 無足夠有效訊號")
        ai_direction = "觀望"
        agreement_count = 0
        conflict_count = 0

    if ai_direction == "觀望" and AI_FORCE_OBSERVE_ON_SPLIT:
        reasons.append("AI 訊號分歧或落在中性區")

    if AI_REQUIRE_SIGNAL_AGREEMENT and ai_direction in {"莊", "閒"} and agreement_count < AI_MIN_AGREEMENT_COUNT:
        reasons.append(f"AI 同方向訊號不足，僅 {agreement_count} 個，未達 {AI_MIN_AGREEMENT_COUNT} 個")

    if AI_BLOCK_CONFLICT_SIGNAL and ai_direction in {"莊", "閒"}:
        # 反向訊號太多，代表各層判斷衝突。
        if conflict_count >= 2 and agreement_count <= conflict_count + 1:
            reasons.append("AI 偵測 point/combo/補牌MC/MC 訊號衝突")

    if gap < AI_MIN_CONFIDENCE_GAP:
        reasons.append(f"AI 判斷主模型差距 {gap * 100:.2f}% 未達 {AI_MIN_CONFIDENCE_GAP * 100:.1f}%")

    if score_edge < AI_MIN_SCORE_EDGE:
        reasons.append(f"AI 莊閒分數差距不足，score_edge={score_edge:.4f}")

    # 若 AI 與主模型相反，預設觀望；除非 AI_OVERRIDE_DIRECTION=1。
    direction_override = False
    if ai_direction in {"莊", "閒"} and current_recommend in {"莊", "閒"} and ai_direction != current_recommend:
        if AI_OVERRIDE_DIRECTION:
            direction_override = True
            reasons.append(f"AI 綜合訊號改判：{current_recommend} → {ai_direction}")
        else:
            reasons.append(f"AI 方向與主模型不同：主模型{current_recommend} / AI{ai_direction}")

    observe = bool(reasons)
    decision_recommend = "觀望" if observe else ai_direction

    signal_summary = " / ".join(
        f"{s.get('name')}:{s.get('direction')}({s.get('gap')}%)"
        for s in signals
    ) or "NO_EFFECTIVE_SIGNAL"

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


# ============================================================
# 主預測函式
# ============================================================

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

    # V9.2：SIM_WEIGHT 改為 AI 綜合決策層權重。
    # 若 USE_AI_DECISION_LAYER=1，就不再把舊 AI 歷史層混進機率，避免追莊追閒。
    sim_w = 0.0 if USE_AI_DECISION_LAYER else float(SIM_WEIGHT)

    comp_w = float(COMPOSITION_MC_WEIGHT) if comp.get("available") else 0.0

    if COMBO_WEIGHT_REQUIRE_AVAILABLE and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0):
        combo_w = 0.0

    if is_tie_point:
        sim_w = min(sim_w, TIE_AI_MAX_WEIGHT)
        comp_w = min(comp_w, COMPOSITION_MC_WEIGHT * 0.50)

    total_weight = max(p_w + combo_w + sim_w + comp_w, 0.0001)
    banker = (
        point["banker_prob"] * p_w
        + combo["banker_prob"] * combo_w
        + ai["banker_prob"] * sim_w
        + comp["banker_prob"] * comp_w
    ) / total_weight

    banker = apply_tie_point_protection(banker, is_tie_point)
    banker = clamp(banker, MIN_OUTPUT_PROB, MAX_OUTPUT_PROB)
    player = 1.0 - banker
    gap = abs(banker - player)

    # 關鍵修正：不再用 banker >= player，避免 50/50 或極小差距自動判莊。
    recommend, _ = prob_direction(banker, player, neutral_gap=MODEL_NEUTRAL_GAP)

    entry_allowed, entry_level, weak_reason = build_entry_decision(is_tie_point=is_tie_point, gap=gap)

    if recommend == "觀望":
        entry_allowed = False
        entry_level = "no_entry"
        weak_reason = f"主模型落在中性區，差距未達 {MODEL_NEUTRAL_GAP * 100:.2f}%，建議觀察一局"

    if (
        REQUIRE_COMBO_SAMPLE_FOR_ENTRY
        and entry_allowed
        and (not combo.get("available") or int(combo.get("sample_size", 0) or 0) <= 0)
        and gap < MIN_GAP_WITHOUT_COMBO
    ):
        entry_allowed = False
        entry_level = "no_entry"
        weak_reason = f"條件資料庫樣本不足，且莊閒差距未達 {MIN_GAP_WITHOUT_COMBO * 100:.1f}%，建議觀察一局"

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
        "model_neutral_gap": MODEL_NEUTRAL_GAP,
        "mc_neutral_gap": MC_NEUTRAL_GAP,
        "ai_signal_min_gap": AI_SIGNAL_MIN_GAP,
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
            "simulation": sim_w,
            "ai_decision": AI_DECISION_WEIGHT if USE_AI_DECISION_LAYER else 0.0,
            "composition_mc": comp_w,
            "total": total_weight,
        },
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
        "ai_decision_layer_enabled": USE_AI_DECISION_LAYER,
        "mode": "POINT_CONDITION_COMBO_COMPOSITION_MC_AI_DECISION_NEUTRAL_GAP_V9_2",
    }

    # 先跑 Monte Carlo，再交給 AI 綜合決策層做最後裁判。
    if USE_MONTE_CARLO:
        mc_result = monte_carlo_verify_from_probs(
            banker_prob=banker,
            player_prob=player,
            seed_key=f"{player_point}:{banker_point}:{combo.get('feature_key','')}:{comp.get('top_scenario','')}:{ai.get('history_adjust',0.0)}",
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

        if MC_DIRECTION_MISMATCH_BLOCK and result["entry_allowed"] and mc_recommend in {"莊", "閒"} and recommend in {"莊", "閒"} and mc_recommend != recommend:
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = "Monte Carlo 模擬方向與主模型不一致，建議觀察一局"
            result["no_observe"] = True
            result["mc_entry_blocked"] = True
    else:
        result["monte_carlo"] = disabled_monte_carlo_result()
        result["mc_entry_blocked"] = False

    # AI 綜合決策層最後裁判：可判莊 / 閒 / 觀望。
    ai_decision = ai_ensemble_decision_layer(
        point=point,
        combo=combo,
        comp=comp,
        mc=result.get("monte_carlo", {}),
        current_recommend=recommend,
        banker_prob=banker,
        player_prob=player,
        gap=gap,
    )
    result["ai_decision"] = ai_decision
    result["ai_decision_recommend"] = ai_decision.get("ai_decision_recommend")
    result["ai_decision_reason"] = ai_decision.get("ai_decision_reason")
    result["ai_signal_summary"] = ai_decision.get("ai_signal_summary")
    result["ai_agreement_count"] = ai_decision.get("ai_agreement_count")
    result["ai_conflict_count"] = ai_decision.get("ai_conflict_count")

    if USE_AI_DECISION_LAYER:
        ai_rec = ai_decision.get("ai_decision_recommend")
        ai_dir = ai_decision.get("ai_decision_direction")

        if ai_rec == "觀望" or ai_decision.get("ai_decision_observe"):
            result["entry_allowed"] = False
            result["entry_level"] = "no_entry"
            result["weak_reason"] = ai_decision.get("ai_decision_reason", "AI 綜合判斷建議觀望")
            result["no_observe"] = True
            result["recommend"] = "觀望"
        elif ai_rec in {"莊", "閒"}:
            # AI 與主模型一致時，照 AI 決策結果；若有開 AI_OVERRIDE_DIRECTION，也可改方向。
            if recommend == "觀望" or ai_rec == recommend or ai_decision.get("ai_direction_override"):
                result["recommend"] = ai_rec

    return result
