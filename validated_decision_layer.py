"""BGS 走勢外驗證決策層。

本模組不訓練、修改或取代任何既有模型。它只使用已結算的時間順序
績效，決定是否採用某個既有模型輸出，並校準前端顯示的信心與機率。
真正的 hard brake／No Bet 永遠優先，不會被本層解除。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import os

from performance_tracker import get_performance_summary
from shoe_composition import KELLY_FRACTION, MAX_BET_FRACTION, MIN_POSITIVE_EV


OUTCOMES = ("B", "P", "T")
VALIDATED_COMPONENT_MIN_SAMPLES = max(
    20,
    min(
        500,
        int(os.getenv("VALIDATED_COMPONENT_MIN_SAMPLES", "40") or "40"),
    ),
)
VALIDATED_COMPONENT_MIN_POSTERIOR = max(
    0.50,
    min(
        0.65,
        float(os.getenv("VALIDATED_COMPONENT_MIN_POSTERIOR", "0.515") or "0.515"),
    ),
)
VALIDATED_COMPONENT_MIN_WILSON = max(
    0.40,
    min(
        0.60,
        float(os.getenv("VALIDATED_COMPONENT_MIN_WILSON", "0.46") or "0.46"),
    ),
)
UNVALIDATED_CONFIDENCE_CAP = max(
    0.50,
    min(
        0.58,
        float(os.getenv("UNVALIDATED_CONFIDENCE_CAP", "0.54") or "0.54"),
    ),
)
VALIDATED_CONFIDENCE_CAP = max(
    0.52,
    min(
        0.70,
        float(os.getenv("VALIDATED_CONFIDENCE_CAP", "0.62") or "0.62"),
    ),
)


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    probabilities = {
        outcome: max(1e-12, float(values.get(outcome, 0.0) or 0.0))
        for outcome in OUTCOMES
    }
    total = sum(probabilities.values()) or 1.0
    return {
        outcome: float(probabilities[outcome] / total)
        for outcome in OUTCOMES
    }


def _hard_brake_active(result: Mapping[str, Any]) -> bool:
    braking = result.get("uncertainty_braking")
    adaptive = result.get("adaptive_ensemble")
    return bool(
        result.get("is_extreme_unseen")
        or result.get("hard_brake_active")
        or (
            isinstance(braking, Mapping)
            and (braking.get("active") or braking.get("is_extreme_unseen"))
        )
        or (
            isinstance(adaptive, Mapping)
            and (
                adaptive.get("hard_brake_active")
                or adaptive.get("circuit_breaker_active")
            )
        )
    )


def _current_components(result: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    raw = result.get("component_probabilities")
    if not isinstance(raw, Mapping):
        raw = dict(result.get("road_support") or {}).get(
            "component_probabilities", {}
        )
    components: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, Mapping):
        return components
    for name, probabilities in raw.items():
        if not isinstance(probabilities, Mapping):
            continue
        try:
            components[str(name)] = _normalize(probabilities)
        except Exception:
            continue
    return components


def _validated_component(
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    components = _current_components(result)
    performance = dict(
        summary.get("component_direction_performance") or {}
    )
    brier_scores = dict(summary.get("component_brier_scores") or {})
    eligible = []
    for name, probabilities in components.items():
        stats = performance.get(name)
        if not isinstance(stats, Mapping):
            continue
        samples = int(stats.get("sample_count", 0) or 0)
        posterior = float(stats.get("posterior_accuracy", 0.5) or 0.5)
        lower_bound = float(stats.get("wilson_lower_bound_90", 0.0) or 0.0)
        if samples < VALIDATED_COMPONENT_MIN_SAMPLES:
            continue
        if posterior < VALIDATED_COMPONENT_MIN_POSTERIOR:
            continue
        if lower_bound < VALIDATED_COMPONENT_MIN_WILSON:
            continue
        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
        try:
            brier = float(brier_scores.get(name, 2.0) or 2.0)
        except Exception:
            brier = 2.0
        eligible.append({
            "name": name,
            "direction": direction,
            "probabilities": probabilities,
            "sample_count": samples,
            "posterior_accuracy": posterior,
            "wilson_lower_bound_90": lower_bound,
            "brier_score": brier,
        })
    if not eligible:
        return {}
    eligible.sort(
        key=lambda item: (
            float(item["wilson_lower_bound_90"]),
            float(item["posterior_accuracy"]),
            -float(item["brier_score"]),
            int(item["sample_count"]),
            str(item["name"]),
        ),
        reverse=True,
    )
    winner = dict(eligible[0])
    winner["eligible_count"] = len(eligible)
    return winner


def _road_aggregate_direction(result: Mapping[str, Any]) -> Dict[str, Any]:
    road = result.get("road_support")
    road_data = dict(road) if isinstance(road, Mapping) else {}
    try:
        banker = max(
            0.02,
            min(0.98, float(road_data.get("banker_probability", 0.5) or 0.5)),
        )
    except Exception:
        banker = 0.5
    return {
        "direction": "B" if banker >= 0.5 else "P",
        "banker_probability": banker,
        "sample_count": int(road_data.get("sample_count", 0) or 0),
    }


def _calibrated_confidence(
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    validated_component: Mapping[str, Any],
    direction_source: str = "",
) -> Dict[str, Any]:
    if validated_component:
        posterior = float(
            validated_component.get("posterior_accuracy", 0.5) or 0.5
        )
        return {
            "confidence": max(0.50, min(VALIDATED_CONFIDENCE_CAP, posterior)),
            "sample_count": int(validated_component.get("sample_count", 0) or 0),
            "source": "validated_component_posterior",
        }

    source = str(
        direction_source
        or result.get("direction_source")
        or "unknown"
    )
    source_stats = dict(
        dict(summary.get("source_direction_performance") or {}).get(source)
        or {}
    )
    source_samples = int(source_stats.get("sample_count", 0) or 0)
    if source_samples >= VALIDATED_COMPONENT_MIN_SAMPLES:
        posterior = float(source_stats.get("posterior_accuracy", 0.5) or 0.5)
        return {
            "confidence": max(0.50, min(VALIDATED_CONFIDENCE_CAP, posterior)),
            "sample_count": source_samples,
            "source": "validated_direction_source_posterior",
        }

    try:
        raw_confidence = float(
            result.get("confidence", result.get("quality_score", 0.5)) or 0.5
        )
    except Exception:
        raw_confidence = 0.5
    return {
        "confidence": max(0.50, min(UNVALIDATED_CONFIDENCE_CAP, raw_confidence)),
        "sample_count": source_samples,
        "source": "unvalidated_confidence_cap",
    }


def _set_direction_distribution(
    result: Dict[str, Any],
    *,
    direction: str,
    confidence: float,
) -> None:
    raw = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {"B": 0.455, "P": 0.455, "T": 0.09}
    )
    result.setdefault("pre_validation_probabilities", dict(raw))
    tie_probability = max(0.04, min(0.18, float(raw["T"])))
    conditional = max(0.50, min(VALIDATED_CONFIDENCE_CAP, float(confidence)))
    bp_mass = 1.0 - tie_probability
    banker_conditional = conditional if direction == "B" else 1.0 - conditional
    banker = bp_mass * banker_conditional
    player = bp_mass * (1.0 - banker_conditional)
    result["probabilities"] = {
        "B": float(banker),
        "P": float(player),
        "T": float(tie_probability),
    }
    result["banker_rate"] = round(banker * 100.0, 2)
    result["player_rate"] = round(player * 100.0, 2)
    result["tie_rate"] = round(tie_probability * 100.0, 2)
    result["recommend"] = direction
    result["recommend_text"] = "莊" if direction == "B" else "閒"
    result["action"] = direction
    result["action_text"] = result["recommend_text"]
    result["internal_recommend"] = direction
    result["internal_action"] = direction
    result["next_round_direction"] = direction
    result["next_round_direction_text"] = result["recommend_text"]
    result["signal_allowed"] = True
    result["direction_edge"] = float(abs(2.0 * conditional - 1.0))
    result["direction_edge_percent"] = round(
        float(result["direction_edge"]) * 100.0, 4
    )


def apply_validated_decision(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    result = dict(prediction or {})
    if _hard_brake_active(result):
        result["decision_validation"] = {
            "active": False,
            "hard_brake_preserved": True,
            "reason": "真正統計硬熔斷優先，驗證層不得恢復方向",
        }
        return result

    action = str(result.get("action") or "O").upper().strip()
    if action not in {"B", "P"}:
        result["decision_validation"] = {
            "active": False,
            "hard_brake_preserved": False,
            "observe_preserved": True,
            "reason": "保留既有非極端觀望條件，不強制改寫模型動作",
        }
        return result

    summary = get_performance_summary(
        venue=str(venue or ""),
        room=str(room or ""),
        limit=5000,
    )
    champion = _validated_component(result, summary)
    original_source = str(result.get("direction_source") or "")
    aggregate_fallback = bool(
        not champion
        and original_source == "adaptive_ensemble_dynamic_champion"
    )

    if champion:
        direction = str(champion["direction"])
        selection_mode = "validated_component"
        selection_name = str(champion["name"])
    elif aggregate_fallback:
        aggregate = _road_aggregate_direction(result)
        direction = str(aggregate["direction"])
        selection_mode = "unvalidated_road_aggregate"
        selection_name = "road_aggregate"
    else:
        direction = action
        selection_mode = "preserve_existing_direction"
        selection_name = original_source or "existing_direction"

    calibration = _calibrated_confidence(
        result,
        summary,
        validated_component=champion,
        direction_source=(
            "validated_decision_road_aggregate"
            if aggregate_fallback
            else original_source
        ),
    )
    calibrated_confidence = float(calibration["confidence"])
    _set_direction_distribution(
        result,
        direction=direction,
        confidence=calibrated_confidence,
    )
    result["pre_validation_direction_source"] = original_source
    result["direction_source"] = (
        "validated_decision_component"
        if champion
        else "validated_decision_road_aggregate"
        if aggregate_fallback
        else original_source
    )
    result["confidence"] = calibrated_confidence
    result["ensemble_confidence"] = calibrated_confidence
    result["quality_score"] = calibrated_confidence
    result["confidence_label"] = (
        "較高"
        if calibrated_confidence >= 0.60
        else "中等"
        if calibrated_confidence >= 0.55
        else "偏低"
    )
    result["signal_status_code"] = "OUT_OF_SAMPLE_VALIDATED_DIRECTION"
    result["signal_status_text"] = "走勢外驗證：正式方向已校準"
    result["signal_reason"] = (
        f"既有模型保持不變；外層依已結算時間順序績效採用 "
        f"{selection_name}，並將信心校準為 {calibrated_confidence:.1%}。"
    )
    result["internal_signal_reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "hard_brake_preserved": False,
        "selection_mode": selection_mode,
        "selection_name": selection_name,
        "direction_before": action,
        "direction_after": direction,
        "confidence_source": str(calibration["source"]),
        "confidence_sample_count": int(calibration["sample_count"]),
        "calibrated_confidence": calibrated_confidence,
        "unvalidated_confidence_cap": UNVALIDATED_CONFIDENCE_CAP,
        "validated_component": dict(champion),
        "performance_sample_count": int(summary.get("sample_count", 0) or 0),
    }
    return result


def _strategy_road_direction(result: Mapping[str, Any]) -> tuple[str, float]:
    """從既有 road context 取得確認方向與強度；它沒有單獨下注權。"""
    road = result.get("road_support")
    if not isinstance(road, Mapping):
        road = result.get("road_context")
    road = dict(road or {})
    try:
        probability = float(road.get("planning_probability", 0.5) or 0.5)
    except (TypeError, ValueError):
        probability = 0.5
    direction = "B" if probability > 0.5 else "P" if probability < 0.5 else ""
    try:
        confidence = float(road.get("confidence_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        consensus = float(road.get("derived_road_consensus", 0.0) or 0.0)
    except (TypeError, ValueError):
        consensus = 0.0
    return direction, max(0.0, min(1.0, max(confidence, consensus)))


def _set_no_bet(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """統一清空所有可被前端誤當成正式下注的欄位。"""
    result.update({
        "action": "O",
        "recommend": "O",
        "internal_action": "O",
        "internal_recommend": "O",
        "next_round_direction": "O",
        "action_text": "觀望／無物理優勢",
        "recommend_text": "觀望／無物理優勢",
        "next_round_direction_text": "觀望／無物理優勢",
        "signal_allowed": False,
        "risk_gate_open": False,
        "kelly_fraction": 0.0,
        "recommended_bet_percentage": 0.0,
        "suggested_bet_amount": 0.0,
        "bet_percentage": 0.0,
        "selected_expected_return": 0.0,
        "selected_expected_return_percent": 0.0,
        "kelly_percentage_applied": 0.0,
        "signal_reason": reason,
        "reason": reason,
    })
    return result


def apply_strategy_decision(
    prediction: Mapping[str, Any],
    *,
    strategy_selection: Mapping[str, Any],
    bankroll: float = 0.0,
) -> Dict[str, Any]:
    """將「策略 Bandit Arm」限制在精確 EV/Kelly 安全網內。

    重要不變量：
    1. 沒有可信 10 維 exact remaining counts，一律 O。
    2. 牌路只可確認或縮小既有物理正 EV 注碼，不能翻轉方向、降低正 EV
       門檻，或把負 EV 變成下注。
    3. 最終注碼永遠同時受既有 fractional ``KELLY_FRACTION`` 計算結果與
       ``MAX_BET_FRACTION`` 硬上限保護。
    """
    result = dict(prediction or {})
    physical = dict(result.get("physical_signal") or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")
    counts = physical.get("remaining_counts")
    trusted = bool(
        physical.get("trusted_exact_counts")
        and physical.get("available")
        and isinstance(counts, (list, tuple))
        and len(counts) == 10
    )
    physical_action = str(physical.get("action") or "O").upper().strip()
    try:
        physical_ev = float(physical.get("selected_expected_return", 0.0) or 0.0)
    except (TypeError, ValueError):
        physical_ev = 0.0
    try:
        base_kelly = max(0.0, float(physical.get("kelly_fraction", 0.0) or 0.0))
    except (TypeError, ValueError):
        base_kelly = 0.0
    required_ev = max(
        float(MIN_POSITIVE_EV),
        float(MIN_POSITIVE_EV) * float(profile.get("minimum_ev_multiplier", 1.0) or 1.0),
    )

    result["decision_strategy_bandit"] = selection
    result["decision_strategy"] = arm
    result["physical_signal"] = physical
    result["strategy_hard_limits"] = {
        "kelly_fraction": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_FRACTION),
        "minimum_positive_ev": float(MIN_POSITIVE_EV),
    }

    if not trusted:
        return _set_no_bet(
            result,
            "尚未提供可信的 10 維精確剩餘牌點計數；B/P/T 路單不能取代物理 EV。",
        )
    if physical_action not in {"B", "P"} or physical_ev < required_ev or base_kelly <= 0.0:
        return _set_no_bet(
            result,
            "精確不放回機率在抽水後未達本策略要求的正 EV／Kelly 條件。",
        )

    road_direction, road_strength = _strategy_road_direction(result)
    multiplier = min(1.0, max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)))
    road_note = "純數學策略，不讀牌路調整注碼。"
    if arm == "ev_road_blend":
        # 路圖同向僅確認、不同向且強度高才縮倉；不允許加碼超過既有 Kelly
        # 結果，因此不會突破 KELLY_FRACTION 的原始風控意義。
        if road_direction and road_direction != physical_action and road_strength >= 0.60:
            multiplier *= 0.65
            road_note = "牌路與物理正 EV 方向衝突，已將既有 Kelly 注碼縮至 65%。"
        elif road_direction == physical_action and road_strength >= 0.60:
            road_note = "牌路與物理正 EV 同向，只確認既有 Kelly 注碼，不額外加碼。"
        else:
            multiplier *= 0.85
            road_note = "牌路確認度不足，已將既有 Kelly 注碼縮至 85%。"
    elif arm == "conservative":
        road_note = "保守策略提高 EV 要求並將既有 Kelly 注碼減半。"

    final_kelly = min(float(MAX_BET_FRACTION), base_kelly * multiplier)
    if final_kelly <= 0.0:
        return _set_no_bet(result, "策略縮放後的下注比例為零，保留觀望。")

    probabilities = dict(physical.get("probabilities") or {})
    result.update({
        "action": physical_action,
        "recommend": physical_action,
        "internal_action": physical_action,
        "internal_recommend": physical_action,
        "next_round_direction": physical_action,
        "action_text": "莊" if physical_action == "B" else "閒",
        "recommend_text": "莊" if physical_action == "B" else "閒",
        "next_round_direction_text": "莊" if physical_action == "B" else "閒",
        "signal_allowed": True,
        "risk_gate_open": True,
        "direction_source": "trusted_exact_ev_with_strategy_bandit",
        "selected_expected_return": physical_ev,
        "selected_expected_return_percent": physical_ev * 100.0,
        "kelly_fraction": final_kelly,
        "kelly_percentage_applied": final_kelly * 100.0,
        "recommended_bet_percentage": final_kelly * 100.0,
        "bet_percentage": final_kelly * 100.0,
        "suggested_bet_amount": max(0.0, float(bankroll or 0.0)) * final_kelly,
        "bankroll": max(0.0, float(bankroll or 0.0)),
        "strategy_weights": {
            "physical_weight": float(profile.get("physical_weight", 1.0) or 1.0),
            "road_weight": float(profile.get("road_weight", 0.0) or 0.0),
            "road_direction": road_direction,
            "road_strength": road_strength,
            "kelly_multiplier": multiplier,
        },
        "strategy_required_ev": required_ev,
        "kelly_cap_enforced": True,
        "banker_rate": round(float(probabilities.get("B", 0.0) or 0.0) * 100.0, 4),
        "player_rate": round(float(probabilities.get("P", 0.0) or 0.0) * 100.0, 4),
        "tie_rate": round(float(probabilities.get("T", 0.0) or 0.0) * 100.0, 4),
        "signal_reason": (
            f"{profile.get('label', arm)}：精確不放回 EV {physical_ev:.3%}；{road_note}"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "mode": "strategy_bandit_physical_ev_gate",
        "selected_strategy_arm": arm,
        "trusted_exact_counts": True,
        "physical_action": physical_action,
        "physical_ev": physical_ev,
        "required_ev": required_ev,
        "base_kelly": base_kelly,
        "final_kelly": final_kelly,
        "max_bet_fraction": float(MAX_BET_FRACTION),
        "road_note": road_note,
    }
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
