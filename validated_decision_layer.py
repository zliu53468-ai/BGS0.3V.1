"""BGS 走勢外驗證決策層。

本模組不訓練、修改或取代任何既有模型。它只使用已結算的時間順序
績效，決定是否採用某個既有模型輸出，並校準前端顯示的信心與機率。
真正的 hard brake／No Bet 永遠優先，不會被本層解除。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import os

from performance_tracker import get_performance_summary


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


__all__ = ["apply_validated_decision"]
