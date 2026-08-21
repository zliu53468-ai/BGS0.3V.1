"""保守型自適應集成。

正常區間只在累積足夠已結算樣本後，依各子模型的歷史 Brier score
做小幅再加權。任一模型回報極端未知區間時，立即執行零信心、
零注碼、No Bet 硬熔斷，不嘗試用其他模型修正該局。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import math
import os

from performance_tracker import get_performance_summary


OUTCOMES = ("B", "P", "T")
ADAPTIVE_ENABLED = os.getenv("ADAPTIVE_ENSEMBLE_ENABLED", "1").strip() == "1"
ADAPTIVE_MIN_SAMPLES = max(50, int(os.getenv("ADAPTIVE_MIN_SAMPLES", "300") or "300"))
ADAPTIVE_MAX_SHARE = max(0.0, min(0.35, float(os.getenv("ADAPTIVE_MAX_SHARE", "0.15") or "0.15")))
ADAPTIVE_TEMPERATURE = max(0.01, min(2.0, float(os.getenv("ADAPTIVE_TEMPERATURE", "0.18") or "0.18")))


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    data = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in OUTCOMES}
    total = sum(data.values())
    return {key: data[key] / total for key in OUTCOMES}


def _components(prediction: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    mapping = {
        "hypergeometric": prediction.get("hypergeometric_probabilities"),
        "monte_carlo": prediction.get("monte_carlo_probabilities"),
        "particle": prediction.get("particle_probabilities"),
        "sequence": prediction.get("sequence_probabilities"),
        "core_before_road": prediction.get("core_probabilities_before_road"),
    }
    result: Dict[str, Dict[str, float]] = {}
    for name, values in mapping.items():
        if isinstance(values, Mapping):
            try:
                result[name] = _normalize(values)
            except Exception:
                pass
    return result


def _is_extreme_unseen(prediction: Mapping[str, Any]) -> bool:
    """接受新舊欄位，避免部署期間因版本差異再次造成訊號斷層。"""
    def flagged(model: Mapping[str, Any]) -> bool:
        braking = model.get("uncertainty_braking")
        predictor_signal = model.get("predictor_signal")
        return bool(
            model.get("is_extreme_unseen")
            or model.get("extreme_uncertainty_signal")
            or model.get("unknown_region_active")
            or (
                isinstance(braking, Mapping)
                and (
                    braking.get("is_extreme_unseen")
                    or braking.get("active")
                )
            )
            or (
                isinstance(predictor_signal, Mapping)
                and (
                    predictor_signal.get("is_extreme_unseen")
                    or predictor_signal.get("extreme_uncertainty")
                )
            )
        )

    if flagged(prediction):
        return True

    # 預留其他模型以巢狀結果接入的契約：任一成員亮起極端標籤，
    # 都優先於正常加權與樣本成熟度，立即觸發全局硬熔斷。
    for collection_name in (
        "model_predictions",
        "component_predictions",
        "ensemble_members",
        "models",
    ):
        collection = prediction.get(collection_name)
        if isinstance(collection, Mapping):
            members = collection.values()
        elif isinstance(collection, (list, tuple)):
            members = collection
        else:
            continue
        if any(
            isinstance(member, Mapping) and flagged(member)
            for member in members
        ):
            return True
    return False


def _base_confidence(prediction: Mapping[str, Any], base: Mapping[str, float]) -> float:
    """統一為 [0, 1]；優先沿用 cMAB 已有 quality_score。"""
    try:
        quality = float(prediction.get("quality_score", 0.0) or 0.0)
    except Exception:
        quality = 0.0
    if quality > 0.0:
        return max(0.0, min(1.0, quality))
    return max(0.0, min(1.0, abs(float(base["B"]) - float(base["P"]))))


def adapt_prediction(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    result = dict(prediction or {})
    base = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {
            "B": float(result.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(result.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(result.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )
    result.setdefault("raw_probabilities", dict(base))
    components = _components(result)
    summary = get_performance_summary(venue=venue, room=room, limit=5000)
    sample_count = int(summary.get("sample_count", 0) or 0)
    historical_scores = dict(summary.get("component_brier_scores") or {})
    extreme_unseen = _is_extreme_unseen(result)
    base_confidence = _base_confidence(result, base)

    if extreme_unseen:
        # 硬熔斷優先於成熟樣本與正常融合；保留原始分數僅供稽核。
        learning_arm = str(result.get("selected_arm") or "").upper().strip()
        result["pre_hard_brake_probabilities"] = dict(base)
        result["pre_hard_brake_recommend"] = str(
            result.get("recommend") or learning_arm
        )
        tie_probability = max(0.0, min(0.30, float(base.get("T", 0.0))))
        neutral_bp = (1.0 - tie_probability) * 0.5
        result["probabilities"] = {
            "B": neutral_bp,
            "P": neutral_bp,
            "T": tie_probability,
        }
        result["banker_rate"] = round(neutral_bp * 100.0, 2)
        result["player_rate"] = round(neutral_bp * 100.0, 2)
        result["tie_rate"] = round(tie_probability * 100.0, 2)
        result["recommend"] = "O"
        result["recommend_text"] = "觀望"
        result["action"] = "O"
        result["action_text"] = "觀望／絕對不下注"
        result["internal_recommend"] = "O"
        result["internal_action"] = "O"
        result["next_round_direction"] = "O"
        result["next_round_direction_text"] = "觀望"
        result["signal_allowed"] = False
        result["signal_status_code"] = "HARD_BRAKE_NO_BET"
        result["signal_status_text"] = "統計混沌硬熔斷：絕對不下注"
        result["signal_reason"] = (
            "任一模型回報極端未知區間；集成層不做方向修正，"
            "本局信心與注碼強制歸零。"
        )
        result["internal_signal_reason"] = result["signal_reason"]
        result["direction_source"] = "adaptive_ensemble_hard_brake"
        result["ensemble_confidence"] = 0.0
        result["confidence"] = 0.0
        result["quality_score"] = 0.0
        result["confidence_label"] = "零信心／硬熔斷"
        result["bet_multiplier"] = 0.0
        result["hard_brake_active"] = True
        # selected_arm 不覆寫：即使不下注，實際結果仍可用固定 1 倍
        # 被動更新模型與共享 context 方差。
        if learning_arm in {"B", "P"}:
            result["selected_arm"] = learning_arm
        result["is_extreme_unseen"] = True
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "statistical_chaos_hard_brake",
            "circuit_breaker_active": True,
            "hard_brake_active": True,
            "sample_count": sample_count,
            "minimum_samples_bypassed_for_safety": True,
            "is_extreme_unseen": True,
            "variance": float(result.get("variance", 0.0) or 0.0),
            "bandit_weight_before": 1.0,
            "bandit_weight_after": 0.0,
            "weight_reduction_ratio": 1.0,
            "weight_reduction_percent": 100.0,
            "alternative_model_weight": 0.0,
            "alternative_components_available": sorted(components),
            "alternative_fusion_attempted": False,
            "fallback_required": True,
            "shadow_backtest_required": True,
            "overall_confidence": 0.0,
            "final_action": "O",
            "bet_multiplier": 0.0,
            "reason": "極端未知區間執行零信心、零注碼硬熔斷",
        }
        return result

    eligible = {
        name: probabilities
        for name, probabilities in components.items()
        if name in historical_scores
        and int(dict(summary.get("component_sample_counts") or {}).get(name, 0) or 0) >= ADAPTIVE_MIN_SAMPLES
    }
    active = bool(
        ADAPTIVE_ENABLED
        and sample_count >= ADAPTIVE_MIN_SAMPLES
        and len(eligible) >= 2
        and ADAPTIVE_MAX_SHARE > 0
    )

    if not active:
        result["ensemble_confidence"] = float(base_confidence)
        result["confidence"] = float(base_confidence)
        result["hard_brake_active"] = False
        result["adaptive_ensemble"] = {
            "active": False,
            "circuit_breaker_active": False,
            "hard_brake_active": False,
            "is_extreme_unseen": False,
            "sample_count": sample_count,
            "minimum_samples": ADAPTIVE_MIN_SAMPLES,
            "reason": "已結算樣本不足或可比較子模型不足",
            "effective_share": 0.0,
            "bandit_weight_before": 1.0,
            "bandit_weight_after": 1.0,
            "weight_reduction_ratio": 0.0,
            "fallback_required": False,
            "overall_confidence": float(base_confidence),
        }
        return result

    raw_weights = {
        name: math.exp(-float(historical_scores[name]) / ADAPTIVE_TEMPERATURE)
        for name in eligible
    }
    weight_total = sum(raw_weights.values()) or 1.0
    weights = {name: value / weight_total for name, value in raw_weights.items()}
    adaptive = {
        outcome: sum(weights[name] * eligible[name][outcome] for name in eligible)
        for outcome in OUTCOMES
    }
    maturity = min(1.0, (sample_count - ADAPTIVE_MIN_SAMPLES + 1) / max(1.0, ADAPTIVE_MIN_SAMPLES * 3.0))
    share = ADAPTIVE_MAX_SHARE * maturity
    blended = _normalize({
        outcome: base[outcome] * (1.0 - share) + adaptive[outcome] * share
        for outcome in OUTCOMES
    })
    result["probabilities"] = blended
    result["banker_rate"] = round(blended["B"] * 100.0, 2)
    result["player_rate"] = round(blended["P"] * 100.0, 2)
    result["tie_rate"] = round(blended["T"] * 100.0, 2)
    result["ensemble_confidence"] = float(base_confidence)
    result["confidence"] = float(base_confidence)
    result["hard_brake_active"] = False
    result["adaptive_ensemble"] = {
        "active": True,
        "circuit_breaker_active": False,
        "hard_brake_active": False,
        "is_extreme_unseen": False,
        "sample_count": sample_count,
        "minimum_samples": ADAPTIVE_MIN_SAMPLES,
        "effective_share": round(share, 6),
        "bandit_weight_before": 1.0,
        "bandit_weight_after": round(1.0 - share, 6),
        "weight_reduction_ratio": round(share, 6),
        "fallback_required": False,
        "overall_confidence": float(base_confidence),
        "weights": {name: round(value, 6) for name, value in weights.items()},
        "component_brier_scores": {
            name: round(float(historical_scores[name]), 6) for name in eligible
        },
    }
    return result


__all__ = ["adapt_prediction"]
