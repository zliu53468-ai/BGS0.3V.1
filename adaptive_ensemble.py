"""保守型自適應集成。

正常區間只在累積足夠已結算樣本後，依各子模型的歷史 Brier score
做小幅再加權。若 cMAB 回報極端未知區間，安全熔斷會立即優先執行，
不受成熟樣本門檻限制，且至少調降 90% cMAB 決策權重。
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
# OOD 動態熔斷後只保留 0～10% cMAB 權重；預設保留 5%（調降 95%）。
EXTREME_UNSEEN_BANDIT_WEIGHT = max(
    0.0,
    min(
        0.10,
        float(os.getenv("EXTREME_UNSEEN_BANDIT_WEIGHT", "0.05") or "0.05"),
    ),
)


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
    braking = prediction.get("uncertainty_braking")
    predictor_signal = prediction.get("predictor_signal")
    return bool(
        prediction.get("is_extreme_unseen")
        or prediction.get("extreme_uncertainty_signal")
        or prediction.get("unknown_region_active")
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


def _base_confidence(prediction: Mapping[str, Any], base: Mapping[str, float]) -> float:
    """統一為 [0, 1]；優先沿用 cMAB 已有 quality_score。"""
    try:
        quality = float(prediction.get("quality_score", 0.0) or 0.0)
    except Exception:
        quality = 0.0
    if quality > 0.0:
        return max(0.0, min(1.0, quality))
    return max(0.0, min(1.0, abs(float(base["B"]) - float(base["P"]))))


def _extreme_alternative_fusion(
    components: Mapping[str, Mapping[str, float]],
    historical_scores: Mapping[str, Any],
) -> tuple[Dict[str, float], Dict[str, float]]:
    """極端區間若仍有獨立子模型，立即融合；沒有則交由 predictor 接管。"""
    if not components:
        return {}, {}

    raw_weights: Dict[str, float] = {}
    for name in components:
        try:
            score = float(historical_scores.get(name))
            raw_weights[name] = math.exp(-score / ADAPTIVE_TEMPERATURE)
        except Exception:
            raw_weights[name] = 1.0
    weight_total = sum(raw_weights.values()) or 1.0
    weights = {name: value / weight_total for name, value in raw_weights.items()}
    probabilities = _normalize({
        outcome: sum(
            weights[name] * float(components[name][outcome])
            for name in components
        )
        for outcome in OUTCOMES
    })
    return probabilities, weights


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
        # 動態熔斷必須早於傳統的最小樣本門檻；風險訊號不能因
        # ADAPTIVE_MIN_SAMPLES 尚未成熟而被忽略。
        bandit_weight = EXTREME_UNSEEN_BANDIT_WEIGHT
        alternative, alternative_weights = _extreme_alternative_fusion(
            components,
            historical_scores,
        )
        has_alternative = bool(alternative)

        if has_alternative:
            blended = _normalize({
                outcome: (
                    base[outcome] * bandit_weight
                    + alternative[outcome] * (1.0 - bandit_weight)
                )
                for outcome in OUTCOMES
            })
            result["probabilities"] = blended
            result["banker_rate"] = round(blended["B"] * 100.0, 2)
            result["player_rate"] = round(blended["P"] * 100.0, 2)
            result["tie_rate"] = round(blended["T"] * 100.0, 2)
            # 沒有可靠的跨模型校準值時，只以方向間距作保守信心。
            overall_confidence = min(
                base_confidence,
                abs(blended["B"] - blended["P"]),
            )
        else:
            # 目前正式 predictor 只啟用 cMAB；此處不把剩餘 5% 再
            # 正規化回 100%，而是把低信心與 fallback_required 傳下去。
            overall_confidence = base_confidence * bandit_weight

        weight_reduction = 1.0 - bandit_weight
        result["ensemble_confidence"] = float(overall_confidence)
        result["is_extreme_unseen"] = True
        result["adaptive_ensemble"] = {
            "active": True,
            "mode": "extreme_unseen_dynamic_circuit_breaker",
            "circuit_breaker_active": True,
            "sample_count": sample_count,
            "minimum_samples_bypassed_for_safety": True,
            "is_extreme_unseen": True,
            "variance": float(result.get("variance", 0.0) or 0.0),
            "bandit_weight_before": 1.0,
            "bandit_weight_after": float(bandit_weight),
            "weight_reduction_ratio": float(weight_reduction),
            "weight_reduction_percent": round(weight_reduction * 100.0, 2),
            "alternative_model_weight": float(1.0 - bandit_weight),
            "alternative_components_available": sorted(components),
            "alternative_weights": {
                name: round(value, 6)
                for name, value in alternative_weights.items()
            },
            "fallback_required": not has_alternative,
            "fusion_deferred_to_short_term": not has_alternative,
            "overall_confidence": float(overall_confidence),
            "reason": (
                "偵測到極端未知特徵，cMAB 權重立即熔斷並交由獨立子模型融合"
                if has_alternative
                else "偵測到極端未知特徵，cMAB 權重立即熔斷並要求短週期接管"
            ),
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
        result["adaptive_ensemble"] = {
            "active": False,
            "circuit_breaker_active": False,
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
    result["adaptive_ensemble"] = {
        "active": True,
        "circuit_breaker_active": False,
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
