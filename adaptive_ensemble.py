"""保守型自適應集成。

只在累積足夠已結算樣本後，依各子模型的歷史 Brier score 做小幅再加權。
主引擎仍保留大多數權重，避免短期輸贏造成權重震盪。
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
        result["adaptive_ensemble"] = {
            "active": False,
            "sample_count": sample_count,
            "minimum_samples": ADAPTIVE_MIN_SAMPLES,
            "reason": "已結算樣本不足或可比較子模型不足",
            "effective_share": 0.0,
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
    result["adaptive_ensemble"] = {
        "active": True,
        "sample_count": sample_count,
        "minimum_samples": ADAPTIVE_MIN_SAMPLES,
        "effective_share": round(share, 6),
        "weights": {name: round(value, 6) for name, value in weights.items()},
        "component_brier_scores": {
            name: round(float(historical_scores[name]), 6) for name in eligible
        },
    }
    return result


__all__ = ["adapt_prediction"]
