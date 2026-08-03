"""BGS V10.7 動態群組集成與後驗模擬。

用途：
- 有限牌組群組（超幾何／蒙地卡羅／粒子）與牌路群組、序列群組平行運算。
- 依資料品質、模型可靠度、完整歷史長度與模型分歧動態決定群組權重。
- 最後以向量化 Dirichlet 後驗模擬估計方向穩定度與信心區間。

本模組只整合既有模型結果，不取得真人桌隱藏牌序，也不保證下一局結果。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence
import math
import os

import numpy as np


OUTCOMES = ("B", "P", "T")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


DYNAMIC_SIMULATIONS = _env_int(
    "DYNAMIC_ENSEMBLE_SIMULATIONS", 6000, 1000, 50000
)
DYNAMIC_MIN_DIRECTION_STABILITY = _env_float(
    "DYNAMIC_MIN_DIRECTION_STABILITY", 0.60, 0.50, 0.90
)
DYNAMIC_ESTIMATED_COMPOSITION_FACTOR = _env_float(
    "DYNAMIC_ESTIMATED_COMPOSITION_FACTOR", 0.45, 0.10, 1.00
)
DYNAMIC_GROUP_CONCENTRATION = _env_float(
    "DYNAMIC_GROUP_CONCENTRATION", 90.0, 12.0, 500.0
)
DYNAMIC_PROBABILITY_CONCENTRATION = _env_float(
    "DYNAMIC_PROBABILITY_CONCENTRATION", 70.0, 10.0, 400.0
)


def _clip01(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return fallback


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("group probability must contain B/P/T")
    array = np.maximum(array, 1e-12)
    total = float(array.sum())
    if total <= 0:
        return np.asarray([0.458597, 0.446247, 0.095156], dtype=np.float64)
    return array / total


def _road_model_disagreement(road_context: Mapping[str, Any]) -> float:
    explicit = road_context.get("model_disagreement")
    if explicit is not None:
        try:
            return max(0.0, min(0.5, float(explicit)))
        except Exception:
            pass

    models = road_context.get("models")
    if not isinstance(models, Mapping):
        return 0.20

    probabilities = []
    weights = []
    for model in models.values():
        if not isinstance(model, Mapping):
            continue
        try:
            probability = float(model.get("banker_probability", 0.5) or 0.5)
            weight = float(model.get("effective_weight", 0.0) or 0.0)
        except Exception:
            continue
        if weight > 0:
            probabilities.append(probability)
            weights.append(weight)

    if len(probabilities) < 2:
        return 0.20

    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= max(1e-12, float(w.sum()))
    center = float(np.sum(p * w))
    return float(np.sqrt(np.sum(w * np.square(p - center))))


def _sequence_quality(
    sequence_probability: np.ndarray,
    history_length: int,
    sequence_meta: Mapping[str, Any],
) -> float:
    support = max(0, int(sequence_meta.get("support", 0) or 0))
    order = max(0, int(sequence_meta.get("order", 0) or 0))
    maturity = min(1.0, history_length / 42.0)
    support_score = min(1.0, support / 18.0)
    order_score = min(1.0, order / 3.0)
    bp_total = max(1e-12, float(sequence_probability[0] + sequence_probability[1]))
    edge = abs(float(sequence_probability[0] - sequence_probability[1])) / bp_total
    edge_score = min(1.0, edge / 0.10)
    return _clip01(
        0.35 * maturity
        + 0.30 * support_score
        + 0.15 * order_score
        + 0.20 * edge_score
    )


def _finite_quality(
    validation_gap: float,
    simulation_uncertainty: float,
    particle_ess_ratio: float,
    composition_quality: str,
) -> float:
    validation_score = 1.0 - min(1.0, max(0.0, validation_gap) / 0.06)
    simulation_score = 1.0 - min(
        1.0, max(0.0, simulation_uncertainty) / 0.025
    )
    particle_score = _clip01(particle_ess_ratio)
    composition = str(composition_quality or "").lower()
    composition_score = (
        1.0
        if composition in {"observed", "actual", "known", "session_actual"}
        else DYNAMIC_ESTIMATED_COMPOSITION_FACTOR
    )
    return _clip01(
        composition_score
        * (
            0.45 * validation_score
            + 0.35 * simulation_score
            + 0.20 * particle_score
        )
    )


def _road_quality(
    road_context: Mapping[str, Any],
    history_length: int,
) -> Dict[str, float]:
    confidence = _clip01(road_context.get("confidence_score"), 0.0)
    uncertainty = max(0.0, float(road_context.get("uncertainty", 1.0) or 1.0))
    uncertainty_score = 1.0 - min(1.0, uncertainty / 0.20)
    maturity = min(1.0, history_length / 42.0)
    disagreement = _road_model_disagreement(road_context)
    agreement_score = 1.0 - min(1.0, disagreement / 0.18)
    quality = _clip01(
        0.38 * confidence
        + 0.22 * uncertainty_score
        + 0.22 * maturity
        + 0.18 * agreement_score
    )
    return {
        "quality": quality,
        "confidence": confidence,
        "uncertainty_score": uncertainty_score,
        "maturity": maturity,
        "model_disagreement": disagreement,
        "agreement_score": agreement_score,
    }


def dynamic_group_ensemble(
    *,
    road_probability: Sequence[float],
    finite_probability: Sequence[float],
    sequence_probability: Sequence[float],
    road_context: Optional[Mapping[str, Any]] = None,
    sequence_meta: Optional[Mapping[str, Any]] = None,
    history_length: int = 0,
    validation_gap: float = 0.0,
    simulation_uncertainty: float = 0.0,
    particle_ess_ratio: float = 1.0,
    composition_quality: str = "estimated",
    road_available: bool = True,
    sequence_available: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """動態整合三個平行群組，並以後驗模擬輸出穩定度。"""
    road = _normalize(road_probability)
    finite = _normalize(finite_probability)
    sequence = _normalize(sequence_probability)
    road_ctx = dict(road_context or {})
    seq_meta = dict(sequence_meta or {})

    road_stats = _road_quality(road_ctx, history_length)
    finite_quality = _finite_quality(
        validation_gap,
        simulation_uncertainty,
        particle_ess_ratio,
        composition_quality,
    )
    sequence_quality = _sequence_quality(
        sequence, history_length, seq_meta
    )

    # 冷啟動只提供先驗，不直接指定方向；真正權重由品質分數調整。
    raw_scores = {
        "road": (
            (0.20 + 0.80 * road_stats["quality"]) * 0.58
            if road_available
            else 0.0
        ),
        "finite": (0.20 + 0.80 * finite_quality) * 0.24,
        "sequence": (
            (0.20 + 0.80 * sequence_quality) * 0.18
            if sequence_available
            else 0.0
        ),
    }

    # 圖片模式只有估計牌值時，有限牌組群組不得壓過實際可觀測的完整牌路。
    if str(composition_quality or "").lower() not in {
        "observed", "actual", "known", "session_actual"
    }:
        raw_scores["finite"] *= DYNAMIC_ESTIMATED_COMPOSITION_FACTOR

    total = sum(raw_scores.values()) or 1.0
    weights = {name: value / total for name, value in raw_scores.items()}

    group_matrix = np.vstack([road, finite, sequence])
    center = (
        road * weights["road"]
        + finite * weights["finite"]
        + sequence * weights["sequence"]
    )
    center = _normalize(center)

    group_qualities = np.asarray(
        [
            max(0.05, road_stats["quality"] if road_available else 0.05),
            max(0.05, finite_quality),
            max(0.05, sequence_quality if sequence_available else 0.05),
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    simulations = DYNAMIC_SIMULATIONS

    weight_alpha = (
        np.asarray(
            [weights["road"], weights["finite"], weights["sequence"]],
            dtype=np.float64,
        )
        * DYNAMIC_GROUP_CONCENTRATION
        + 1.0
    )
    sampled_weights = rng.dirichlet(weight_alpha, size=simulations)

    sampled_groups = []
    for probability, quality in zip(group_matrix, group_qualities):
        concentration = (
            DYNAMIC_PROBABILITY_CONCENTRATION * (0.35 + 0.65 * quality)
        )
        alpha = probability * concentration + 1.0
        sampled_groups.append(rng.dirichlet(alpha, size=simulations))

    posterior = (
        sampled_groups[0] * sampled_weights[:, [0]]
        + sampled_groups[1] * sampled_weights[:, [1]]
        + sampled_groups[2] * sampled_weights[:, [2]]
    )
    posterior /= posterior.sum(axis=1, keepdims=True)

    posterior_mean = posterior.mean(axis=0)
    low = np.quantile(posterior, 0.025, axis=0)
    high = np.quantile(posterior, 0.975, axis=0)

    bp_total = np.maximum(1e-12, posterior[:, 0] + posterior[:, 1])
    bp_difference = (posterior[:, 0] - posterior[:, 1]) / bp_total
    center_direction = "B" if posterior_mean[0] >= posterior_mean[1] else "P"
    direction_stability = float(
        np.mean(bp_difference >= 0.0)
        if center_direction == "B"
        else np.mean(bp_difference < 0.0)
    )
    difference_low, difference_high = (
        float(v) for v in np.quantile(bp_difference, [0.025, 0.975])
    )
    interval_crosses_zero = difference_low <= 0.0 <= difference_high
    group_disagreement = float(
        np.sqrt(
            np.sum(
                np.asarray(
                    [weights["road"], weights["finite"], weights["sequence"]]
                )
                * np.square(
                    group_matrix[:, 0]
                    - float(np.sum(
                        np.asarray(
                            [weights["road"], weights["finite"], weights["sequence"]]
                        )
                        * group_matrix[:, 0]
                    ))
                )
            )
        )
    )

    return {
        "probabilities": {
            "B": float(posterior_mean[0]),
            "P": float(posterior_mean[1]),
            "T": float(posterior_mean[2]),
        },
        "center_before_simulation": {
            "B": float(center[0]),
            "P": float(center[1]),
            "T": float(center[2]),
        },
        "weights": {name: float(value) for name, value in weights.items()},
        "raw_scores": {name: float(value) for name, value in raw_scores.items()},
        "quality": {
            "road": float(road_stats["quality"]),
            "finite": float(finite_quality),
            "sequence": float(sequence_quality),
            "road_model_disagreement": float(
                road_stats["model_disagreement"]
            ),
            "group_disagreement": group_disagreement,
            "composition_quality": str(composition_quality or "estimated"),
        },
        "posterior": {
            "simulations": simulations,
            "direction": center_direction,
            "direction_stability": direction_stability,
            "minimum_direction_stability": DYNAMIC_MIN_DIRECTION_STABILITY,
            "probability_interval_95": {
                OUTCOMES[index]: {
                    "low": float(low[index]),
                    "high": float(high[index]),
                }
                for index in range(3)
            },
            "bp_difference_mean": float(np.mean(bp_difference)),
            "bp_difference_std": float(np.std(bp_difference, ddof=1)),
            "bp_difference_interval_95": {
                "low": difference_low,
                "high": difference_high,
            },
            "bp_difference_interval_crosses_zero": interval_crosses_zero,
        },
    }


__all__ = [
    "DYNAMIC_MIN_DIRECTION_STABILITY",
    "DYNAMIC_SIMULATIONS",
    "dynamic_group_ensemble",
]
