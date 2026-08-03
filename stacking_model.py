"""BGS V10.8 受限制機率 Stacking 與後驗穩定度。

五個群組：
1. 全歷史機率擬合
2. 全盤牌路規劃
3. 近期牌路專家
4. 有限牌組機率（超幾何／蒙地卡羅／粒子）
5. 額外序列模型

權重會依品質調整，但同時受最低／最高邊界約束，避免近期趨勢再次占到七成以上，
也避免圖片模式的有限牌組因估計牌值被重複降權。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence
import os

import numpy as np


OUTCOMES = ("B", "P", "T")
GROUPS = ("global_history", "road_planning", "recent_road", "finite", "sequence")


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


STACKING_SIMULATIONS = _env_int("STACKING_SIMULATIONS", 6000, 1000, 50000)
STACKING_MIN_STABILITY = _env_float("STACKING_MIN_STABILITY", 0.58, 0.50, 0.90)
STACKING_PROBABILITY_CONCENTRATION = _env_float(
    "STACKING_PROBABILITY_CONCENTRATION", 95.0, 12.0, 500.0
)

BASE_PRIORS = {
    "global_history": 0.30,
    "road_planning": 0.30,
    "recent_road": 0.15,
    "finite": 0.20,
    "sequence": 0.05,
}

DEFAULT_BOUNDS = {
    "global_history": (0.20, 0.40),
    "road_planning": (0.20, 0.40),
    "recent_road": (0.05, 0.25),
    "finite": (0.12, 0.30),
    "sequence": (0.02, 0.10),
}


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("probability vector must contain B/P/T")
    array = np.maximum(array, 1e-12)
    total = float(array.sum())
    return array / total


def _clip01(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return fallback


def _bounded_weights(
    scores: Mapping[str, float],
    availability: Mapping[str, bool],
) -> Dict[str, float]:
    bounds: Dict[str, list[float]] = {}
    for name in GROUPS:
        if availability.get(name, True):
            minimum, maximum = DEFAULT_BOUNDS[name]
        else:
            minimum = maximum = 0.0
        bounds[name] = [float(minimum), float(maximum)]

    # 極早期資料不足時，確保仍有足夠的最大容量可分配到 1。
    maximum_total = sum(value[1] for value in bounds.values())
    if maximum_total < 1.0:
        for name, emergency_maximum in (
            ("finite", 0.70),
            ("recent_road", 0.45),
            ("sequence", 0.20),
            ("global_history", 0.45),
            ("road_planning", 0.45),
        ):
            if availability.get(name, True):
                bounds[name][1] = max(bounds[name][1], emergency_maximum)
            maximum_total = sum(value[1] for value in bounds.values())
            if maximum_total >= 1.0:
                break

    minimum_total = sum(value[0] for value in bounds.values())
    if minimum_total > 1.0:
        scale = 1.0 / minimum_total
        for value in bounds.values():
            value[0] *= scale

    weights = {name: bounds[name][0] for name in GROUPS}
    remaining = max(0.0, 1.0 - sum(weights.values()))
    active = {name for name in GROUPS if bounds[name][1] > weights[name] + 1e-12}

    while remaining > 1e-12 and active:
        score_total = sum(max(1e-9, float(scores.get(name, 0.0))) for name in active)
        proposed = {
            name: remaining * max(1e-9, float(scores.get(name, 0.0))) / score_total
            for name in active
        }
        saturated = []
        for name in active:
            capacity = bounds[name][1] - weights[name]
            if proposed[name] >= capacity - 1e-12:
                weights[name] += capacity
                remaining -= capacity
                saturated.append(name)
        if saturated:
            for name in saturated:
                active.remove(name)
            continue
        for name in active:
            weights[name] += proposed[name]
        remaining = 0.0

    if remaining > 1e-9:
        # 理論上只會在所有 max 設得太低時發生；把殘量放入目前可用群組中。
        fallback = next((name for name in GROUPS if availability.get(name, True)), "finite")
        weights[fallback] += remaining

    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def constrained_stacking(
    *,
    probabilities: Mapping[str, Sequence[float]],
    qualities: Mapping[str, float],
    availability: Optional[Mapping[str, bool]] = None,
    composition_quality: str = "estimated",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    available = {name: True for name in GROUPS}
    if availability:
        available.update({name: bool(value) for name, value in availability.items()})

    normalized = {
        name: _normalize(probabilities[name])
        for name in GROUPS
    }

    quality_scores = {name: _clip01(qualities.get(name), 0.0) for name in GROUPS}
    composition = str(composition_quality or "estimated").lower()
    finite_information_factor = (
        1.0
        if composition in {"observed", "actual", "known", "session_actual"}
        else 0.70
    )

    scores: Dict[str, float] = {}
    for name in GROUPS:
        quality_multiplier = 0.55 + 0.45 * quality_scores[name]
        score = BASE_PRIORS[name] * quality_multiplier
        if name == "finite":
            # 只在這裡降一次，避免 V10.7 的重複降權。
            score *= finite_information_factor
        scores[name] = score if available.get(name, True) else 0.0

    weights = _bounded_weights(scores, available)
    center = np.zeros(3, dtype=np.float64)
    for name in GROUPS:
        center += normalized[name] * weights[name]
    center = _normalize(center)

    rng = np.random.default_rng(seed)
    posterior_groups: Dict[str, np.ndarray] = {}
    for name in GROUPS:
        quality = max(0.05, quality_scores[name])
        concentration = STACKING_PROBABILITY_CONCENTRATION * (0.35 + 0.65 * quality)
        alpha = normalized[name] * concentration + 1.0
        posterior_groups[name] = rng.dirichlet(alpha, size=STACKING_SIMULATIONS)

    posterior = np.zeros((STACKING_SIMULATIONS, 3), dtype=np.float64)
    for name in GROUPS:
        posterior += posterior_groups[name] * weights[name]
    posterior /= posterior.sum(axis=1, keepdims=True)

    mean = posterior.mean(axis=0)
    low = np.quantile(posterior, 0.025, axis=0)
    high = np.quantile(posterior, 0.975, axis=0)

    bp_total = np.maximum(1e-12, posterior[:, 0] + posterior[:, 1])
    bp_difference = (posterior[:, 0] - posterior[:, 1]) / bp_total
    direction = "B" if mean[0] >= mean[1] else "P"
    direction_stability = float(
        np.mean(bp_difference >= 0.0)
        if direction == "B"
        else np.mean(bp_difference < 0.0)
    )
    difference_low, difference_high = (
        float(value) for value in np.quantile(bp_difference, [0.025, 0.975])
    )

    group_directions = {
        name: "B" if normalized[name][0] >= normalized[name][1] else "P"
        for name in GROUPS
        if available.get(name, True)
    }
    weighted_agreement = sum(
        weights[name]
        for name, group_direction in group_directions.items()
        if group_direction == direction
    )
    contributions = {
        name: {
            "weight": float(weights[name]),
            "B": float(normalized[name][0] * weights[name]),
            "P": float(normalized[name][1] * weights[name]),
            "T": float(normalized[name][2] * weights[name]),
            "direction": group_directions.get(name, ""),
            "quality": float(quality_scores[name]),
        }
        for name in GROUPS
    }

    return {
        "engine": "CONSTRAINED_FULL_HISTORY_STACKING_V10_8",
        "probabilities": {key: float(mean[index]) for index, key in enumerate(OUTCOMES)},
        "center_before_simulation": {key: float(center[index]) for index, key in enumerate(OUTCOMES)},
        "weights": {name: float(weights[name]) for name in GROUPS},
        "scores": {name: float(scores[name]) for name in GROUPS},
        "qualities": {name: float(quality_scores[name]) for name in GROUPS},
        "availability": dict(available),
        "bounds": {
            name: {
                "minimum": DEFAULT_BOUNDS[name][0] if available.get(name, True) else 0.0,
                "maximum": DEFAULT_BOUNDS[name][1] if available.get(name, True) else 0.0,
            }
            for name in GROUPS
        },
        "contributions": contributions,
        "direction": direction,
        "weighted_agreement": float(weighted_agreement),
        "composition_quality": composition,
        "finite_information_factor": float(finite_information_factor),
        "posterior": {
            "simulations": STACKING_SIMULATIONS,
            "direction": direction,
            "direction_stability": direction_stability,
            "minimum_direction_stability": STACKING_MIN_STABILITY,
            "probability_interval_95": {
                key: {"low": float(low[index]), "high": float(high[index])}
                for index, key in enumerate(OUTCOMES)
            },
            "bp_difference_mean": float(np.mean(bp_difference)),
            "bp_difference_std": float(np.std(bp_difference, ddof=1)),
            "bp_difference_interval_95": {
                "low": difference_low,
                "high": difference_high,
            },
            "bp_difference_interval_crosses_zero": difference_low <= 0.0 <= difference_high,
        },
    }


__all__ = ["constrained_stacking", "GROUPS", "STACKING_MIN_STABILITY"]
