"""BGS V10.9 牌路先行但受限制的規則導向 Stacking。

有限牌組是唯一可提供真實因果偏移的主要群組；全歷史、牌路規劃、近期牌路與
額外序列只作弱輔助。截圖／真人桌沒有真實剩餘牌時，各群組偏移會先收縮回
標準 8 副牌先驗，再套用路型總權重硬上限，避免把視覺路型當成強因果訊號。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import os

import numpy as np


OUTCOMES = ("B", "P", "T")
GROUPS = ("global_history", "road_planning", "recent_road", "finite", "sequence")
ROAD_GROUPS = ("global_history", "road_planning", "recent_road")
ROAD_FIRST_GROUPS = ("road_planning", "recent_road")
BASELINE_PRIOR = np.asarray([0.458597, 0.446247, 0.095156], dtype=np.float64)
RELIABLE_COMPOSITION_LABELS = {"observed", "actual", "known", "session_actual", "virtual_shoe"}


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
STACKING_ROAD_TOTAL_CAP_RELIABLE = _env_float(
    "STACKING_ROAD_TOTAL_CAP_RELIABLE", 0.22, 0.05, 0.40
)
STACKING_ROAD_TOTAL_CAP_ESTIMATED = _env_float(
    "STACKING_ROAD_TOTAL_CAP_ESTIMATED", 0.28, 0.05, 0.40
)
STACKING_ROAD_FIRST_ENABLED = os.getenv(
    "STACKING_ROAD_FIRST_ENABLED", "1"
).strip() == "1"
STACKING_ROAD_FIRST_MIN_QUALITY = _env_float(
    "STACKING_ROAD_FIRST_MIN_QUALITY", 0.62, 0.40, 0.95
)
STACKING_ROAD_FIRST_PLANNING_BOOST = _env_float(
    "STACKING_ROAD_FIRST_PLANNING_BOOST", 1.35, 1.0, 2.0
)
STACKING_ROAD_FIRST_RECENT_BOOST = _env_float(
    "STACKING_ROAD_FIRST_RECENT_BOOST", 1.20, 1.0, 2.0
)
STACKING_SEQUENCE_MAX_RELIABLE = _env_float(
    "STACKING_SEQUENCE_MAX_RELIABLE", 0.10, 0.0, 0.25
)
STACKING_SEQUENCE_MAX_ESTIMATED = _env_float(
    "STACKING_SEQUENCE_MAX_ESTIMATED", 0.06, 0.0, 0.15
)
STACKING_FINITE_MIN_RELIABLE = _env_float(
    "STACKING_FINITE_MIN_RELIABLE", 0.68, 0.40, 0.95
)
STACKING_FINITE_MIN_ESTIMATED = _env_float(
    "STACKING_FINITE_MIN_ESTIMATED", 0.66, 0.45, 0.95
)
STACKING_ESTIMATED_FINITE_INFORMATION_FACTOR = _env_float(
    "STACKING_ESTIMATED_FINITE_INFORMATION_FACTOR", 0.55, 0.05, 1.0
)
STACKING_ESTIMATED_FINITE_SIGNAL_SHARE = _env_float(
    "STACKING_ESTIMATED_FINITE_SIGNAL_SHARE", 0.25, 0.0, 1.0
)
STACKING_ESTIMATED_ROAD_SIGNAL_SHARE = _env_float(
    "STACKING_ESTIMATED_ROAD_SIGNAL_SHARE", 0.30, 0.0, 1.0
)
STACKING_ESTIMATED_SEQUENCE_SIGNAL_SHARE = _env_float(
    "STACKING_ESTIMATED_SEQUENCE_SIGNAL_SHARE", 0.20, 0.0, 1.0
)

RELIABLE_BASE_PRIORS = {
    "global_history": 0.10,
    "road_planning": 0.12,
    "recent_road": 0.07,
    "finite": 0.63,
    "sequence": 0.08,
}
ESTIMATED_BASE_PRIORS = {
    "global_history": 0.10,
    "road_planning": 0.14,
    "recent_road": 0.08,
    "finite": 0.62,
    "sequence": 0.06,
}

# 相容舊匯入名稱；實際執行會依 composition_quality 選擇設定。
BASE_PRIORS = dict(RELIABLE_BASE_PRIORS)


def _bounds_for_mode(reliable_composition: bool) -> Dict[str, Tuple[float, float]]:
    if reliable_composition:
        return {
            "global_history": (0.0, 0.10),
            "road_planning": (0.0, 0.12),
            "recent_road": (0.0, 0.08),
            "finite": (STACKING_FINITE_MIN_RELIABLE, 0.96),
            "sequence": (0.0, STACKING_SEQUENCE_MAX_RELIABLE),
        }
    return {
        "global_history": (0.0, 0.10),
        "road_planning": (0.0, 0.18),
        "recent_road": (0.0, 0.12),
        "finite": (STACKING_FINITE_MIN_ESTIMATED, 0.98),
        "sequence": (0.0, STACKING_SEQUENCE_MAX_ESTIMATED),
    }


DEFAULT_BOUNDS = _bounds_for_mode(True)


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


def _blend_with_prior(values: np.ndarray, signal_share: float) -> np.ndarray:
    share = max(0.0, min(1.0, float(signal_share)))
    return _normalize(BASELINE_PRIOR * (1.0 - share) + values * share)


def _redistribute(
    weights: Dict[str, float],
    amount: float,
    bounds: Mapping[str, Tuple[float, float]],
    targets: Sequence[str],
) -> float:
    remaining = max(0.0, float(amount))
    for name in targets:
        if remaining <= 1e-12:
            break
        maximum = float(bounds[name][1])
        capacity = max(0.0, maximum - weights.get(name, 0.0))
        take = min(capacity, remaining)
        weights[name] = weights.get(name, 0.0) + take
        remaining -= take
    return remaining


def _bounded_weights(
    scores: Mapping[str, float],
    availability: Mapping[str, bool],
    bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    road_total_cap: Optional[float] = None,
) -> Dict[str, float]:
    selected_bounds = dict(bounds or DEFAULT_BOUNDS)
    cap = float(
        STACKING_ROAD_TOTAL_CAP_RELIABLE if road_total_cap is None else road_total_cap
    )
    active_bounds: Dict[str, list[float]] = {}
    for name in GROUPS:
        minimum, maximum = selected_bounds[name]
        if not availability.get(name, True):
            minimum = maximum = 0.0
        active_bounds[name] = [float(minimum), float(maximum)]

    road_minimum = sum(active_bounds[name][0] for name in ROAD_GROUPS)
    if road_minimum > cap and road_minimum > 0:
        scale = cap / road_minimum
        for name in ROAD_GROUPS:
            active_bounds[name][0] *= scale

    minimum_total = sum(value[0] for value in active_bounds.values())
    if minimum_total > 1.0:
        scale = 1.0 / minimum_total
        for value in active_bounds.values():
            value[0] *= scale

    maximum_total = sum(value[1] for value in active_bounds.values())
    if maximum_total < 1.0:
        for name in ("finite", "sequence", "global_history", "road_planning", "recent_road"):
            if availability.get(name, True):
                active_bounds[name][1] += 1.0 - maximum_total
                maximum_total = 1.0
                break

    weights = {name: active_bounds[name][0] for name in GROUPS}
    remaining = max(0.0, 1.0 - sum(weights.values()))
    active = {name for name in GROUPS if active_bounds[name][1] > weights[name] + 1e-12}
    while remaining > 1e-12 and active:
        score_total = sum(max(1e-9, float(scores.get(name, 0.0))) for name in active)
        proposed = {
            name: remaining * max(1e-9, float(scores.get(name, 0.0))) / score_total
            for name in active
        }
        saturated = []
        for name in active:
            capacity = active_bounds[name][1] - weights[name]
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

    road_total = sum(weights[name] for name in ROAD_GROUPS)
    if road_total > cap + 1e-12:
        excess = road_total - cap
        scale = cap / max(1e-12, road_total)
        for name in ROAD_GROUPS:
            weights[name] *= scale
        leftover = _redistribute(
            weights,
            excess,
            {name: tuple(active_bounds[name]) for name in GROUPS},
            ("finite", "sequence"),
        )
        if leftover > 1e-9:
            # 只有使用者把非路型 max 設得過低時才可能發生。
            weights["finite"] += leftover

    total = sum(weights.values()) or 1.0
    normalized = {name: max(0.0, value / total) for name, value in weights.items()}
    # 浮點修正殘量優先放入 finite，不改變路型上限。
    residual = 1.0 - sum(normalized.values())
    normalized["finite"] = normalized.get("finite", 0.0) + residual
    return normalized


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

    raw_normalized = {name: _normalize(probabilities[name]) for name in GROUPS}
    quality_scores = {name: _clip01(qualities.get(name), 0.0) for name in GROUPS}
    composition = str(composition_quality or "estimated").lower().strip()
    reliable_composition = composition in RELIABLE_COMPOSITION_LABELS

    effective_probabilities = {name: values.copy() for name, values in raw_normalized.items()}
    signal_shares = {name: 1.0 for name in GROUPS}
    if not reliable_composition:
        for name in ROAD_GROUPS:
            signal_shares[name] = STACKING_ESTIMATED_ROAD_SIGNAL_SHARE
            effective_probabilities[name] = _blend_with_prior(
                raw_normalized[name], STACKING_ESTIMATED_ROAD_SIGNAL_SHARE
            )
        signal_shares["finite"] = STACKING_ESTIMATED_FINITE_SIGNAL_SHARE
        effective_probabilities["finite"] = _blend_with_prior(
            raw_normalized["finite"], STACKING_ESTIMATED_FINITE_SIGNAL_SHARE
        )
        signal_shares["sequence"] = STACKING_ESTIMATED_SEQUENCE_SIGNAL_SHARE
        effective_probabilities["sequence"] = _blend_with_prior(
            raw_normalized["sequence"], STACKING_ESTIMATED_SEQUENCE_SIGNAL_SHARE
        )

    priors = RELIABLE_BASE_PRIORS if reliable_composition else ESTIMATED_BASE_PRIORS
    finite_information_factor = (
        1.0 if reliable_composition else STACKING_ESTIMATED_FINITE_INFORMATION_FACTOR
    )
    scores: Dict[str, float] = {}
    for name in GROUPS:
        quality_multiplier = 0.60 + 0.40 * quality_scores[name]
        score = float(priors[name]) * quality_multiplier
        if name == "finite":
            score *= finite_information_factor
        scores[name] = score if available.get(name, True) else 0.0

    planning_direction = (
        "B"
        if raw_normalized["road_planning"][0] >= raw_normalized["road_planning"][1]
        else "P"
    )
    recent_direction = (
        "B"
        if raw_normalized["recent_road"][0] >= raw_normalized["recent_road"][1]
        else "P"
    )
    road_first_quality = min(
        quality_scores["road_planning"],
        quality_scores["recent_road"],
    )
    road_first_active = bool(
        STACKING_ROAD_FIRST_ENABLED
        and available.get("road_planning", False)
        and available.get("recent_road", False)
        and planning_direction == recent_direction
        and road_first_quality >= STACKING_ROAD_FIRST_MIN_QUALITY
    )
    if road_first_active:
        scores["road_planning"] *= STACKING_ROAD_FIRST_PLANNING_BOOST
        scores["recent_road"] *= STACKING_ROAD_FIRST_RECENT_BOOST

    bounds = _bounds_for_mode(reliable_composition)
    road_total_cap = (
        STACKING_ROAD_TOTAL_CAP_RELIABLE
        if reliable_composition
        else STACKING_ROAD_TOTAL_CAP_ESTIMATED
    )
    weights = _bounded_weights(scores, available, bounds=bounds, road_total_cap=road_total_cap)
    center = np.zeros(3, dtype=np.float64)
    for name in GROUPS:
        center += effective_probabilities[name] * weights[name]
    center = _normalize(center)

    rng = np.random.default_rng(seed)
    posterior_groups: Dict[str, np.ndarray] = {}
    for name in GROUPS:
        quality = max(0.05, quality_scores[name])
        concentration = STACKING_PROBABILITY_CONCENTRATION * (0.35 + 0.65 * quality)
        alpha = effective_probabilities[name] * concentration + 1.0
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
        np.mean(bp_difference >= 0.0) if direction == "B" else np.mean(bp_difference < 0.0)
    )
    difference_low, difference_high = (
        float(value) for value in np.quantile(bp_difference, [0.025, 0.975])
    )

    group_directions = {
        name: "B" if effective_probabilities[name][0] >= effective_probabilities[name][1] else "P"
        for name in GROUPS
        if available.get(name, True)
    }
    weighted_agreement = sum(
        weights[name]
        for name, group_direction in group_directions.items()
        if group_direction == direction
    )
    effective_signal_weights = {
        name: float(weights[name] * signal_shares[name]) for name in GROUPS
    }
    implicit_prior_weights = {
        name: float(weights[name] * (1.0 - signal_shares[name])) for name in GROUPS
    }
    baseline_anchor_weight = float(sum(implicit_prior_weights.values()))
    contributions = {
        name: {
            "weight": float(weights[name]),
            "B": float(effective_probabilities[name][0] * weights[name]),
            "P": float(effective_probabilities[name][1] * weights[name]),
            "T": float(effective_probabilities[name][2] * weights[name]),
            "direction": group_directions.get(name, ""),
            "quality": float(quality_scores[name]),
            "signal_share": float(signal_shares[name]),
            "effective_signal_weight": effective_signal_weights[name],
            "implicit_prior_weight": implicit_prior_weights[name],
        }
        for name in GROUPS
    }
    road_total_weight = float(sum(weights[name] for name in ROAD_GROUPS))
    road_effective_signal_weight = float(
        sum(effective_signal_weights[name] for name in ROAD_GROUPS)
    )

    return {
        "engine": "CONSTRAINED_RULE_AWARE_STACKING_V10_9",
        "probabilities": {key: float(mean[index]) for index, key in enumerate(OUTCOMES)},
        "center_before_simulation": {key: float(center[index]) for index, key in enumerate(OUTCOMES)},
        "baseline_prior": {key: float(BASELINE_PRIOR[index]) for index, key in enumerate(OUTCOMES)},
        "raw_group_probabilities": {
            name: {key: float(raw_normalized[name][index]) for index, key in enumerate(OUTCOMES)}
            for name in GROUPS
        },
        "effective_group_probabilities": {
            name: {key: float(effective_probabilities[name][index]) for index, key in enumerate(OUTCOMES)}
            for name in GROUPS
        },
        "weights": {name: float(weights[name]) for name in GROUPS},
        "road_total_weight": road_total_weight,
        "road_total_cap": float(road_total_cap),
        "road_effective_signal_weight": road_effective_signal_weight,
        "road_first": {
            "enabled": bool(STACKING_ROAD_FIRST_ENABLED),
            "active": bool(road_first_active),
            "direction": planning_direction if road_first_active else "",
            "planning_direction": planning_direction,
            "recent_direction": recent_direction,
            "direction_agrees": bool(planning_direction == recent_direction),
            "quality": float(road_first_quality),
            "minimum_quality": float(STACKING_ROAD_FIRST_MIN_QUALITY),
            "planning_boost": (
                float(STACKING_ROAD_FIRST_PLANNING_BOOST)
                if road_first_active
                else 1.0
            ),
            "recent_boost": (
                float(STACKING_ROAD_FIRST_RECENT_BOOST)
                if road_first_active
                else 1.0
            ),
            "road_total_weight": road_total_weight,
            "road_total_cap": float(road_total_cap),
        },
        "finite_effective_signal_weight": float(effective_signal_weights["finite"]),
        "sequence_effective_signal_weight": float(effective_signal_weights["sequence"]),
        "baseline_anchor_weight": baseline_anchor_weight,
        "effective_signal_weights": effective_signal_weights,
        "implicit_prior_weights": implicit_prior_weights,
        "sequence_weight_cap": float(bounds["sequence"][1]),
        "scores": {name: float(scores[name]) for name in GROUPS},
        "qualities": {name: float(quality_scores[name]) for name in GROUPS},
        "availability": dict(available),
        "bounds": {
            name: {
                "minimum": float(bounds[name][0]) if available.get(name, True) else 0.0,
                "maximum": float(bounds[name][1]) if available.get(name, True) else 0.0,
            }
            for name in GROUPS
        },
        "contributions": contributions,
        "direction": direction,
        "weighted_agreement": float(weighted_agreement),
        "composition_quality": composition,
        "reliable_composition": reliable_composition,
        "finite_information_factor": float(finite_information_factor),
        "estimated_signal_shares": {name: float(signal_shares[name]) for name in GROUPS},
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
            "bp_difference_interval_95": {"low": difference_low, "high": difference_high},
            "bp_difference_interval_crosses_zero": difference_low <= 0.0 <= difference_high,
        },
    }


__all__ = [
    "constrained_stacking",
    "GROUPS",
    "ROAD_GROUPS",
    "ROAD_FIRST_GROUPS",
    "STACKING_MIN_STABILITY",
    "BASELINE_PRIOR",
]
