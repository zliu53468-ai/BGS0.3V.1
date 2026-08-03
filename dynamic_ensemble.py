"""BGS V10.8 舊介面相容層。

V10.7 的三群組 ``dynamic_group_ensemble`` 已被五群組受限制 Stacking 取代。
保留此函式只為避免其他舊程式匯入失敗；新主引擎直接呼叫 ``stacking_model``。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from stacking_model import constrained_stacking


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
    road = dict(road_context or {})
    planning_b = float(road.get("planning_probability", road_probability[0]) or road_probability[0])
    recent_b = float(road.get("recent_probability", road_probability[0]) or road_probability[0])
    tie = float(finite_probability[2])

    planning_three = [(1.0 - tie) * planning_b, (1.0 - tie) * (1.0 - planning_b), tie]
    recent_three = [(1.0 - tie) * recent_b, (1.0 - tie) * (1.0 - recent_b), tie]

    result = constrained_stacking(
        probabilities={
            "global_history": road_probability,
            "road_planning": planning_three,
            "recent_road": recent_three,
            "finite": finite_probability,
            "sequence": sequence_probability,
        },
        qualities={
            "global_history": float(road.get("confidence_score", 0.4) or 0.4),
            "road_planning": float(road.get("planning_reliability", 0.4) or 0.4),
            "recent_road": float(road.get("recent_reliability", 0.4) or 0.4),
            "finite": max(0.0, 1.0 - validation_gap / 0.06),
            "sequence": min(1.0, int(dict(sequence_meta or {}).get("support", 0) or 0) / 18.0),
        },
        availability={
            "global_history": history_length >= 10,
            "road_planning": road_available,
            "recent_road": road_available,
            "finite": True,
            "sequence": sequence_available,
        },
        composition_quality=composition_quality,
        seed=seed,
    )
    return {
        **result,
        "compatibility_mode": True,
        "legacy_function": "dynamic_group_ensemble",
    }


__all__ = ["dynamic_group_ensemble"]
