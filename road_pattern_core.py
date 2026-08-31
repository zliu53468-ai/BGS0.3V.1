"""Production Road-Primary wrapper: exact V1 core + capped human derived roads.

The underlying Big-Road predictor is preserved exactly in road_pattern_v1_core.
This wrapper adds only one auxiliary stage after V1 has already produced its
B/P probability: a human-style Big Eye / Small Road / Cockroach Ask-Road model.

The derived-road auxiliary now has two sublayers:
1. Human R/U sequence patterns (recent rhythm, replay, N-gram, pattern break).
2. Standard six-row display geometry (vertical drop, horizontal tail, collision
   turns, column-height rhythm and stair/shape structure).

Both sublayers come from the same Big-Road history, so their combined formal
reliability remains capped and cannot dominate a strong V1 signal.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import math

from derived_road_geometry import score_geometry_ask_road_scenarios
from derived_road_markov import (
    MAX_DERIVED_ROAD_RELIABILITY,
    score_ask_road_scenarios,
)
from road_model import build_standard_derived_roads
from road_pattern_v1_core import (
    COMPONENT_WEIGHTS,
    MODEL_ID,
    NGRAM_ORDERS,
    OUTCOMES,
    REPLAY_LENGTHS,
    WINDOW_WEIGHTS,
    VERSION as V1_VERSION,
    forecast_road_pattern as forecast_v1_road_pattern,
    normalize_bp,
)

VERSION = f"{V1_VERSION}|HUMAN-DERIVED-ASK-V2.1-GEOMETRY"


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _logit(probability: float) -> float:
    p = _clip(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    x = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-x))


def _scenario_marks(
    sequence: Sequence[str],
    current_derived: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, str]]:
    names = ("big_eye", "small_road", "cockroach_road")
    result: dict[str, dict[str, str]] = {"B": {}, "P": {}}
    base_lengths = {
        name: len(list(current_derived.get(name) or [])) for name in names
    }
    for side in ("B", "P"):
        scenario = build_standard_derived_roads(list(sequence) + [side])
        for name in names:
            values = list(scenario.get(name) or [])
            result[side][name] = (
                str(values[-1]) if len(values) > base_lengths[name] else ""
            )
    return result


def _combine_derived_layers(
    human: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine sequence and grid-shape Ask-Road evidence under one 18% cap."""
    human_likelihood = dict(human.get("likelihood") or {})
    geometry_likelihood = dict(geometry.get("likelihood") or {})
    human_p_b = _clip(human_likelihood.get("B", 0.5), 0.20, 0.80)
    geometry_p_b = _clip(geometry_likelihood.get("B", 0.5), 0.20, 0.80)
    human_rel = _clip(
        human.get("reliability", 0.0), 0.0, MAX_DERIVED_ROAD_RELIABILITY
    )
    geometry_rel = _clip(geometry.get("reliability", 0.0), 0.0, 0.10)

    # Sequence reading remains primary inside the derived-road subsystem.
    human_weight = 0.78 * human_rel
    geometry_weight = 0.22 * geometry_rel
    if human_weight + geometry_weight <= 1e-12:
        combined_p_b = 0.5
    else:
        combined_logit = (
            human_weight * _logit(human_p_b)
            + geometry_weight * _logit(geometry_p_b)
        ) / (human_weight + geometry_weight)
        combined_p_b = _clip(_sigmoid(combined_logit), 0.20, 0.80)

    human_side = 1 if human_p_b >= 0.5 else -1
    geometry_side = 1 if geometry_p_b >= 0.5 else -1
    layers_agree = (
        human_rel <= 0.0
        or geometry_rel <= 0.0
        or human_side == geometry_side
    )
    agreement_factor = 1.0 if layers_agree else 0.80
    combined_rel = min(
        MAX_DERIVED_ROAD_RELIABILITY,
        max(human_rel, 0.65 * geometry_rel)
        + (0.15 * geometry_rel if layers_agree else 0.0),
    ) * agreement_factor

    human_active = set(human.get("active_roads") or [])
    geometry_active = set(geometry.get("active_roads") or [])
    active_roads = sorted(human_active | geometry_active)
    if len(active_roads) < 2:
        combined_rel = 0.0

    result = dict(human)
    result.update(
        {
            "model_id": "HUMAN-DERIVED-ASK-ROAD-V2.1-GEOMETRY",
            "likelihood": {
                "B": float(combined_p_b),
                "P": float(1.0 - combined_p_b),
            },
            "reliability": float(combined_rel),
            "raw_reliability": float(combined_rel),
            "max_reliability": float(MAX_DERIVED_ROAD_RELIABILITY),
            "active_roads": active_roads,
            "active_road_count": len(active_roads),
            "sequence_layer": dict(human),
            "geometry_layer": dict(geometry),
            "layer_agreement": bool(layers_agree),
            "sequence_layer_p_b": float(human_p_b),
            "geometry_layer_p_b": float(geometry_p_b),
            "sequence_layer_reliability": float(human_rel),
            "geometry_layer_reliability": float(geometry_rel),
            "semantics": (
                "human_RU_sequence_plus_standard_six_row_geometry_ask_road_"
                "with_single_shared_history_reliability_cap"
            ),
        }
    )
    return result


def _derived_analysis(sequence: Sequence[str]) -> dict[str, Any]:
    standard = build_standard_derived_roads(sequence)
    derived = {
        "big_eye": list(standard.get("big_eye") or []),
        "small_road": list(standard.get("small_road") or []),
        "cockroach_road": list(standard.get("cockroach_road") or []),
    }
    scenarios = _scenario_marks(sequence, derived)
    human = score_ask_road_scenarios(derived, scenarios)
    geometry = score_geometry_ask_road_scenarios(derived, scenarios)
    analysis = _combine_derived_layers(human, geometry)
    analysis["derived_roads"] = derived
    analysis["rule_version"] = standard.get("rule_version")
    analysis["encoding"] = dict(standard.get("encoding") or {})
    return analysis


def fuse_v1_with_derived(
    v1_banker_probability: float,
    derived_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a capped Ask-Road log-odds correction to a V1 probability.

    Strong V1 signals are intentionally hard to overturn. Derived roads gain
    their highest relative influence only when V1 is near neutral and at least
    two derived roads are active with non-trivial Ask-Road separation.
    """
    base_p_b = _clip(v1_banker_probability, 0.37, 0.63)
    likelihood = dict(derived_analysis.get("likelihood") or {})
    derived_p_b = _clip(likelihood.get("B", 0.5), 0.20, 0.80)
    reliability = _clip(
        derived_analysis.get("reliability", 0.0),
        0.0,
        MAX_DERIVED_ROAD_RELIABILITY,
    )
    active_roads = int(derived_analysis.get("active_road_count", 0) or 0)
    separation = abs(derived_p_b - 0.5) * 2.0
    ambiguity = _clip(1.0 - abs(base_p_b - 0.5) / 0.10)

    if active_roads < 2 or separation < 0.06:
        effective_weight = 0.0
    else:
        effective_weight = reliability * (0.35 + 0.65 * ambiguity)

    final_logit = _logit(base_p_b) + effective_weight * _logit(derived_p_b)
    final_p_b = _clip(_sigmoid(final_logit), 0.37, 0.63)
    return {
        "base_p_b": float(base_p_b),
        "base_p_p": float(1.0 - base_p_b),
        "derived_p_b": float(derived_p_b),
        "derived_p_p": float(1.0 - derived_p_b),
        "derived_reliability": float(reliability),
        "derived_effective_weight": float(effective_weight),
        "main_ambiguity": float(ambiguity),
        "derived_separation": float(separation),
        "final_p_b": float(final_p_b),
        "final_p_p": float(1.0 - final_p_b),
        "direction_override": (
            (base_p_b >= 0.5) != (final_p_b >= 0.5)
            and abs(base_p_b - 0.5) < 0.06
        ),
        "semantics": "v1_logit_plus_capped_human_derived_ask_road_auxiliary",
    }


def forecast_road_pattern(history: str | Iterable[Any] | None) -> dict[str, Any]:
    base = dict(forecast_v1_road_pattern(history))
    sequence = normalize_bp(history)
    derived = _derived_analysis(sequence)
    base_probabilities = dict(base.get("probabilities") or {})
    base_p_b = _clip(base_probabilities.get("B", 0.5), 0.37, 0.63)
    fusion = fuse_v1_with_derived(base_p_b, derived)
    p_b = float(fusion["final_p_b"])
    p_p = float(fusion["final_p_p"])
    direction = "B" if p_b >= p_p else "P"
    confidence = max(p_b, p_p)

    base.update(
        {
            "model_id": MODEL_ID,
            "version": VERSION,
            "direction": direction,
            "action": direction,
            "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
            "confidence": float(confidence),
            "selected_win_probability": float(confidence),
            "margin": float(abs(p_b - p_p)),
            "v1_direction": str(base.get("direction") or "B"),
            "v1_probabilities": {
                "B": float(base_p_b),
                "P": float(1.0 - base_p_b),
                "T": 0.0,
            },
            "v1_model_id": MODEL_ID,
            "v1_version": V1_VERSION,
            "derived_ask_road": derived,
            "derived_ask_road_fusion": fusion,
            "derived_direction_weight": float(fusion["derived_effective_weight"]),
            "derived_direction_authority": "capped_auxiliary_only",
            "direction_authority": "road_pattern_v1_plus_human_derived_ask_road_geometry",
            "semantics": (
                "road_v1_primary_plus_capped_human_derived_sequence_and_geometry_"
                "ask_road_auxiliary"
            ),
        }
    )
    return base


__all__ = [
    "MODEL_ID",
    "VERSION",
    "V1_VERSION",
    "WINDOW_WEIGHTS",
    "COMPONENT_WEIGHTS",
    "NGRAM_ORDERS",
    "REPLAY_LENGTHS",
    "normalize_bp",
    "fuse_v1_with_derived",
    "forecast_road_pattern",
]
