"""Direct Markov prediction model for BGS.

The model keeps the original first 29D design as an explainable context:
21 road features + 8 Laplace-smoothed Markov transition probabilities.
No LinUCB, CUSUM, stacking, adaptive ensemble, or cross-resonance layer is used.
Markov is the primary direction predictor; the road model only provides a small
structural calibration and confidence adjustment.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple
import math

from road_model import ROAD_FEATURE_NAMES

MODEL_VERSION = "ROAD-MARKOV-29D-DIRECT-V1"
MARKOV_WINDOW_SIZE = 36
MARKOV_ALPHA = 1.0
MARKOV_FEATURE_NAMES = (
    "markov_p_b_given_b", "markov_p_p_given_b",
    "markov_p_b_given_p", "markov_p_p_given_p",
    "markov_p_b_given_bb", "markov_p_b_given_pp",
    "markov_p_p_given_bb", "markov_p_p_given_pp",
)
FEATURE_NAMES = ROAD_FEATURE_NAMES + MARKOV_FEATURE_NAMES
CONTEXT_DIM = len(FEATURE_NAMES)


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _clean_bp(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("actual_outcome")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P"}:
            result.append(value)
    return result[-2000:]


def _pair(banker_count: int, player_count: int, alpha: float = MARKOV_ALPHA) -> Tuple[float, float, int]:
    support = int(banker_count + player_count)
    denominator = support + 2.0 * alpha
    return (
        (banker_count + alpha) / denominator,
        (player_count + alpha) / denominator,
        support,
    )


def extract_markov_features(
    history: Iterable[Any],
    *,
    window_size: int = MARKOV_WINDOW_SIZE,
    alpha: float = MARKOV_ALPHA,
) -> Dict[str, Any]:
    sequence = _clean_bp(history)[-max(3, int(window_size)):]

    bb = bp = pb = pp = 0
    for previous, current in zip(sequence, sequence[1:]):
        if previous == "B" and current == "B":
            bb += 1
        elif previous == "B" and current == "P":
            bp += 1
        elif previous == "P" and current == "B":
            pb += 1
        else:
            pp += 1

    p_b_b, p_p_b, support_b = _pair(bb, bp, alpha)
    p_b_p, p_p_p, support_p = _pair(pb, pp, alpha)

    b_after_bb = p_after_bb = b_after_pp = p_after_pp = 0
    for first, second, current in zip(sequence, sequence[1:], sequence[2:]):
        if first == "B" and second == "B":
            if current == "B":
                b_after_bb += 1
            else:
                p_after_bb += 1
        elif first == "P" and second == "P":
            if current == "B":
                b_after_pp += 1
            else:
                p_after_pp += 1

    p_b_bb, p_p_bb, support_bb = _pair(b_after_bb, p_after_bb, alpha)
    p_b_pp, p_p_pp, support_pp = _pair(b_after_pp, p_after_pp, alpha)

    values = [
        p_b_b, p_p_b,
        p_b_p, p_p_p,
        p_b_bb, p_b_pp,
        p_p_bb, p_p_pp,
    ]
    return {
        "sequence": sequence,
        "values": values,
        "feature_dict": dict(zip(MARKOV_FEATURE_NAMES, values)),
        "supports": {
            "B": support_b,
            "P": support_p,
            "BB": support_bb,
            "PP": support_pp,
        },
        "window_size": max(3, int(window_size)),
        "alpha": float(alpha),
        "sample_count": len(sequence),
    }


def _active_markov_probability(markov: Mapping[str, Any]) -> Dict[str, Any]:
    sequence = list(markov.get("sequence") or [])
    features = dict(markov.get("feature_dict") or {})
    supports = dict(markov.get("supports") or {})

    if not sequence:
        return {
            "state": "",
            "order": 0,
            "banker_probability": 0.5,
            "player_probability": 0.5,
            "support": 0,
            "reliability": 0.0,
        }

    last = sequence[-1]
    first_b = float(features["markov_p_b_given_b"] if last == "B" else features["markov_p_b_given_p"])
    first_support = int(supports.get(last, 0) or 0)
    first_reliability = first_support / (first_support + 6.0)

    state = last
    order = 1
    active_b = first_b
    active_support = first_support

    # Exact first-29D semantics: second-order state exists only for BB/PP.
    if len(sequence) >= 2:
        tail2 = "".join(sequence[-2:])
        if tail2 == "BB":
            second_b = float(features["markov_p_b_given_bb"])
            second_support = int(supports.get("BB", 0) or 0)
            second_reliability = second_support / (second_support + 5.0)
            active_b = second_reliability * second_b + (1.0 - second_reliability) * first_b
            state = "BB"
            order = 2
            active_support = second_support
        elif tail2 == "PP":
            second_b = float(features["markov_p_b_given_pp"])
            second_support = int(supports.get("PP", 0) or 0)
            second_reliability = second_support / (second_support + 5.0)
            active_b = second_reliability * second_b + (1.0 - second_reliability) * first_b
            state = "PP"
            order = 2
            active_support = second_support

    reliability = 1.0 - (1.0 - first_reliability) * (
        1.0 - active_support / (active_support + 6.0)
    )
    active_b = _clip(active_b, 0.02, 0.98)
    return {
        "state": state,
        "order": order,
        "banker_probability": float(active_b),
        "player_probability": float(1.0 - active_b),
        "support": int(active_support),
        "first_order_support": int(first_support),
        "reliability": float(_clip(reliability)),
    }


def predict_markov(
    history: Iterable[Any],
    *,
    road_context: Mapping[str, Any],
) -> Dict[str, Any]:
    markov = extract_markov_features(history)
    active = _active_markov_probability(markov)

    road_features = list(road_context.get("road_features") or [])
    if len(road_features) != len(ROAD_FEATURE_NAMES):
        road_features = [0.0] * len(ROAD_FEATURE_NAMES)
        road_features[0] = 1.0

    markov_values = list(markov["values"])
    context = [float(v) for v in road_features] + [float(v) for v in markov_values]
    if len(context) != CONTEXT_DIM:
        raise RuntimeError("29D Road+Markov context dimension mismatch")

    markov_b = float(active["banker_probability"])
    road_b = _clip(road_context.get("banker_probability", 0.5), 0.35, 0.65)
    road_confidence = _clip(road_context.get("confidence_score", 0.0))

    # Markov is explicitly primary. Road receives only 10-25% calibration share.
    markov_reliability = float(active["reliability"])
    road_weight = _clip(0.10 + 0.15 * road_confidence * (1.0 - 0.45 * markov_reliability), 0.10, 0.25)
    markov_weight = 1.0 - road_weight
    final_b = markov_weight * markov_b + road_weight * road_b
    final_b = _clip(final_b, 0.02, 0.98)

    direction = "B" if final_b >= 0.5 else "P"
    selected_probability = final_b if direction == "B" else 1.0 - final_b

    # Confidence measures evidence/edge, not a guaranteed win probability.
    # Short histories are deliberately maturity-capped.
    edge = abs(final_b - 0.5) * 2.0
    maturity = min(1.0, float(markov["sample_count"]) / 18.0)
    edge_strength = min(1.0, edge / 0.35)
    confidence = _clip(
        0.28
        + 0.28 * markov_reliability
        + 0.18 * edge_strength
        + 0.12 * maturity
        + 0.06 * road_confidence,
        0.0,
        0.82,
    )

    return {
        "model_version": MODEL_VERSION,
        "engine": "DIRECT_MARKOV_PRIMARY",
        "model_core": "original_29d_features_markov_primary_road_calibration",
        "context_dim": CONTEXT_DIM,
        "context_feature_names": list(FEATURE_NAMES),
        "context_vector": [round(float(v), 10) for v in context],
        "markov_features": dict(markov["feature_dict"]),
        "markov_state": {
            "window_size": markov["window_size"],
            "sample_count": markov["sample_count"],
            "supports": dict(markov["supports"]),
            "active_state": active["state"],
            "active_order": active["order"],
            "active_support": active["support"],
            "reliability": active["reliability"],
        },
        "markov_predict": {
            "direction": "B" if markov_b >= 0.5 else "P",
            "banker_probability": markov_b,
            "player_probability": 1.0 - markov_b,
            "state": active["state"],
            "order": active["order"],
            "support": active["support"],
            "reliability": active["reliability"],
        },
        "road_predict": {
            "direction": str(road_context.get("direction") or ("B" if road_b >= 0.5 else "P")),
            "banker_probability": road_b,
            "player_probability": 1.0 - road_b,
            "confidence": road_confidence,
        },
        "fusion": {
            "mode": "markov_primary_road_calibration",
            "markov_weight": float(markov_weight),
            "road_weight": float(road_weight),
        },
        "direction": direction,
        "banker_probability": float(final_b),
        "player_probability": float(1.0 - final_b),
        "selected_probability": float(selected_probability),
        "confidence": float(confidence),
        "edge": float(edge),
    }


__all__ = [
    "MODEL_VERSION",
    "MARKOV_ALPHA",
    "MARKOV_WINDOW_SIZE",
    "MARKOV_FEATURE_NAMES",
    "FEATURE_NAMES",
    "CONTEXT_DIM",
    "extract_markov_features",
    "predict_markov",
]
