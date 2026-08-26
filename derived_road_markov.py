"""Support-aware Markov model for baccarat derived-road color sequences.

The derived roads are deterministic transformations of the Big Road. Their red
and blue marks do NOT mean Banker/Player. Internally we encode:
    R = red / regular
    U = blue / irregular ("U" avoids confusing blue with Banker "B")

The model estimates the next derived-road color for Big Eye Boy, Small Road and
Cockroach Pig, then scores the color emitted under hypothetical next-Banker and
next-Player Big-Road extensions.

Because these roads are derived from the same B/P history used by the main
Markov model, their formal fusion reliability is deliberately capped.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence
import math

DERIVED_SYMBOLS = ("R", "U")
DERIVED_MAX_ORDER = 3
DERIVED_SUPPORT_THRESHOLD = 3
DERIVED_BACKOFF_ALPHA = 0.75
DERIVED_PRIOR_STRENGTH = 2.0
MAX_DERIVED_ROAD_RELIABILITY = 0.18

ROAD_WEIGHTS = {
    "big_eye": 1.00,
    "small_road": 0.85,
    "cockroach_road": 0.70,
}


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _clean(values: Sequence[Any]) -> list[str]:
    return [
        str(value).upper().strip()
        for value in values
        if str(value).upper().strip() in DERIVED_SYMBOLS
    ]


def _normalize_pair(values: Mapping[str, float]) -> Dict[str, float]:
    red = max(1e-12, float(values.get("R", 0.0) or 0.0))
    blue = max(1e-12, float(values.get("U", 0.0) or 0.0))
    total = red + blue
    return {"R": red / total, "U": blue / total}


def _build_transition_table(sequence: Sequence[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"R": 0.0, "U": 0.0}
    )
    values = _clean(sequence)
    for index, target in enumerate(values):
        for order in range(1, DERIVED_MAX_ORDER + 1):
            if index < order:
                continue
            context = "".join(values[index - order:index])
            table[f"O{order}|{context}"][target] += 1.0
    return {key: dict(counts) for key, counts in table.items()}


def _posterior(counts: Mapping[str, float]) -> Dict[str, float]:
    """Beta smoothing: P(c|ctx)=(N_c+beta*0.5)/(N+beta)."""
    support = sum(
        float(counts.get(symbol, 0.0) or 0.0)
        for symbol in DERIVED_SYMBOLS
    )
    denominator = support + DERIVED_PRIOR_STRENGTH
    if denominator <= 1e-12:
        return {"R": 0.5, "U": 0.5}
    prior_each = DERIVED_PRIOR_STRENGTH * 0.5
    return {
        symbol: (
            float(counts.get(symbol, 0.0) or 0.0) + prior_each
        ) / denominator
        for symbol in DERIVED_SYMBOLS
    }


def predict_next_derived_mark(sequence: Sequence[Any]) -> Dict[str, Any]:
    """Predict next red/blue mark with support-aware variable-order backoff."""
    values = _clean(sequence)
    table = _build_transition_table(values)
    highest = min(DERIVED_MAX_ORDER, len(values))

    selected_order = 0
    selected_counts = {"R": 0.0, "U": 0.0}
    selected_probability = {"R": 0.5, "U": 0.5}
    penalty = 1.0
    backoff_steps = 0
    contexts: Dict[str, Any] = {}

    for order in range(highest, 0, -1):
        context = "".join(values[-order:])
        key = f"O{order}|{context}"
        counts = dict(table.get(key, {"R": 0.0, "U": 0.0}))
        support = sum(counts.values())
        probabilities = _posterior(counts)
        qualifies = support >= DERIVED_SUPPORT_THRESHOLD
        contexts[f"order_{order}"] = {
            "key": key,
            "support": float(support),
            "support_threshold": int(DERIVED_SUPPORT_THRESHOLD),
            "qualifies": bool(qualifies),
            "probabilities": dict(probabilities),
            "counts": counts,
        }
        if qualifies:
            selected_order = order
            selected_counts = counts
            selected_probability = probabilities
            break
        if order > 1:
            penalty *= DERIVED_BACKOFF_ALPHA
            backoff_steps += 1

    if selected_order == 0:
        if highest >= 1:
            key = f"O1|{values[-1]}"
            selected_counts = dict(table.get(key, {"R": 0.0, "U": 0.0}))
            selected_probability = _posterior(selected_counts)
            selected_order = 1
        else:
            penalty = 0.0

    probabilities = _normalize_pair({
        symbol: (
            (1.0 - penalty) * 0.5
            + penalty * float(selected_probability[symbol])
        )
        for symbol in DERIVED_SYMBOLS
    })

    support = sum(selected_counts.values())
    maturity = min(1.0, len(values) / 12.0)
    support_reliability = (
        support / (support + DERIVED_SUPPORT_THRESHOLD)
        if support > 0.0 else 0.0
    )
    confidence = _clip(maturity * support_reliability * penalty)

    return {
        "probabilities": probabilities,
        "direction": "R" if probabilities["R"] >= probabilities["U"] else "U",
        "selected_order": int(selected_order),
        "support": float(support),
        "support_threshold": int(DERIVED_SUPPORT_THRESHOLD),
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(penalty),
        "confidence": float(confidence),
        "sample_count": len(values),
        "contexts": contexts,
        "semantics": "next_derived_color_probability_not_banker_player_probability",
    }


def score_ask_road_scenarios(
    derived_roads: Mapping[str, Sequence[Any]],
    scenario_marks: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Score hypothetical Banker/Player via standard derived-road Markov.

    log L(side) = weighted mean of log P_r(mark_r(side)); each q_r is the
    confidence of the R/U Markov for that derived road.  The road likelihood
    power is capped because all three roads are transforms of the same Big Road.
    """
    models = {
        name: predict_next_derived_mark(list(derived_roads.get(name) or []))
        for name in ROAD_WEIGHTS
    }

    log_scores = {"B": math.log(0.5), "P": math.log(0.5)}
    weighted_sum = {"B": 0.0, "P": 0.0}
    weight_sum = {"B": 0.0, "P": 0.0}
    scenario_details: Dict[str, Any] = {"B": {}, "P": {}}
    active_roads: set[str] = set()

    for side in ("B", "P"):
        for name, base_weight in ROAD_WEIGHTS.items():
            mark = str(
                (scenario_marks.get(side) or {}).get(name) or ""
            ).upper().strip()
            model = models[name]
            confidence = float(model.get("confidence", 0.0) or 0.0)

            if mark not in DERIVED_SYMBOLS or confidence <= 0.0:
                scenario_details[side][name] = {
                    "mark": mark,
                    "active": False,
                    "reason": "road_not_mature_or_no_mark_emitted",
                }
                continue

            probability = max(
                1e-6,
                float(model["probabilities"].get(mark, 0.5) or 0.5),
            )
            effective_weight = float(base_weight) * confidence
            weighted_sum[side] += effective_weight * math.log(probability)
            weight_sum[side] += effective_weight
            active_roads.add(name)
            scenario_details[side][name] = {
                "mark": mark,
                "active": True,
                "mark_probability": float(probability),
                "model_confidence": confidence,
                "effective_weight": float(effective_weight),
            }

        if weight_sum[side] > 1e-12:
            log_scores[side] = weighted_sum[side] / weight_sum[side]

    banker_score = math.exp(log_scores["B"])
    player_score = math.exp(log_scores["P"])
    total = banker_score + player_score
    if total <= 1e-12:
        likelihood = {"B": 0.5, "P": 0.5}
    else:
        likelihood = {
            "B": banker_score / total,
            "P": player_score / total,
        }

    if active_roads:
        active_weight_total = sum(ROAD_WEIGHTS[name] for name in active_roads)
        mean_confidence = (
            sum(
                ROAD_WEIGHTS[name] * float(models[name]["confidence"])
                for name in active_roads
            ) / max(1e-12, active_weight_total)
        )
    else:
        mean_confidence = 0.0

    active_fraction = len(active_roads) / 3.0
    separation = abs(likelihood["B"] - 0.5) * 2.0
    reliability = min(
        MAX_DERIVED_ROAD_RELIABILITY,
        MAX_DERIVED_ROAD_RELIABILITY
        * mean_confidence
        * active_fraction
        * (0.50 + 0.50 * separation),
    )

    return {
        "likelihood": likelihood,
        "reliability": float(reliability),
        "max_reliability": float(MAX_DERIVED_ROAD_RELIABILITY),
        "active_roads": sorted(active_roads),
        "active_road_count": len(active_roads),
        "models": models,
        "scenario_marks": {
            side: dict(scenario_marks.get(side) or {})
            for side in ("B", "P")
        },
        "scenario_details": scenario_details,
        "log_scores": log_scores,
        "semantics": (
            "derived_road_markov_ask_road_likelihood_capped_due_to_shared_history"
        ),
    }


__all__ = [
    "DERIVED_MAX_ORDER",
    "DERIVED_SUPPORT_THRESHOLD",
    "DERIVED_BACKOFF_ALPHA",
    "MAX_DERIVED_ROAD_RELIABILITY",
    "predict_next_derived_mark",
    "score_ask_road_scenarios",
]
