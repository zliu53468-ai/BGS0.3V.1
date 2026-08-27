"""Run-length structural hazard model for baccarat Big Road.

This module learns from the *observed Big-Road run structure* instead of assigning
fixed names such as dragon/stair/chop to a road.  Ties do not create a new run.

Two related views are exposed:
1) completed-column height structure: H=[h1,h2,...] and delta-H symbols
   UP / DOWN / EQUAL;
2) next-hand duration hazard for the currently active column:

       h(l, x) = P(TURN next | current run length=l, structural context=x)

Training is leakage-safe within the current shoe: only completed historical runs
are used as duration outcomes.  For a completed run of final length L, every
at-risk length l contributes CONTINUE when l<L and TURN when l=L.

This is a stochastic calibration channel, not a guarantee of a baccarat turn.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence
import math

HAZARD_SUPPORT_THRESHOLD = 4
HAZARD_BACKOFF_ALPHA = 0.75
HAZARD_PRIOR_STRENGTH = 6.0
MAX_HAZARD_RELIABILITY = 0.15
DELTA_MAX_ORDER = 3
DELTA_SUPPORT_THRESHOLD = 3

_EVENTS = ("CONTINUE", "TURN")
_DELTAS = ("UP", "DOWN", "EQUAL")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _clean_bp(history: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in history:
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


def build_runs(history: Iterable[Any]) -> list[tuple[str, int]]:
    values = _clean_bp(history)
    if not values:
        return []
    runs: list[tuple[str, int]] = []
    side = values[0]
    length = 1
    for value in values[1:]:
        if value == side:
            length += 1
        else:
            runs.append((side, length))
            side = value
            length = 1
    runs.append((side, length))
    return runs


def _height_delta(previous: int, current: int) -> str:
    if current > previous:
        return "UP"
    if current < previous:
        return "DOWN"
    return "EQUAL"


def encode_height_deltas(heights: Sequence[int]) -> list[str]:
    values = [max(1, int(value)) for value in heights]
    return [
        _height_delta(previous, current)
        for previous, current in zip(values, values[1:])
    ]


def _length_bucket(length: int) -> str:
    value = max(1, int(length))
    return str(value) if value <= 5 else "6+"


def _previous_structure(previous_heights: Sequence[int]) -> tuple[int, str, str]:
    heights = [max(1, int(value)) for value in previous_heights]
    previous_height = heights[-1] if heights else 0
    deltas = encode_height_deltas(heights)
    delta1 = deltas[-1] if deltas else "NA"
    delta2 = deltas[-2] if len(deltas) >= 2 else "NA"
    return previous_height, delta1, delta2


def _hazard_contexts(
    *,
    side: str,
    current_length: int,
    previous_heights: Sequence[int],
) -> list[tuple[str, str]]:
    previous_height, delta1, delta2 = _previous_structure(previous_heights)
    cur = _length_bucket(current_length)
    prev = _length_bucket(previous_height) if previous_height > 0 else "0"
    side = str(side or "").upper().strip()
    return [
        (
            "full",
            f"HZF|side={side or 'NA'}|cur={cur}|prev={prev}|d1={delta1}|d2={delta2}",
        ),
        (
            "structure",
            f"HZS|cur={cur}|prev={prev}|d1={delta1}|d2={delta2}",
        ),
        ("shape", f"HZP|cur={cur}|prev={prev}|d1={delta1}"),
        ("length", f"HZL|cur={cur}"),
        ("global", "HZG|GLOBAL"),
    ]


def _new_event_counts() -> dict[str, float]:
    return {"CONTINUE": 0.0, "TURN": 0.0}


def _build_hazard_table(runs: Sequence[tuple[str, int]]) -> dict[str, dict[str, float]]:
    """Build hazard counts from completed runs only.

    The final run in ``runs`` is the currently active column and is deliberately
    excluded because its terminal length is not known yet.
    """
    completed = list(runs[:-1])
    table: dict[str, dict[str, float]] = defaultdict(_new_event_counts)
    completed_heights = [length for _, length in completed]

    for run_index, (side, final_length) in enumerate(completed):
        previous_heights = completed_heights[:run_index]
        final_length = max(1, int(final_length))
        for at_risk_length in range(1, final_length + 1):
            event = "CONTINUE" if at_risk_length < final_length else "TURN"
            for _, key in _hazard_contexts(
                side=side,
                current_length=at_risk_length,
                previous_heights=previous_heights,
            ):
                table[key][event] += 1.0

    return {key: dict(counts) for key, counts in table.items()}


def _event_posterior(counts: Mapping[str, float]) -> dict[str, float]:
    support = sum(float(counts.get(event, 0.0) or 0.0) for event in _EVENTS)
    denominator = support + HAZARD_PRIOR_STRENGTH
    prior_each = HAZARD_PRIOR_STRENGTH * 0.5
    if denominator <= 1e-12:
        return {"CONTINUE": 0.5, "TURN": 0.5}
    return {
        event: (float(counts.get(event, 0.0) or 0.0) + prior_each) / denominator
        for event in _EVENTS
    }


def _delta_markov(heights: Sequence[int]) -> dict[str, Any]:
    """Diagnostic 1..3 order Markov over completed-column height changes."""
    deltas = encode_height_deltas(heights)
    table: dict[str, dict[str, float]] = defaultdict(
        lambda: {symbol: 0.0 for symbol in _DELTAS}
    )
    for index, target in enumerate(deltas):
        for order in range(1, DELTA_MAX_ORDER + 1):
            if index < order:
                continue
            context = ">".join(deltas[index - order:index])
            table[f"D{order}|{context}"][target] += 1.0

    selected_order = 0
    selected_counts = {symbol: 0.0 for symbol in _DELTAS}
    penalty = 1.0
    backoff_steps = 0
    for order in range(min(DELTA_MAX_ORDER, len(deltas)), 0, -1):
        context = ">".join(deltas[-order:])
        counts = dict(table.get(f"D{order}|{context}", selected_counts))
        support = sum(counts.values())
        if support >= DELTA_SUPPORT_THRESHOLD:
            selected_order = order
            selected_counts = counts
            break
        if order > 1:
            penalty *= HAZARD_BACKOFF_ALPHA
            backoff_steps += 1

    support = sum(selected_counts.values())
    prior_strength = 3.0
    denominator = support + prior_strength
    raw = {
        symbol: (selected_counts[symbol] + prior_strength / 3.0) / denominator
        if denominator > 0.0 else 1.0 / 3.0
        for symbol in _DELTAS
    }
    probability = {
        symbol: (1.0 - penalty) / 3.0 + penalty * raw[symbol]
        for symbol in _DELTAS
    }
    total = sum(probability.values())
    probability = {symbol: probability[symbol] / total for symbol in _DELTAS}
    return {
        "height_deltas": deltas,
        "selected_order": int(selected_order),
        "support": float(support),
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(penalty),
        "probabilities": probability,
        "direction": max(probability, key=probability.get) if probability else "EQUAL",
        "semantics": "next_completed_column_height_relation_not_next_hand_bp_probability",
    }


def analyze_run_length_hazard(history: Iterable[Any]) -> dict[str, Any]:
    runs = build_runs(history)
    if not runs:
        return {
            "available": False,
            "likelihood": {"B": 0.5, "P": 0.5},
            "reliability": 0.0,
            "max_reliability": float(MAX_HAZARD_RELIABILITY),
            "reason": "no_big_road_runs",
        }

    current_side, current_length = runs[-1]
    completed_runs = runs[:-1]
    completed_heights = [int(length) for _, length in completed_runs]
    table = _build_hazard_table(runs)
    current_contexts = _hazard_contexts(
        side=current_side,
        current_length=current_length,
        previous_heights=completed_heights,
    )

    selected_tier = "prior"
    selected_key = ""
    selected_counts = _new_event_counts()
    selected_probability = {"CONTINUE": 0.5, "TURN": 0.5}
    penalty = 1.0
    backoff_steps = 0
    context_diagnostics: dict[str, Any] = {}

    for tier_index, (tier, key) in enumerate(current_contexts):
        counts = dict(table.get(key, _new_event_counts()))
        support = sum(counts.values())
        posterior = _event_posterior(counts)
        qualifies = support >= HAZARD_SUPPORT_THRESHOLD
        context_diagnostics[tier] = {
            "key": key,
            "support": float(support),
            "support_threshold": int(HAZARD_SUPPORT_THRESHOLD),
            "qualifies": bool(qualifies),
            "counts": counts,
            "posterior": posterior,
        }
        if qualifies:
            selected_tier = tier
            selected_key = key
            selected_counts = counts
            selected_probability = posterior
            break
        if tier_index < len(current_contexts) - 1:
            penalty *= HAZARD_BACKOFF_ALPHA
            backoff_steps += 1

    # Global is the final data-driven fallback even when its support is below K.
    if selected_tier == "prior":
        global_counts = dict(table.get("HZG|GLOBAL", _new_event_counts()))
        global_support = sum(global_counts.values())
        if global_support > 0.0:
            selected_tier = "global_fallback"
            selected_key = "HZG|GLOBAL"
            selected_counts = global_counts
            selected_probability = _event_posterior(global_counts)
        else:
            penalty = 0.0

    continue_probability = (
        (1.0 - penalty) * 0.5
        + penalty * float(selected_probability["CONTINUE"])
    )
    turn_probability = 1.0 - continue_probability

    # Reliability is based only on observed structural support and maturity.
    support = sum(selected_counts.values())
    support_factor = (
        support / (support + HAZARD_SUPPORT_THRESHOLD)
        if support > 0.0 else 0.0
    )
    maturity = min(1.0, len(completed_runs) / 8.0)
    separation = abs(continue_probability - turn_probability)
    reliability = min(
        MAX_HAZARD_RELIABILITY,
        MAX_HAZARD_RELIABILITY
        * support_factor
        * maturity
        * penalty
        * (0.65 + 0.35 * separation),
    )

    if current_side == "B":
        likelihood = {"B": continue_probability, "P": turn_probability}
    else:
        likelihood = {"P": continue_probability, "B": turn_probability}

    all_heights = [int(length) for _, length in runs]
    return {
        "available": bool(support > 0.0 and len(completed_runs) > 0),
        "current_side": current_side,
        "current_run_length": int(current_length),
        "run_lengths": all_heights,
        "completed_run_lengths": completed_heights,
        "height_deltas": encode_height_deltas(completed_heights),
        "continue_probability": float(continue_probability),
        "turn_probability": float(turn_probability),
        "likelihood": {
            "B": float(likelihood["B"]),
            "P": float(likelihood["P"]),
        },
        "selected_context_tier": selected_tier,
        "selected_context": selected_key,
        "support": float(support),
        "support_threshold": int(HAZARD_SUPPORT_THRESHOLD),
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(penalty),
        "reliability": float(reliability),
        "max_reliability": float(MAX_HAZARD_RELIABILITY),
        "context_diagnostics": context_diagnostics,
        "height_delta_markov": _delta_markov(completed_heights),
        "turn_pressure": float(turn_probability - 0.5),
        "semantics": (
            "run_length_duration_hazard_from_completed_big_road_columns_"
            "not_deterministic_turn_probability"
        ),
    }


__all__ = [
    "HAZARD_SUPPORT_THRESHOLD",
    "HAZARD_BACKOFF_ALPHA",
    "MAX_HAZARD_RELIABILITY",
    "build_runs",
    "encode_height_deltas",
    "analyze_run_length_hazard",
]
