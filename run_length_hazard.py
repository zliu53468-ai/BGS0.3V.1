"""Run-length structural hazard model for baccarat Big Road.

This module learns from the observed Big-Road run structure instead of assigning
fixed labels to a road. Ties do not create a new run.

The production contract is calibration-only:
- ``analyze_run_length_hazard`` estimates continue/turn probabilities from
  completed columns;
- ``lstm_hazard_confidence_calibration`` converts unusually high turn pressure
  into a *small* confidence reduction for the LSTM layer;
- neither function may create or overwrite a B/P direction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

HAZARD_SUPPORT_THRESHOLD = 4
HAZARD_BACKOFF_ALPHA = 0.75
HAZARD_PRIOR_STRENGTH = 6.0
MAX_HAZARD_RELIABILITY = 0.15
DELTA_MAX_ORDER = 3
DELTA_SUPPORT_THRESHOLD = 3
LSTM_HAZARD_MIN_FACTOR = 0.92
LSTM_HAZARD_PRESSURE_START = 0.56
LSTM_HAZARD_PRESSURE_FULL = 0.78

_EVENTS = ("CONTINUE", "TURN")
_DELTAS = ("UP", "DOWN", "EQUAL")


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _clean_bp(history: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in history:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
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
            side, length = value, 1
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
    return [_height_delta(previous, current) for previous, current in zip(values, values[1:])]


def _length_bucket(length: int) -> str:
    value = max(1, int(length))
    return str(value) if value <= 5 else "6+"


def _previous_structure(previous_heights: Sequence[int]) -> tuple[int, str, str]:
    heights = [max(1, int(value)) for value in previous_heights]
    previous_height = heights[-1] if heights else 0
    deltas = encode_height_deltas(heights)
    return previous_height, deltas[-1] if deltas else "NA", deltas[-2] if len(deltas) >= 2 else "NA"


def _hazard_contexts(*, side: str, current_length: int, previous_heights: Sequence[int]) -> list[tuple[str, str]]:
    previous_height, delta1, delta2 = _previous_structure(previous_heights)
    cur = _length_bucket(current_length)
    prev = _length_bucket(previous_height) if previous_height > 0 else "0"
    side = str(side or "").upper().strip()
    return [
        ("full", f"HZF|side={side or 'NA'}|cur={cur}|prev={prev}|d1={delta1}|d2={delta2}"),
        ("structure", f"HZS|cur={cur}|prev={prev}|d1={delta1}|d2={delta2}"),
        ("shape", f"HZP|cur={cur}|prev={prev}|d1={delta1}"),
        ("length", f"HZL|cur={cur}"),
        ("global", "HZG|GLOBAL"),
    ]


def _new_event_counts() -> dict[str, float]:
    return {"CONTINUE": 0.0, "TURN": 0.0}


def _build_hazard_table(runs: Sequence[tuple[str, int]]) -> dict[str, dict[str, float]]:
    completed = list(runs[:-1])
    table: dict[str, dict[str, float]] = defaultdict(_new_event_counts)
    completed_heights = [length for _, length in completed]
    for run_index, (side, final_length) in enumerate(completed):
        previous_heights = completed_heights[:run_index]
        for at_risk_length in range(1, max(1, int(final_length)) + 1):
            event = "CONTINUE" if at_risk_length < final_length else "TURN"
            for _, key in _hazard_contexts(side=side, current_length=at_risk_length, previous_heights=previous_heights):
                table[key][event] += 1.0
    return {key: dict(counts) for key, counts in table.items()}


def _event_posterior(counts: Mapping[str, float]) -> dict[str, float]:
    support = sum(float(counts.get(event, 0.0) or 0.0) for event in _EVENTS)
    denominator = support + HAZARD_PRIOR_STRENGTH
    prior_each = HAZARD_PRIOR_STRENGTH * 0.5
    if denominator <= 1e-12:
        return {"CONTINUE": 0.5, "TURN": 0.5}
    return {event: (float(counts.get(event, 0.0) or 0.0) + prior_each) / denominator for event in _EVENTS}


def _delta_markov(heights: Sequence[int]) -> dict[str, Any]:
    deltas = encode_height_deltas(heights)
    table: dict[str, dict[str, float]] = defaultdict(lambda: {symbol: 0.0 for symbol in _DELTAS})
    for index, target in enumerate(deltas):
        for order in range(1, DELTA_MAX_ORDER + 1):
            if index >= order:
                table[f"D{order}|{'>'.join(deltas[index-order:index])}"][target] += 1.0
    selected_order = 0
    selected_counts = {symbol: 0.0 for symbol in _DELTAS}
    penalty, backoff_steps = 1.0, 0
    for order in range(min(DELTA_MAX_ORDER, len(deltas)), 0, -1):
        counts = dict(table.get(f"D{order}|{'>'.join(deltas[-order:])}", selected_counts))
        support = sum(counts.values())
        if support >= DELTA_SUPPORT_THRESHOLD:
            selected_order, selected_counts = order, counts
            break
        if order > 1:
            penalty *= HAZARD_BACKOFF_ALPHA
            backoff_steps += 1
    support = sum(selected_counts.values())
    prior_strength = 3.0
    denominator = support + prior_strength
    raw = {symbol: (selected_counts[symbol] + prior_strength / 3.0) / denominator if denominator > 0.0 else 1.0 / 3.0 for symbol in _DELTAS}
    probability = {symbol: (1.0 - penalty) / 3.0 + penalty * raw[symbol] for symbol in _DELTAS}
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
        return {"available": False, "likelihood": {"B": 0.5, "P": 0.5}, "continue_probability": 0.5, "turn_probability": 0.5, "reliability": 0.0, "max_reliability": float(MAX_HAZARD_RELIABILITY), "turn_pressure": 0.0, "reason": "no_big_road_runs"}
    current_side, current_length = runs[-1]
    completed_runs = runs[:-1]
    completed_heights = [int(length) for _, length in completed_runs]
    table = _build_hazard_table(runs)
    current_contexts = _hazard_contexts(side=current_side, current_length=current_length, previous_heights=completed_heights)
    selected_tier, selected_key = "prior", ""
    selected_counts = _new_event_counts()
    selected_probability = {"CONTINUE": 0.5, "TURN": 0.5}
    penalty, backoff_steps = 1.0, 0
    context_diagnostics: dict[str, Any] = {}
    for tier_index, (tier, key) in enumerate(current_contexts):
        counts = dict(table.get(key, _new_event_counts()))
        support = sum(counts.values())
        posterior = _event_posterior(counts)
        qualifies = support >= HAZARD_SUPPORT_THRESHOLD
        context_diagnostics[tier] = {"key": key, "support": float(support), "support_threshold": int(HAZARD_SUPPORT_THRESHOLD), "qualifies": bool(qualifies), "counts": counts, "posterior": posterior}
        if qualifies:
            selected_tier, selected_key, selected_counts, selected_probability = tier, key, counts, posterior
            break
        if tier_index < len(current_contexts) - 1:
            penalty *= HAZARD_BACKOFF_ALPHA
            backoff_steps += 1
    if selected_tier == "prior":
        global_counts = dict(table.get("HZG|GLOBAL", _new_event_counts()))
        if sum(global_counts.values()) > 0.0:
            selected_tier, selected_key, selected_counts, selected_probability = "global_fallback", "HZG|GLOBAL", global_counts, _event_posterior(global_counts)
        else:
            penalty = 0.0
    continue_probability = (1.0 - penalty) * 0.5 + penalty * float(selected_probability["CONTINUE"])
    turn_probability = 1.0 - continue_probability
    support = sum(selected_counts.values())
    support_factor = support / (support + HAZARD_SUPPORT_THRESHOLD) if support > 0.0 else 0.0
    maturity = min(1.0, len(completed_runs) / 8.0)
    separation = abs(continue_probability - turn_probability)
    reliability = min(MAX_HAZARD_RELIABILITY, MAX_HAZARD_RELIABILITY * support_factor * maturity * penalty * (0.65 + 0.35 * separation))
    likelihood = {current_side: continue_probability, "P" if current_side == "B" else "B": turn_probability}
    return {
        "available": bool(support > 0.0 and completed_runs),
        "current_side": current_side,
        "current_run_length": int(current_length),
        "run_lengths": [int(length) for _, length in runs],
        "completed_run_lengths": completed_heights,
        "height_deltas": encode_height_deltas(completed_heights),
        "continue_probability": float(continue_probability),
        "turn_probability": float(turn_probability),
        "likelihood": {"B": float(likelihood["B"]), "P": float(likelihood["P"])},
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
        "semantics": "run_length_duration_hazard_from_completed_big_road_columns_not_deterministic_turn_probability",
    }


def lstm_hazard_confidence_calibration(history: Iterable[Any] | None = None, *, analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Soft one-way LSTM confidence factor; never changes direction."""
    hazard = dict(analysis or analyze_run_length_hazard(history or []))
    turn_probability = _clip(float(hazard.get("turn_probability", 0.5) or 0.5))
    reliability = _clip(float(hazard.get("reliability", 0.0) or 0.0))
    max_reliability = max(1e-9, float(hazard.get("max_reliability", MAX_HAZARD_RELIABILITY) or MAX_HAZARD_RELIABILITY))
    support_ratio = _clip(reliability / max_reliability)
    pressure = _clip((turn_probability - LSTM_HAZARD_PRESSURE_START) / max(1e-9, LSTM_HAZARD_PRESSURE_FULL - LSTM_HAZARD_PRESSURE_START))
    penalty = (1.0 - LSTM_HAZARD_MIN_FACTOR) * pressure * (0.35 + 0.65 * support_ratio)
    factor = _clip(1.0 - penalty, LSTM_HAZARD_MIN_FACTOR, 1.0)
    return {
        "applied": bool(factor < 0.999999),
        "confidence_factor": float(factor),
        "penalty": float(1.0 - factor),
        "turn_probability": float(turn_probability),
        "turn_pressure": float(turn_probability - 0.5),
        "hazard_reliability": float(reliability),
        "hazard_support_ratio": float(support_ratio),
        "min_factor": float(LSTM_HAZARD_MIN_FACTOR),
        "direction_override": False,
        "semantics": "soft_transition_margin_downweight_only_no_direction_vote",
    }


__all__ = ["HAZARD_SUPPORT_THRESHOLD", "HAZARD_BACKOFF_ALPHA", "MAX_HAZARD_RELIABILITY", "LSTM_HAZARD_MIN_FACTOR", "build_runs", "encode_height_deltas", "analyze_run_length_hazard", "lstm_hazard_confidence_calibration"]
