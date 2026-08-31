"""Road-primary baccarat B/P forecasting for 50-70 hand shoes.

The formal direction is derived only from the observed Big-Road B/P sequence.
No shoe composition, cut-card position, LSTM, LinUCB, HSMM or hazard signal can
vote on B/P here.

The model deliberately uses orientation-invariant evidence: historical patterns
are expressed as SAME/SWITCH relationships, so swapping every B and P in a shoe
swaps the forecast instead of creating a fixed Player/Banker prior.

Components
----------
1. Multi-window transition behaviour over the latest 6/10/16/24 resolved hands.
2. Pattern replay: match recent SAME/SWITCH signatures against earlier prefixes.
3. N-gram backoff, orders 2-5, with recency decay and support shrinkage.
4. Pattern survival for alternating/double/dragon-like run structures.

This is an inspectable sequence model, not evidence that baccarat is
predictable or profitable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math

MODEL_ID = "ROAD-PATTERN-PRIMARY-V1"
VERSION = "ROAD-PATTERN-50-70-V1"
OUTCOMES = ("B", "P")
WINDOW_WEIGHTS = {6: 0.32, 10: 0.28, 16: 0.23, 24: 0.17}
COMPONENT_WEIGHTS = {
    "multi_window": 0.30,
    "pattern_replay": 0.30,
    "ngram": 0.25,
    "pattern_survival": 0.15,
}
NGRAM_ORDERS = (2, 3, 4, 5)
REPLAY_LENGTHS = (3, 4, 5, 6, 8)
RECENCY_DECAY = 0.965
MAX_DIRECTION_EDGE = 0.13
MIN_COMPONENT_RELIABILITY = 0.02


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def normalize_bp(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = "".join(
            ch for ch in history.upper() if not ch.isspace() and ch not in ",|"
        )
        if compact and all(ch in {"B", "P", "T"} for ch in compact):
            return [ch for ch in compact if ch in OUTCOMES][-500:]
        raw_items: Iterable[Any] = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        raw_items = history
    result: list[str] = []
    for item in raw_items:
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
        if value in OUTCOMES:
            result.append(value)
    return result[-500:]


def _opposite(side: str) -> str:
    return "P" if side == "B" else "B"


def _runs(sequence: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for side in sequence:
        if runs and runs[-1][0] == side:
            runs[-1] = (side, runs[-1][1] + 1)
        else:
            runs.append((side, 1))
    return runs


def _relation_signature(values: Sequence[str]) -> tuple[int, ...]:
    """Orientation-invariant transition signature: 1=same, 0=switch."""
    return tuple(1 if left == right else 0 for left, right in zip(values, values[1:]))


def _same_probability_to_b(last_side: str, p_same: float) -> float:
    p_same = _clip(p_same)
    return p_same if last_side == "B" else 1.0 - p_same


def _support_reliability(support: float, *, half_saturation: float) -> float:
    support = max(0.0, float(support))
    return _clip(support / (support + max(1e-9, float(half_saturation))))


def _weighted_same_probability(
    observations: Sequence[tuple[bool, float]],
    *,
    prior_strength: float = 3.0,
) -> tuple[float, float]:
    same = 0.5 * max(0.0, prior_strength)
    switch = 0.5 * max(0.0, prior_strength)
    support = 0.0
    for is_same, weight in observations:
        w = max(0.0, float(weight))
        support += w
        if is_same:
            same += w
        else:
            switch += w
    total = same + switch
    return (same / total if total > 1e-12 else 0.5), support


def _multi_window_component(sequence: Sequence[str]) -> dict[str, Any]:
    if len(sequence) < 2:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "windows": {},
        }
    last = sequence[-1]
    numerator = 0.0
    denominator = 0.0
    diagnostics: dict[str, Any] = {}
    total_support = 0.0
    for size, base_weight in WINDOW_WEIGHTS.items():
        values = list(sequence[-size:])
        transitions = list(zip(values, values[1:]))
        same_count = sum(left == right for left, right in transitions)
        switch_count = len(transitions) - same_count
        support = len(transitions)
        p_same = (same_count + 2.0) / (support + 4.0) if support else 0.5
        reliability = _support_reliability(support, half_saturation=max(3.0, size * 0.45))
        p_b = _same_probability_to_b(last, p_same)
        weight = base_weight * reliability
        numerator += weight * (p_b - 0.5)
        denominator += weight
        total_support += support * base_weight
        diagnostics[str(size)] = {
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_b": float(p_b),
            "p_p": float(1.0 - p_b),
            "support": int(support),
            "reliability": float(reliability),
            "base_weight": float(base_weight),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.38, 0.62)
    reliability = _clip(denominator / max(1e-12, sum(WINDOW_WEIGHTS.values())))
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "windows": diagnostics,
        "semantics": "recent_same_switch_rates_6_10_16_24_orientation_invariant",
    }


def _pattern_replay_component(sequence: Sequence[str]) -> dict[str, Any]:
    n = len(sequence)
    if n < 4:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "matches": {},
        }
    last = sequence[-1]
    numerator = 0.0
    denominator = 0.0
    total_support = 0.0
    details: dict[str, Any] = {}
    for length in REPLAY_LENGTHS:
        if n < length + 1:
            continue
        target_signature = _relation_signature(sequence[-length:])
        observations: list[tuple[bool, float]] = []
        # start+length must be < n so a historical next outcome exists.
        for start in range(0, n - length):
            window = sequence[start : start + length]
            if _relation_signature(window) != target_signature:
                continue
            next_side = sequence[start + length]
            is_same = next_side == window[-1]
            age = (n - 1) - (start + length)
            observations.append((is_same, RECENCY_DECAY ** max(0, age)))
        p_same, support = _weighted_same_probability(observations, prior_strength=3.0)
        reliability = _support_reliability(support, half_saturation=3.5 + 0.4 * length)
        p_b = _same_probability_to_b(last, p_same)
        # Longer signatures are more specific, but only when they have support.
        specificity = 0.55 + 0.45 * (length / max(REPLAY_LENGTHS))
        weight = specificity * reliability
        numerator += weight * (p_b - 0.5)
        denominator += weight
        total_support += support
        details[str(length)] = {
            "signature": "".join("S" if value else "X" for value in target_signature),
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_b": float(p_b),
            "support": float(support),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.37, 0.63)
    reliability = _clip(denominator / max(1.0, len(REPLAY_LENGTHS) * 0.65))
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "matches": details,
        "semantics": "historical_same_switch_signature_replay_not_raw_BP_chasing",
    }


def _ngram_component(sequence: Sequence[str]) -> dict[str, Any]:
    n = len(sequence)
    if n < 3:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0.0,
            "orders": {},
        }
    numerator = 0.0
    denominator = 0.0
    total_support = 0.0
    details: dict[str, Any] = {}
    for order in NGRAM_ORDERS:
        context_len = order - 1
        if n <= context_len:
            continue
        target_signature = _relation_signature(sequence[-order:]) if order > 1 else ()
        # For order k, compare the last k-1 SAME/SWITCH relations to prior
        # contexts, then ask whether the next result stays with the context tail.
        current_context = sequence[-order:]
        current_rel = _relation_signature(current_context)
        observations: list[tuple[bool, float]] = []
        for start in range(0, n - order):
            prior = sequence[start : start + order]
            if _relation_signature(prior) != current_rel:
                continue
            next_side = sequence[start + order]
            is_same = next_side == prior[-1]
            age = (n - 1) - (start + order)
            observations.append((is_same, RECENCY_DECAY ** max(0, age)))
        p_same, support = _weighted_same_probability(observations, prior_strength=3.5)
        reliability = _support_reliability(support, half_saturation=4.0 + order)
        p_b = _same_probability_to_b(sequence[-1], p_same)
        order_weight = (0.70 + 0.30 * order / max(NGRAM_ORDERS)) * reliability
        numerator += order_weight * (p_b - 0.5)
        denominator += order_weight
        total_support += support
        details[str(order)] = {
            "relation_context": "".join("S" if value else "X" for value in target_signature),
            "p_same": float(p_same),
            "p_switch": float(1.0 - p_same),
            "p_b": float(p_b),
            "support": float(support),
            "reliability": float(reliability),
        }
    edge = numerator / denominator if denominator > 1e-12 else 0.0
    p_b = _clip(0.5 + edge, 0.38, 0.62)
    reliability = _clip(denominator / max(1.0, len(NGRAM_ORDERS) * 0.75))
    return {
        "p_b": float(p_b),
        "p_p": float(1.0 - p_b),
        "reliability": float(reliability),
        "support": float(total_support),
        "orders": details,
        "semantics": "orientation_invariant_relation_ngram_orders_2_to_5",
    }


def _completed_run_lengths(sequence: Sequence[str]) -> list[int]:
    runs = _runs(sequence)
    return [length for _, length in runs[:-1]] if len(runs) > 1 else []


def _empirical_run_survival(sequence: Sequence[str], current_length: int) -> tuple[float, int]:
    lengths = _completed_run_lengths(sequence)
    eligible = [length for length in lengths if length >= current_length]
    if not eligible:
        return 0.5, 0
    survived = sum(length > current_length for length in eligible)
    # Beta(2,2) shrinkage keeps tiny in-shoe samples weak.
    return (survived + 2.0) / (len(eligible) + 4.0), len(eligible)


def _pattern_survival_component(sequence: Sequence[str]) -> dict[str, Any]:
    if not sequence:
        return {
            "p_b": 0.5,
            "p_p": 0.5,
            "reliability": 0.0,
            "support": 0,
            "pattern": "COLD_START",
            "survival_probability": 0.5,
        }
    runs = _runs(sequence)
    last_side, current_run = runs[-1]
    recent_lengths = [length for _, length in runs[-5:]]
    pattern = "GENERIC"
    desired_same: bool | None = None
    base_strength = 0.0

    if len(recent_lengths) >= 4 and all(length == 1 for length in recent_lengths[-4:]):
        pattern = "SINGLE_JUMP"
        desired_same = False
        base_strength = 0.70
    elif len(recent_lengths) >= 4 and all(length == 2 for length in recent_lengths[-4:-1]):
        pattern = "DOUBLE_JUMP"
        desired_same = current_run < 2
        base_strength = 0.62
    elif current_run >= 3:
        pattern = "DRAGON"
        desired_same = True
        base_strength = 0.52

    empirical_same, support = _empirical_run_survival(sequence, current_run)
    support_reliability = _support_reliability(support, half_saturation=4.0)

    if desired_same is None:
        p_same = empirical_same
        reliability = 0.35 * support_reliability
    else:
        rule_same = 0.62 if desired_same else 0.38
        # Pattern rules are a weak prior; in-shoe empirical survival takes over
        # only as support accumulates.
        empirical_weight = 0.55 * support_reliability
        p_same = (1.0 - empirical_weight) * rule_same + empirical_weight * empirical_same
        reliability = _clip(base_strength * (0.45 + 0.55 * support_reliability))

    p_b = _same_probability_to_b(last_side, p_same)
    return {
        "p_b": float(_clip(p_b, 0.38, 0.62)),
        "p_p": float(1.0 - _clip(p_b, 0.38, 0.62)),
        "reliability": float(reliability),
        "support": int(support),
        "pattern": pattern,
        "current_run_length": int(current_run),
        "desired_relation": (
            "SAME" if desired_same is True else "SWITCH" if desired_same is False else "EMPIRICAL"
        ),
        "survival_probability": float(p_same),
        "empirical_run_survival": float(empirical_same),
        "semantics": "weak_pattern_prior_blended_with_in_shoe_run_survival",
    }


def forecast_road_pattern(history: str | Iterable[Any] | None) -> dict[str, Any]:
    sequence = normalize_bp(history)
    components = {
        "multi_window": _multi_window_component(sequence),
        "pattern_replay": _pattern_replay_component(sequence),
        "ngram": _ngram_component(sequence),
        "pattern_survival": _pattern_survival_component(sequence),
    }

    numerator = 0.0
    denominator = 0.0
    used: dict[str, Any] = {}
    for name, base_weight in COMPONENT_WEIGHTS.items():
        component = components[name]
        reliability = _clip(component.get("reliability", 0.0))
        if reliability < MIN_COMPONENT_RELIABILITY:
            effective = 0.0
        else:
            effective = base_weight * reliability
        p_b = _clip(component.get("p_b", 0.5))
        numerator += effective * (p_b - 0.5)
        denominator += effective
        used[name] = {
            "base_weight": float(base_weight),
            "reliability": float(reliability),
            "effective_weight": float(effective),
            "p_b": float(p_b),
            "p_p": float(1.0 - p_b),
        }

    raw_edge = numerator / denominator if denominator > 1e-12 else 0.0
    # Overall support maturity prevents the first few hands from producing a
    # large edge. There is no forced follow-last or forced alternation.
    maturity = _clip(len(sequence) / 20.0)
    final_edge = max(-MAX_DIRECTION_EDGE, min(MAX_DIRECTION_EDGE, raw_edge * (0.35 + 0.65 * maturity)))
    p_b = _clip(0.5 + final_edge, 0.37, 0.63)
    p_p = 1.0 - p_b
    direction = "B" if p_b >= p_p else "P"
    confidence = max(p_b, p_p)

    runs = _runs(sequence)
    current_run = runs[-1][1] if runs else 0
    return {
        "model_id": MODEL_ID,
        "version": VERSION,
        "available": True,
        "direction": direction,
        "action": direction,
        "probabilities": {"B": float(p_b), "P": float(p_p), "T": 0.0},
        "confidence": float(confidence),
        "selected_win_probability": float(confidence),
        "margin": float(abs(p_b - p_p)),
        "sequence_length": len(sequence),
        "big_road_sequence": "".join(sequence[-24:]),
        "current_run_length": int(current_run),
        "maturity": float(maturity),
        "raw_edge": float(raw_edge),
        "final_edge": float(final_edge),
        "effective_weight_sum": float(denominator),
        "components": components,
        "component_weights": used,
        "pattern": str(components["pattern_survival"].get("pattern") or "GENERIC"),
        "pattern_survival_score": float(
            components["pattern_survival"].get("survival_probability", 0.5) or 0.5
        ),
        "direction_authority": "road_pattern_core_only",
        "shoe_direction_weight": 0.0,
        "lstm_direction_weight": 0.0,
        "linucb_direction_weight": 0.0,
        "hazard_direction_weight": 0.0,
        "semantics": "road_only_multiwindow_pattern_replay_ngram_pattern_survival",
    }


__all__ = [
    "MODEL_ID",
    "VERSION",
    "WINDOW_WEIGHTS",
    "COMPONENT_WEIGHTS",
    "NGRAM_ORDERS",
    "REPLAY_LENGTHS",
    "normalize_bp",
    "forecast_road_pattern",
]
