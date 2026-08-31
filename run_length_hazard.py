"""Run-length structural hazard model for baccarat Big Road.

This module learns from the *observed Big-Road run structure* instead of assigning
fixed names such as dragon/stair/chop to a road. Ties do not create a new run.

Two related views are exposed:
1) completed-column height structure: H=[h1,h2,...] and delta-H symbols
   UP / DOWN / EQUAL;
2) next-hand duration hazard for the currently active column:

       h(l, x) = P(TURN next | current run length=l, structural context=x)

Training is leakage-safe within the current shoe: only completed historical runs
are used as duration outcomes. For a completed run of final length L, every
at-risk length l contributes CONTINUE when l<L and TURN when l=L.

The raw context posterior is blended with a local length-aware hazard prior around
the historical break-length region. This keeps turn probability continuous near
critical run lengths while preserving the original context/backoff model.

This is a stochastic calibration channel, not a guarantee of a baccarat turn.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence
import math

HAZARD_SUPPORT_THRESHOLD = 4
HAZARD_TRANSITION_SUPPORT_BOOST = 1
HAZARD_BACKOFF_ALPHA = 0.88
HAZARD_PRIOR_STRENGTH = 6.0
MAX_HAZARD_RELIABILITY = 0.25
DELTA_MAX_ORDER = 3
DELTA_SUPPORT_THRESHOLD = 3

LENGTH_PRIOR_STRENGTH = 4.0
LENGTH_SMOOTH_BLEND_MIN = 0.24
LENGTH_SMOOTH_BLEND_MAX = 0.48
TRANSITION_STABILITY_WINDOW = 5

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


def _transition_stability(heights: Sequence[int]) -> float:
    """Return 0..1 convergence of recent completed-run height changes."""
    values = [
        max(1, int(value))
        for value in list(heights)[-TRANSITION_STABILITY_WINDOW:]
    ]
    if len(values) < 3:
        return 0.50

    diffs = [
        abs(right - left)
        for left, right in zip(values, values[1:])
    ]
    if len(diffs) < 2:
        return 0.50

    recent_count = min(2, len(diffs))
    recent = diffs[-recent_count:]
    earlier = diffs[:-recent_count] or diffs[:1]
    recent_mean = sum(recent) / len(recent)
    earlier_mean = sum(earlier) / len(earlier)

    convergence = _clip(
        0.50
        + (earlier_mean - recent_mean)
        / max(1.0, earlier_mean + recent_mean)
    )
    recent_spread = max(recent) - min(recent)
    dispersion_stability = 1.0 - _clip(
        recent_spread / max(1.0, max(recent) if recent else 1.0)
    )
    low_motion = 1.0 - _clip(recent_mean / 4.0)

    return _clip(
        0.45 * convergence
        + 0.35 * dispersion_stability
        + 0.20 * low_motion
    )


def _critical_length_reference(
    completed_heights: Sequence[int],
    current_length: int,
) -> dict[str, float]:
    """Estimate the historical break region without creating a hard threshold."""
    recent = [
        max(1, int(value))
        for value in list(completed_heights)[-12:]
    ]
    if not recent:
        return {
            "critical_run_length": float(max(1, current_length)),
            "critical_scale": 1.25,
            "critical_proximity": 0.0,
        }

    ordered = sorted(recent)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = float(ordered[midpoint])
    else:
        median = 0.5 * float(ordered[midpoint - 1] + ordered[midpoint])
    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)
    spread = math.sqrt(max(0.0, variance))

    critical = 0.55 * median + 0.45 * mean
    scale = max(1.50, min(3.50, spread if spread > 1e-9 else 1.50))
    proximity = math.exp(
        -abs(float(current_length) - critical) / max(1e-9, scale)
    )
    return {
        "critical_run_length": float(critical),
        "critical_scale": float(scale),
        "critical_proximity": float(_clip(proximity)),
    }


def _logistic_turn_prior(
    length: int,
    *,
    critical_length: float,
    scale: float,
) -> float:
    """Smooth length-aware prior bounded away from 0/1."""
    z = (
        float(max(1, int(length))) - float(critical_length)
    ) / max(0.90, float(scale))
    z = max(-12.0, min(12.0, z))
    logistic = 1.0 / (1.0 + math.exp(-z))
    return _clip(0.20 + 0.60 * logistic)


def _smoothed_length_hazard(
    completed_heights: Sequence[int],
    current_length: int,
    critical: Mapping[str, float],
) -> dict[str, float]:
    """Kernel-smooth empirical hazard across adjacent at-risk lengths."""
    heights = [max(1, int(value)) for value in completed_heights]
    if not heights:
        return {
            "turn_probability": 0.5,
            "support": 0.0,
            "length_prior": 0.5,
        }

    critical_length = float(
        critical.get("critical_run_length", current_length)
    )
    scale = float(critical.get("critical_scale", 1.25))
    current_prior = _logistic_turn_prior(
        current_length,
        critical_length=critical_length,
        scale=scale,
    )

    numer = 0.0
    denom = 0.0
    for offset, kernel_weight in (
        (-2, 0.55), (-1, 1.0), (0, 1.80), (1, 1.0), (2, 0.55)
    ):
        at_risk = max(1, int(current_length) + offset)
        risk = sum(1 for final_length in heights if final_length >= at_risk)
        turns = sum(1 for final_length in heights if final_length == at_risk)
        prior = _logistic_turn_prior(
            at_risk,
            critical_length=critical_length,
            scale=scale,
        )
        posterior = (
            turns + LENGTH_PRIOR_STRENGTH * prior
        ) / max(1e-9, risk + LENGTH_PRIOR_STRENGTH)
        support_weight = 0.45 + 0.55 * (
            risk / max(1.0, risk + LENGTH_PRIOR_STRENGTH)
        )
        weight = kernel_weight * support_weight
        numer += weight * posterior
        denom += weight

    smoothed = numer / denom if denom > 1e-12 else current_prior
    current_support = sum(
        1 for final_length in heights if final_length >= current_length
    )
    return {
        "turn_probability": float(_clip(smoothed)),
        "support": float(current_support),
        "length_prior": float(current_prior),
    }


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
    """Build hazard counts from completed runs only."""
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
            "raw_turn_probability": 0.5,
            "smoothed_turn_probability": 0.5,
            "turn_probability": 0.5,
            "turn_pressure": 0.0,
            "critical_proximity": 0.0,
            "transition_stability": 0.5,
            "reason": "no_big_road_runs",
        }

    current_side, current_length = runs[-1]
    completed_runs = runs[:-1]
    completed_heights = [int(length) for _, length in completed_runs]
    transition_stability = _transition_stability(completed_heights)
    critical = _critical_length_reference(completed_heights, current_length)
    critical_proximity = _clip(
        float(critical.get("critical_proximity", 0.0) or 0.0)
    )

    support_boost = int(
        round(
            HAZARD_TRANSITION_SUPPORT_BOOST
            * (1.0 - transition_stability)
        )
    )
    if critical_proximity >= 0.75:
        support_boost += 1
    effective_support_threshold = max(
        HAZARD_SUPPORT_THRESHOLD,
        min(
            HAZARD_SUPPORT_THRESHOLD + HAZARD_TRANSITION_SUPPORT_BOOST + 1,
            HAZARD_SUPPORT_THRESHOLD + support_boost,
        ),
    )

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
        qualifies = support >= effective_support_threshold
        context_diagnostics[tier] = {
            "key": key,
            "support": float(support),
            "support_threshold": int(effective_support_threshold),
            "base_support_threshold": int(HAZARD_SUPPORT_THRESHOLD),
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

    # Backoff now changes reliability, not the probability itself. Sparse
    # contexts are blended with the global parent below instead of being pulled
    # directly toward 0.5, which preserves useful transition structure.
    raw_continue_probability = float(selected_probability["CONTINUE"])
    raw_turn_probability = float(selected_probability["TURN"])

    # Hierarchical parent smoothing keeps a sparse context from taking control
    # in a single step.  The global completed-run hazard is the parent; more
    # specific contexts earn more weight only as their at-risk support grows.
    global_counts_for_smoothing = dict(table.get("HZG|GLOBAL", _new_event_counts()))
    global_support_for_smoothing = sum(global_counts_for_smoothing.values())
    global_turn_probability = float(
        _event_posterior(global_counts_for_smoothing)["TURN"]
    ) if global_support_for_smoothing > 0.0 else 0.5

    support = sum(selected_counts.values())
    if selected_tier in {"global", "global_fallback", "prior"}:
        context_specificity_weight = 1.0
    else:
        context_specificity_weight = _clip(
            support / max(1e-9, support + 1.50 * effective_support_threshold),
            0.25,
            0.72,
        )
    context_smoothed_turn_probability = _clip(
        context_specificity_weight * raw_turn_probability
        + (1.0 - context_specificity_weight) * global_turn_probability
    )

    length_hazard = _smoothed_length_hazard(
        completed_heights,
        current_length,
        critical,
    )

    support_factor = (
        support / (support + 0.75 * effective_support_threshold)
        if support > 0.0 else 0.0
    )
    length_blend = _clip(
        LENGTH_SMOOTH_BLEND_MIN
        + 0.14 * critical_proximity
        + 0.06 * (1.0 - support_factor)
        + 0.03 * (1.0 - transition_stability),
        LENGTH_SMOOTH_BLEND_MIN,
        LENGTH_SMOOTH_BLEND_MAX,
    )
    smoothed_turn_probability = _clip(
        (1.0 - length_blend) * context_smoothed_turn_probability
        + length_blend * float(length_hazard["turn_probability"])
    )
    turn_probability = smoothed_turn_probability
    continue_probability = 1.0 - turn_probability

    maturity = min(1.0, len(completed_runs) / 8.0)
    separation = abs(continue_probability - turn_probability)
    stability_factor = 0.90 + 0.10 * transition_stability
    backoff_reliability = 0.85 + 0.15 * penalty
    reliability = min(
        MAX_HAZARD_RELIABILITY,
        MAX_HAZARD_RELIABILITY
        * support_factor
        * maturity
        * backoff_reliability
        * stability_factor
        * (0.75 + 0.25 * separation),
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
        "raw_turn_probability": float(raw_turn_probability),
        "context_smoothed_turn_probability": float(context_smoothed_turn_probability),
        "global_parent_turn_probability": float(global_turn_probability),
        "context_specificity_weight": float(context_specificity_weight),
        "smoothed_turn_probability": float(smoothed_turn_probability),
        "length_aware_turn_probability": float(length_hazard["turn_probability"]),
        "length_aware_turn_prior": float(length_hazard["length_prior"]),
        "length_aware_support": float(length_hazard["support"]),
        "critical_run_length": float(critical["critical_run_length"]),
        "critical_scale": float(critical["critical_scale"]),
        "critical_proximity": float(critical_proximity),
        "transition_stability": float(transition_stability),
        "length_smoothing_weight": float(length_blend),
        "likelihood": {
            "B": float(likelihood["B"]),
            "P": float(likelihood["P"]),
        },
        "selected_context_tier": selected_tier,
        "selected_context": selected_key,
        "support": float(support),
        "support_threshold": int(effective_support_threshold),
        "base_support_threshold": int(HAZARD_SUPPORT_THRESHOLD),
        "backoff_steps": int(backoff_steps),
        "backoff_penalty": float(penalty),
        "backoff_reliability_factor": float(backoff_reliability),
        "reliability": float(reliability),
        "max_reliability": float(MAX_HAZARD_RELIABILITY),
        "context_diagnostics": context_diagnostics,
        "height_delta_markov": _delta_markov(completed_heights),
        "turn_pressure": float(turn_probability - 0.5),
        "semantics": (
            "smoothed_run_length_duration_hazard_from_completed_big_road_columns_"
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
