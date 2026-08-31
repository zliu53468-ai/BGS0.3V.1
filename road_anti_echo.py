"""Fresh-switch calibration for the Road V1 probability.

V1 models SAME/SWITCH relationships. Immediately after a long run breaks, SAME
inertia learned from the old run can be re-anchored to the single newly observed
opposite result. This module removes that last-hand echo without installing a
mechanical reversal rule.

Only a fresh switch is eligible: current run length == 1 after a prior run >= 2.
The exact V1 output is preserved for diagnostics; this layer sits after V1 and is
orientation symmetric under B<->P mirroring.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

MODEL_VERSION = "FRESH-SWITCH-ANTI-ECHO-V1"
MIN_PREVIOUS_RUN = 2
MIN_CONTEXT_CASES = 2
CONTEXT_SUPPORT_THRESHOLD = 2.0
MAX_ECHO_SHRINK = 0.72
MAX_CONTEXT_RESIDUAL_WEIGHT = 0.10


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


def _clean(sequence: Sequence[Any]) -> list[str]:
    return [
        str(value).upper().strip()
        for value in sequence
        if str(value).upper().strip() in {"B", "P"}
    ][-500:]


def _runs(sequence: Sequence[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for side in sequence:
        if result and result[-1][0] == side:
            result[-1] = (side, result[-1][1] + 1)
        else:
            result.append((side, 1))
    return result


def _length_bucket(length: int) -> str:
    value = max(1, int(length))
    if value <= 2:
        return str(value)
    if value == 3:
        return "3"
    return "4+"


def _fresh_switch_context(
    sequence: Sequence[str],
    previous_run_length: int,
) -> dict[str, Any]:
    """Estimate whether a newly switched side gets a second hand in this shoe.

    Maturity is decided by *case count*, not recency-decayed effective support.
    Recency decay is used only for probability/reliability strength. This avoids
    incorrectly backing off when two genuine comparable cases have decayed to an
    effective weight slightly below 2.0.
    """
    runs = _runs(sequence)
    if len(runs) < 3:
        return {
            "p_new_continue": 0.5,
            "p_old_return": 0.5,
            "support": 0.0,
            "case_count": 0,
            "exact_support": 0.0,
            "exact_case_count": 0,
            "reliability": 0.0,
            "context_tier": "none",
        }

    target_bucket = _length_bucket(previous_run_length)
    exact: list[tuple[bool, float]] = []
    global_obs: list[tuple[bool, float]] = []
    last_historical_index = len(runs) - 2

    # Exclude the current unfinished run. For each completed historical run,
    # final length >=2 means the newly switched side continued at least once.
    for index in range(1, len(runs) - 1):
        prior_length = int(runs[index - 1][1])
        new_length = int(runs[index][1])
        continued = new_length >= 2
        age = max(0, last_historical_index - index)
        weight = 0.94 ** age
        observation = (continued, weight)
        global_obs.append(observation)
        if _length_bucket(prior_length) == target_bucket:
            exact.append(observation)

    exact_support = sum(weight for _, weight in exact)
    exact_case_count = len(exact)
    if exact_case_count >= MIN_CONTEXT_CASES:
        selected = exact
        context_tier = "matched_previous_run_bucket"
        backoff_factor = 1.0
    elif len(global_obs) >= MIN_CONTEXT_CASES:
        selected = global_obs
        context_tier = "global_fresh_switch_backoff"
        backoff_factor = 0.62
    else:
        selected = []
        context_tier = "none"
        backoff_factor = 0.0

    support = sum(weight for _, weight in selected)
    case_count = len(selected)
    continued_weight = sum(weight for continued, weight in selected if continued)
    p_continue = (
        (continued_weight + 2.0) / (support + 4.0)
        if support > 0.0
        else 0.5
    )
    support_reliability = support / (support + 4.0) if support > 0.0 else 0.0
    separation = abs(p_continue - 0.5) * 2.0
    reliability = _clip(
        backoff_factor
        * support_reliability
        * (0.55 + 0.45 * separation)
    )
    return {
        "p_new_continue": float(p_continue),
        "p_old_return": float(1.0 - p_continue),
        "support": float(support),
        "case_count": int(case_count),
        "exact_support": float(exact_support),
        "exact_case_count": int(exact_case_count),
        "minimum_context_cases": int(MIN_CONTEXT_CASES),
        "reliability": float(reliability),
        "context_tier": context_tier,
        "target_previous_run_bucket": target_bucket,
        "semantics": "in_shoe_probability_new_side_gets_second_hand_after_fresh_switch",
    }


def calibrate_fresh_switch(
    sequence: Sequence[Any],
    banker_probability: float,
    *,
    v1_components: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shrink inherited last-side echo; never impose an automatic reversal."""
    values = _clean(sequence)
    base_p_b = _clip(banker_probability, 0.37, 0.63)
    runs = _runs(values)
    if len(runs) < 2:
        return {
            "model_version": MODEL_VERSION,
            "applied": False,
            "fresh_switch": False,
            "base_p_b": float(base_p_b),
            "final_p_b": float(base_p_b),
            "final_p_p": float(1.0 - base_p_b),
            "echo_shrink": 0.0,
            "context_residual_weight": 0.0,
            "reason": "insufficient_runs",
        }

    current_side, current_run = runs[-1]
    previous_side, previous_run = runs[-2]
    fresh_switch = current_run == 1 and previous_run >= MIN_PREVIOUS_RUN
    base_p_current = base_p_b if current_side == "B" else 1.0 - base_p_b
    base_favors_current = base_p_current > 0.5

    if not fresh_switch or not base_favors_current:
        return {
            "model_version": MODEL_VERSION,
            "applied": False,
            "fresh_switch": bool(fresh_switch),
            "current_side": current_side,
            "previous_side": previous_side,
            "current_run_length": int(current_run),
            "previous_run_length": int(previous_run),
            "base_p_b": float(base_p_b),
            "base_p_current_side": float(base_p_current),
            "final_p_b": float(base_p_b),
            "final_p_p": float(1.0 - base_p_b),
            "echo_shrink": 0.0,
            "context_residual_weight": 0.0,
            "reason": (
                "not_fresh_switch"
                if not fresh_switch
                else "base_not_chasing_new_side"
            ),
        }

    context = _fresh_switch_context(values, previous_run)
    context_rel = _clip(context.get("reliability", 0.0))
    p_new_continue = _clip(context.get("p_new_continue", 0.5), 0.20, 0.80)
    context_separation = abs(p_new_continue - 0.5) * 2.0

    previous_run_strength = _clip((previous_run - 1) / 4.0)
    base_current_edge = _clip((base_p_current - 0.5) / 0.13)
    inherited_echo_strength = previous_run_strength * base_current_edge

    contextual_continue_support = (
        context_rel * context_separation
        if p_new_continue >= 0.5
        else 0.0
    )
    echo_shrink = _clip(
        MAX_ECHO_SHRINK
        * inherited_echo_strength
        * (1.0 - 0.75 * contextual_continue_support),
        0.0,
        MAX_ECHO_SHRINK,
    )
    shrunk_logit = _logit(base_p_b) * (1.0 - echo_shrink)
    shrunk_p_b = _clip(_sigmoid(shrunk_logit), 0.37, 0.63)

    context_weight = 0.0
    final_logit = _logit(shrunk_p_b)
    if int(context.get("case_count", 0) or 0) >= MIN_CONTEXT_CASES:
        context_weight = min(
            MAX_CONTEXT_RESIDUAL_WEIGHT,
            MAX_CONTEXT_RESIDUAL_WEIGHT * context_rel * context_separation,
        )
        context_p_b = (
            p_new_continue
            if current_side == "B"
            else 1.0 - p_new_continue
        )
        final_logit += context_weight * _logit(context_p_b)

    final_p_b = _clip(_sigmoid(final_logit), 0.37, 0.63)
    # This is calibration, not an anti-follow betting rule. Material V1 edges
    # cannot be forced to the opposite side solely by Anti-Echo.
    if (
        (base_p_b >= 0.5) != (final_p_b >= 0.5)
        and abs(base_p_b - 0.5) >= 0.025
    ):
        final_p_b = 0.500001 if base_p_b >= 0.5 else 0.499999

    components = dict(v1_components or {})
    return {
        "model_version": MODEL_VERSION,
        "applied": bool(abs(final_p_b - base_p_b) > 1e-12),
        "fresh_switch": True,
        "current_side": current_side,
        "previous_side": previous_side,
        "current_run_length": int(current_run),
        "previous_run_length": int(previous_run),
        "base_p_b": float(base_p_b),
        "base_p_current_side": float(base_p_current),
        "shrunk_p_b": float(shrunk_p_b),
        "final_p_b": float(final_p_b),
        "final_p_p": float(1.0 - final_p_b),
        "echo_shrink": float(echo_shrink),
        "inherited_echo_strength": float(inherited_echo_strength),
        "context_residual_weight": float(context_weight),
        "context": context,
        "v1_component_snapshot": {
            name: {
                "p_b": float(dict(item).get("p_b", 0.5) or 0.5),
                "reliability": float(
                    dict(item).get("reliability", 0.0) or 0.0
                ),
            }
            for name, item in components.items()
            if isinstance(item, Mapping)
        },
        "direction_override": bool(
            (base_p_b >= 0.5) != (final_p_b >= 0.5)
            and abs(base_p_b - 0.5) < 0.025
        ),
        "semantics": "fresh_switch_shrink_old_same_inertia_then_small_in_shoe_context_residual",
    }


__all__ = [
    "MODEL_VERSION",
    "MIN_PREVIOUS_RUN",
    "MIN_CONTEXT_CASES",
    "CONTEXT_SUPPORT_THRESHOLD",
    "MAX_ECHO_SHRINK",
    "MAX_CONTEXT_RESIDUAL_WEIGHT",
    "calibrate_fresh_switch",
]
