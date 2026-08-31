"""Fresh-switch calibration for the Road V1 probability.

The V1 core deliberately models SAME/SWITCH relationships.  A subtle failure
mode appears immediately after a long run breaks: a high SAME rate learned from
the *old* run can be re-anchored to the one newly observed opposite result.  That
looks like "see Banker -> Banker / see Player -> Player" even though no explicit
follow-last rule exists.

This module does not predict the opposite side by rule.  It only:
1. detects a fresh switch (current run length == 1 after a prior run >= 2);
2. shrinks inherited last-side momentum toward 50/50 when there is little
   in-shoe evidence for what usually happens after a comparable switch;
3. optionally applies a very small context residual when the same shoe contains
   enough completed, comparable fresh-switch examples.

The original V1 result is preserved in diagnostics and this layer is orientation
symmetric under B<->P mirroring.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

MODEL_VERSION = "FRESH-SWITCH-ANTI-ECHO-V1"
MIN_PREVIOUS_RUN = 2
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
    runs: list[tuple[str, int]] = []
    for side in sequence:
        if runs and runs[-1][0] == side:
            runs[-1] = (side, runs[-1][1] + 1)
        else:
            runs.append((side, 1))
    return runs


def _length_bucket(length: int) -> str:
    value = max(1, int(length))
    if value <= 2:
        return str(value)
    if value == 3:
        return "3"
    return "4+"


def _fresh_switch_context(sequence: Sequence[str], previous_run_length: int) -> dict[str, Any]:
    """Estimate whether a newly switched side gets a second hand in this shoe.

    A completed new run of length >=2 means NEW_CONTINUE; length ==1 means the
    table immediately returned to the previous side.  Exact previous-run-length
    buckets are preferred; low support backs off to all historical switches with
    a reliability penalty.  Beta(2,2) shrinkage keeps tiny samples weak.
    """
    runs = _runs(sequence)
    if len(runs) < 3:
        return {
            "p_new_continue": 0.5,
            "support": 0.0,
            "exact_support": 0.0,
            "reliability": 0.0,
            "context_tier": "none",
        }

    # Exclude the current unfinished fresh run. Historical new runs are indexes
    # 1 .. len(runs)-2, because their final length is already known.
    target_bucket = _length_bucket(previous_run_length)
    exact: list[tuple[bool, float]] = []
    global_obs: list[tuple[bool, float]] = []
    last_historical_index = len(runs) - 2
    for index in range(1, len(runs) - 1):
        prior_length = int(runs[index - 1][1])
        new_length = int(runs[index][1])
        continued = new_length >= 2
        age = max(0, last_historical_index - index)
        weight = 0.94 ** age
        global_obs.append((continued, weight))
        if _length_bucket(prior_length) == target_bucket:
            exact.append((continued, weight))

    exact_support = sum(weight for _, weight in exact)
    if exact_support >= CONTEXT_SUPPORT_THRESHOLD:
        selected = exact
        context_tier = "matched_previous_run_bucket"
        backoff_factor = 1.0
    else:
        selected = global_obs
        context_tier = "global_fresh_switch_backoff" if selected else "none"
        backoff_factor = 0.62 if selected else 0.0

    support = sum(weight for _, weight in selected)
    continued_weight = sum(weight for continued, weight in selected if continued)
    p_continue = (continued_weight + 2.0) / (support + 4.0) if support > 0.0 else 0.5
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
        "exact_support": float(exact_support),
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
    """Remove inherited last-hand echo without installing an opposite-side rule."""
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
            "reason": "not_fresh_switch" if not fresh_switch else "base_not_chasing_new_side",
        }

    context = _fresh_switch_context(values, previous_run)
    context_rel = _clip(context.get("reliability", 0.0))
    p_new_continue = _clip(context.get("p_new_continue", 0.5), 0.20, 0.80)
    context_separation = abs(p_new_continue - 0.5) * 2.0

    # The longer the run that just broke, and the stronger V1 is already leaning
    # toward the single new result, the more suspicious the inherited echo is.
    previous_run_strength = _clip((previous_run - 1) / 4.0)
    base_current_edge = _clip((base_p_current - 0.5) / 0.13)
    inherited_echo_strength = previous_run_strength * base_current_edge

    # Reliable same-shoe context that genuinely supports the new side should
    # preserve more of the V1 edge. Without such context, shrink toward neutral.
    contextual_continue_support = (
        context_rel * context_separation if p_new_continue >= 0.5 else 0.0
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

    # Context may add a small residual either toward NEW_CONTINUE or OLD_RETURN.
    # It can only be meaningful with actual same-shoe support and remains capped.
    context_weight = 0.0
    final_logit = _logit(shrunk_p_b)
    if float(context.get("support", 0.0) or 0.0) >= CONTEXT_SUPPORT_THRESHOLD:
        context_weight = min(
            MAX_CONTEXT_RESIDUAL_WEIGHT,
            MAX_CONTEXT_RESIDUAL_WEIGHT * context_rel * context_separation,
        )
        context_p_b = p_new_continue if current_side == "B" else 1.0 - p_new_continue
        final_logit += context_weight * _logit(context_p_b)

    final_p_b = _clip(_sigmoid(final_logit), 0.37, 0.63)
    # Anti-echo is a calibrator, not a reversal rule. A flip is permitted only
    # when the original V1 result was already very close to neutral.
    if (base_p_b >= 0.5) != (final_p_b >= 0.5) and abs(base_p_b - 0.5) >= 0.025:
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
                "reliability": float(dict(item).get("reliability", 0.0) or 0.0),
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
    "CONTEXT_SUPPORT_THRESHOLD",
    "MAX_ECHO_SHRINK",
    "MAX_CONTEXT_RESIDUAL_WEIGHT",
    "calibrate_fresh_switch",
]
