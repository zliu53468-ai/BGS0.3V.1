"""Public Markov interface plus early-shoe Global Prior smoothing helpers.

The existing V3.3 Markov implementation is preserved byte-for-byte in
`_markov_model_v33_core.py`.  This module re-exports that implementation and
adds only the static Global Prior utilities required by the dynamic prediction
policy.  The Markov model output itself is not silently replaced or retrained.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from _markov_model_v33_core import *  # noqa: F401,F403
from _markov_model_v33_core import __all__ as _CORE_ALL

# Requested large-sample static baccarat baseline.  These are policy priors,
# not runtime-learned transition counts.
GLOBAL_PRIOR_PROBABILITIES = {"B": 0.458, "P": 0.446, "T": 0.096}
GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS = 30

# Under a stationary large-sample baseline, the next-hand marginal remains the
# same regardless of the previous result.  Keeping the matrix explicit lets the
# policy consume a proper transition prior while also exposing the expected
# Big-Road CONTINUE/TURN balance below.
GLOBAL_PRIOR_TRANSITION_MATRIX = {
    "START": dict(GLOBAL_PRIOR_PROBABILITIES),
    "B": dict(GLOBAL_PRIOR_PROBABILITIES),
    "P": dict(GLOBAL_PRIOR_PROBABILITIES),
    "T": dict(GLOBAL_PRIOR_PROBABILITIES),
}

_GLOBAL_BP_MASS = (
    GLOBAL_PRIOR_PROBABILITIES["B"] + GLOBAL_PRIOR_PROBABILITIES["P"]
)
GLOBAL_ROAD_NORMAL_EXPECTATION = {
    "after_B": {
        "CONTINUE": GLOBAL_PRIOR_PROBABILITIES["B"] / _GLOBAL_BP_MASS,
        "TURN": GLOBAL_PRIOR_PROBABILITIES["P"] / _GLOBAL_BP_MASS,
    },
    "after_P": {
        "CONTINUE": GLOBAL_PRIOR_PROBABILITIES["P"] / _GLOBAL_BP_MASS,
        "TURN": GLOBAL_PRIOR_PROBABILITIES["B"] / _GLOBAL_BP_MASS,
    },
}


def _normalize_global(values: Mapping[str, Any]) -> dict[str, float]:
    raw = {
        side: max(1e-12, float(values.get(side, 0.0) or 0.0))
        for side in ("B", "P", "T")
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return dict(GLOBAL_PRIOR_PROBABILITIES)
    return {side: raw[side] / total for side in raw}


def global_prior_for_history(history: Iterable[Any] | str | None = None) -> dict[str, float]:
    """Return the configured global transition prior for the current state."""
    last = "START"
    if history is not None:
        values = list(history) if not isinstance(history, str) else list(history)
        for item in reversed(values):
            raw = item.get("outcome") if isinstance(item, Mapping) else item
            value = str(raw or "").upper().strip()
            if value in {"B", "P", "T"}:
                last = value
                break
    return dict(GLOBAL_PRIOR_TRANSITION_MATRIX.get(last, GLOBAL_PRIOR_PROBABILITIES))


def global_prior_alpha(current_hand: int) -> float:
    """alpha=current_hand/30 in the early shoe; alpha=1 afterwards."""
    hand = max(0, int(current_hand or 0))
    if hand >= GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS:
        return 1.0
    return max(0.0, min(1.0, hand / float(GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS)))


def blend_with_global_prior(
    local_probabilities: Mapping[str, Any],
    current_hand: int,
    *,
    history: Iterable[Any] | str | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply Final_P=(1-alpha)*P_global + alpha*P_local for hands 1..30."""
    local = _normalize_global(local_probabilities)
    global_probs = global_prior_for_history(history)
    hand = max(0, int(current_hand or 0))
    alpha = global_prior_alpha(hand)
    applied = bool(0 < hand <= GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS)

    if applied:
        final = _normalize_global({
            side: (1.0 - alpha) * global_probs[side] + alpha * local[side]
            for side in ("B", "P", "T")
        })
    else:
        final = dict(local)

    return final, {
        "applied": applied,
        "current_hand": hand,
        "max_rounds": int(GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS),
        "alpha": float(alpha),
        "global_prior": dict(global_probs),
        "local_probability": dict(local),
        "blended_probability": dict(final),
        "transition_matrix": {
            key: dict(value) for key, value in GLOBAL_PRIOR_TRANSITION_MATRIX.items()
        },
        "road_normal_expectation": {
            key: dict(value) for key, value in GLOBAL_ROAD_NORMAL_EXPECTATION.items()
        },
        "formula": "Final_P=(1-alpha)*P_global+alpha*P_local",
    }


__all__ = list(_CORE_ALL) + [
    "GLOBAL_PRIOR_PROBABILITIES",
    "GLOBAL_PRIOR_SMOOTH_MAX_ROUNDS",
    "GLOBAL_PRIOR_TRANSITION_MATRIX",
    "GLOBAL_ROAD_NORMAL_EXPECTATION",
    "global_prior_for_history",
    "global_prior_alpha",
    "blend_with_global_prior",
]
