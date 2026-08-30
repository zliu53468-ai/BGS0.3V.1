"""Road-only short-shoe prediction policy for BGS.

Public helper names and outward fields are preserved. The original 12-round
ShortShoePredictor remains the local model. Formal B/P probability is produced
from Local Markov + full-shoe Global Base Probability + NumPy regression, then
passed through an Anti-Lock + Anti-Chase Direction Controller.

The controller has two jobs only:
1. Global history cannot lock the decision when Local + Regression agree against it.
2. One newly opened B/P round cannot by itself flip the formal direction.

The compatibility penalty payload is retained but has no authority to force O.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np

from performance_tracker import get_resolved_records

POLICY_VERSION = "ROAD-ONLY-W12-GLOBAL-REGRESSION-ANTI-LOCK-CHASE-V5"

OUTCOMES = ("B", "P")
WINDOW_SIZE = 12
MARKOV_MAX_ORDER = 1
MARKOV_DECAY = 0.93
MARKOV_PRIOR_STRENGTH = 0.50
MARKOV_MIN_EFFECTIVE_SUPPORT = 0.0
MIN_HISTORY_FOR_SCORING = 4

ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2

PENALTY_CONSECUTIVE_MISSES = 2
PENALTY_MIN_OBSERVE_ROUNDS = 0
RECOVERY_WINDOW = 0
RECOVERY_MIN_HITS = 0
RECOVERY_CONFIDENCE = 0.50

REGRESSION_LOCAL_WINDOW = 8
REGRESSION_GLOBAL_STRONG_SLOPE = 0.20
REGRESSION_LOCAL_STRONG_SLOPE = 0.35
REGRESSION_SLOPE_SCALE = 1.25

# The old trend regime allowed Global + Regression to own 85% of the vote.
# These weights deliberately prevent full-shoe history from permanently locking
# one side while retaining Regression as the independent turning-point signal.
ENSEMBLE_NORMAL_WEIGHTS = (0.40, 0.20, 0.40)
ENSEMBLE_TREND_WEIGHTS = (0.30, 0.25, 0.45)
ENSEMBLE_REVERSAL_WEIGHTS = (0.45, 0.0, 0.55)
ENSEMBLE_ANTI_LOCK_WEIGHTS = (0.45, 0.10, 0.45)

ANTI_LOCK_MIN_COMPONENT_EDGE = 0.025
ANTI_CHASE_FAST_COMPONENT_EDGE = 0.050
ANTI_CHASE_PREVIOUS_WEIGHT = 0.65
ANTI_CHASE_CURRENT_WEIGHT = 0.35
ANTI_CHASE_DIRECTION_EPSILON = 0.0005

GLOBAL_LOCAL_WEIGHT = ENSEMBLE_NORMAL_WEIGHTS[0]
GLOBAL_TREND_WEIGHT = ENSEMBLE_NORMAL_WEIGHTS[1]

EARLY_SHOE_MAX_ROUNDS = 20
LATE_SHOE_MIN_ROUNDS = 41
MIN_DIRECTION_CONFIDENCE = 0.50
EARLY_MIN_DIRECTION_CONFIDENCE = 0.50
PHYSICAL_MIN_EV = 0.0
EARLY_ACTIVE_MAX_ROUNDS = 30
TEMPERATURE_SCALING_MAX_ROUNDS = 0
EARLY_TEMPERATURE = 1.0

_INSTALLED = False


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _side_from_probability(p_b: float) -> str:
    return "B" if float(p_b) >= 0.5 else "P"


def normalize_big_road(history: str | Iterable[Any] | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = (
            history.replace("|", "")
            .replace(",", "")
            .replace(" ", "")
            .upper()
        )
        if compact and all(char in {"B", "P", "T"} for char in compact):
            return [char for char in compact if char in OUTCOMES][-2000:]
        raw_items: Iterable[Any] = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        raw_items = history

    cleaned: list[str] = []
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
            cleaned.append(value)
    return cleaned[-2000:]


def _weighted_outcome_counts(
    sequence: Sequence[str],
    decay: float,
) -> dict[str, float]:
    counts = {
        "B": float(MARKOV_PRIOR_STRENGTH),
        "P": float(MARKOV_PRIOR_STRENGTH),
    }
    n = len(sequence)
    for index, outcome in enumerate(sequence):
        age = max(0, n - 1 - index)
        counts[outcome] += decay ** age
    return counts


def _transition_counts_for_last(
    sequence: Sequence[str],
    decay: float,
) -> tuple[dict[str, float], float, str]:
    counts = {
        "B": float(MARKOV_PRIOR_STRENGTH),
        "P": float(MARKOV_PRIOR_STRENGTH),
    }
    if len(sequence) < 2:
        return counts, 0.0, sequence[-1] if sequence else ""

    state = sequence[-1]
    support = 0.0
    n = len(sequence)
    for index in range(1, n):
        if sequence[index - 1] != state:
            continue
        age = max(0, n - 1 - index)
        weight = decay ** age
        counts[sequence[index]] += weight
        support += weight
    return counts, float(support), state


def _same_switch_probabilities(
    sequence: Sequence[str],
    decay: float,
) -> tuple[float, float, float, float]:
    same = float(MARKOV_PRIOR_STRENGTH)
    switch = float(MARKOV_PRIOR_STRENGTH)
    raw_same = 0.0
    raw_switch = 0.0
    n = len(sequence)
    for index in range(1, n):
        age = max(0, n - 1 - index)
        weight = decay ** age
        if sequence[index] == sequence[index - 1]:
            same += weight
            raw_same += weight
        else:
            switch += weight
            raw_switch += weight
    total = same + switch
    if total <= 1e-12:
        return 0.5, 0.5, raw_same, raw_switch
    return same / total, switch / total, raw_same, raw_switch


class ShortShoePredictor:
    """Existing 12-round B/P local model; core math is unchanged."""

    def __init__(self, window_size: int = WINDOW_SIZE, decay: float = MARKOV_DECAY):
        self.window_size = max(4, int(window_size or WINDOW_SIZE))
        self.decay = _clip(decay, 0.50, 1.0)

    def predict(self, history: str | Iterable[Any] | None) -> dict[str, Any]:
        full_sequence = normalize_big_road(history)
        sequence = full_sequence[-self.window_size:]
        n = len(sequence)

        global_counts = _weighted_outcome_counts(sequence, self.decay)
        global_total = sum(global_counts.values()) or 1.0
        global_probs = {
            "B": global_counts["B"] / global_total,
            "P": global_counts["P"] / global_total,
        }

        if n == 0:
            probabilities = {"B": 0.5, "P": 0.5}
            transition_counts = {
                "B": float(MARKOV_PRIOR_STRENGTH),
                "P": float(MARKOV_PRIOR_STRENGTH),
            }
            support = 0.0
            state_key = ""
            same_prob = 0.5
            switch_prob = 0.5
            raw_same = 0.0
            raw_switch = 0.0
        elif n == 1:
            probabilities = dict(global_probs)
            transition_counts = {
                "B": float(MARKOV_PRIOR_STRENGTH),
                "P": float(MARKOV_PRIOR_STRENGTH),
            }
            support = 0.0
            state_key = sequence[-1]
            same_prob = 0.5
            switch_prob = 0.5
            raw_same = 0.0
            raw_switch = 0.0
        else:
            transition_counts, support, state_key = _transition_counts_for_last(
                sequence,
                self.decay,
            )
            transition_total = sum(transition_counts.values()) or 1.0
            transition_probs = {
                "B": transition_counts["B"] / transition_total,
                "P": transition_counts["P"] / transition_total,
            }

            same_prob, switch_prob, raw_same, raw_switch = _same_switch_probabilities(
                sequence,
                self.decay,
            )
            last = sequence[-1]
            opposite = "P" if last == "B" else "B"
            regime_probs = {
                last: same_prob,
                opposite: switch_prob,
            }

            context_weight = min(0.70, 0.45 + 0.08 * min(3.0, support))
            regime_weight = 1.0 - context_weight
            probabilities = {
                side: (
                    context_weight * transition_probs[side]
                    + regime_weight * regime_probs[side]
                )
                for side in OUTCOMES
            }

        total = probabilities["B"] + probabilities["P"]
        if total <= 1e-12:
            probabilities = {"B": 0.5, "P": 0.5}
        else:
            probabilities = {
                "B": probabilities["B"] / total,
                "P": probabilities["P"] / total,
            }

        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"
        confidence_prob = float(max(probabilities["B"], probabilities["P"]))
        margin = float(abs(probabilities["B"] - probabilities["P"]))

        return {
            "model": "short_shoe_sliding_window_markov",
            "version": POLICY_VERSION,
            "window_size": int(self.window_size),
            "window_sequence": "".join(sequence),
            "window_rounds": len(sequence),
            "history_rounds": len(full_sequence),
            "sequence_length": len(full_sequence),
            "decay": float(self.decay),
            "max_order": 1,
            "selected_order": 1 if n >= 2 else 0,
            "state_key": state_key,
            "effective_support": float(support),
            "transition_counts": {
                "B": float(transition_counts["B"]),
                "P": float(transition_counts["P"]),
            },
            "global_counts": {
                "B": float(global_counts["B"]),
                "P": float(global_counts["P"]),
            },
            "global_probabilities": dict(global_probs),
            "probabilities": {
                "B": float(probabilities["B"]),
                "P": float(probabilities["P"]),
                "T": 0.0,
            },
            "direction": direction,
            "confidence": confidence_prob,
            "confidence_prob": confidence_prob,
            "margin": margin,
            "same_probability": float(same_prob),
            "switch_probability": float(switch_prob),
            "same_support": float(raw_same),
            "switch_support": float(raw_switch),
            "order_diagnostics": [
                {
                    "order": 1 if n >= 2 else 0,
                    "context": state_key,
                    "effective_support": float(support),
                    "counts": dict(transition_counts),
                    "probabilities": {
                        "B": float(probabilities["B"]),
                        "P": float(probabilities["P"]),
                    },
                    "window_size": int(self.window_size),
                }
            ],
        }


def decayed_markov_forecast(
    history: str | Iterable[Any] | None,
    *,
    decay: float = MARKOV_DECAY,
    max_order: int = MARKOV_MAX_ORDER,
) -> dict[str, Any]:
    del max_order
    return ShortShoePredictor(window_size=WINDOW_SIZE, decay=decay).predict(history)


def _full_shoe_base_probability(sequence: Sequence[str]) -> tuple[int, int, float]:
    total_b = sum(1 for value in sequence if value == "B")
    total_p = sum(1 for value in sequence if value == "P")
    resolved = total_b + total_p
    global_p_b = float(total_b / resolved) if resolved > 0 else 0.5
    return total_b, total_p, global_p_b


def _least_squares_line(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    n = int(x.size)
    if n < 2 or y.size != x.size:
        base = float(y[-1]) if y.size else 0.0
        return 0.0, base, 0.0

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return 0.0, y_mean, 0.0

    slope = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, _clip(r_squared, 0.0, 1.0)


def regression_analysis_model(
    history: str | Iterable[Any] | None,
) -> dict[str, Any]:
    sequence = normalize_big_road(history)
    n = len(sequence)
    if n == 0:
        return {
            "available": False,
            "rounds": 0,
            "encoded_path": [],
            "cumulative_path": [],
            "global_slope": 0.0,
            "local_slope": 0.0,
            "global_intercept": 0.0,
            "local_intercept": 0.0,
            "global_r_squared": 0.0,
            "local_r_squared": 0.0,
            "global_next_cumulative": 0.0,
            "local_next_cumulative": 0.0,
            "aligned_strong_trend": False,
            "structural_reversal": False,
            "regime": "normal",
            "regression_signal_slope": 0.0,
            "regression_p_b": 0.5,
            "regression_p_p": 0.5,
            "local_window": 0,
        }

    encoded = np.asarray(
        [1.0 if value == "B" else -1.0 for value in sequence],
        dtype=np.float64,
    )
    cumulative = np.cumsum(encoded)
    x = np.arange(1, n + 1, dtype=np.float64)

    global_slope, global_intercept, global_r2 = _least_squares_line(x, cumulative)
    global_next = float(global_intercept + global_slope * (n + 1))

    local_window = min(REGRESSION_LOCAL_WINDOW, n)
    local_x = x[-local_window:]
    local_y = cumulative[-local_window:]
    local_slope, local_intercept, local_r2 = _least_squares_line(local_x, local_y)
    local_next = float(local_intercept + local_slope * (n + 1))

    global_strong = abs(global_slope) >= REGRESSION_GLOBAL_STRONG_SLOPE
    local_strong = abs(local_slope) >= REGRESSION_LOCAL_STRONG_SLOPE
    same_direction = bool(
        abs(global_slope) > 1e-12
        and abs(local_slope) > 1e-12
        and global_slope * local_slope > 0.0
    )
    opposite_direction = bool(
        abs(global_slope) > 1e-12
        and abs(local_slope) > 1e-12
        and global_slope * local_slope < 0.0
    )

    aligned_strong_trend = bool(global_strong and local_strong and same_direction)
    structural_reversal = bool(global_strong and local_strong and opposite_direction)

    if structural_reversal:
        regime = "structural_reversal"
        regression_signal_slope = local_slope
    elif aligned_strong_trend:
        regime = "aligned_strong_trend"
        regression_signal_slope = 0.50 * global_slope + 0.50 * local_slope
    else:
        regime = "normal"
        regression_signal_slope = 0.65 * global_slope + 0.35 * local_slope

    regression_p_b = float(
        0.5 + 0.5 * np.tanh(REGRESSION_SLOPE_SCALE * regression_signal_slope)
    )
    regression_p_b = _clip(regression_p_b, 0.001, 0.999)

    return {
        "available": True,
        "rounds": int(n),
        "encoded_path": [int(value) for value in encoded.tolist()],
        "cumulative_path": [float(value) for value in cumulative.tolist()],
        "global_slope": float(global_slope),
        "local_slope": float(local_slope),
        "global_intercept": float(global_intercept),
        "local_intercept": float(local_intercept),
        "global_r_squared": float(global_r2),
        "local_r_squared": float(local_r2),
        "global_next_cumulative": float(global_next),
        "local_next_cumulative": float(local_next),
        "global_strong": bool(global_strong),
        "local_strong": bool(local_strong),
        "slope_same_direction": bool(same_direction),
        "slope_opposite_direction": bool(opposite_direction),
        "aligned_strong_trend": bool(aligned_strong_trend),
        "structural_reversal": bool(structural_reversal),
        "regime": regime,
        "regression_signal_slope": float(regression_signal_slope),
        "regression_p_b": float(regression_p_b),
        "regression_p_p": float(1.0 - regression_p_b),
        "local_window": int(local_window),
        "global_strong_threshold": float(REGRESSION_GLOBAL_STRONG_SLOPE),
        "local_strong_threshold": float(REGRESSION_LOCAL_STRONG_SLOPE),
    }


def _normalize_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    values = [max(0.0, float(value)) for value in weights[:3]]
    while len(values) < 3:
        values.append(0.0)
    total = sum(values)
    if total <= 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return values[0] / total, values[1] / total, values[2] / total


def _adaptive_ensemble_weights(
    *,
    local_p_b: float,
    global_p_b: float,
    regression_p_b: float,
    regression: Mapping[str, Any],
) -> tuple[float, float, float, str, bool]:
    local_side = _side_from_probability(local_p_b)
    global_side = _side_from_probability(global_p_b)
    regression_side = _side_from_probability(regression_p_b)
    local_edge = abs(local_p_b - 0.5)
    regression_edge = abs(regression_p_b - 0.5)

    local_regression_consensus = bool(
        local_side == regression_side
        and local_edge >= ANTI_LOCK_MIN_COMPONENT_EDGE
        and regression_edge >= ANTI_LOCK_MIN_COMPONENT_EDGE
    )
    anti_lock = bool(
        local_regression_consensus
        and local_side != global_side
    )

    if bool(regression.get("structural_reversal")):
        weights = ENSEMBLE_REVERSAL_WEIGHTS
        regime = "structural_reversal"
    elif anti_lock:
        weights = ENSEMBLE_ANTI_LOCK_WEIGHTS
        regime = "anti_lock_local_regression_consensus"
    elif bool(regression.get("aligned_strong_trend")):
        weights = ENSEMBLE_TREND_WEIGHTS
        regime = "aligned_strong_trend"
    else:
        weights = ENSEMBLE_NORMAL_WEIGHTS
        regime = "normal"

    w1, w2, w3 = _normalize_weights(weights)
    return w1, w2, w3, regime, anti_lock


def _component_snapshot(sequence: Sequence[str]) -> dict[str, Any]:
    seq = list(sequence)
    local_forecast = decayed_markov_forecast(seq)
    local_probabilities = local_forecast.get("probabilities")
    if not isinstance(local_probabilities, Mapping):
        local_probabilities = {"B": 0.5, "P": 0.5}
    local_p_b = _clip(local_probabilities.get("B", 0.5), 0.0, 1.0)
    local_p_p = _clip(local_probabilities.get("P", 0.5), 0.0, 1.0)
    local_total = local_p_b + local_p_p
    if local_total <= 1e-12:
        local_p_b, local_p_p = 0.5, 0.5
    else:
        local_p_b, local_p_p = local_p_b / local_total, local_p_p / local_total

    total_b, total_p, global_p_b = _full_shoe_base_probability(seq)
    global_p_p = 1.0 - global_p_b
    regression = regression_analysis_model(seq)
    regression_p_b = _clip(regression.get("regression_p_b", 0.5), 0.0, 1.0)
    regression_p_p = 1.0 - regression_p_b

    w1, w2, w3, ensemble_regime, anti_lock = _adaptive_ensemble_weights(
        local_p_b=local_p_b,
        global_p_b=global_p_b,
        regression_p_b=regression_p_b,
        regression=regression,
    )

    raw_p_b = w1 * local_p_b + w2 * global_p_b + w3 * regression_p_b
    raw_p_p = w1 * local_p_p + w2 * global_p_p + w3 * regression_p_p
    total = raw_p_b + raw_p_p
    if total <= 1e-12:
        raw_p_b, raw_p_p = 0.5, 0.5
    else:
        raw_p_b, raw_p_p = raw_p_b / total, raw_p_p / total

    return {
        "local_forecast": local_forecast,
        "local_p_b": float(local_p_b),
        "local_p_p": float(local_p_p),
        "global_p_b": float(global_p_b),
        "global_p_p": float(global_p_p),
        "regression_p_b": float(regression_p_b),
        "regression_p_p": float(regression_p_p),
        "regression_analysis": dict(regression),
        "weights": {"local": float(w1), "global": float(w2), "regression": float(w3)},
        "ensemble_regime": ensemble_regime,
        "anti_lock_applied": bool(anti_lock),
        "raw_p_b": float(raw_p_b),
        "raw_p_p": float(raw_p_p),
        "raw_direction": _side_from_probability(raw_p_b),
        "total_b": int(total_b),
        "total_p": int(total_p),
    }


def _anti_lock_anti_chase_controller(
    sequence: Sequence[str],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    raw_p_b = float(current.get("raw_p_b", 0.5) or 0.5)
    raw_p_p = 1.0 - raw_p_b
    raw_direction = _side_from_probability(raw_p_b)

    local_p_b = float(current.get("local_p_b", 0.5) or 0.5)
    regression_p_b = float(current.get("regression_p_b", 0.5) or 0.5)
    global_p_b = float(current.get("global_p_b", 0.5) or 0.5)
    local_side = _side_from_probability(local_p_b)
    regression_side = _side_from_probability(regression_p_b)
    global_side = _side_from_probability(global_p_b)
    local_edge = abs(local_p_b - 0.5)
    regression_edge = abs(regression_p_b - 0.5)

    if len(sequence) < 2:
        return {
            "mode": "initial_no_hysteresis",
            "raw_direction": raw_direction,
            "previous_raw_direction": "",
            "final_direction": raw_direction,
            "raw_p_b": raw_p_b,
            "final_p_b": raw_p_b,
            "final_p_p": raw_p_p,
            "one_step_flip": False,
            "local_side": local_side,
            "regression_side": regression_side,
            "global_side": global_side,
            "local_regression_consensus": local_side == regression_side,
            "prior_support_count": 0,
            "immediate_switch_allowed": False,
            "latest_outcome": sequence[-1] if sequence else "",
        }

    previous = _component_snapshot(sequence[:-1])
    previous_raw_p_b = float(previous.get("raw_p_b", 0.5) or 0.5)
    previous_raw_direction = _side_from_probability(previous_raw_p_b)
    one_step_flip = raw_direction != previous_raw_direction

    previous_local_side = _side_from_probability(
        float(previous.get("local_p_b", 0.5) or 0.5)
    )
    previous_regression_side = _side_from_probability(
        float(previous.get("regression_p_b", 0.5) or 0.5)
    )
    prior_support_count = int(previous_local_side == raw_direction) + int(
        previous_regression_side == raw_direction
    )

    local_regression_consensus = bool(
        local_side == regression_side == raw_direction
        and local_edge >= ANTI_LOCK_MIN_COMPONENT_EDGE
        and regression_edge >= ANTI_LOCK_MIN_COMPONENT_EDGE
    )
    strong_local_regression_consensus = bool(
        local_side == regression_side == raw_direction
        and local_edge >= ANTI_CHASE_FAST_COMPONENT_EDGE
        and regression_edge >= ANTI_CHASE_FAST_COMPONENT_EDGE
    )
    structural_reversal = bool(
        (current.get("regression_analysis") or {}).get("structural_reversal")
    )

    # A fast switch needs independent confirmation.  One freshly opened result
    # cannot flip both short-horizon components and immediately become a bet.
    immediate_switch_allowed = bool(
        one_step_flip
        and strong_local_regression_consensus
        and (structural_reversal or prior_support_count >= 1)
    )

    final_p_b = raw_p_b
    mode = "raw_ensemble_confirmed"
    if one_step_flip and not immediate_switch_allowed:
        final_p_b = (
            ANTI_CHASE_PREVIOUS_WEIGHT * previous_raw_p_b
            + ANTI_CHASE_CURRENT_WEIGHT * raw_p_b
        )
        # Ensure a single new result cannot cross the formal direction boundary.
        if previous_raw_direction == "B" and final_p_b < 0.5 + ANTI_CHASE_DIRECTION_EPSILON:
            final_p_b = 0.5 + ANTI_CHASE_DIRECTION_EPSILON
        elif previous_raw_direction == "P" and final_p_b > 0.5 - ANTI_CHASE_DIRECTION_EPSILON:
            final_p_b = 0.5 - ANTI_CHASE_DIRECTION_EPSILON
        mode = "anti_chase_one_round_hysteresis"
    elif immediate_switch_allowed:
        mode = "confirmed_fast_switch"

    final_p_b = _clip(final_p_b, 0.001, 0.999)
    final_p_p = 1.0 - final_p_b
    final_direction = _side_from_probability(final_p_b)

    return {
        "mode": mode,
        "raw_direction": raw_direction,
        "previous_raw_direction": previous_raw_direction,
        "final_direction": final_direction,
        "raw_p_b": float(raw_p_b),
        "previous_raw_p_b": float(previous_raw_p_b),
        "final_p_b": float(final_p_b),
        "final_p_p": float(final_p_p),
        "one_step_flip": bool(one_step_flip),
        "local_side": local_side,
        "regression_side": regression_side,
        "global_side": global_side,
        "local_edge": float(local_edge),
        "regression_edge": float(regression_edge),
        "local_regression_consensus": bool(local_regression_consensus),
        "strong_local_regression_consensus": bool(strong_local_regression_consensus),
        "prior_support_count": int(prior_support_count),
        "structural_reversal": bool(structural_reversal),
        "immediate_switch_allowed": bool(immediate_switch_allowed),
        "latest_outcome": sequence[-1] if sequence else "",
        "previous_weight": float(ANTI_CHASE_PREVIOUS_WEIGHT),
        "current_weight": float(ANTI_CHASE_CURRENT_WEIGHT),
    }


def global_trend_bias_correction(
    history: str | Iterable[Any] | None,
    local_forecast: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = normalize_big_road(history)
    current = _component_snapshot(sequence)

    # Reuse the caller's local forecast exactly so the original 12-hand model is
    # not recomputed differently from the public prediction path.
    if isinstance(local_forecast.get("probabilities"), Mapping):
        supplied = dict(local_forecast.get("probabilities") or {})
        supplied_b = _clip(supplied.get("B", 0.5), 0.0, 1.0)
        supplied_p = _clip(supplied.get("P", 0.5), 0.0, 1.0)
        supplied_total = supplied_b + supplied_p
        if supplied_total > 1e-12:
            supplied_b, supplied_p = supplied_b / supplied_total, supplied_p / supplied_total
            current["local_forecast"] = dict(local_forecast)
            current["local_p_b"] = float(supplied_b)
            current["local_p_p"] = float(supplied_p)
            regression = dict(current.get("regression_analysis") or {})
            w1, w2, w3, regime, anti_lock = _adaptive_ensemble_weights(
                local_p_b=supplied_b,
                global_p_b=float(current.get("global_p_b", 0.5) or 0.5),
                regression_p_b=float(current.get("regression_p_b", 0.5) or 0.5),
                regression=regression,
            )
            current["weights"] = {
                "local": float(w1),
                "global": float(w2),
                "regression": float(w3),
            }
            current["ensemble_regime"] = regime
            current["anti_lock_applied"] = bool(anti_lock)
            raw_p_b = (
                w1 * supplied_b
                + w2 * float(current.get("global_p_b", 0.5) or 0.5)
                + w3 * float(current.get("regression_p_b", 0.5) or 0.5)
            )
            current["raw_p_b"] = float(_clip(raw_p_b, 0.0, 1.0))
            current["raw_p_p"] = float(1.0 - current["raw_p_b"])
            current["raw_direction"] = _side_from_probability(current["raw_p_b"])

    controller = _anti_lock_anti_chase_controller(sequence, current)
    final_p_b = float(controller["final_p_b"])
    final_p_p = float(controller["final_p_p"])
    direction = str(controller["final_direction"])
    confidence_prob = max(final_p_b, final_p_p)

    total_b = int(current.get("total_b", 0) or 0)
    total_p = int(current.get("total_p", 0) or 0)
    global_p_b = float(current.get("global_p_b", 0.5) or 0.5)
    if len(sequence) >= 2:
        _, _, previous_global_p_b = _full_shoe_base_probability(sequence[:-1])
    else:
        previous_global_p_b = 0.5
    global_velocity_b = float(global_p_b - previous_global_p_b)
    if global_velocity_b > 1e-12:
        global_shift_direction = "toward_B"
    elif global_velocity_b < -1e-12:
        global_shift_direction = "toward_P"
    else:
        global_shift_direction = "flat"

    weights = dict(current.get("weights") or {})
    regression = dict(current.get("regression_analysis") or {})
    return {
        "applied": True,
        "mode": "global_local_regression_anti_lock_anti_chase",
        "ensemble_regime": str(current.get("ensemble_regime") or "normal"),
        "local_weight": float(weights.get("local", 0.0) or 0.0),
        "global_weight": float(weights.get("global", 0.0) or 0.0),
        "regression_weight": float(weights.get("regression", 0.0) or 0.0),
        "ensemble_weights": weights,
        "anti_lock_applied": bool(current.get("anti_lock_applied", False)),
        "direction_controller": dict(controller),
        "total_rounds": int(total_b + total_p),
        "total_b": total_b,
        "total_p": total_p,
        "global_p_b": global_p_b,
        "global_p_p": float(1.0 - global_p_b),
        "previous_global_p_b": float(previous_global_p_b),
        "global_probability_velocity_b": float(global_velocity_b),
        "global_shift_direction": global_shift_direction,
        "local_p_b": float(current.get("local_p_b", 0.5) or 0.5),
        "local_p_p": float(current.get("local_p_p", 0.5) or 0.5),
        "regression_p_b": float(current.get("regression_p_b", 0.5) or 0.5),
        "regression_p_p": float(current.get("regression_p_p", 0.5) or 0.5),
        "regression_analysis": regression,
        "raw_ensemble_p_b": float(current.get("raw_p_b", 0.5) or 0.5),
        "raw_ensemble_p_p": float(current.get("raw_p_p", 0.5) or 0.5),
        "raw_ensemble_direction": str(current.get("raw_direction") or direction),
        "final_p_b": float(final_p_b),
        "final_p_p": float(final_p_p),
        "final_direction": direction,
        "final_confidence_prob": float(confidence_prob),
        "formula": "Adaptive(Local,Global,Regression) -> AntiLock -> AntiChaseDirectionController",
    }


def _replay_penalty_state(sequence: Sequence[str]) -> dict[str, Any]:
    del sequence
    return {
        "active": False,
        "force_observe": False,
        "observe_remaining": 0,
        "recovery_pending": False,
        "consecutive_misses": 0,
        "trigger_count": 0,
        "official_scored": 0,
        "official_hits": 0,
        "official_accuracy": 0.0,
        "virtual_sample_count": 0,
        "recent_virtual_sample_count": 0,
        "recent_virtual_hits": 0,
        "recent_virtual_accuracy": 0.0,
        "minimum_observe_rounds": 0,
        "recovery_window": 0,
        "recovery_min_hits": 0,
        "recovery_confidence": 0.50,
        "semantics": "disabled_anti_lock_anti_chase_no_observe_gate",
    }


def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    sequence = normalize_big_road(history)
    local_forecast = decayed_markov_forecast(sequence)
    correction = global_trend_bias_correction(sequence, local_forecast)
    penalty = _replay_penalty_state(sequence)

    forecast = dict(local_forecast)
    probabilities = {
        "B": float(correction["final_p_b"]),
        "P": float(correction["final_p_p"]),
        "T": 0.0,
    }
    direction = str(correction["final_direction"])
    confidence_prob = float(correction["final_confidence_prob"])
    forecast.update({
        "probabilities": probabilities,
        "direction": direction,
        "confidence": confidence_prob,
        "confidence_prob": confidence_prob,
        "margin": float(abs(probabilities["B"] - probabilities["P"])),
        "local_model_probabilities": dict(local_forecast.get("probabilities") or {}),
        "global_trend_bias_correction": dict(correction),
        "regression_analysis": dict(correction.get("regression_analysis") or {}),
        "ensemble_weights": dict(correction.get("ensemble_weights") or {}),
        "direction_controller": dict(correction.get("direction_controller") or {}),
    })

    return {
        **forecast,
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "latent_direction": direction,
        "confidence_prob": confidence_prob,
        "penalty_observe": penalty,
        "global_trend_bias_correction": dict(correction),
        "regression_analysis": dict(correction.get("regression_analysis") or {}),
        "ensemble_weights": dict(correction.get("ensemble_weights") or {}),
        "direction_controller": dict(correction.get("direction_controller") or {}),
        "big_road_sequence": str(local_forecast.get("window_sequence") or ""),
    }


def shoe_progress_policy(rounds: int) -> dict[str, Any]:
    value = max(0, int(rounds or 0))
    if value <= 0:
        phase = "UNKNOWN"
    elif value <= EARLY_SHOE_MAX_ROUNDS:
        phase = "EARLY"
    elif value < LATE_SHOE_MIN_ROUNDS:
        phase = "MID"
    else:
        phase = "LATE"
    return {
        "rounds": value,
        "phase": phase,
        "shoe_weight_factor": 0.0,
        "road_weight_factor": 1.0,
    }


def recent_user_direction_feedback(
    user_id: str,
    *,
    limit: int = ONLINE_WINDOW,
) -> dict[str, Any]:
    raw_user = str(user_id or "")
    if not raw_user:
        return {
            "available": False,
            "sample_count": 0,
            "correct_count": 0,
            "accuracy": 0.0,
            "consecutive_losses": 0,
            "triggered": False,
        }

    uid_key = sha256(raw_user.encode("utf-8")).hexdigest()[:24]
    try:
        records = get_resolved_records(limit=5000)
    except Exception:
        records = []

    recent: list[dict[str, Any]] = []
    for record in reversed(records):
        if str(record.get("uid_key") or "") != uid_key:
            continue
        actual = str(record.get("actual_outcome") or "").upper().strip()
        predicted = str(
            record.get("adaptive_only_direction")
            or record.get("action")
            or record.get("recommend")
            or ""
        ).upper().strip()
        if actual not in OUTCOMES or predicted not in OUTCOMES:
            continue
        recent.append({
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
        })
        if len(recent) >= max(1, int(limit)):
            break

    correct = sum(1 for item in recent if item["correct"])
    consecutive_losses = 0
    for item in recent:
        if item["correct"]:
            break
        consecutive_losses += 1
    return {
        "available": bool(recent),
        "sample_count": len(recent),
        "correct_count": int(correct),
        "accuracy": float(correct / max(1, len(recent))),
        "consecutive_losses": int(consecutive_losses),
        "triggered": False,
        "window": int(ONLINE_WINDOW),
        "loss_trigger": int(ONLINE_CONSECUTIVE_LOSS_TRIGGER),
        "semantics": "diagnostic_only_anti_lock_anti_chase_no_observe_gate",
    }


def install_dynamic_prediction_policy() -> bool:
    global _INSTALLED
    _INSTALLED = True
    return True


__all__ = [
    "POLICY_VERSION",
    "WINDOW_SIZE",
    "MARKOV_MAX_ORDER",
    "MARKOV_DECAY",
    "MIN_DIRECTION_CONFIDENCE",
    "EARLY_MIN_DIRECTION_CONFIDENCE",
    "PHYSICAL_MIN_EV",
    "EARLY_ACTIVE_MAX_ROUNDS",
    "TEMPERATURE_SCALING_MAX_ROUNDS",
    "EARLY_TEMPERATURE",
    "GLOBAL_LOCAL_WEIGHT",
    "GLOBAL_TREND_WEIGHT",
    "REGRESSION_LOCAL_WINDOW",
    "REGRESSION_GLOBAL_STRONG_SLOPE",
    "REGRESSION_LOCAL_STRONG_SLOPE",
    "ShortShoePredictor",
    "normalize_big_road",
    "decayed_markov_forecast",
    "regression_analysis_model",
    "global_trend_bias_correction",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
