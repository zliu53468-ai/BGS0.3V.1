"""Road-only short-shoe prediction policy for BGS.

Public helper names and return fields are preserved for the existing runtime.
The original 12-round ShortShoePredictor remains the local model.  A separate
Global Trend Bias Correction layer fuses that local B/P probability with the
full-shoe B/P base probability using the requested fixed weights:

    Final_P(B) = 0.40 * Local_P(B) + 0.60 * Global_P_B

Ties and exact remaining-card composition are not used by the formal direction.
The compatibility penalty payload is retained but has no authority to force O.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import math

from performance_tracker import get_resolved_records

POLICY_VERSION = "ROAD-ONLY-SHORT-SHOE-W12-GLOBAL-TREND-V3"

OUTCOMES = ("B", "P")
WINDOW_SIZE = 12
MARKOV_MAX_ORDER = 1
MARKOV_DECAY = 0.93
MARKOV_PRIOR_STRENGTH = 0.50
MARKOV_MIN_EFFECTIVE_SUPPORT = 0.0
MIN_HISTORY_FOR_SCORING = 4

ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2

# Compatibility constants retained. Replay penalty never forces O.
PENALTY_CONSECUTIVE_MISSES = 2
PENALTY_MIN_OBSERVE_ROUNDS = 0
RECOVERY_WINDOW = 0
RECOVERY_MIN_HITS = 0
RECOVERY_CONFIDENCE = 0.50

# Requested Global-Local ensemble weights.
GLOBAL_LOCAL_WEIGHT = 0.40
GLOBAL_TREND_WEIGHT = 0.60

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


def normalize_big_road(history: str | Iterable[Any] | None) -> list[str]:
    """Return chronological B/P outcomes; ties and non-outcomes are ignored."""
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
    """Return P(same), P(switch), raw same support and raw switch support."""
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
    """Micro sliding-window B/P predictor with a hard 12-round memory limit."""

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

            # Existing 12-hand local core logic remains unchanged.
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
    """Compatibility entrypoint backed by the unchanged 12-round local model."""
    del max_order
    return ShortShoePredictor(window_size=WINDOW_SIZE, decay=decay).predict(history)


def _full_shoe_base_probability(sequence: Sequence[str]) -> tuple[int, int, float]:
    total_b = sum(1 for value in sequence if value == "B")
    total_p = sum(1 for value in sequence if value == "P")
    resolved = total_b + total_p
    global_p_b = float(total_b / resolved) if resolved > 0 else 0.5
    return total_b, total_p, global_p_b


def global_trend_bias_correction(
    history: str | Iterable[Any] | None,
    local_forecast: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse the 12-hand local probability with the full-shoe 60% global anchor."""
    sequence = normalize_big_road(history)
    total_b, total_p, global_p_b = _full_shoe_base_probability(sequence)
    global_p_p = 1.0 - global_p_b

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

    final_p_b = (
        GLOBAL_LOCAL_WEIGHT * local_p_b
        + GLOBAL_TREND_WEIGHT * global_p_b
    )
    final_p_p = (
        GLOBAL_LOCAL_WEIGHT * local_p_p
        + GLOBAL_TREND_WEIGHT * global_p_p
    )
    final_total = final_p_b + final_p_p
    if final_total <= 1e-12:
        final_p_b, final_p_p = 0.5, 0.5
    else:
        final_p_b, final_p_p = final_p_b / final_total, final_p_p / final_total

    direction = "B" if final_p_b >= final_p_p else "P"
    confidence_prob = max(final_p_b, final_p_p)

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

    return {
        "applied": True,
        "mode": "global_local_40_60",
        "local_weight": float(GLOBAL_LOCAL_WEIGHT),
        "global_weight": float(GLOBAL_TREND_WEIGHT),
        "total_rounds": int(total_b + total_p),
        "total_b": int(total_b),
        "total_p": int(total_p),
        "global_p_b": float(global_p_b),
        "global_p_p": float(global_p_p),
        "previous_global_p_b": float(previous_global_p_b),
        "global_probability_velocity_b": float(global_velocity_b),
        "global_shift_direction": global_shift_direction,
        "local_p_b": float(local_p_b),
        "local_p_p": float(local_p_p),
        "final_p_b": float(final_p_b),
        "final_p_p": float(final_p_p),
        "final_direction": direction,
        "final_confidence_prob": float(confidence_prob),
        "formula": "Final_P(B)=0.40*Local_P(B)+0.60*Global_P_B",
    }


def _replay_penalty_state(sequence: Sequence[str]) -> dict[str, Any]:
    """Compatibility payload only; no replay penalty may force O."""
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
        "semantics": "disabled_global_trend_bias_correction_no_observe_gate",
    }


def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """Return the 12-hand local forecast fused with the full-shoe 60% anchor."""
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
    })

    return {
        **forecast,
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "latent_direction": direction,
        "confidence_prob": confidence_prob,
        "penalty_observe": penalty,
        "global_trend_bias_correction": dict(correction),
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
    """Compatibility diagnostic only; it never gates the global/local forecast."""
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
        "semantics": "diagnostic_only_global_trend_filter_has_no_observe_gate",
    }


def install_dynamic_prediction_policy() -> bool:
    """Compatibility installation hook used by runtime_app."""
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
    "ShortShoePredictor",
    "normalize_big_road",
    "decayed_markov_forecast",
    "global_trend_bias_correction",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
