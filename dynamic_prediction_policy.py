"""Road-only dynamic prediction policy for BGS.

This module deliberately keeps the public installation hook and helper names used
by the runtime, but the decision model is now based only on the Big Road B/P
sequence.  Exact card composition, OCR output format and transport interfaces are
not required by this policy.

Core policy:
- variable-order Markov chain (orders 1..4),
- exponential time decay so recent transitions dominate,
- deterministic replay of a penalty/observe state,
- after two consecutive directional misses, force O for at least three resolved
  B/P rounds while continuing to make virtual forecasts and update the chain,
- resume only after the recent virtual hit rate is calibrated again.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import math

from performance_tracker import get_resolved_records

POLICY_VERSION = "ROAD-ONLY-DECAY-MARKOV-PENALTY-V1"

OUTCOMES = ("B", "P")
MARKOV_MAX_ORDER = 4
MARKOV_DECAY = 0.93
MARKOV_PRIOR_STRENGTH = 0.75
MARKOV_MIN_EFFECTIVE_SUPPORT = 1.25
MIN_HISTORY_FOR_SCORING = 4

ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2

PENALTY_CONSECUTIVE_MISSES = 2
PENALTY_MIN_OBSERVE_ROUNDS = 3
RECOVERY_WINDOW = 3
RECOVERY_MIN_HITS = 2
RECOVERY_CONFIDENCE = 0.56

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
    """Return only chronological B/P outcomes; ties and non-outcomes are ignored."""
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


def _global_decayed_counts(sequence: Sequence[str], decay: float) -> dict[str, float]:
    counts = {
        "B": float(MARKOV_PRIOR_STRENGTH),
        "P": float(MARKOV_PRIOR_STRENGTH),
    }
    n = len(sequence)
    for index, outcome in enumerate(sequence):
        age = max(0, n - 1 - index)
        counts[outcome] += decay ** age
    return counts


def _context_decayed_counts(
    sequence: Sequence[str],
    *,
    order: int,
    decay: float,
) -> tuple[dict[str, float], float, str]:
    counts = {
        "B": float(MARKOV_PRIOR_STRENGTH),
        "P": float(MARKOV_PRIOR_STRENGTH),
    }
    n = len(sequence)
    if order <= 0 or n <= order:
        return counts, 0.0, ""

    context = tuple(sequence[-order:])
    support = 0.0
    for index in range(order, n):
        if tuple(sequence[index - order:index]) != context:
            continue
        age = max(0, n - 1 - index)
        weight = decay ** age
        counts[sequence[index]] += weight
        support += weight
    return counts, float(support), "".join(context)


def decayed_markov_forecast(
    history: str | Iterable[Any] | None,
    *,
    decay: float = MARKOV_DECAY,
    max_order: int = MARKOV_MAX_ORDER,
) -> dict[str, Any]:
    """Forecast the next B/P result from a time-decayed variable-order Markov chain."""
    sequence = normalize_big_road(history)
    decay = _clip(decay, 0.50, 0.999)
    max_order = max(1, min(8, int(max_order or MARKOV_MAX_ORDER)))

    global_counts = _global_decayed_counts(sequence, decay)
    global_total = sum(global_counts.values()) or 1.0
    global_probs = {
        "B": global_counts["B"] / global_total,
        "P": global_counts["P"] / global_total,
    }

    selected_order = 0
    selected_context = ""
    selected_support = 0.0
    selected_counts = dict(global_counts)
    selected_probs = dict(global_probs)
    order_diagnostics: list[dict[str, Any]] = []

    highest = min(max_order, max(1, len(sequence) - 1))
    for order in range(highest, 0, -1):
        counts, support, context = _context_decayed_counts(
            sequence,
            order=order,
            decay=decay,
        )
        total = sum(counts.values()) or 1.0
        probabilities = {
            "B": counts["B"] / total,
            "P": counts["P"] / total,
        }
        order_diagnostics.append({
            "order": order,
            "context": context,
            "effective_support": float(support),
            "counts": dict(counts),
            "probabilities": dict(probabilities),
        })
        if support >= MARKOV_MIN_EFFECTIVE_SUPPORT:
            selected_order = order
            selected_context = context
            selected_support = support
            selected_counts = counts
            selected_probs = probabilities
            break

    if selected_order == 0:
        counts, support, context = _context_decayed_counts(
            sequence,
            order=1,
            decay=decay,
        )
        if len(sequence) >= 2:
            total = sum(counts.values()) or 1.0
            one_step = {
                "B": counts["B"] / total,
                "P": counts["P"] / total,
            }
            blend = _clip(support / max(MARKOV_MIN_EFFECTIVE_SUPPORT, 1e-9))
            selected_probs = {
                side: blend * one_step[side] + (1.0 - blend) * global_probs[side]
                for side in OUTCOMES
            }
            selected_order = 1
            selected_context = context
            selected_support = support
            selected_counts = counts

    total_prob = selected_probs["B"] + selected_probs["P"]
    if total_prob <= 1e-12:
        selected_probs = {"B": 0.5, "P": 0.5}
    else:
        selected_probs = {
            "B": selected_probs["B"] / total_prob,
            "P": selected_probs["P"] / total_prob,
        }

    direction = "B" if selected_probs["B"] >= selected_probs["P"] else "P"
    confidence = float(max(selected_probs.values()))
    margin = float(abs(selected_probs["B"] - selected_probs["P"]))

    return {
        "model": "time_decay_variable_order_markov",
        "version": POLICY_VERSION,
        "sequence_length": len(sequence),
        "decay": float(decay),
        "max_order": int(max_order),
        "selected_order": int(selected_order),
        "state_key": selected_context,
        "effective_support": float(selected_support),
        "transition_counts": {
            "B": float(selected_counts["B"]),
            "P": float(selected_counts["P"]),
        },
        "global_counts": {
            "B": float(global_counts["B"]),
            "P": float(global_counts["P"]),
        },
        "global_probabilities": dict(global_probs),
        "probabilities": {
            "B": float(selected_probs["B"]),
            "P": float(selected_probs["P"]),
            "T": 0.0,
        },
        "direction": direction,
        "confidence": confidence,
        "margin": margin,
        "order_diagnostics": order_diagnostics,
    }


def _replay_penalty_state(sequence: Sequence[str]) -> dict[str, Any]:
    """Replay model forecasts to derive the current deterministic observe state."""
    consecutive_misses = 0
    observe_remaining = 0
    recovery_pending = False
    virtual_results: list[bool] = []
    official_scored = 0
    official_hits = 0
    triggers = 0
    last_virtual_confidence = 0.5

    for index in range(MIN_HISTORY_FOR_SCORING, len(sequence)):
        prefix = sequence[:index]
        actual = sequence[index]
        forecast = decayed_markov_forecast(prefix)
        predicted = str(forecast["direction"])
        correct = predicted == actual
        last_virtual_confidence = float(forecast["confidence"])

        if observe_remaining > 0 or recovery_pending:
            virtual_results.append(bool(correct))
            if observe_remaining > 0:
                observe_remaining -= 1

            if observe_remaining <= 0:
                window = virtual_results[-RECOVERY_WINDOW:]
                hits = sum(1 for item in window if item)
                calibrated = (
                    len(window) >= RECOVERY_WINDOW
                    and hits >= RECOVERY_MIN_HITS
                )
                if calibrated:
                    recovery_pending = False
                    consecutive_misses = 0
                else:
                    recovery_pending = True
            continue

        official_scored += 1
        if correct:
            official_hits += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= PENALTY_CONSECUTIVE_MISSES:
                triggers += 1
                observe_remaining = PENALTY_MIN_OBSERVE_ROUNDS
                recovery_pending = False
                virtual_results = []
                consecutive_misses = 0

    recent_virtual = virtual_results[-RECOVERY_WINDOW:]
    recent_virtual_hits = sum(1 for item in recent_virtual if item)
    active = observe_remaining > 0 or recovery_pending
    return {
        "active": bool(active),
        "force_observe": bool(active),
        "observe_remaining": int(observe_remaining),
        "recovery_pending": bool(recovery_pending),
        "consecutive_misses": int(consecutive_misses),
        "trigger_count": int(triggers),
        "official_scored": int(official_scored),
        "official_hits": int(official_hits),
        "official_accuracy": float(official_hits / max(1, official_scored)),
        "virtual_sample_count": len(virtual_results),
        "recent_virtual_sample_count": len(recent_virtual),
        "recent_virtual_hits": int(recent_virtual_hits),
        "recent_virtual_accuracy": float(
            recent_virtual_hits / max(1, len(recent_virtual))
        ),
        "minimum_observe_rounds": int(PENALTY_MIN_OBSERVE_ROUNDS),
        "recovery_window": int(RECOVERY_WINDOW),
        "recovery_min_hits": int(RECOVERY_MIN_HITS),
        "recovery_confidence": float(RECOVERY_CONFIDENCE),
        "semantics": (
            "two_consecutive_misses_then_minimum_three_virtual_rounds_"
            "and_recover_only_after_virtual_hit_rate_recalibrates"
        ),
    }


def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """Return road-only next-round probabilities plus penalty-observe state."""
    sequence = normalize_big_road(history)
    forecast = decayed_markov_forecast(sequence)
    penalty = _replay_penalty_state(sequence)
    direction = str(forecast["direction"])
    action = "O" if penalty["active"] else direction
    return {
        **forecast,
        "action": action,
        "action_text": "觀望" if action == "O" else ("莊" if action == "B" else "閒"),
        "latent_direction": direction,
        "penalty_observe": penalty,
        "big_road_sequence": "".join(sequence),
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
    """Compatibility diagnostic only; it does not vote on the next direction."""
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
        "triggered": bool(
            consecutive_losses >= ONLINE_CONSECUTIVE_LOSS_TRIGGER
        ),
        "window": int(ONLINE_WINDOW),
        "loss_trigger": int(ONLINE_CONSECUTIVE_LOSS_TRIGGER),
        "semantics": "diagnostic_only_road_model_replay_controls_penalty_state",
    }


def install_dynamic_prediction_policy() -> bool:
    """Compatibility installation hook used by runtime_app.

    The predictor imports and calls :func:`road_only_policy` directly, so no
    monkey-patching of OCR, screenshot, transport or quant-engine code is needed.
    """
    global _INSTALLED
    _INSTALLED = True
    return True


__all__ = [
    "POLICY_VERSION",
    "MARKOV_MAX_ORDER",
    "MARKOV_DECAY",
    "MIN_DIRECTION_CONFIDENCE",
    "EARLY_MIN_DIRECTION_CONFIDENCE",
    "PHYSICAL_MIN_EV",
    "EARLY_ACTIVE_MAX_ROUNDS",
    "TEMPERATURE_SCALING_MAX_ROUNDS",
    "EARLY_TEMPERATURE",
    "normalize_big_road",
    "decayed_markov_forecast",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
