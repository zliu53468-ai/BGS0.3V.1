"""Road-only short-shoe prediction policy for BGS.

Public helper names and return fields are preserved for the existing runtime.
The formal decision uses only the OCR-derived chronological Big Road B/P history.
Exact remaining-card counts are not required.

The existing 12-round predictor is kept unchanged.  A Bias & Momentum Adjuster
runs outside that core and only corrects the final direction/confidence when the
full-shoe B/P skew and short-term momentum satisfy the requested conditions.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import math

from performance_tracker import get_resolved_records

POLICY_VERSION = "ROAD-ONLY-SHORT-SHOE-W12-BIAS-MOMENTUM-V2"

OUTCOMES = ("B", "P")
WINDOW_SIZE = 12
MARKOV_MAX_ORDER = 1
MARKOV_DECAY = 0.93
MARKOV_PRIOR_STRENGTH = 0.50
MARKOV_MIN_EFFECTIVE_SUPPORT = 0.0
MIN_HISTORY_FOR_SCORING = 4

ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2

# Compatibility constants retained.  The historical replay penalty has no
# authority to force O; the short sliding window remains the base predictor.
PENALTY_CONSECUTIVE_MISSES = 2
PENALTY_MIN_OBSERVE_ROUNDS = 0
RECOVERY_WINDOW = 0
RECOVERY_MIN_HITS = 0
RECOVERY_CONFIDENCE = 0.50

# Bias & Momentum Adjuster constants required by the patch.
BIAS_MIN_ROUNDS = 16
BIAS_MIN_DELTA = 6
MOMENTUM_WINDOW = 6
MOMENTUM_MIN_HITS = 5
CATCHUP_WINDOW = 4
CATCHUP_MIN_HITS = 3
TREND_CONFIDENCE_MULTIPLIER = 1.05
COUNTERTREND_CONFIDENCE_MULTIPLIER = 0.90
CATCHUP_MIN_CONFIDENCE = 0.53

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

            # Existing 12-hand core logic is intentionally unchanged.
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
    """Compatibility entrypoint backed by the 12-round ShortShoePredictor."""
    del max_order
    return ShortShoePredictor(window_size=WINDOW_SIZE, decay=decay).predict(history)


def _trailing_run(sequence: Sequence[str], side: str) -> int:
    run = 0
    for outcome in reversed(sequence):
        if outcome != side:
            break
        run += 1
    return run


def bias_momentum_adjuster(
    history: str | Iterable[Any] | None,
    base_forecast: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the requested full-shoe skew and short-momentum corrections."""
    sequence = normalize_big_road(history)
    total_rounds = len(sequence)
    banker_count = sum(1 for value in sequence if value == "B")
    player_count = sum(1 for value in sequence if value == "P")
    delta = abs(banker_count - player_count)

    skew_active = bool(total_rounds > 15 and delta >= BIAS_MIN_DELTA)
    if banker_count > player_count:
        advantage_side = "B"
        weak_side = "P"
    elif player_count > banker_count:
        advantage_side = "P"
        weak_side = "B"
    else:
        advantage_side = ""
        weak_side = ""

    recent6 = sequence[-MOMENTUM_WINDOW:]
    recent6_b = sum(1 for value in recent6 if value == "B")
    recent6_p = sum(1 for value in recent6 if value == "P")
    if recent6_b >= MOMENTUM_MIN_HITS:
        momentum_side = "B"
    elif recent6_p >= MOMENTUM_MIN_HITS:
        momentum_side = "P"
    else:
        momentum_side = ""

    recent4 = sequence[-CATCHUP_WINDOW:]
    weak_recent4 = (
        sum(1 for value in recent4 if value == weak_side)
        if weak_side in OUTCOMES
        else 0
    )
    weak_trailing_run = (
        _trailing_run(sequence, weak_side)
        if weak_side in OUTCOMES
        else 0
    )
    catchup_active = bool(
        skew_active
        and weak_side in OUTCOMES
        and weak_recent4 >= CATCHUP_MIN_HITS
        and weak_trailing_run >= CATCHUP_MIN_HITS
    )

    base_direction = str(base_forecast.get("direction") or "").upper().strip()
    if base_direction not in OUTCOMES:
        base_direction = "B"
    try:
        base_confidence = float(
            base_forecast.get(
                "confidence_prob",
                base_forecast.get("confidence", 0.50),
            )
            or 0.50
        )
    except (TypeError, ValueError):
        base_confidence = 0.50
    base_confidence = _clip(base_confidence, 0.0, 0.999)

    adjusted_direction = base_direction
    adjusted_confidence = base_confidence
    mode = "base_12_round_unchanged"
    applied = False

    # Scenario 2 has priority: the weak side is actively catching up with a
    # consecutive 3/4 or 4/4 burst inside the latest four rounds.
    if catchup_active:
        adjusted_direction = weak_side
        adjusted_confidence = max(
            CATCHUP_MIN_CONFIDENCE,
            base_confidence if base_direction == weak_side else CATCHUP_MIN_CONFIDENCE,
        )
        adjusted_confidence = _clip(adjusted_confidence, 0.50, 0.999)
        mode = "catchup_momentum_override"
        applied = True

    # Scenario 1: full-shoe skew remains wide and the latest six-hand momentum
    # still belongs to the already-advantaged side.
    elif skew_active and momentum_side == advantage_side and advantage_side in OUTCOMES:
        factor = (
            TREND_CONFIDENCE_MULTIPLIER
            if base_direction == advantage_side
            else COUNTERTREND_CONFIDENCE_MULTIPLIER
        )
        weighted_confidence = _clip(base_confidence * factor, 0.0, 0.999)
        if weighted_confidence < 0.50:
            adjusted_direction = "P" if base_direction == "B" else "B"
            adjusted_confidence = 1.0 - weighted_confidence
        else:
            adjusted_direction = base_direction
            adjusted_confidence = weighted_confidence
        adjusted_confidence = _clip(adjusted_confidence, 0.50, 0.999)
        mode = (
            "advantage_trend_boost"
            if base_direction == advantage_side
            else "weak_side_suppression"
        )
        applied = True

    adjusted_probabilities = {
        "B": (
            adjusted_confidence
            if adjusted_direction == "B"
            else 1.0 - adjusted_confidence
        ),
        "P": (
            adjusted_confidence
            if adjusted_direction == "P"
            else 1.0 - adjusted_confidence
        ),
        "T": 0.0,
    }

    return {
        "applied": bool(applied),
        "mode": mode,
        "total_rounds": int(total_rounds),
        "banker_count": int(banker_count),
        "player_count": int(player_count),
        "delta": int(delta),
        "skew_active": bool(skew_active),
        "advantage_side": advantage_side,
        "weak_side": weak_side,
        "recent6": "".join(recent6),
        "recent6_banker_count": int(recent6_b),
        "recent6_player_count": int(recent6_p),
        "momentum_side": momentum_side,
        "recent4": "".join(recent4),
        "weak_recent4_count": int(weak_recent4),
        "weak_trailing_run": int(weak_trailing_run),
        "catchup_active": bool(catchup_active),
        "base_direction": base_direction,
        "base_confidence_prob": float(base_confidence),
        "adjusted_direction": adjusted_direction,
        "adjusted_confidence_prob": float(adjusted_confidence),
        "adjusted_probabilities": dict(adjusted_probabilities),
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
        "semantics": "disabled_bias_momentum_adjuster_no_observe_gate",
    }


def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """Return 12-hand forecast after the external Bias & Momentum correction."""
    sequence = normalize_big_road(history)
    base_forecast = decayed_markov_forecast(sequence)
    adjustment = bias_momentum_adjuster(sequence, base_forecast)
    penalty = _replay_penalty_state(sequence)

    forecast = dict(base_forecast)
    direction = str(adjustment["adjusted_direction"])
    confidence_prob = float(adjustment["adjusted_confidence_prob"])
    probabilities = dict(adjustment["adjusted_probabilities"])
    forecast.update({
        "probabilities": probabilities,
        "direction": direction,
        "confidence": confidence_prob,
        "confidence_prob": confidence_prob,
        "margin": float(abs(probabilities["B"] - probabilities["P"])),
        "bias_momentum_adjuster": dict(adjustment),
    })

    return {
        **forecast,
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "latent_direction": direction,
        "confidence_prob": confidence_prob,
        "penalty_observe": penalty,
        "bias_momentum_adjuster": dict(adjustment),
        "big_road_sequence": str(base_forecast.get("window_sequence") or ""),
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
    """Compatibility diagnostic only; it never gates the short-window forecast."""
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
        "semantics": "diagnostic_only_no_observe_gate",
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
    "ShortShoePredictor",
    "normalize_big_road",
    "decayed_markov_forecast",
    "bias_momentum_adjuster",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "install_dynamic_prediction_policy",
]
