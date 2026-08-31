"""BGS prediction compatibility layer for the production LSTM-shoe-cut core.

Formal direction has exactly one owner: ``lstm_road_model.predict_lstm_road``.
That model fuses Big-Road LSTM sequence evidence, exact remaining shoe
composition (when available), and 50-70 hand cut-card depth weighting.

Legacy Markov / LinUCB / road forecaster helpers remain import-compatible and
are diagnostic only.  They cannot override the formal B/P direction.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import math

import numpy as np

from contextual_bandit import (
    CONTEXT_DIM,
    CONTEXT_FEATURE_NAMES,
    LINUCB_ALPHA,
    LINUCB_RIDGE,
    make_scope_key,
    predict_bandit,
    update_bandit,
)
from lstm_road_model import (
    MODEL_ID as LSTM_MODEL_ID,
    MIN_HISTORY as LSTM_MIN_HISTORY,
    WINDOW_SIZE as LSTM_WINDOW_SIZE,
    predict_lstm_road,
)
from performance_tracker import get_resolved_records
from road_forecaster import VERSION as FORECASTER_VERSION

POLICY_VERSION = f"LSTM-SHOE-CUT-PRODUCTION-V2|{FORECASTER_VERSION}"
OUTCOMES = ("B", "P")
WINDOW_SIZE = LSTM_WINDOW_SIZE
MARKOV_MAX_ORDER = 1
MARKOV_DECAY = 0.93
MIN_DIRECTION_CONFIDENCE = 0.48
EARLY_MIN_DIRECTION_CONFIDENCE = 0.48
PHYSICAL_MIN_EV = 0.0
EARLY_ACTIVE_MAX_ROUNDS = 30
TEMPERATURE_SCALING_MAX_ROUNDS = 0
EARLY_TEMPERATURE = 1.0
GLOBAL_LOCAL_WEIGHT = 0.0
GLOBAL_TREND_WEIGHT = 0.0
REGRESSION_LOCAL_WINDOW = 8
REGRESSION_GLOBAL_STRONG_SLOPE = 0.20
REGRESSION_LOCAL_STRONG_SLOPE = 0.35
ONLINE_WINDOW = 5
ONLINE_CONSECUTIVE_LOSS_TRIGGER = 2
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
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return 0.0, y_mean, 0.0
    slope = float(np.dot(centered_x, y - y_mean) / denominator)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, _clip(r2)


def regression_analysis_model(
    history: str | Iterable[Any] | None,
) -> dict[str, Any]:
    """Legacy cumulative-slope diagnostic; never owns formal direction."""
    sequence = normalize_big_road(history)
    n = len(sequence)
    if n == 0:
        return {
            "available": False,
            "rounds": 0,
            "global_slope": 0.0,
            "local_slope": 0.0,
            "global_r_squared": 0.0,
            "local_r_squared": 0.0,
            "regime": "diagnostic_only",
            "regression_p_b": 0.5,
            "regression_p_p": 0.5,
            "diagnostic_only": True,
        }
    encoded = np.asarray(
        [1.0 if side == "B" else -1.0 for side in sequence],
        dtype=np.float64,
    )
    cumulative = np.cumsum(encoded)
    x = np.arange(1, n + 1, dtype=np.float64)
    global_slope, global_intercept, global_r2 = _least_squares_line(x, cumulative)
    local_window = min(REGRESSION_LOCAL_WINDOW, n)
    local_slope, local_intercept, local_r2 = _least_squares_line(
        x[-local_window:],
        cumulative[-local_window:],
    )
    diagnostic_slope = 0.15 * global_slope + 0.15 * local_slope
    p_b = _clip(0.5 + 0.5 * math.tanh(diagnostic_slope), 0.01, 0.99)
    return {
        "available": True,
        "rounds": n,
        "global_slope": float(global_slope),
        "local_slope": float(local_slope),
        "global_intercept": float(global_intercept),
        "local_intercept": float(local_intercept),
        "global_r_squared": float(global_r2),
        "local_r_squared": float(local_r2),
        "regime": "diagnostic_only",
        "regression_p_b": float(p_b),
        "regression_p_p": float(1.0 - p_b),
        "local_window": local_window,
        "diagnostic_only": True,
    }


def time_decay_markov_fallback(
    history: str | Iterable[Any] | None,
    *,
    decay: float = MARKOV_DECAY,
) -> dict[str, Any]:
    """Legacy diagnostic only.  It is no longer a formal fallback."""
    sequence = normalize_big_road(history)
    decay = _clip(decay, 0.50, 0.999)
    context = sequence[-1] if sequence else None
    context_counts = {"B": 0.0, "P": 0.0}
    global_counts = {"B": 0.0, "P": 0.0}
    if len(sequence) >= 2:
        last_pair_index = len(sequence) - 2
        for index, (left, right) in enumerate(zip(sequence, sequence[1:])):
            age = max(0, last_pair_index - index)
            weight = decay ** age
            global_counts[right] += weight
            if context is not None and left == context:
                context_counts[right] += weight
    if sequence:
        newest_index = len(sequence) - 1
        for index, side in enumerate(sequence):
            age = newest_index - index
            global_counts[side] += 0.20 * (decay ** age)
    context_support = sum(context_counts.values())
    global_support = sum(global_counts.values())
    alpha = 1.25
    context_p = {
        side: (context_counts[side] + alpha) / (context_support + 2.0 * alpha)
        for side in OUTCOMES
    }
    global_p = {
        side: (
            (global_counts[side] + alpha) / (global_support + 2.0 * alpha)
            if global_support > 0.0
            else 0.5
        )
        for side in OUTCOMES
    }
    specificity = (
        context_support / (context_support + 3.0)
        if context_support > 0.0
        else 0.0
    )
    p_b = specificity * context_p["B"] + (1.0 - specificity) * global_p["B"]
    p_p = specificity * context_p["P"] + (1.0 - specificity) * global_p["P"]
    total = p_b + p_p
    if total <= 1e-12:
        p_b = p_p = 0.5
    else:
        p_b, p_p = p_b / total, p_p / total
    direction = "B" if p_b >= p_p else "P"
    confidence = p_b if direction == "B" else p_p
    return {
        "available": True,
        "model_id": "TIME-DECAY-MARKOV-ORDER1-DIAGNOSTIC",
        "direction": direction,
        "action": direction,
        "action_text": "莊" if direction == "B" else "閒",
        "probabilities": {"B": float(p_b), "P": float(p_p), "T": 0.0},
        "selected_win_probability": float(confidence),
        "confidence": float(confidence),
        "context": context,
        "context_counts": context_counts,
        "global_counts": global_counts,
        "context_support": float(context_support),
        "global_support": float(global_support),
        "context_specificity": float(specificity),
        "decay": float(decay),
        "selected_order": 1 if context is not None else 0,
        "history_rounds": len(sequence),
        "fallback_only": False,
        "diagnostic_only": True,
        "formal_direction_weight": 0.0,
        "semantics": "legacy_time_decay_markov_diagnostic_only",
    }


def linucb_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> dict[str, Any]:
    """Legacy contextual road stack retained for diagnostics and feedback only."""
    scope_key = make_scope_key(
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    result = predict_bandit(
        history=history,
        shoe_context=dict(shoe_context or {}),
        scope_key=scope_key,
    )
    regression = regression_analysis_model(history)
    result.update(
        {
            "action": str(result["direction"]),
            "action_text": "莊" if result["direction"] == "B" else "閒",
            "latent_direction": str(result["direction"]),
            "confidence_prob": float(result["selected_win_probability"]),
            "margin": abs(
                float(result["probabilities"]["B"])
                - float(result["probabilities"]["P"])
            ),
            "regression_analysis": regression,
            "penalty_observe": {
                "active": False,
                "force_observe": False,
                "observe_remaining": 0,
                "semantics": "two_arm_only_no_third_arm",
            },
            "big_road_sequence": "".join(normalize_big_road(history)[-WINDOW_SIZE:]),
            "state_key": "LINUCB_DIAGNOSTIC",
            "transition_counts": {},
            "effective_support": float(result["road_forecaster"]["effective_support"]),
            "decay": 0.0,
            "selected_order": 0,
            "max_order": 0,
            "order_diagnostics": [],
            "global_probabilities": {"B": 0.5, "P": 0.5},
            "policy_source": "road_forecaster_diagnostic",
            "diagnostic_only": True,
            "formal_direction_weight": 0.0,
        }
    )
    return result


def lstm_primary_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    allow_online_update: bool = True,
) -> dict[str, Any]:
    """Sole formal policy: LSTM + exact shoe + cut-card fusion."""
    scope_key = make_scope_key(
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    diagnostic = linucb_policy(
        history,
        shoe_context=shoe_context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    legacy_markov = time_decay_markov_fallback(history)
    lstm = predict_lstm_road(
        history,
        scope_key=scope_key,
        shoe_context=dict(shoe_context or {}),
        allow_online_update=allow_online_update,
    )

    raw_probabilities = dict(lstm.get("probabilities") or {})
    p_b = _clip(raw_probabilities.get("B", 0.5))
    p_p = _clip(raw_probabilities.get("P", 0.5))
    total = p_b + p_p
    if total <= 1e-12:
        p_b = p_p = 0.5
    else:
        p_b, p_p = p_b / total, p_p / total
    direction = str(lstm.get("direction") or "").upper().strip()
    if direction not in OUTCOMES:
        direction = "B" if p_b >= p_p else "P"
    confidence = p_b if direction == "B" else p_p
    formal_source = "lstm_road_model"

    result = dict(diagnostic)
    result.update(
        {
            "direction": direction,
            "selected_arm": direction,
            "action": direction,
            "action_text": "莊" if direction == "B" else "閒",
            "latent_direction": direction,
            "probabilities": {"B": float(p_b), "P": float(p_p), "T": 0.0},
            "selected_win_probability": float(confidence),
            "confidence": float(confidence),
            "confidence_prob": float(confidence),
            "margin": abs(float(p_b) - float(p_p)),
            "scope_key": scope_key,
            "formal_direction_source": formal_source,
            "policy_source": formal_source,
            "lstm_primary": True,
            "lstm_shoe_cut_fusion": True,
            "lstm_model_id": LSTM_MODEL_ID,
            "lstm_min_history": int(LSTM_MIN_HISTORY),
            "lstm": dict(lstm),
            # Retained because predictor/API historically exposes this block.
            # It has no formal direction weight in V2.
            "fallback_markov": dict(legacy_markov),
            "fallback_active": False,
            "fallback_reason": "",
            "road_forecaster_diagnostic": {
                "direction": str(diagnostic.get("direction") or ""),
                "probabilities": dict(diagnostic.get("probabilities") or {}),
                "selected_win_probability": float(
                    diagnostic.get("selected_win_probability", 0.5) or 0.5
                ),
            },
            "selection_reason": "single_lstm_shoe_cut_fusion",
            "state_key": "LSTM_SHOE_CUT_FUSION",
            "big_road_sequence": "".join(normalize_big_road(history)[-WINDOW_SIZE:]),
            "diagnostic_only": False,
            "formal_direction_weight": 1.0,
        }
    )
    return result


class ShortShoePredictor:
    """Legacy class name mapped to the formal LSTM-shoe-cut policy."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        decay: float = MARKOV_DECAY,
    ) -> None:
        self.window_size = max(4, int(window_size or WINDOW_SIZE))
        self.decay = float(decay)

    def predict(self, history: str | Iterable[Any] | None) -> dict[str, Any]:
        result = lstm_primary_policy(history)
        result["window_size"] = self.window_size
        result["window_rounds"] = min(
            len(normalize_big_road(history)),
            self.window_size,
        )
        result["history_rounds"] = len(normalize_big_road(history))
        result["sequence_length"] = len(normalize_big_road(history))
        result["model"] = "short_shoe_lstm_shoe_cut_compatibility"
        return result


def decayed_markov_forecast(
    history: str | Iterable[Any] | None,
    *,
    decay: float = MARKOV_DECAY,
    max_order: int = MARKOV_MAX_ORDER,
) -> dict[str, Any]:
    del max_order
    return time_decay_markov_fallback(history, decay=decay)


def global_trend_bias_correction(
    history: str | Iterable[Any] | None,
    local_forecast: Mapping[str, Any],
) -> dict[str, Any]:
    del local_forecast
    policy = lstm_primary_policy(history)
    p_b = float(policy["probabilities"]["B"])
    p_p = float(policy["probabilities"]["P"])
    return {
        "applied": True,
        "mode": "lstm_shoe_cut_formal_compatibility",
        "ensemble_regime": "lstm_shoe_cut_fusion",
        "local_weight": 0.0,
        "global_weight": 0.0,
        "regression_weight": 0.0,
        "ensemble_weights": {
            "lstm_shoe_cut": 1.0,
            "time_decay_markov": 0.0,
            "road_forecaster": 0.0,
        },
        "anti_lock_applied": True,
        "direction_controller": {
            "mode": "single_lstm_shoe_cut_fusion",
            "final_direction": policy["direction"],
            "final_p_b": p_b,
            "final_p_p": p_p,
        },
        "total_rounds": len(normalize_big_road(history)),
        "total_b": sum(side == "B" for side in normalize_big_road(history)),
        "total_p": sum(side == "P" for side in normalize_big_road(history)),
        "global_p_b": 0.5,
        "global_p_p": 0.5,
        "local_p_b": p_b,
        "local_p_p": p_p,
        "regression_p_b": float(
            policy["regression_analysis"].get("regression_p_b", 0.5)
        ),
        "regression_p_p": float(
            policy["regression_analysis"].get("regression_p_p", 0.5)
        ),
        "regression_analysis": dict(policy["regression_analysis"]),
        "raw_ensemble_p_b": p_b,
        "raw_ensemble_p_p": p_p,
        "raw_ensemble_direction": policy["direction"],
        "final_p_b": p_b,
        "final_p_p": p_p,
        "final_direction": policy["direction"],
        "final_confidence_prob": float(policy["selected_win_probability"]),
        "formula": "balanced_masked_LSTM + exact_shoe_logit + cut_depth_weighting",
    }


def road_only_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> dict[str, Any]:
    """Compatibility name; behavior is the formal LSTM-shoe-cut fusion."""
    return lstm_primary_policy(
        history,
        shoe_context=shoe_context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )


def shoe_progress_policy(rounds: int) -> dict[str, Any]:
    value = max(0, int(rounds or 0))
    phase = "EARLY" if value < 20 else "MID" if value < 50 else "TARGET_50_70"
    return {
        "rounds": value,
        "phase": phase,
        "shoe_weight_factor": 1.0,
        "road_weight_factor": 0.0,
        "formal_model": "lstm_shoe_cut_fusion",
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
            record.get("action") or record.get("recommend") or ""
        ).upper().strip()
        if actual not in OUTCOMES or predicted not in OUTCOMES:
            continue
        recent.append({"correct": predicted == actual})
        if len(recent) >= max(1, int(limit)):
            break
    correct = sum(item["correct"] for item in recent)
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
        "consecutive_losses": consecutive_losses,
        "triggered": False,
        "window": ONLINE_WINDOW,
        "loss_trigger": ONLINE_CONSECUTIVE_LOSS_TRIGGER,
        "semantics": "diagnostic_feedback_only_formal_direction_is_lstm_shoe_cut",
    }


def record_online_feedback(
    *,
    scope_key: str,
    action: str,
    context_vector: Sequence[float],
    actual_outcome: str,
) -> dict[str, Any]:
    """Preserve bandit settlement API; it cannot alter formal direction."""
    return update_bandit(
        scope_key=scope_key,
        action=action,
        context_vector=context_vector,
        actual_outcome=actual_outcome,
        clear_pending=True,
    )


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
    "CONTEXT_DIM",
    "CONTEXT_FEATURE_NAMES",
    "LINUCB_ALPHA",
    "LINUCB_RIDGE",
    "ShortShoePredictor",
    "normalize_big_road",
    "time_decay_markov_fallback",
    "decayed_markov_forecast",
    "regression_analysis_model",
    "global_trend_bias_correction",
    "linucb_policy",
    "lstm_primary_policy",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "record_online_feedback",
    "install_dynamic_prediction_policy",
]
