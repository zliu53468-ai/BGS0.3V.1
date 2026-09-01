"""Compatibility policy layer for the Single-Brain LinUCB production core.

Formal B/P direction, probability and confidence come only from Contextual LinUCB.
Legacy road-pattern, Markov, regression and LSTM entry points remain available for
API/import compatibility, but they are diagnostics with zero formal direction
weight and cannot override the LinUCB argmax.
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
from performance_tracker import get_resolved_records

POLICY_VERSION = "LINUCB-32D-BBB-WEB-PARITY-V10"
OUTCOMES = ("B", "P")
WINDOW_SIZE = 24
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
        values: Iterable[Any] = list(history.replace("|", "").replace(",", "").replace(" ", ""))
    else:
        values = history
    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in OUTCOMES:
            result.append(value)
    return result[-2000:]


def _least_squares_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    n = int(x.size)
    if n < 2 or y.size != x.size:
        base = float(y[-1]) if y.size else 0.0
        return 0.0, base, 0.0
    x_mean = float(np.mean(x)); y_mean = float(np.mean(y))
    centered_x = x - x_mean
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 1e-12:
        return 0.0, y_mean, 0.0
    slope = float(np.dot(centered_x, y - y_mean) / denominator)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2)); ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, _clip(r2)


def regression_analysis_model(history: str | Iterable[Any] | None) -> dict[str, Any]:
    sequence = normalize_big_road(history)
    n = len(sequence)
    if n == 0:
        return {"available": False, "rounds": 0, "global_slope": 0.0, "local_slope": 0.0, "global_r_squared": 0.0, "local_r_squared": 0.0, "regime": "diagnostic_only", "regression_p_b": 0.5, "regression_p_p": 0.5, "diagnostic_only": True, "formal_direction_weight": 0.0}
    encoded = np.asarray([1.0 if side == "B" else -1.0 for side in sequence], dtype=np.float64)
    cumulative = np.cumsum(encoded); x = np.arange(1, n + 1, dtype=np.float64)
    global_slope, global_intercept, global_r2 = _least_squares_line(x, cumulative)
    local_window = min(REGRESSION_LOCAL_WINDOW, n)
    local_slope, local_intercept, local_r2 = _least_squares_line(x[-local_window:], cumulative[-local_window:])
    p_b = _clip(0.5 + 0.5 * math.tanh(0.15 * global_slope + 0.15 * local_slope), 0.01, 0.99)
    return {"available": True, "rounds": n, "global_slope": float(global_slope), "local_slope": float(local_slope), "global_intercept": float(global_intercept), "local_intercept": float(local_intercept), "global_r_squared": float(global_r2), "local_r_squared": float(local_r2), "regime": "diagnostic_only", "regression_p_b": float(p_b), "regression_p_p": float(1.0 - p_b), "local_window": local_window, "diagnostic_only": True, "formal_direction_weight": 0.0}


def time_decay_markov_fallback(history: str | Iterable[Any] | None, *, decay: float = MARKOV_DECAY) -> dict[str, Any]:
    sequence = normalize_big_road(history); decay = _clip(decay, 0.50, 0.999)
    context = sequence[-1] if sequence else None
    context_counts = {"B": 0.0, "P": 0.0}; global_counts = {"B": 0.0, "P": 0.0}
    if len(sequence) >= 2:
        last_pair_index = len(sequence) - 2
        for index, (left, right) in enumerate(zip(sequence, sequence[1:])):
            age = max(0, last_pair_index - index); weight = decay ** age
            global_counts[right] += weight
            if context is not None and left == context:
                context_counts[right] += weight
    context_support = sum(context_counts.values()); global_support = sum(global_counts.values()); alpha = 1.25
    context_p = {side: (context_counts[side] + alpha) / (context_support + 2.0 * alpha) for side in OUTCOMES}
    global_p = {side: ((global_counts[side] + alpha) / (global_support + 2.0 * alpha) if global_support > 0 else 0.5) for side in OUTCOMES}
    specificity = context_support / (context_support + 3.0) if context_support > 0 else 0.0
    p_b = specificity * context_p["B"] + (1.0 - specificity) * global_p["B"]
    p_p = specificity * context_p["P"] + (1.0 - specificity) * global_p["P"]
    total = p_b + p_p; p_b, p_p = (0.5, 0.5) if total <= 1e-12 else (p_b / total, p_p / total)
    direction = "B" if p_b >= p_p else "P"; confidence = max(p_b, p_p)
    return {"available": True, "model_id": "TIME-DECAY-MARKOV-ORDER1-DIAGNOSTIC", "direction": direction, "action": direction, "action_text": "莊" if direction == "B" else "閒", "probabilities": {"B": float(p_b), "P": float(p_p), "T": 0.0}, "selected_win_probability": float(confidence), "confidence": float(confidence), "context": context, "context_counts": context_counts, "global_counts": global_counts, "context_support": float(context_support), "global_support": float(global_support), "decay": float(decay), "selected_order": 1 if context is not None else 0, "history_rounds": len(sequence), "diagnostic_only": True, "formal_direction_weight": 0.0, "semantics": "legacy_markov_diagnostic_only_contextual_linucb_is_formal"}


def _road_pattern_diagnostic(history: str | Iterable[Any] | None) -> dict[str, Any]:
    try:
        from road_pattern_core import forecast_road_pattern
        value = dict(forecast_road_pattern(history) or {})
        value["diagnostic_only"] = True; value["formal_direction_weight"] = 0.0
        return value
    except Exception as exc:
        return {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0, "reason": f"diagnostic_unavailable:{type(exc).__name__}"}


def linucb_policy(history: str | Iterable[Any] | None, *, shoe_context: Mapping[str, Any] | None = None, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> dict[str, Any]:
    scope_key = make_scope_key(user_id=user_id, venue=venue, room=room, shoe_id=shoe_id)
    result = predict_bandit(history=history, shoe_context=dict(shoe_context or {}), scope_key=scope_key)
    direction = str(result.get("direction") or "B").upper().strip()
    if direction not in OUTCOMES:
        direction = "B"
    probabilities = dict(result.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    confidence = float(result.get("selected_win_probability", probabilities.get(direction, 0.5)) or 0.5)
    result.update({
        "direction": direction, "selected_arm": direction, "action": direction, "action_text": "莊" if direction == "B" else "閒", "latent_direction": direction,
        "confidence_prob": confidence, "margin": abs(float(probabilities.get("B", 0.5)) - float(probabilities.get("P", 0.5))),
        "regression_analysis": regression_analysis_model(history), "penalty_observe": {"active": False, "force_observe": False, "observe_remaining": 0},
        "big_road_sequence": "".join(normalize_big_road(history)[-WINDOW_SIZE:]), "state_key": "LINUCB_SINGLE_BRAIN_V5",
        "policy_source": "contextual_linucb", "formal_direction_source": "contextual_linucb", "diagnostic_only": False, "formal_direction_weight": 1.0,
        "linucb_direction_weight": 1.0, "road_pattern_direction_weight": 0.0, "road_forecaster_direction_weight": 0.0,
        "derived_road_direction_weight": 0.0, "geometry_direction_weight": 0.0, "anti_echo_direction_weight": 0.0,
        "shoe_direction_weight": 0.0, "card_composition_direction_weight": 0.0, "lstm_direction_weight": 0.0,
        "road_primary": False, "lstm_primary": False, "lstm_shoe_cut_fusion": False,
        "lstm": {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0, "reason": "disabled_single_brain_linucb"},
    })
    return result


def road_only_policy(history: str | Iterable[Any] | None, *, shoe_context: Mapping[str, Any] | None = None, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "") -> dict[str, Any]:
    result = linucb_policy(history, shoe_context=shoe_context, user_id=user_id, venue=venue, room=room, shoe_id=shoe_id)
    result["fallback_markov"] = time_decay_markov_fallback(history)
    result["road_pattern"] = _road_pattern_diagnostic(history)
    result["road_pattern_model_id"] = str(result["road_pattern"].get("model_id") or "")
    result["road_forecaster_diagnostic"] = {"available": False, "direction": None, "probabilities": {}, "selected_win_probability": 0.5, "diagnostic_only": True, "formal_direction_weight": 0.0}
    result["road_forecaster"] = {"available": False, "diagnostic_only": True, "formal_direction_weight": 0.0}
    result["fallback_active"] = False; result["fallback_reason"] = ""
    return result


def lstm_primary_policy(history: str | Iterable[Any] | None, *, shoe_context: Mapping[str, Any] | None = None, user_id: str = "", venue: str = "", room: str = "", shoe_id: str = "", allow_online_update: bool = True) -> dict[str, Any]:
    del allow_online_update
    return road_only_policy(history, shoe_context=shoe_context, user_id=user_id, venue=venue, room=room, shoe_id=shoe_id)


class ShortShoePredictor:
    def __init__(self, window_size: int = WINDOW_SIZE, decay: float = MARKOV_DECAY) -> None:
        self.window_size = max(4, int(window_size or WINDOW_SIZE)); self.decay = float(decay)
    def predict(self, history: str | Iterable[Any] | None) -> dict[str, Any]:
        result = road_only_policy(history); sequence = normalize_big_road(history)
        result["window_size"] = self.window_size; result["window_rounds"] = min(len(sequence), self.window_size)
        result["history_rounds"] = len(sequence); result["sequence_length"] = len(sequence); result["model"] = "short_shoe_contextual_linucb_single_brain"
        return result


def decayed_markov_forecast(history: str | Iterable[Any] | None, *, decay: float = MARKOV_DECAY, max_order: int = MARKOV_MAX_ORDER) -> dict[str, Any]:
    del max_order
    return time_decay_markov_fallback(history, decay=decay)


def global_trend_bias_correction(history: str | Iterable[Any] | None, local_forecast: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(local_forecast or {}); probabilities = dict(supplied.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    p_b = _clip(probabilities.get("B", 0.5)); p_p = _clip(probabilities.get("P", 0.5)); total = p_b + p_p
    p_b, p_p = (0.5, 0.5) if total <= 1e-12 else (p_b / total, p_p / total)
    direction = str(supplied.get("direction") or ("B" if p_b >= p_p else "P")).upper().strip()
    if direction not in OUTCOMES: direction = "B" if p_b >= p_p else "P"
    sequence = normalize_big_road(history)
    return {"applied": False, "mode": "single_brain_linucb_passthrough", "ensemble_regime": "disabled_external_ensemble", "local_weight": 0.0, "global_weight": 0.0, "regression_weight": 0.0, "ensemble_weights": {"contextual_linucb": 1.0, "road_pattern_core": 0.0, "time_decay_markov": 0.0, "regression": 0.0}, "anti_lock_applied": False, "direction_controller": {"mode": "contextual_linucb", "final_direction": direction, "final_p_b": p_b, "final_p_p": p_p}, "total_rounds": len(sequence), "total_b": sum(side == "B" for side in sequence), "total_p": sum(side == "P" for side in sequence), "global_p_b": 0.5, "global_p_p": 0.5, "local_p_b": p_b, "local_p_p": p_p, "raw_ensemble_p_b": p_b, "raw_ensemble_p_p": p_p, "raw_ensemble_direction": direction, "final_p_b": p_b, "final_p_p": p_p, "final_direction": direction, "final_confidence_prob": float(probabilities.get(direction, 0.5)), "formula": "passthrough_contextual_linucb_only", "diagnostic_only": True, "formal_direction_weight": 0.0}


def shoe_progress_policy(rounds: int) -> dict[str, Any]:
    value = max(0, int(rounds or 0)); phase = "EARLY" if value < 20 else "MID" if value < 50 else "TARGET_50_70"
    return {"rounds": value, "phase": phase, "shoe_weight_factor": 1.0, "road_weight_factor": 0.0, "formal_model": "contextual_linucb", "semantics": "shoe_information_is_context_feature_not_separate_vote"}


def recent_user_direction_feedback(user_id: str, *, limit: int = ONLINE_WINDOW) -> dict[str, Any]:
    raw_user = str(user_id or "")
    if not raw_user:
        return {"available": False, "sample_count": 0, "correct_count": 0, "accuracy": 0.0, "consecutive_losses": 0, "triggered": False}
    uid_key = sha256(raw_user.encode("utf-8")).hexdigest()[:24]
    try: records = get_resolved_records(limit=5000)
    except Exception: records = []
    recent: list[dict[str, Any]] = []
    for record in reversed(records):
        if str(record.get("uid_key") or "") != uid_key: continue
        actual = str(record.get("actual_outcome") or "").upper().strip(); predicted = str(record.get("action") or record.get("recommend") or "").upper().strip()
        if actual not in OUTCOMES or predicted not in OUTCOMES: continue
        recent.append({"correct": predicted == actual})
        if len(recent) >= max(1, int(limit)): break
    correct = sum(item["correct"] for item in recent); consecutive_losses = 0
    for item in recent:
        if item["correct"]: break
        consecutive_losses += 1
    return {"available": bool(recent), "sample_count": len(recent), "correct_count": int(correct), "accuracy": float(correct / max(1, len(recent))), "consecutive_losses": consecutive_losses, "triggered": False, "window": ONLINE_WINDOW, "loss_trigger": ONLINE_CONSECUTIVE_LOSS_TRIGGER, "semantics": "diagnostic_feedback_formal_direction_is_contextual_linucb"}


def record_online_feedback(*, scope_key: str, action: str, context_vector: Sequence[float], actual_outcome: str) -> dict[str, Any]:
    return update_bandit(scope_key=scope_key, action=action, context_vector=context_vector, actual_outcome=actual_outcome, clear_pending=True)


def install_dynamic_prediction_policy() -> bool:
    global _INSTALLED
    _INSTALLED = True
    return True


__all__ = [
    "POLICY_VERSION", "WINDOW_SIZE", "MARKOV_MAX_ORDER", "MARKOV_DECAY", "MIN_DIRECTION_CONFIDENCE", "EARLY_MIN_DIRECTION_CONFIDENCE",
    "PHYSICAL_MIN_EV", "EARLY_ACTIVE_MAX_ROUNDS", "TEMPERATURE_SCALING_MAX_ROUNDS", "EARLY_TEMPERATURE", "GLOBAL_LOCAL_WEIGHT", "GLOBAL_TREND_WEIGHT",
    "REGRESSION_LOCAL_WINDOW", "REGRESSION_GLOBAL_STRONG_SLOPE", "REGRESSION_LOCAL_STRONG_SLOPE", "CONTEXT_DIM", "CONTEXT_FEATURE_NAMES", "LINUCB_ALPHA", "LINUCB_RIDGE",
    "ShortShoePredictor", "normalize_big_road", "time_decay_markov_fallback", "decayed_markov_forecast", "regression_analysis_model", "global_trend_bias_correction",
    "linucb_policy", "lstm_primary_policy", "road_only_policy", "shoe_progress_policy", "recent_user_direction_feedback", "record_online_feedback", "install_dynamic_prediction_policy",
]
