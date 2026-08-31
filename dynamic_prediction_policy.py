"""BGS 動態決策相容層：正式方向由因果式 road_forecaster 產生。

保留舊模組常用 helper 名稱，避免其他程式 import 失效；但所有正式 P/B
方向經既有 contextual_bandit 入口轉接 forecaster 機率 argmax。
LinUCB 與事後迴歸僅供診斷，不能覆蓋正式下一手方向。
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
from road_forecaster import VERSION as FORECASTER_VERSION

POLICY_VERSION = FORECASTER_VERSION
OUTCOMES = ("B", "P")
WINDOW_SIZE = 12
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


def _least_squares_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
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


def regression_analysis_model(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """事後累積斜率只作診斷，不進入正式 forecaster 或決定下一手方向。"""
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
            "regime": "neutral",
            "regression_p_b": 0.5,
            "regression_p_p": 0.5,
            "diagnostic_only": True,
        }
    encoded = np.asarray([1.0 if side == "B" else -1.0 for side in sequence], dtype=np.float64)
    cumulative = np.cumsum(encoded)
    x = np.arange(1, n + 1, dtype=np.float64)
    global_slope, global_intercept, global_r2 = _least_squares_line(x, cumulative)
    local_window = min(REGRESSION_LOCAL_WINDOW, n)
    local_slope, local_intercept, local_r2 = _least_squares_line(
        x[-local_window:], cumulative[-local_window:]
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


# 保留 linucb_policy 名稱與參數；predict_bandit 已轉接真正的下一手 forecaster。
# 16 維 legacy context、LinUCB 與下方 regression 均沒有方向否決權。
def linucb_policy(
    history: str | Iterable[Any] | None,
    *,
    shoe_context: Mapping[str, Any] | None = None,
    user_id: str = "",
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
) -> dict[str, Any]:
    scope_key = make_scope_key(
        user_id=user_id, venue=venue, room=room, shoe_id=shoe_id
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
            "state_key": "LINUCB",
            "transition_counts": {},
            "effective_support": float(result["road_forecaster"]["effective_support"]),
            "decay": 0.0,
            "selected_order": 0,
            "max_order": 0,
            "order_diagnostics": [],
            "global_probabilities": {"B": 0.5, "P": 0.5},
            "policy_source": "road_forecaster",
        }
    )
    return result


class ShortShoePredictor:
    """舊類別名稱相容層；正式內容為下一手 road_forecaster。"""

    def __init__(self, window_size: int = WINDOW_SIZE, decay: float = MARKOV_DECAY):
        self.window_size = max(4, int(window_size or WINDOW_SIZE))
        self.decay = float(decay)

    def predict(self, history: str | Iterable[Any] | None) -> dict[str, Any]:
        result = linucb_policy(history)
        result["window_size"] = self.window_size
        result["window_rounds"] = min(len(normalize_big_road(history)), self.window_size)
        result["history_rounds"] = len(normalize_big_road(history))
        result["sequence_length"] = len(normalize_big_road(history))
        result["model"] = "short_shoe_road_forecaster_compatibility"
        return result


def decayed_markov_forecast(
    history: str | Iterable[Any] | None,
    *,
    decay: float = MARKOV_DECAY,
    max_order: int = MARKOV_MAX_ORDER,
) -> dict[str, Any]:
    del decay, max_order
    return ShortShoePredictor().predict(history)


def global_trend_bias_correction(
    history: str | Iterable[Any] | None,
    local_forecast: Mapping[str, Any],
) -> dict[str, Any]:
    del local_forecast
    policy = linucb_policy(history)
    p_b = float(policy["probabilities"]["B"])
    p_p = float(policy["probabilities"]["P"])
    return {
        "applied": True,
        "mode": "road_forecaster_formal_direction_compatibility",
        "ensemble_regime": "road_forecaster",
        "local_weight": 0.0,
        "global_weight": 0.0,
        "regression_weight": 0.0,
        "ensemble_weights": {"linucb": 0.0, "road_forecaster": 1.0},
        "anti_lock_applied": False,
        "direction_controller": {
            "mode": "road_forecaster_probability_argmax",
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
        "regression_p_b": float(policy["regression_analysis"].get("regression_p_b", 0.5)),
        "regression_p_p": float(policy["regression_analysis"].get("regression_p_p", 0.5)),
        "regression_analysis": dict(policy["regression_analysis"]),
        "raw_ensemble_p_b": p_b,
        "raw_ensemble_p_p": p_p,
        "raw_ensemble_direction": policy["direction"],
        "final_p_b": p_b,
        "final_p_p": p_p,
        "final_direction": policy["direction"],
        "final_confidence_prob": float(policy["selected_win_probability"]),
        "formula": "argmax(causal_road_forecaster.p_b, causal_road_forecaster.p_p)",
    }


def road_only_policy(history: str | Iterable[Any] | None) -> dict[str, Any]:
    """保留舊入口名稱與簽名；呼叫正式下一手 forecaster。"""
    return linucb_policy(history, shoe_context={})


def shoe_progress_policy(rounds: int) -> dict[str, Any]:
    value = max(0, int(rounds or 0))
    phase = "EARLY" if value <= 20 else "MID" if value < 41 else "LATE"
    return {
        "rounds": value,
        "phase": phase,
        "shoe_weight_factor": 1.0,
        "road_weight_factor": 0.25,
    }


def recent_user_direction_feedback(
    user_id: str, *, limit: int = ONLINE_WINDOW
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
        predicted = str(record.get("action") or record.get("recommend") or "").upper().strip()
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
        "semantics": "diagnostic_only_formal_direction_is_road_forecaster",
    }


def record_online_feedback(
    *,
    scope_key: str,
    action: str,
    context_vector: Sequence[float],
    actual_outcome: str,
) -> dict[str, Any]:
    """保留結算 API；LinUCB 診斷更新，forecaster 於下一次已揭曉 history 重播訓練。"""
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
    "decayed_markov_forecast",
    "regression_analysis_model",
    "global_trend_bias_correction",
    "linucb_policy",
    "road_only_policy",
    "shoe_progress_policy",
    "recent_user_direction_feedback",
    "record_online_feedback",
    "install_dynamic_prediction_policy",
]
