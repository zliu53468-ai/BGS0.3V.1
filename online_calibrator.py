"""B/P/T 線上機率校準器。

採用保守的預測／實際比例修正，且只有在已結算樣本達門檻後才啟用。
它改善的是機率校準與過度自信問題，不會取得真人桌未公開牌序。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import math
import os

from performance_tracker import get_performance_summary


OUTCOMES = ("B", "P", "T")
BASELINE = {"B": 0.458597, "P": 0.446247, "T": 0.095156}
CALIBRATION_ENABLED = os.getenv("ONLINE_CALIBRATION_ENABLED", "1").strip() == "1"
CALIBRATION_MIN_GLOBAL = max(50, int(os.getenv("CALIBRATION_MIN_GLOBAL", "250") or "250"))
CALIBRATION_MIN_VENUE = max(50, int(os.getenv("CALIBRATION_MIN_VENUE", "180") or "180"))
CALIBRATION_MIN_ROOM = max(30, int(os.getenv("CALIBRATION_MIN_ROOM", "100") or "100"))
CALIBRATION_MAX_STRENGTH = max(0.0, min(0.50, float(os.getenv("CALIBRATION_MAX_STRENGTH", "0.25") or "0.25")))
TIE_SIGNAL_MIN_SAMPLES = max(100, int(os.getenv("TIE_SIGNAL_MIN_SAMPLES", "500") or "500"))
TIE_SIGNAL_MIN_PROBABILITY = max(0.10, min(0.30, float(os.getenv("TIE_SIGNAL_MIN_PROBABILITY", "0.125") or "0.125")))
TIE_SIGNAL_MIN_EDGE = max(0.0, min(0.15, float(os.getenv("TIE_SIGNAL_MIN_EDGE", "0.015") or "0.015")))
MIN_DIRECTION_EDGE = max(0.0, min(0.20, float(os.getenv("HG_MIN_DIRECTION_EDGE", "0.016") or "0.016")))


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    data = {key: max(1e-12, float(values.get(key, 0.0) or 0.0)) for key in OUTCOMES}
    total = sum(data.values())
    return {key: data[key] / total for key in OUTCOMES}


def _best_summary(venue: str, room: str) -> tuple[str, Dict[str, Any], int]:
    scopes = [
        ("room", get_performance_summary(venue=venue, room=room, limit=5000), CALIBRATION_MIN_ROOM),
        ("venue", get_performance_summary(venue=venue, room="", limit=5000), CALIBRATION_MIN_VENUE),
        ("global", get_performance_summary(venue="", room="", limit=10000), CALIBRATION_MIN_GLOBAL),
    ]
    for name, summary, minimum in scopes:
        if int(summary.get("sample_count", 0) or 0) >= minimum:
            return name, summary, minimum
    return "none", scopes[-1][1], CALIBRATION_MIN_GLOBAL


def calibrate_prediction(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    result = dict(prediction or {})
    raw = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {
            "B": float(result.get("banker_rate", 0.0) or 0.0) / 100.0,
            "P": float(result.get("player_rate", 0.0) or 0.0) / 100.0,
            "T": float(result.get("tie_rate", 0.0) or 0.0) / 100.0,
        }
    )
    result.setdefault("raw_probabilities", dict(raw))
    scope, summary, minimum = _best_summary(str(venue or ""), str(room or ""))
    sample_count = int(summary.get("sample_count", 0) or 0)
    active = bool(CALIBRATION_ENABLED and scope != "none" and CALIBRATION_MAX_STRENGTH > 0)

    calibrated = dict(raw)
    strength = 0.0
    if active:
        empirical = _normalize(dict(summary.get("empirical_probabilities") or BASELINE))
        mean_predicted = _normalize(dict(summary.get("mean_predicted_probabilities") or BASELINE))
        maturity = min(1.0, (sample_count - minimum + 1) / max(1.0, minimum * 3.0))
        strength = CALIBRATION_MAX_STRENGTH * maturity
        ratio_corrected = {}
        for key in OUTCOMES:
            ratio = max(0.55, min(1.80, empirical[key] / max(1e-9, mean_predicted[key])))
            ratio_corrected[key] = raw[key] * (ratio ** strength)
        calibrated = _normalize(ratio_corrected)

    result["calibrated_probabilities"] = dict(calibrated)
    result["probabilities"] = dict(calibrated)
    result["banker_rate"] = round(calibrated["B"] * 100.0, 2)
    result["player_rate"] = round(calibrated["P"] * 100.0, 2)
    result["tie_rate"] = round(calibrated["T"] * 100.0, 2)

    b = calibrated["B"]
    p = calibrated["P"]
    t = calibrated["T"]
    bp_total = max(1e-12, b + p)
    b_no_tie = b / bp_total
    p_no_tie = p / bp_total
    bp_direction = "B" if b >= p else "P"
    bp_edge = abs(b_no_tie - p_no_tie)

    tie_ev = t * 8.0 - (b + p)
    tie_break_even = 1.0 / 9.0  # 8:1 賠付的理論損益兩平機率
    tie_signal_allowed = bool(
        active
        and sample_count >= TIE_SIGNAL_MIN_SAMPLES
        and t >= TIE_SIGNAL_MIN_PROBABILITY
        and t - tie_break_even >= TIE_SIGNAL_MIN_EDGE
        and tie_ev > 0.0
    )

    core_quality_ok = bool(
        float(result.get("uncertainty", 1.0) or 1.0)
        <= float(result.get("max_signal_uncertainty", 0.012) or 0.012)
        if "max_signal_uncertainty" in result
        else True
    )
    existing_signal = bool(result.get("signal_allowed"))
    existing_direction = str(result.get("recommend") or bp_direction).upper()
    direction_consistent = existing_direction == bp_direction
    bp_signal_allowed = bool(
        existing_signal
        and direction_consistent
        and bp_edge >= MIN_DIRECTION_EDGE
        and core_quality_ok
    )

    if tie_signal_allowed:
        recommend = "T"
        action = "T"
        signal_reason = "和局校準樣本、機率門檻與期望值同時通過保守訊號條件"
        signal_status = "和局訊號已開放"
    else:
        recommend = bp_direction
        action = bp_direction if bp_signal_allowed else "O"
        signal_reason = str(result.get("signal_reason") or "")
        if active:
            signal_reason = (
                (signal_reason + "；") if signal_reason else ""
            ) + f"三方機率已依 {scope} 歷史樣本做保守校準"
        signal_status = str(result.get("signal_status_text") or ("方向訊號已開放" if action in {"B", "P"} else "等待更明確訊號"))

    result["recommend"] = recommend
    result["recommend_text"] = {"B": "莊", "P": "閒", "T": "和"}[recommend]
    result["action"] = action
    result["action_text"] = {"B": "莊", "P": "閒", "T": "和", "O": "觀望"}[action]
    result["signal_allowed"] = action in OUTCOMES
    result["tie_signal_allowed"] = tie_signal_allowed
    result["direction_edge"] = float(bp_edge)
    result["direction_edge_percent"] = round(bp_edge * 100.0, 4)
    result["no_tie_probabilities"] = {"B": b_no_tie, "P": p_no_tie}
    result["signal_reason"] = signal_reason or "目前資料尚未形成正式方向訊號"
    result["signal_status_text"] = signal_status
    result["expected_values"] = {
        "B": b * 0.95 - p,
        "P": p - b,
        "T": tie_ev,
    }
    result["calibration"] = {
        "active": active,
        "scope": scope,
        "sample_count": sample_count,
        "minimum_samples": minimum,
        "strength": round(strength, 6),
        "brier_score": summary.get("brier_score"),
        "log_loss": summary.get("log_loss"),
        "tie_signal_min_samples": TIE_SIGNAL_MIN_SAMPLES,
    }
    return result


__all__ = ["calibrate_prediction"]
