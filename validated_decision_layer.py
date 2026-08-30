"""Validated decision layer for the road-only BGS ensemble.

Public function signatures and outward JSON fields are preserved. Final B/P
probabilities are produced by the three-way Local + Global + NumPy Regression
ensemble in dynamic_prediction_policy.py, converted to virtual EV, then passed
to MoneyManagementModel. This layer never changes the formal B/P direction to O.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import os

from money_management import (
    KELLY_FRACTION,
    MAX_BET_RATIO,
    MIN_POSITIVE_EV,
    MoneyManagementModel,
)

OUTCOMES = ("B", "P", "T")
_MONEY = MoneyManagementModel()

VALIDATED_COMPONENT_MIN_SAMPLES = max(
    20,
    min(500, int(os.getenv("VALIDATED_COMPONENT_MIN_SAMPLES", "40") or "40")),
)
UNVALIDATED_CONFIDENCE_CAP = max(
    0.50,
    min(0.99, float(os.getenv("UNVALIDATED_CONFIDENCE_CAP", "0.99") or "0.99")),
)
VALIDATED_CONFIDENCE_CAP = max(
    0.50,
    min(0.99, float(os.getenv("VALIDATED_CONFIDENCE_CAP", "0.99") or "0.99")),
)


def _normalize(values: Mapping[str, Any]) -> Dict[str, float]:
    raw = {
        outcome: max(0.0, float(values.get(outcome, 0.0) or 0.0))
        for outcome in OUTCOMES
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return {"B": 0.5, "P": 0.5, "T": 0.0}
    return {outcome: float(raw[outcome] / total) for outcome in OUTCOMES}


def _short_window_forecast(result: Mapping[str, Any]) -> Dict[str, Any]:
    dynamic = result.get("dynamic_prediction_policy")
    if not isinstance(dynamic, Mapping):
        return {}
    forecast = dynamic.get("forecast")
    return dict(forecast) if isinstance(forecast, Mapping) else {}


def _global_trend_state(result: Mapping[str, Any]) -> Dict[str, Any]:
    forecast = _short_window_forecast(result)
    state = forecast.get("global_trend_bias_correction")
    if isinstance(state, Mapping):
        return dict(state)

    dynamic = result.get("dynamic_prediction_policy")
    if isinstance(dynamic, Mapping):
        state = dynamic.get("global_trend_bias_correction")
        if isinstance(state, Mapping):
            return dict(state)

    state = result.get("global_trend_bias_correction")
    return dict(state) if isinstance(state, Mapping) else {}


def _regression_state(result: Mapping[str, Any]) -> Dict[str, Any]:
    correction = _global_trend_state(result)
    state = correction.get("regression_analysis")
    if isinstance(state, Mapping):
        return dict(state)

    forecast = _short_window_forecast(result)
    state = forecast.get("regression_analysis")
    if isinstance(state, Mapping):
        return dict(state)

    dynamic = result.get("dynamic_prediction_policy")
    if isinstance(dynamic, Mapping):
        state = dynamic.get("regression_analysis")
        if isinstance(state, Mapping):
            return dict(state)

    state = result.get("regression_analysis")
    return dict(state) if isinstance(state, Mapping) else {}


def _model_direction_and_confidence(
    result: Mapping[str, Any],
) -> tuple[str, float, Dict[str, float], str]:
    forecast = _short_window_forecast(result)
    correction = _global_trend_state(result)

    if isinstance(forecast.get("probabilities"), Mapping):
        raw_probabilities = dict(forecast.get("probabilities") or {})
        source = (
            "global_local_regression_ensemble"
            if correction
            else "short_window_forecast"
        )
    elif isinstance(result.get("probabilities"), Mapping):
        raw_probabilities = dict(result.get("probabilities") or {})
        source = "prediction_probabilities"
    elif isinstance(result.get("economic_probs"), Mapping):
        raw_probabilities = dict(result.get("economic_probs") or {})
        source = "economic_probabilities"
    else:
        raw_probabilities = {"B": 0.5, "P": 0.5, "T": 0.0}
        source = "neutral_fallback"

    probabilities = _normalize(raw_probabilities)
    bp_mass = probabilities["B"] + probabilities["P"]
    if bp_mass <= 1e-12:
        probabilities = {"B": 0.5, "P": 0.5, "T": 0.0}
    else:
        probabilities = {
            "B": probabilities["B"] / bp_mass,
            "P": probabilities["P"] / bp_mass,
            "T": 0.0,
        }

    direction = str(
        forecast.get("direction")
        or correction.get("final_direction")
        or result.get("direction")
        or result.get("adaptive_only_direction")
        or result.get("action")
        or result.get("recommend")
        or ""
    ).upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    confidence_prob = float(probabilities[direction])
    if confidence_prob < 0.5:
        direction = "P" if direction == "B" else "B"
        confidence_prob = float(probabilities[direction])

    return direction, confidence_prob, probabilities, source


def _virtual_money(
    *,
    direction: str,
    confidence_prob: float,
    probabilities: Mapping[str, Any],
    bankroll: float,
) -> tuple[Dict[str, Any], float]:
    p_win = max(0.0, min(1.0, float(confidence_prob)))
    payout = 0.95 if direction == "B" else 1.0
    virtual_ev = float(p_win * payout - (1.0 - p_win))

    money = dict(
        _MONEY.allocate(
            direction=direction,
            probabilities=probabilities,
            final_weight=p_win,
            bankroll=max(0.0, float(bankroll or 0.0)),
        )
    )

    if virtual_ev > 0.0 and not bool(money.get("bet_allowed", False)):
        ratio = min(
            float(MAX_BET_RATIO),
            max(
                0.0,
                float(
                    _MONEY.kelly_fraction(
                        side=direction,
                        probabilities=probabilities,
                    )
                    or 0.0
                ),
            ),
        )
        if ratio > 0.0:
            bankroll_value = max(0.0, float(bankroll or 0.0))
            money.update({
                "virtual_ev": virtual_ev,
                "virtual_ev_percent": virtual_ev * 100.0,
                "expected_value_per_unit": virtual_ev,
                "kelly_fraction": ratio,
                "final_bet_ratio": ratio,
                "pre_tie_adjusted_ratio": ratio,
                "adjusted_ratio": ratio,
                "bet_percentage": ratio * 100.0,
                "bet_amount": bankroll_value * ratio,
                "bet_allowed": True,
                "reason": "positive_regression_ensemble_virtual_ev_quarter_kelly_fallback",
            })

    money["virtual_ev"] = virtual_ev
    money["virtual_ev_percent"] = virtual_ev * 100.0
    money["ensemble_probability_input"] = p_win
    return money, virtual_ev


def _set_direction_distribution(
    result: Dict[str, Any],
    *,
    direction: str,
    probabilities: Mapping[str, Any],
) -> None:
    normalized = _normalize(probabilities)
    bp_mass = normalized["B"] + normalized["P"]
    if bp_mass <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker = normalized["B"] / bp_mass
        player = normalized["P"] / bp_mass
    result["probabilities"] = {"B": banker, "P": player, "T": 0.0}
    result["economic_probs"] = {"B": banker, "P": player, "T": 0.0}
    result["banker_rate"] = round(banker * 100.0, 4)
    result["player_rate"] = round(player * 100.0, 4)
    result["tie_rate"] = 0.0
    result["direction"] = direction
    result["direction_text"] = "莊" if direction == "B" else "閒"


def _apply_direction_without_observe(
    result: Dict[str, Any],
    *,
    bankroll: float,
    strategy_selection: Mapping[str, Any] | None = None,
    strategy_multiplier: float = 1.0,
) -> Dict[str, Any]:
    direction, confidence_prob, probabilities, source = (
        _model_direction_and_confidence(result)
    )
    correction = _global_trend_state(result)
    regression = _regression_state(result)
    money, virtual_ev = _virtual_money(
        direction=direction,
        confidence_prob=confidence_prob,
        probabilities=probabilities,
        bankroll=bankroll,
    )

    base_ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    ratio = min(
        float(MAX_BET_RATIO),
        max(0.0, base_ratio * max(0.0, min(1.0, strategy_multiplier))),
    )
    amount = max(0.0, float(bankroll or 0.0)) * ratio
    money.update({
        "final_bet_ratio": ratio,
        "bet_percentage": ratio * 100.0,
        "bet_amount": amount,
        "bet_allowed": bool(money.get("bet_allowed", False) and ratio > 0.0),
        "strategy_multiplier": max(0.0, min(1.0, strategy_multiplier)),
        "virtual_ev": virtual_ev,
        "virtual_ev_percent": virtual_ev * 100.0,
    })

    result.setdefault(
        "pre_validation_probabilities",
        dict(result.get("probabilities") or {}),
    )
    _set_direction_distribution(
        result,
        direction=direction,
        probabilities=probabilities,
    )

    selection = dict(strategy_selection or {})
    if strategy_selection is not None:
        result["decision_strategy_bandit"] = selection
        result["decision_strategy"] = str(
            selection.get("selected_arm") or "math_only"
        )

    text = "莊" if direction == "B" else "閒"
    weights = dict(correction.get("ensemble_weights") or {})
    regime = str(correction.get("ensemble_regime") or "normal")
    result.update({
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
        "action_text": text,
        "recommend_text": text,
        "next_round_direction_text": text,
        "decision": direction,
        "decision_text": text,
        "skip": False,
        "skip_reason": "",
        "force_observe": False,
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": False,
        "confidence": confidence_prob,
        "confidence_prob": confidence_prob,
        "ensemble_confidence": confidence_prob,
        "quality_score": confidence_prob,
        "confidence_label": (
            "較高" if confidence_prob >= 0.60
            else "中等" if confidence_prob >= 0.54
            else "偏低"
        ),
        "selected_expected_return": virtual_ev,
        "selected_expected_return_percent": virtual_ev * 100.0,
        "kelly_fraction": ratio,
        "kelly_percentage_applied": ratio * 100.0,
        "recommended_bet_percentage": ratio * 100.0,
        "bet_percentage": ratio * 100.0,
        "final_bet_ratio": ratio,
        "suggested_bet_amount": amount,
        "bet_amount": amount,
        "bet_allowed": bool(money.get("bet_allowed", False)),
        "money_management": money,
        "global_trend_bias_correction": correction,
        "regression_analysis": regression,
        "bias_momentum_adjuster": {
            "applied": False,
            "mode": "replaced_by_global_local_regression_ensemble",
        },
        "direction_source": "global_local_regression_ensemble_virtual_ev",
        "signal_reason": (
            f"RegressionEnsemble[{regime}] {text} confidence={confidence_prob:.3%}；"
            f"w={weights}；"
            f"GlobalSlope={float(regression.get('global_slope', 0.0) or 0.0):+.4f}；"
            f"LocalSlope={float(regression.get('local_slope', 0.0) or 0.0):+.4f}；"
            f"virtualEV={virtual_ev:.3%}；Kelly={ratio:.3%}。"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["model_virtual_signal"] = {
        "available": True,
        "source": source,
        "action": direction,
        "confidence_prob": confidence_prob,
        "probabilities": probabilities,
        "selected_expected_return": virtual_ev,
        "kelly_fraction": ratio,
        "trusted_exact_counts": False,
        "exact_card_counts_required": False,
        "global_trend_bias_correction": correction,
        "regression_analysis": regression,
        "ensemble_weights": weights,
    }
    result["decision_validation"] = {
        "active": True,
        "mode": "global_local_regression_virtual_ev_no_observe",
        "direction": direction,
        "confidence_prob": confidence_prob,
        "virtual_ev": virtual_ev,
        "confidence_source": source,
        "global_trend_bias_correction": correction,
        "regression_analysis": regression,
        "ensemble_regime": regime,
        "ensemble_weights": weights,
        "exact_card_counts_required": False,
        "virtual_ev_fallback_used": True,
        "observe_gate_enabled": False,
        "base_kelly_ratio": base_ratio,
        "final_kelly_ratio": ratio,
    }
    return result


def apply_validated_decision(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    del venue, room
    result = dict(prediction or {})
    bankroll = max(0.0, float(result.get("bankroll", 0.0) or 0.0))
    return _apply_direction_without_observe(result, bankroll=bankroll)


def _strategy_road_direction(result: Mapping[str, Any]) -> tuple[str, float]:
    direction, confidence_prob, _, _ = _model_direction_and_confidence(result)
    return direction, max(0.0, min(1.0, abs(confidence_prob - 0.5) * 2.0))


def apply_strategy_decision(
    prediction: Mapping[str, Any],
    *,
    strategy_selection: Mapping[str, Any],
    bankroll: float = 0.0,
) -> Dict[str, Any]:
    result = dict(prediction or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")

    multiplier = min(
        1.0,
        max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)),
    )
    road_direction, road_strength = _strategy_road_direction(result)
    road_note = "三方 Local/Global/Regression 融合概率轉虛擬 EV 後進入 1/4 Kelly。"
    if arm == "conservative":
        multiplier *= 0.50
        road_note = "保守策略將三方融合概率的 1/4 Kelly 再縮半。"
    elif arm == "ev_road_blend" and road_strength < 0.20:
        multiplier *= 0.75
        road_note = "三方融合方向幅度偏弱，1/4 Kelly 再縮至 75%。"

    result = _apply_direction_without_observe(
        result,
        bankroll=max(0.0, float(bankroll or 0.0)),
        strategy_selection=selection,
        strategy_multiplier=multiplier,
    )
    correction = _global_trend_state(result)
    ensemble_weights = dict(correction.get("ensemble_weights") or {})
    result["strategy_weights"] = {
        "model_probability_weight": 1.0,
        "local_model_weight": float(ensemble_weights.get("local", 0.0) or 0.0),
        "global_trend_weight": float(ensemble_weights.get("global", 0.0) or 0.0),
        "regression_weight": float(ensemble_weights.get("regression", 0.0) or 0.0),
        "ensemble_regime": str(correction.get("ensemble_regime") or "normal"),
        "road_direction": road_direction,
        "road_strength": road_strength,
        "kelly_multiplier": multiplier,
    }
    result["strategy_hard_limits"] = {
        "kelly_fraction": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_RATIO),
        "minimum_positive_ev": 0.0,
        "money_management_min_positive_ev": float(MIN_POSITIVE_EV),
        "exact_card_counts_required": False,
    }
    result["strategy_required_ev"] = 0.0
    result["kelly_cap_enforced"] = True
    result["decision_validation"].update({
        "mode": "global_local_regression_virtual_ev_quarter_kelly_no_observe",
        "selected_strategy_arm": arm,
        "quarter_kelly_multiplier": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_RATIO),
        "road_note": road_note,
    })
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
