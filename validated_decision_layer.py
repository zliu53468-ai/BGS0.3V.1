"""Validated decision layer for the short-shoe road-only BGS model.

Public function signatures and outward JSON fields are preserved.  The final B/P
probabilities come from the 12-hand local predictor after the requested Global
Trend Bias Correction (40% local + 60% full-shoe base probability).  The fused
confidence is converted directly to virtual EV and passed to MoneyManagementModel.
This layer never changes the formal B/P direction to O/observe.
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
    min(
        500,
        int(os.getenv("VALIDATED_COMPONENT_MIN_SAMPLES", "40") or "40"),
    ),
)
UNVALIDATED_CONFIDENCE_CAP = max(
    0.50,
    min(
        0.99,
        float(os.getenv("UNVALIDATED_CONFIDENCE_CAP", "0.99") or "0.99"),
    ),
)
VALIDATED_CONFIDENCE_CAP = max(
    0.50,
    min(
        0.99,
        float(os.getenv("VALIDATED_CONFIDENCE_CAP", "0.99") or "0.99"),
    ),
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


def _model_direction_and_confidence(
    result: Mapping[str, Any],
) -> tuple[str, float, Dict[str, float], str]:
    forecast = _short_window_forecast(result)
    correction = _global_trend_state(result)

    raw_probabilities: Mapping[str, Any]
    if isinstance(forecast.get("probabilities"), Mapping):
        raw_probabilities = dict(forecast.get("probabilities") or {})
        source = (
            "global_trend_40_60_forecast"
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

    # Confidence is exactly the fused probability of the chosen B/P direction.
    confidence_prob = float(probabilities[direction])
    if confidence_prob < 0.5:
        direction = "P" if direction == "B" else "B"
        confidence_prob = float(probabilities[direction])

    probabilities = {
        "B": float(probabilities["B"]),
        "P": float(probabilities["P"]),
        "T": 0.0,
    }
    return direction, confidence_prob, probabilities, source


def _virtual_money(
    *,
    direction: str,
    confidence_prob: float,
    probabilities: Mapping[str, Any],
    bankroll: float,
) -> tuple[Dict[str, Any], float]:
    """Convert fused Final_P to virtual EV and obtain quarter-Kelly sizing."""
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

    # Preserve the existing compatibility rule: if EV is positive but the money
    # module's legacy minimum-EV threshold blocks sizing, use its quarter-Kelly
    # fraction instead. Negative/zero EV is still transmitted as-is; it never
    # changes the formal B/P prediction into O.
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
                "reason": "positive_global_trend_virtual_ev_quarter_kelly_fallback",
            })

    money["virtual_ev"] = virtual_ev
    money["virtual_ev_percent"] = virtual_ev * 100.0
    money["global_trend_probability_input"] = p_win
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
        "bias_momentum_adjuster": {
            "applied": False,
            "mode": "replaced_by_global_trend_bias_correction",
        },
        "direction_source": "global_trend_40_60_virtual_ev",
        "signal_reason": (
            f"GlobalTrend40/60 {text} confidence={confidence_prob:.3%}；"
            f"GlobalPB={float(correction.get('global_p_b', 0.5) or 0.5):.3%}；"
            f"velocityB={float(correction.get('global_probability_velocity_b', 0.0) or 0.0):+.5f}；"
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
    }
    result["decision_validation"] = {
        "active": True,
        "mode": "global_trend_40_60_virtual_ev_no_observe",
        "direction": direction,
        "confidence_prob": confidence_prob,
        "virtual_ev": virtual_ev,
        "confidence_source": source,
        "global_trend_bias_correction": correction,
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
    """Pass fused Global/Local Final_P directly through virtual EV without O."""
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
    """Apply strategy sizing without changing the fused B/P signal to O."""
    result = dict(prediction or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")

    multiplier = min(
        1.0,
        max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)),
    )
    road_direction, road_strength = _strategy_road_direction(result)
    road_note = "Global 60% + Local 40% 融合概率轉虛擬 EV 後進入 1/4 Kelly。"
    if arm == "conservative":
        multiplier *= 0.50
        road_note = "保守策略將融合概率的 1/4 Kelly 再縮半。"
    elif arm == "ev_road_blend" and road_strength < 0.20:
        multiplier *= 0.75
        road_note = "融合方向幅度偏弱，1/4 Kelly 再縮至 75%。"

    result = _apply_direction_without_observe(
        result,
        bankroll=max(0.0, float(bankroll or 0.0)),
        strategy_selection=selection,
        strategy_multiplier=multiplier,
    )
    result["strategy_weights"] = {
        "model_probability_weight": 1.0,
        "global_trend_weight": 0.60,
        "local_model_weight": 0.40,
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
        "mode": "global_trend_40_60_virtual_ev_quarter_kelly_no_observe",
        "selected_strategy_arm": arm,
        "quarter_kelly_multiplier": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_RATIO),
        "road_note": road_note,
    })
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
