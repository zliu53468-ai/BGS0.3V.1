"""Validated decision layer for the short-shoe road-only BGS model.

Public function signatures and outward JSON fields are preserved.  The final B/P
direction and confidence come from the 12-hand predictor after its external Bias
& Momentum Adjuster.  That adjusted confidence is converted directly to virtual
EV and passed to MoneyManagementModel.  This layer does not convert a model
signal to O/observe.
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


def _bias_momentum_state(result: Mapping[str, Any]) -> Dict[str, Any]:
    forecast = _short_window_forecast(result)
    state = forecast.get("bias_momentum_adjuster")
    if isinstance(state, Mapping):
        return dict(state)

    dynamic = result.get("dynamic_prediction_policy")
    if isinstance(dynamic, Mapping):
        state = dynamic.get("bias_momentum_adjuster")
        if isinstance(state, Mapping):
            return dict(state)

    state = result.get("bias_momentum_adjuster")
    return dict(state) if isinstance(state, Mapping) else {}


def _model_direction_and_confidence(
    result: Mapping[str, Any],
) -> tuple[str, float, Dict[str, float], str]:
    forecast = _short_window_forecast(result)
    adjustment = _bias_momentum_state(result)

    raw_probabilities: Mapping[str, Any]
    if isinstance(forecast.get("probabilities"), Mapping):
        raw_probabilities = dict(forecast.get("probabilities") or {})
        source = (
            "bias_momentum_adjusted_forecast"
            if adjustment.get("applied")
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
        or adjustment.get("adjusted_direction")
        or result.get("direction")
        or result.get("adaptive_only_direction")
        or result.get("action")
        or result.get("recommend")
        or ""
    ).upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    confidence_candidates = (
        forecast.get("confidence_prob"),
        adjustment.get("adjusted_confidence_prob"),
        result.get("confidence_prob"),
        forecast.get("confidence"),
        result.get("confidence"),
        probabilities.get(direction),
    )
    confidence_prob = 0.5
    for raw in confidence_candidates:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0.5:
            confidence_prob = max(0.5, min(0.999, value))
            break

    probabilities = {
        "B": confidence_prob if direction == "B" else 1.0 - confidence_prob,
        "P": confidence_prob if direction == "P" else 1.0 - confidence_prob,
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
    """Convert adjusted model confidence to virtual EV and quarter-Kelly sizing."""
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

    # Keep the existing positive-EV quarter-Kelly compatibility fallback.
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
                "reason": "positive_virtual_ev_quarter_kelly_fallback",
            })

    money["virtual_ev"] = virtual_ev
    money["virtual_ev_percent"] = virtual_ev * 100.0
    return money, virtual_ev


def _set_direction_distribution(
    result: Dict[str, Any],
    *,
    direction: str,
    confidence: float,
) -> None:
    conditional = max(0.50, min(0.999, float(confidence)))
    banker = conditional if direction == "B" else 1.0 - conditional
    player = 1.0 - banker
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
) -> Dict[str, Any]:
    direction, confidence_prob, probabilities, source = (
        _model_direction_and_confidence(result)
    )
    adjustment = _bias_momentum_state(result)
    money, virtual_ev = _virtual_money(
        direction=direction,
        confidence_prob=confidence_prob,
        probabilities=probabilities,
        bankroll=bankroll,
    )

    ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    amount = float(money.get("bet_amount", bankroll * ratio) or 0.0)
    text = "莊" if direction == "B" else "閒"

    result.setdefault(
        "pre_validation_probabilities",
        dict(result.get("probabilities") or {}),
    )
    _set_direction_distribution(
        result,
        direction=direction,
        confidence=confidence_prob,
    )

    if strategy_selection is not None:
        selection = dict(strategy_selection or {})
        result["decision_strategy_bandit"] = selection
        result["decision_strategy"] = str(
            selection.get("selected_arm") or "math_only"
        )

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
        "bias_momentum_adjuster": adjustment,
        "direction_source": (
            "bias_momentum_adjusted_virtual_ev"
            if adjustment.get("applied")
            else "short_window_confidence_virtual_ev"
        ),
        "signal_reason": (
            f"12局短窗 {text} confidence={confidence_prob:.3%}；"
            f"BiasMomentum={str(adjustment.get('mode') or 'base')}；"
            f"virtualEV={virtual_ev:.3%}；Kelly={ratio:.3%}。"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "mode": "bias_momentum_virtual_ev_no_observe",
        "direction": direction,
        "confidence_prob": confidence_prob,
        "virtual_ev": virtual_ev,
        "confidence_source": source,
        "bias_momentum_adjuster": adjustment,
        "exact_card_counts_required": False,
        "virtual_ev_fallback_used": True,
        "observe_gate_enabled": False,
        "final_kelly_ratio": ratio,
    }
    return result


def apply_validated_decision(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    """Always pass adjusted 12-hand B/P confidence through virtual EV."""
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
    """Apply strategy sizing without converting the adjusted B/P signal to O."""
    result = dict(prediction or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")

    direction, confidence_prob, probabilities, source = (
        _model_direction_and_confidence(result)
    )
    adjustment = _bias_momentum_state(result)
    bankroll_value = max(0.0, float(bankroll or 0.0))
    money, virtual_ev = _virtual_money(
        direction=direction,
        confidence_prob=confidence_prob,
        probabilities=probabilities,
        bankroll=bankroll_value,
    )

    multiplier = min(
        1.0,
        max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)),
    )
    road_direction, road_strength = _strategy_road_direction(result)
    road_note = "Bias/Momentum 校正勝率轉虛擬 EV 後進入 1/4 Kelly。"
    if arm == "conservative":
        multiplier *= 0.50
        road_note = "保守策略將 1/4 Kelly 再縮半。"
    elif arm == "ev_road_blend" and road_strength < 0.20:
        multiplier *= 0.75
        road_note = "短窗方向幅度偏弱，1/4 Kelly 再縮至 75%。"

    base_ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    final_ratio = min(MAX_BET_RATIO, max(0.0, base_ratio * multiplier))
    amount = bankroll_value * final_ratio
    text = "莊" if direction == "B" else "閒"

    money.update({
        "final_bet_ratio": final_ratio,
        "bet_percentage": final_ratio * 100.0,
        "bet_amount": amount,
        "bet_allowed": bool(
            money.get("bet_allowed", False) and final_ratio > 0.0
        ),
        "strategy_multiplier": multiplier,
        "virtual_ev": virtual_ev,
        "virtual_ev_percent": virtual_ev * 100.0,
    })

    _set_direction_distribution(
        result,
        direction=direction,
        confidence=confidence_prob,
    )
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
        "direction_source": (
            "bias_momentum_adjusted_virtual_ev_quarter_kelly"
            if adjustment.get("applied")
            else "short_window_confidence_virtual_ev_quarter_kelly"
        ),
        "confidence": confidence_prob,
        "confidence_prob": confidence_prob,
        "selected_expected_return": virtual_ev,
        "selected_expected_return_percent": virtual_ev * 100.0,
        "kelly_fraction": final_ratio,
        "kelly_percentage_applied": final_ratio * 100.0,
        "recommended_bet_percentage": final_ratio * 100.0,
        "bet_percentage": final_ratio * 100.0,
        "final_bet_ratio": final_ratio,
        "suggested_bet_amount": amount,
        "bet_amount": amount,
        "bet_allowed": bool(money.get("bet_allowed", False)),
        "bankroll": bankroll_value,
        "money_management": money,
        "bias_momentum_adjuster": adjustment,
        "decision_strategy_bandit": selection,
        "decision_strategy": arm,
        "strategy_weights": {
            "model_probability_weight": 1.0,
            "road_direction": road_direction,
            "road_strength": road_strength,
            "kelly_multiplier": multiplier,
        },
        "strategy_hard_limits": {
            "kelly_fraction": float(KELLY_FRACTION),
            "max_bet_fraction": float(MAX_BET_RATIO),
            "minimum_positive_ev": 0.0,
            "money_management_min_positive_ev": float(MIN_POSITIVE_EV),
            "exact_card_counts_required": False,
        },
        "strategy_required_ev": 0.0,
        "kelly_cap_enforced": True,
        "signal_reason": (
            f"{profile.get('label', arm)}：{text} confidence "
            f"{confidence_prob:.3%}；BiasMomentum="
            f"{str(adjustment.get('mode') or 'base')}；"
            f"虛擬 EV {virtual_ev:.3%}；{road_note}"
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
        "kelly_fraction": final_ratio,
        "trusted_exact_counts": False,
        "exact_card_counts_required": False,
        "bias_momentum_adjuster": adjustment,
    }
    result["decision_validation"] = {
        "active": True,
        "mode": "bias_momentum_virtual_ev_quarter_kelly_no_observe",
        "selected_strategy_arm": arm,
        "exact_card_counts_required": False,
        "virtual_ev_fallback_used": True,
        "model_direction": direction,
        "confidence_prob": confidence_prob,
        "model_virtual_ev": virtual_ev,
        "required_ev": 0.0,
        "quarter_kelly_multiplier": float(KELLY_FRACTION),
        "base_kelly_ratio": base_ratio,
        "final_kelly_ratio": final_ratio,
        "max_bet_fraction": float(MAX_BET_RATIO),
        "road_note": road_note,
        "bias_momentum_adjuster": adjustment,
        "observe_gate_enabled": False,
    }
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
