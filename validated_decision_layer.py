"""Validated decision layer for the short-shoe road-only BGS model.

Public function signatures and compatibility fields are preserved.  When an
exact physical-card EV is unavailable, the layer uses the short-window model's
``confidence_prob`` as the predicted win probability, converts it to a virtual
EV, and forwards the signal to ``MoneyManagementModel`` for sizing.
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

# Retained as public/configuration compatibility constants.  The short-shoe
# confidence is no longer replaced by a long-horizon performance posterior.
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


def _hard_brake_active(result: Mapping[str, Any]) -> bool:
    braking = result.get("uncertainty_braking")
    adaptive = result.get("adaptive_ensemble")
    return bool(
        result.get("is_extreme_unseen")
        or result.get("hard_brake_active")
        or (
            isinstance(braking, Mapping)
            and (braking.get("active") or braking.get("is_extreme_unseen"))
        )
        or (
            isinstance(adaptive, Mapping)
            and (
                adaptive.get("hard_brake_active")
                or adaptive.get("circuit_breaker_active")
            )
        )
    )


def _set_no_bet(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    result.update({
        "action": "O",
        "recommend": "O",
        "internal_action": "O",
        "internal_recommend": "O",
        "next_round_direction": "O",
        "action_text": "觀望",
        "recommend_text": "觀望",
        "next_round_direction_text": "觀望",
        "signal_allowed": False,
        "risk_gate_open": False,
        "mandatory_bet": False,
        "kelly_fraction": 0.0,
        "recommended_bet_percentage": 0.0,
        "suggested_bet_amount": 0.0,
        "bet_amount": 0.0,
        "bet_percentage": 0.0,
        "final_bet_ratio": 0.0,
        "selected_expected_return": 0.0,
        "selected_expected_return_percent": 0.0,
        "kelly_percentage_applied": 0.0,
        "signal_reason": reason,
        "reason": reason,
    })
    if isinstance(result.get("money_management"), Mapping):
        money = dict(result.get("money_management") or {})
        money.update({
            "bet_allowed": False,
            "bet_amount": 0.0,
            "bet_percentage": 0.0,
            "final_bet_ratio": 0.0,
            "reason": reason,
        })
        result["money_management"] = money
    return result


def _short_window_forecast(result: Mapping[str, Any]) -> Dict[str, Any]:
    dynamic = result.get("dynamic_prediction_policy")
    if not isinstance(dynamic, Mapping):
        return {}
    forecast = dynamic.get("forecast")
    return dict(forecast) if isinstance(forecast, Mapping) else {}


def _model_direction_and_confidence(
    result: Mapping[str, Any],
) -> tuple[str, float, Dict[str, float], str]:
    forecast = _short_window_forecast(result)

    raw_probabilities: Mapping[str, Any]
    if isinstance(forecast.get("probabilities"), Mapping):
        raw_probabilities = dict(forecast.get("probabilities") or {})
        source = "short_window_forecast"
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

    # Confidence is the probability of the chosen direction, so rebuild a clean
    # B/P distribution from that value for virtual-EV and money sizing.
    probabilities = {
        "B": confidence_prob if direction == "B" else 1.0 - confidence_prob,
        "P": confidence_prob if direction == "P" else 1.0 - confidence_prob,
        "T": 0.0,
    }
    return direction, confidence_prob, probabilities, source


def _physical_ev_available(result: Mapping[str, Any]) -> bool:
    physical = result.get("physical_signal")
    if not isinstance(physical, Mapping):
        return False
    action = str(physical.get("action") or "").upper().strip()
    try:
        ev = float(physical.get("selected_expected_return", 0.0) or 0.0)
    except (TypeError, ValueError):
        ev = 0.0
    return bool(physical.get("available") and action in {"B", "P"} and ev > 0.0)


def _virtual_money(
    *,
    direction: str,
    confidence_prob: float,
    probabilities: Mapping[str, Any],
    bankroll: float,
) -> tuple[Dict[str, Any], float]:
    """Convert model confidence to virtual EV and obtain quarter-Kelly sizing."""
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

    # ``MoneyManagementModel.allocate`` retains its own minimum-EV gate for
    # backward compatibility.  The requested fallback rule is strictly EV > 0,
    # so for a small positive virtual EV we still use the model's quarter-Kelly
    # calculation instead of replacing the signal with a zero bet.
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
    result["banker_rate"] = round(banker * 100.0, 2)
    result["player_rate"] = round(player * 100.0, 2)
    result["tie_rate"] = 0.0
    result["direction"] = direction
    result["direction_text"] = "莊" if direction == "B" else "閒"


def apply_validated_decision(
    prediction: Mapping[str, Any],
    *,
    venue: str = "",
    room: str = "",
) -> Dict[str, Any]:
    """Use short-window confidence as virtual EV when physical EV is absent."""
    del venue, room
    result = dict(prediction or {})

    if _hard_brake_active(result):
        result["decision_validation"] = {
            "active": False,
            "hard_brake_preserved": True,
            "reason": "統計硬熔斷優先，驗證層不得恢復方向",
        }
        return result

    if _physical_ev_available(result):
        result["decision_validation"] = {
            "active": True,
            "mode": "physical_ev_preserved",
            "exact_card_counts_required": False,
            "virtual_ev_fallback_used": False,
        }
        return result

    direction, confidence_prob, probabilities, source = _model_direction_and_confidence(result)
    bankroll = max(0.0, float(result.get("bankroll", 0.0) or 0.0))
    money, virtual_ev = _virtual_money(
        direction=direction,
        confidence_prob=confidence_prob,
        probabilities=probabilities,
        bankroll=bankroll,
    )

    result.setdefault(
        "pre_validation_probabilities",
        dict(result.get("probabilities") or {}),
    )
    _set_direction_distribution(
        result,
        direction=direction,
        confidence=confidence_prob,
    )
    result["confidence"] = confidence_prob
    result["confidence_prob"] = confidence_prob
    result["ensemble_confidence"] = confidence_prob
    result["quality_score"] = confidence_prob
    result["confidence_label"] = (
        "較高" if confidence_prob >= 0.60
        else "中等" if confidence_prob >= 0.54
        else "偏低"
    )

    if virtual_ev <= 0.0 or not bool(money.get("bet_allowed", False)):
        result["decision_validation"] = {
            "active": True,
            "mode": "short_window_virtual_ev_gate",
            "direction": direction,
            "confidence_prob": confidence_prob,
            "virtual_ev": virtual_ev,
            "confidence_source": source,
            "exact_card_counts_required": False,
            "virtual_ev_fallback_used": True,
        }
        return _set_no_bet(
            result,
            "短窗模型虛擬 EV ≤ 0，維持觀望。",
        )

    ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    amount = float(money.get("bet_amount", bankroll * ratio) or 0.0)
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
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": False,
        "selected_expected_return": virtual_ev,
        "selected_expected_return_percent": virtual_ev * 100.0,
        "kelly_fraction": ratio,
        "kelly_percentage_applied": ratio * 100.0,
        "recommended_bet_percentage": ratio * 100.0,
        "bet_percentage": ratio * 100.0,
        "final_bet_ratio": ratio,
        "suggested_bet_amount": amount,
        "bet_amount": amount,
        "money_management": money,
        "direction_source": "short_window_confidence_virtual_ev",
        "signal_reason": (
            f"12局短窗模型 {text} confidence={confidence_prob:.3%}；"
            f"virtualEV={virtual_ev:.3%}；"
            f"Kelly={ratio:.3%}。"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "mode": "short_window_confidence_virtual_ev",
        "direction": direction,
        "confidence_prob": confidence_prob,
        "virtual_ev": virtual_ev,
        "confidence_source": source,
        "exact_card_counts_required": False,
        "virtual_ev_fallback_used": True,
        "final_kelly_ratio": ratio,
    }
    return result


def _strategy_road_direction(result: Mapping[str, Any]) -> tuple[str, float]:
    direction, confidence_prob, _, _ = _model_direction_and_confidence(result)
    return direction, max(0.0, min(1.0, abs(confidence_prob - 0.5) * 2.0))


def apply_strategy_decision(
    prediction: Mapping[str, Any],
    *,
    strategy_selection: Mapping[str, Any],
    bankroll: float = 0.0,
) -> Dict[str, Any]:
    """Apply strategy scaling to physical EV or short-window virtual EV fallback."""
    result = dict(prediction or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")

    if _hard_brake_active(result):
        result["decision_strategy_bandit"] = selection
        result["decision_strategy"] = arm
        return _set_no_bet(result, "統計硬熔斷仍在作用，維持觀望。")

    if _physical_ev_available(result):
        result["decision_strategy_bandit"] = selection
        result["decision_strategy"] = arm
        result["decision_validation"] = {
            "active": True,
            "mode": "physical_ev_preserved",
            "selected_strategy_arm": arm,
            "exact_card_counts_required": False,
            "virtual_ev_fallback_used": False,
        }
        return result

    direction, confidence_prob, probabilities, source = _model_direction_and_confidence(result)
    bankroll_value = max(0.0, float(bankroll or 0.0))
    money, virtual_ev = _virtual_money(
        direction=direction,
        confidence_prob=confidence_prob,
        probabilities=probabilities,
        bankroll=bankroll_value,
    )

    result["decision_strategy_bandit"] = selection
    result["decision_strategy"] = arm
    result["strategy_hard_limits"] = {
        "kelly_fraction": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_RATIO),
        "minimum_positive_ev": 0.0,
        "money_management_min_positive_ev": float(MIN_POSITIVE_EV),
        "exact_card_counts_required": False,
    }
    result["model_virtual_signal"] = {
        "available": True,
        "source": source,
        "action": direction,
        "confidence_prob": confidence_prob,
        "probabilities": probabilities,
        "selected_expected_return": virtual_ev,
        "kelly_fraction": float(money.get("kelly_fraction", 0.0) or 0.0),
        "trusted_exact_counts": False,
        "exact_card_counts_required": False,
    }

    if virtual_ev <= 0.0 or not bool(money.get("bet_allowed", False)):
        result["decision_validation"] = {
            "active": True,
            "mode": "short_window_virtual_ev_gate",
            "selected_strategy_arm": arm,
            "model_direction": direction,
            "confidence_prob": confidence_prob,
            "model_virtual_ev": virtual_ev,
            "required_ev": 0.0,
            "exact_card_counts_required": False,
        }
        return _set_no_bet(
            result,
            "短窗模型虛擬 EV ≤ 0，維持觀望。",
        )

    multiplier = min(
        1.0,
        max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)),
    )
    road_direction, road_strength = _strategy_road_direction(result)
    road_note = "12局短窗 confidence 轉虛擬 EV 後進入 1/4 Kelly。"
    if arm == "conservative":
        multiplier *= 0.50
        road_note = "保守策略將 1/4 Kelly 再縮半。"
    elif arm == "ev_road_blend" and road_strength < 0.20:
        multiplier *= 0.75
        road_note = "短窗方向幅度偏弱，1/4 Kelly 再縮至 75%。"

    base_ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    final_ratio = min(MAX_BET_RATIO, max(0.0, base_ratio * multiplier))
    if final_ratio <= 0.0:
        return _set_no_bet(result, "策略縮放後下注比例為零，維持觀望。")

    amount = bankroll_value * final_ratio
    text = "莊" if direction == "B" else "閒"
    money.update({
        "final_bet_ratio": final_ratio,
        "bet_percentage": final_ratio * 100.0,
        "bet_amount": amount,
        "bet_allowed": True,
        "strategy_multiplier": multiplier,
        "virtual_ev": virtual_ev,
        "virtual_ev_percent": virtual_ev * 100.0,
    })

    result.update({
        "action": direction,
        "recommend": direction,
        "internal_action": direction,
        "internal_recommend": direction,
        "next_round_direction": direction,
        "action_text": text,
        "recommend_text": text,
        "next_round_direction_text": text,
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": False,
        "direction_source": "short_window_confidence_virtual_ev_quarter_kelly",
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
        "bankroll": bankroll_value,
        "money_management": money,
        "strategy_weights": {
            "model_probability_weight": 1.0,
            "road_direction": road_direction,
            "road_strength": road_strength,
            "kelly_multiplier": multiplier,
        },
        "strategy_required_ev": 0.0,
        "kelly_cap_enforced": True,
        "banker_rate": round(probabilities["B"] * 100.0, 4),
        "player_rate": round(probabilities["P"] * 100.0, 4),
        "tie_rate": 0.0,
        "signal_reason": (
            f"{profile.get('label', arm)}：12局短窗 confidence "
            f"{confidence_prob:.3%}；虛擬 EV {virtual_ev:.3%}；{road_note}"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "mode": "short_window_confidence_virtual_ev_quarter_kelly",
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
    }
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
