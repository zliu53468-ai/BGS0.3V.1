"""Exploit-only validation and defensive Kelly bridge.

The stable V5 implementation lives in validated_decision_layer_base_v5.py.
This bridge preserves its public API while forcing sizing to use posterior arm
means only, never UCB exploration bonuses.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import math

import validated_decision_layer_base_v5 as _base

MIN_BET_RATIO = _base.MIN_BET_RATIO
MAX_BET_RATIO = _base.MAX_BET_RATIO
KELLY_FRACTION = _base.KELLY_FRACTION
MIN_POSITIVE_EV = _base.MIN_POSITIVE_EV

_ORIGINAL_MODEL_DIRECTION = _base._model_direction_and_confidence


def _extract_mean_scores(result: Mapping[str, Any]) -> dict[str, float] | None:
    direct = result.get("mean_scores")
    if isinstance(direct, Mapping) and "B" in direct and "P" in direct:
        return {
            "B": float(direct.get("B", 0.0) or 0.0),
            "P": float(direct.get("P", 0.0) or 0.0),
        }

    scores = result.get("bandit_scores")
    if isinstance(scores, Mapping):
        b = scores.get("B")
        p = scores.get("P")
        if isinstance(b, Mapping) and isinstance(p, Mapping):
            return {
                "B": float(b.get("mean", 0.0) or 0.0),
                "P": float(p.get("mean", 0.0) or 0.0),
            }

    dynamic = result.get("dynamic_prediction_policy")
    if isinstance(dynamic, Mapping):
        forecast = dynamic.get("forecast")
        if isinstance(forecast, Mapping):
            means = forecast.get("mean_scores")
            if isinstance(means, Mapping) and "B" in means and "P" in means:
                return {
                    "B": float(means.get("B", 0.0) or 0.0),
                    "P": float(means.get("P", 0.0) or 0.0),
                }
    return None


def _softmax_means(mean_scores: Mapping[str, Any]) -> Dict[str, float]:
    mean_b = float(mean_scores.get("B", 0.0) or 0.0)
    mean_p = float(mean_scores.get("P", 0.0) or 0.0)
    shift = max(mean_b, mean_p)
    exp_b = math.exp(max(-40.0, min(40.0, mean_b - shift)))
    exp_p = math.exp(max(-40.0, min(40.0, mean_p - shift)))
    total = exp_b + exp_p
    if total <= 1e-12:
        return {"B": 0.5, "P": 0.5, "T": 0.0}
    return {"B": exp_b / total, "P": exp_p / total, "T": 0.0}


def _model_direction_and_confidence_exploit(
    result: Mapping[str, Any],
) -> tuple[str, float, Dict[str, float], str]:
    direction = str(
        result.get("direction")
        or result.get("action")
        or result.get("recommend")
        or ""
    ).upper().strip()

    means = _extract_mean_scores(result)
    if means is None:
        return _ORIGINAL_MODEL_DIRECTION(result)

    probabilities = _softmax_means(means)
    if direction not in {"B", "P"}:
        forecast = _base._short_window_forecast(result)
        direction = str(forecast.get("direction") or "").upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    confidence_prob = float(probabilities.get(direction, 0.5))
    return (
        direction,
        confidence_prob,
        probabilities,
        "contextual_linucb_exploit_only_mean_softmax",
    )


def _virtual_money_defensive(
    *,
    direction: str,
    confidence_prob: float,
    probabilities: Mapping[str, Any],
    bankroll: float,
) -> tuple[Dict[str, Any], float]:
    p_win = max(0.0, min(1.0, float(confidence_prob)))
    payout = 0.95 if direction == "B" else 1.0
    pure_ev = float(p_win * payout - (1.0 - p_win))
    bankroll_value = max(0.0, float(bankroll or 0.0))

    if pure_ev > 0.0:
        money = dict(
            _base._MONEY.allocate(
                direction=direction,
                probabilities=probabilities,
                final_weight=p_win,
                bankroll=bankroll_value,
            )
        )
        ratio = min(
            float(MAX_BET_RATIO),
            max(
                float(MIN_BET_RATIO),
                float(money.get("final_bet_ratio", MIN_BET_RATIO) or MIN_BET_RATIO),
            ),
        )
        sizing_mode = "EXPLOIT_ONLY_KELLY"
    else:
        money = {}
        ratio = float(MIN_BET_RATIO)
        sizing_mode = "PURE_EV_NON_POSITIVE_FORCE_MIN_BET"

    money.update(
        {
            "pure_ev": pure_ev,
            "virtual_ev": pure_ev,
            "virtual_ev_percent": pure_ev * 100.0,
            "expected_value_per_unit": pure_ev,
            "kelly_probability_source": "softmax_of_exploit_only_mean_scores",
            "exploration_term_used_for_kelly": False,
            "sizing_mode": sizing_mode,
            "kelly_fraction": ratio,
            "final_bet_ratio": ratio,
            "pre_tie_adjusted_ratio": ratio,
            "adjusted_ratio": ratio,
            "bet_percentage": ratio * 100.0,
            "bet_amount": bankroll_value * ratio,
            "bet_allowed": True,
            "mandatory_bet": True,
            "ensemble_probability_input": p_win,
            "reason": (
                "single_brain_exploit_only_kelly"
                if pure_ev > 0.0
                else "pure_ev_non_positive_force_min_bet"
            ),
        }
    )
    return money, pure_ev


_base._model_direction_and_confidence = _model_direction_and_confidence_exploit
_base._virtual_money = _virtual_money_defensive

apply_validated_decision = _base.apply_validated_decision
apply_strategy_decision = _base.apply_strategy_decision

__all__ = ["apply_strategy_decision", "apply_validated_decision"]
