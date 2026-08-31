"""Exploit-only Kelly bridge for the Single-Brain LinUCB predictor.

Direction remains the UCB argmax produced by the stable V5 predictor. Bet sizing
uses only the two arm posterior means exposed by contextual_bandit.py.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Union
import math

import predictor_base_v5 as _base

OUTCOMES = _base.OUTCOMES
MODEL_VERSION = _base.MODEL_VERSION
MIN_BET_RATIO = _base.MIN_BET_RATIO
MAX_BET_RATIO = _base.MAX_BET_RATIO


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _mean_scores_from_result(result: Mapping[str, Any]) -> dict[str, float]:
    direct = result.get("mean_scores")
    if isinstance(direct, Mapping) and "B" in direct and "P" in direct:
        return {
            "B": float(direct.get("B", 0.0) or 0.0),
            "P": float(direct.get("P", 0.0) or 0.0),
        }

    bandit_scores = result.get("bandit_scores")
    if isinstance(bandit_scores, Mapping):
        b = bandit_scores.get("B")
        p = bandit_scores.get("P")
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
            scores = forecast.get("scores")
            if isinstance(scores, Mapping):
                b = scores.get("B")
                p = scores.get("P")
                if isinstance(b, Mapping) and isinstance(p, Mapping):
                    return {
                        "B": float(b.get("mean", 0.0) or 0.0),
                        "P": float(p.get("mean", 0.0) or 0.0),
                    }

    return {"B": 0.0, "P": 0.0}


def _softmax_means(mean_scores: Mapping[str, Any]) -> dict[str, float]:
    mean_b = float(mean_scores.get("B", 0.0) or 0.0)
    mean_p = float(mean_scores.get("P", 0.0) or 0.0)
    shift = max(mean_b, mean_p)
    exp_b = math.exp(max(-40.0, min(40.0, mean_b - shift)))
    exp_p = math.exp(max(-40.0, min(40.0, mean_p - shift)))
    total = exp_b + exp_p
    if total <= 1e-12:
        return {"B": 0.5, "P": 0.5, "T": 0.0}
    return {"B": exp_b / total, "P": exp_p / total, "T": 0.0}


def _defensive_money(
    *,
    direction: str,
    probabilities: Mapping[str, Any],
    confidence: float,
    bankroll: float,
) -> tuple[dict[str, Any], float, float]:
    payout = 0.95 if direction == "B" else 1.0
    pure_ev = float(confidence * payout - (1.0 - confidence))

    if pure_ev > 0.0:
        money = dict(
            _base._MONEY.allocate(
                direction=direction,
                probabilities=probabilities,
                final_weight=confidence,
                bankroll=bankroll,
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

    amount = bankroll * ratio
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
            "bet_amount": amount,
            "bet_allowed": True,
            "mandatory_bet": True,
        }
    )
    return money, pure_ev, ratio


def predict(
    history: Union[str, Iterable[Any], None] = None,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
    road_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = deepcopy(
        _base.predict(
            history=deepcopy(history),
            venue=venue,
            room=room,
            shoe_id=shoe_id,
            user_id=user_id,
            run_seed=run_seed,
            shoe_context=deepcopy(dict(shoe_context or {})),
            road_context=deepcopy(dict(road_context or {})),
        )
    )

    direction = str(
        result.get("action")
        or result.get("recommend")
        or result.get("direction")
        or "B"
    ).upper().strip()
    if direction not in {"B", "P"}:
        direction = "B"

    direction_probabilities_ucb = deepcopy(
        dict(result.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    )
    mean_scores = _mean_scores_from_result(result)
    exploit_probabilities = _softmax_means(mean_scores)
    confidence = _clip(exploit_probabilities.get(direction, 0.5), 0.0, 1.0)

    bankroll = max(
        0.0,
        float((shoe_context or {}).get("bankroll", result.get("bankroll", 0.0)) or 0.0),
    )
    money, pure_ev, final_ratio = _defensive_money(
        direction=direction,
        probabilities=exploit_probabilities,
        confidence=confidence,
        bankroll=bankroll,
    )
    bet_percentage = final_ratio * 100.0
    amount = bankroll * final_ratio

    result.update(
        {
            "mean_scores": mean_scores,
            "exploit_probabilities": exploit_probabilities,
            "direction_probabilities_ucb": direction_probabilities_ucb,
            "raw_direction_probabilities": direction_probabilities_ucb,
            "probabilities": exploit_probabilities,
            "final_probs": exploit_probabilities,
            "economic_probs": exploit_probabilities,
            "direction_probs": direction_probabilities_ucb,
            "banker_rate": round(exploit_probabilities["B"] * 100.0, 4),
            "player_rate": round(exploit_probabilities["P"] * 100.0, 4),
            "tie_rate": 0.0,
            "confidence": confidence,
            "confidence_prob": confidence,
            "raw_model_confidence": confidence,
            "pattern_calibrated_confidence": confidence,
            "ensemble_confidence": confidence,
            "quality_score": confidence,
            "selected_expected_return": pure_ev,
            "selected_expected_return_percent": pure_ev * 100.0,
            "pure_ev": pure_ev,
            "money_management": money,
            "kelly_fraction": final_ratio,
            "pre_tie_adjusted_ratio": final_ratio,
            "adjusted_ratio": final_ratio,
            "final_bet_ratio": final_ratio,
            "bet_percentage": bet_percentage,
            "recommended_bet_percentage": bet_percentage,
            "suggested_bet_amount": amount,
            "bet_amount": amount,
            "bet_allowed": True,
            "mandatory_bet": True,
            "probability_semantics": "softmax_of_exploit_only_mean_scores_for_sizing",
            "direction_score_semantics": "UCB_argmax_including_exploration",
            "exploration_term_used_for_direction": True,
            "exploration_term_used_for_kelly": False,
            "defensive_min_bet_active": pure_ev <= 0.0,
            "signal_reason": (
                f"Direction=UCB-{direction}；ExploitP={confidence:.3%}；"
                f"PureEV={pure_ev:.3%}；Kelly={bet_percentage:.2f}%"
            ),
        }
    )

    decision_gate = result.get("decision_gate")
    if isinstance(decision_gate, dict):
        decision_gate.update(
            {
                "direction": direction,
                "decision": direction,
                "resolved_confidence": confidence,
                "expected_net_ev": pure_ev,
                "allowed": True,
                "penalty_observe": False,
            }
        )

    context_meta = result.get("context_metadata")
    if isinstance(context_meta, dict):
        context_meta.update(
            {
                "direction_probability_source": "linucb_ucb_argmax",
                "kelly_probability_source": "exploit_only_mean_softmax",
                "tie_memory_semantics": "TIE_FREEZE_BRAIN",
            }
        )
    return result


# The stable virtual-shoe implementation already consumes physical cards before
# feedback. Point depletion therefore still occurs on T; only LinUCB memory is
# frozen by contextual_bandit.py.
_base.predict = predict


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    output = deepcopy(_base.run_virtual_round(deepcopy(dict(session or {})), run_seed=run_seed))
    prediction = output.get("prediction")
    if isinstance(prediction, dict):
        actual = str(prediction.get("virtual_outcome") or "").upper().strip()
        if actual == "T":
            prediction["bandit_learning_applied"] = False
            update = prediction.get("bandit_update")
            if isinstance(update, dict):
                update.update(
                    {
                        "updated": True,
                        "reward": 0.0,
                        "forgetting": 1.0,
                        "memory_decay_applied": False,
                        "directional_sample_applied": False,
                        "reason": "TIE_FREEZE_BRAIN",
                    }
                )
            prediction["tie_physical_cards_consumed"] = True
            prediction["tie_memory_frozen"] = True
    return output


parse_point_observation = _base.parse_point_observation

__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
