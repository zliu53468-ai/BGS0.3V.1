"""Validated decision layer for the road-only BGS model.

The public functions are unchanged.  Exact remaining-card composition is no
longer a prerequisite for a B/P decision.  The layer validates the road-model
probability, converts it to virtual EV, and delegates sizing to
``MoneyManagementModel`` (one-quarter Kelly).
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
from performance_tracker import get_performance_summary

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
        0.65,
        float(os.getenv("UNVALIDATED_CONFIDENCE_CAP", "0.58") or "0.58"),
    ),
)
VALIDATED_CONFIDENCE_CAP = max(
    0.52,
    min(
        0.75,
        float(os.getenv("VALIDATED_CONFIDENCE_CAP", "0.68") or "0.68"),
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


def _calibrated_confidence(
    result: Mapping[str, Any],
    *,
    venue: str,
    room: str,
) -> tuple[float, int, str]:
    probabilities = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {"B": 0.5, "P": 0.5, "T": 0.0}
    )
    resolved = probabilities["B"] + probabilities["P"]
    raw = (
        max(probabilities["B"], probabilities["P"]) / resolved
        if resolved > 1e-12 else 0.5
    )
    source = str(result.get("direction_source") or "")
    try:
        summary = get_performance_summary(
            venue=str(venue or ""),
            room=str(room or ""),
            limit=5000,
        )
        stats = dict(
            dict(summary.get("source_direction_performance") or {}).get(source)
            or {}
        )
        samples = int(stats.get("sample_count", 0) or 0)
        if samples >= VALIDATED_COMPONENT_MIN_SAMPLES:
            posterior = float(stats.get("posterior_accuracy", raw) or raw)
            return (
                max(0.50, min(VALIDATED_CONFIDENCE_CAP, posterior)),
                samples,
                "validated_direction_source_posterior",
            )
    except Exception:
        pass
    return (
        max(0.50, min(UNVALIDATED_CONFIDENCE_CAP, raw)),
        0,
        "road_model_probability_cap",
    )


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
    """Calibrate confidence without reviving an existing observe/hard-brake state."""
    result = dict(prediction or {})
    if _hard_brake_active(result):
        result["decision_validation"] = {
            "active": False,
            "hard_brake_preserved": True,
            "reason": "統計硬熔斷優先，驗證層不得恢復方向",
        }
        return result

    action = str(result.get("action") or "O").upper().strip()
    if action not in {"B", "P"}:
        result["decision_validation"] = {
            "active": False,
            "hard_brake_preserved": False,
            "observe_preserved": True,
            "reason": "保留時間衰減馬可夫的觀望／校正狀態",
        }
        return result

    direction = str(result.get("direction") or action).upper().strip()
    if direction not in {"B", "P"}:
        direction = action
    confidence, samples, source = _calibrated_confidence(
        result,
        venue=venue,
        room=room,
    )
    result.setdefault(
        "pre_validation_probabilities",
        dict(result.get("probabilities") or {}),
    )
    _set_direction_distribution(
        result,
        direction=direction,
        confidence=confidence,
    )
    result["confidence"] = confidence
    result["ensemble_confidence"] = confidence
    result["quality_score"] = confidence
    result["confidence_label"] = (
        "較高" if confidence >= 0.60
        else "中等" if confidence >= 0.54
        else "偏低"
    )
    result["decision_validation"] = {
        "active": True,
        "mode": "road_model_probability_validation",
        "direction_before": action,
        "direction_after": direction,
        "confidence_source": source,
        "confidence_sample_count": samples,
        "calibrated_confidence": confidence,
        "exact_card_counts_required": False,
    }
    return result


def _strategy_road_direction(result: Mapping[str, Any]) -> tuple[str, float]:
    probabilities = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else {"B": 0.5, "P": 0.5, "T": 0.0}
    )
    bp_mass = probabilities["B"] + probabilities["P"]
    if bp_mass <= 1e-12:
        return "", 0.0
    banker = probabilities["B"] / bp_mass
    direction = "B" if banker >= 0.5 else "P"
    strength = abs(banker - 0.5) * 2.0
    return direction, max(0.0, min(1.0, strength))


def apply_strategy_decision(
    prediction: Mapping[str, Any],
    *,
    strategy_selection: Mapping[str, Any],
    bankroll: float = 0.0,
) -> Dict[str, Any]:
    """Apply strategy scaling to road-model virtual EV and quarter Kelly.

    Exact ``remaining_counts`` / ``observed_cards`` are intentionally not used as
    a gate. A model probability such as P(B)=0.53 is converted to a virtual EV
    after commission and then passed through ``MoneyManagementModel``.
    """
    result = dict(prediction or {})
    selection = dict(strategy_selection or {})
    profile = dict(selection.get("profile") or {})
    arm = str(selection.get("selected_arm") or "math_only")

    penalty = dict(
        dict(result.get("dynamic_prediction_policy") or {}).get(
            "penalty_observe", {}
        )
        or dict(result.get("decision_gate") or {}).get("penalty_observe", {})
        or {}
    )
    force_observe = bool(
        result.get("force_observe")
        or result.get("post_reset_vacuum_active")
        or (
            isinstance(penalty, Mapping)
            and penalty.get("active")
        )
    )
    if force_observe:
        result["decision_strategy_bandit"] = selection
        result["decision_strategy"] = arm
        result["decision_validation"] = {
            "active": True,
            "mode": "penalty_observe_preserved",
            "exact_card_counts_required": False,
        }
        return _set_no_bet(
            result,
            "連錯 2 局後的懲罰觀望仍在進行；本局只做虛擬下注與轉移矩陣更新。",
        )

    probabilities = _normalize(
        result.get("probabilities")
        if isinstance(result.get("probabilities"), Mapping)
        else dict(result.get("economic_probs") or {})
    )
    direction = str(
        result.get("direction")
        or result.get("adaptive_only_direction")
        or result.get("action")
        or "O"
    ).upper().strip()
    if direction not in {"B", "P"}:
        direction = "B" if probabilities["B"] >= probabilities["P"] else "P"

    model_confidence = max(probabilities["B"], probabilities["P"])
    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=model_confidence,
        bankroll=max(0.0, float(bankroll or 0.0)),
    )
    virtual_ev = float(money.get("virtual_ev", 0.0) or 0.0)
    required_ev = max(
        float(MIN_POSITIVE_EV),
        float(MIN_POSITIVE_EV)
        * max(0.0, float(profile.get("minimum_ev_multiplier", 1.0) or 1.0)),
    )

    result["decision_strategy_bandit"] = selection
    result["decision_strategy"] = arm
    result["strategy_hard_limits"] = {
        "kelly_fraction": float(KELLY_FRACTION),
        "max_bet_fraction": float(MAX_BET_RATIO),
        "minimum_positive_ev": float(MIN_POSITIVE_EV),
        "exact_card_counts_required": False,
    }
    result["model_virtual_signal"] = {
        "available": True,
        "source": "time_decay_markov_big_road_probability",
        "action": direction,
        "probabilities": probabilities,
        "selected_expected_return": virtual_ev,
        "kelly_fraction": float(money.get("kelly_fraction", 0.0) or 0.0),
        "trusted_exact_counts": False,
        "exact_card_counts_required": False,
    }

    if virtual_ev < required_ev or not bool(money.get("bet_allowed", False)):
        result["decision_validation"] = {
            "active": True,
            "mode": "road_model_virtual_ev_gate",
            "selected_strategy_arm": arm,
            "model_direction": direction,
            "model_virtual_ev": virtual_ev,
            "required_ev": required_ev,
            "exact_card_counts_required": False,
        }
        return _set_no_bet(
            result,
            "牌路模型虛擬 EV 未達本策略正期望門檻，維持觀望。",
        )

    multiplier = min(
        1.0,
        max(0.0, float(profile.get("kelly_multiplier", 1.0) or 1.0)),
    )
    road_direction, road_strength = _strategy_road_direction(result)
    road_note = "純牌路模型機率直接進入 1/4 Kelly。"
    if arm == "conservative":
        multiplier *= 0.50
        road_note = "保守策略將 1/4 Kelly 再縮半。"
    elif arm == "ev_road_blend":
        if road_strength < 0.20:
            multiplier *= 0.75
            road_note = "牌路優勢幅度偏弱，1/4 Kelly 再縮至 75%。"

    base_ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)
    final_ratio = min(MAX_BET_RATIO, max(0.0, base_ratio * multiplier))
    if final_ratio <= 0.0:
        return _set_no_bet(result, "策略縮放後下注比例為零，維持觀望。")

    bankroll_value = max(0.0, float(bankroll or 0.0))
    amount = bankroll_value * final_ratio
    text = "莊" if direction == "B" else "閒"
    money.update({
        "final_bet_ratio": final_ratio,
        "bet_percentage": final_ratio * 100.0,
        "bet_amount": amount,
        "bet_allowed": True,
        "strategy_multiplier": multiplier,
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
        "direction_source": "road_model_virtual_ev_quarter_kelly",
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
        "strategy_required_ev": required_ev,
        "kelly_cap_enforced": True,
        "banker_rate": round(probabilities["B"] * 100.0, 4),
        "player_rate": round(probabilities["P"] * 100.0, 4),
        "tie_rate": round(probabilities["T"] * 100.0, 4),
        "signal_reason": (
            f"{profile.get('label', arm)}：牌路模型虛擬 EV "
            f"{virtual_ev:.3%}；{road_note}"
        ),
    })
    result["reason"] = result["signal_reason"]
    result["decision_validation"] = {
        "active": True,
        "mode": "road_model_virtual_ev_quarter_kelly",
        "selected_strategy_arm": arm,
        "exact_card_counts_required": False,
        "model_direction": direction,
        "model_virtual_ev": virtual_ev,
        "required_ev": required_ev,
        "quarter_kelly_multiplier": float(KELLY_FRACTION),
        "base_kelly_ratio": base_ratio,
        "final_kelly_ratio": final_ratio,
        "max_bet_fraction": float(MAX_BET_RATIO),
        "road_note": road_note,
    }
    return result


__all__ = ["apply_strategy_decision", "apply_validated_decision"]
