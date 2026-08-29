"""BGS predictor using only Big Road B/P sequence features.

The public API is unchanged.  OCR/screenshot adapters may still pass shoe and road
context, but the formal next-round decision is produced solely from the
chronological Big Road B/P sequence by the time-decayed Markov policy.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import (
    POLICY_VERSION,
    normalize_big_road,
    recent_user_direction_feedback,
    road_only_policy,
)
from money_management import MAX_BET_RATIO, MoneyManagementModel
from road_model import ROAD_FEATURE_NAMES, build_road_context

OUTCOMES = ("B", "P", "T")
_MONEY = MoneyManagementModel()
MODEL_VERSION = POLICY_VERSION


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
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
            result.append(value)
    return result[-2000:]


def _history_values(history: Union[str, Iterable[Any], None]) -> List[Any]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in OUTCOMES for char in compact):
            return list(compact)
        return [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    return list(history)


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        banker = max(
            0.0,
            min(1.0, float(road.get("banker_probability", 0.5) or 0.5)),
        )
    except (TypeError, ValueError):
        banker = 0.5
    try:
        player = max(
            0.0,
            min(
                1.0,
                float(road.get("player_probability", 1.0 - banker) or 0.0),
            ),
        )
    except (TypeError, ValueError):
        player = 1.0 - banker
    total = banker + player
    if total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / total, player / total
    try:
        confidence = max(
            0.0,
            min(1.0, float(road.get("confidence_score", 0.0) or 0.0)),
        )
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "direction": "B" if banker >= player else "P",
        "banker_probability": float(banker),
        "player_probability": float(player),
        "confidence": float(confidence),
        "decision_weight": 0.0,
        "diagnostic_only": True,
    }


def _zero_money(money: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    result = dict(money or {})
    result.update({
        "bet_allowed": False,
        "mandatory_bet": False,
        "bet_percentage": 0.0,
        "bet_amount": 0.0,
        "final_bet_ratio": 0.0,
        "pre_tie_adjusted_ratio": 0.0,
        "adjusted_ratio": 0.0,
        "reason": str(reason),
    })
    return result


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
    del run_seed

    raw_history = _normalize_outcome_history(_history_values(history))
    big_road = normalize_big_road(raw_history)

    supplied_road = dict(road_context or {})
    if (
        isinstance(supplied_road.get("road_features"), list)
        and len(supplied_road.get("road_features") or []) == len(ROAD_FEATURE_NAMES)
    ):
        road = supplied_road
    else:
        road = build_road_context(
            raw_history,
            grid_cells=list(supplied_road.get("grid_cells") or []),
            initial_image_count=int(
                supplied_road.get("initial_image_count", 0) or 0
            ),
            manual_count=int(supplied_road.get("manual_count", 0) or 0),
        )
        if supplied_road:
            road["scan_metadata"] = supplied_road

    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))

    policy = road_only_policy(big_road)
    probabilities = dict(policy["probabilities"])
    direction = str(policy["direction"])
    confidence = float(policy["confidence"])
    penalty = dict(policy.get("penalty_observe") or {})
    penalty_active = bool(penalty.get("active", False))

    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )

    too_short = len(big_road) < 4
    positive_virtual_ev = bool(money.get("bet_allowed", False))
    if penalty_active:
        action = "O"
        observe_reason = "penalty_observe_after_two_consecutive_misses"
        money = _zero_money(money, observe_reason)
    elif too_short:
        action = "O"
        observe_reason = "insufficient_big_road_history"
        money = _zero_money(money, observe_reason)
    elif not positive_virtual_ev:
        action = "O"
        observe_reason = "model_probability_virtual_ev_not_positive"
        money = _zero_money(money, observe_reason)
    else:
        action = direction
        observe_reason = ""

    text = "觀望" if action == "O" else ("莊" if action == "B" else "閒")
    direction_text = "莊" if direction == "B" else "閒"
    bet_allowed = bool(action in {"B", "P"} and money.get("bet_allowed", False))

    if penalty_active:
        signal_status_code = "PENALTY_OBSERVE"
        signal_status_text = (
            f"觀望校正：連錯 2 局後虛擬下注中；"
            f"最少觀望剩餘 {int(penalty.get('observe_remaining', 0) or 0)} 局"
        )
    elif too_short:
        signal_status_code = "ROAD_HISTORY_WARMUP"
        signal_status_text = "觀望：大路 B/P 樣本不足 4 局"
    elif not bet_allowed:
        signal_status_code = "ROAD_MODEL_NO_POSITIVE_VIRTUAL_EV"
        signal_status_text = "觀望：模型方向存在，但虛擬 EV 未達正期望門檻"
    else:
        signal_status_code = "ROAD_MODEL_QUARTER_KELLY"
        signal_status_text = (
            f"時間衰減馬可夫：{direction_text} "
            f"{confidence:.1%}；1/4 Kelly "
            f"{float(money.get('bet_percentage', 0.0) or 0.0):.2f}%"
        )

    fingerprint = sha256(
        "|".join((
            "".join(big_road),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
            POLICY_VERSION,
        )).encode("utf-8")
    ).hexdigest()[:24]

    road_predict = _road_diagnostic(road)
    p_b = float(probabilities["B"])
    p_p = float(probabilities["P"])
    p_t = float(probabilities.get("T", 0.0) or 0.0)
    feedback = recent_user_direction_feedback(user_id)

    markov_predict = {
        "direction": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "pattern_calibrated_probabilities": {
            "B": p_b,
            "P": p_p,
            "T": p_t,
        },
        "state": {
            "direction_context": str(policy.get("state_key") or ""),
            "density": "RoadOnly",
            "tie_trigger": "Ignored",
        },
        "state_key": str(policy.get("state_key") or ""),
        "transition_counts": dict(policy.get("transition_counts") or {}),
        "effective_support": float(policy.get("effective_support", 0.0) or 0.0),
        "entropy_bits": 0.0,
        "base_weight": confidence,
        "raw_final_weight": confidence,
        "final_weight": confidence,
        "pattern_survival_score": 1.0,
        "decay": float(policy.get("decay", 0.0) or 0.0),
        "retention_lambda": float(policy.get("decay", 0.0) or 0.0),
        "decay_intensity": float(1.0 - float(policy.get("decay", 0.0) or 0.0)),
        "selected_order": int(policy.get("selected_order", 0) or 0),
        "support_threshold": 0,
        "backoff_steps": max(
            0,
            int(policy.get("max_order", 0) or 0)
            - int(policy.get("selected_order", 0) or 0),
        ),
        "backoff_penalty": 1.0,
        "focus_applied": True,
        "regime_profile": {
            "change_point": penalty_active,
            "penalty_observe": penalty,
        },
        "prior": dict(policy.get("global_probabilities") or {}),
        "prior_strength": 0.75,
        "order_diagnostics": list(policy.get("order_diagnostics") or []),
    }

    component_probabilities = {
        "decayed_markov": {"B": p_b, "P": p_p, "T": p_t},
        "road_only_model": {"B": p_b, "P": p_p, "T": p_t},
    }

    remaining_cards = context.get("remaining_cards")
    try:
        remaining_cards_value = int(remaining_cards or 0)
    except (TypeError, ValueError):
        remaining_cards_value = 0

    neutral_shoe = {
        "direction": direction,
        "probabilities": {"B": 0.5, "P": 0.5, "T": 0.0},
        "bp_conditional_probabilities": {"B": 0.5, "P": 0.5},
        "expected_remaining_cards": float(remaining_cards_value),
        "expected_remaining_decks": 0.0,
        "expected_remaining_counts": [],
        "remaining_count_std": [],
        "remaining_card_state": {
            "available": False,
            "conditioned_rounds": len(big_road),
            "source": "road_only_no_exact_card_dependency",
        },
        "conditioned_rounds": len(big_road),
        "particle_count": 0,
        "reliability": 0.0,
        "fusion_weight": 0.0,
        "target_remaining_cards": (
            remaining_cards_value if remaining_cards_value > 0 else None
        ),
        "depth_constraint_applied": False,
        "depth_constraint": {
            "applied": False,
            "reason": "road_only_model_does_not_use_card_depth",
        },
        "shoe_tendency": {},
        "inference_semantics": "diagnostic_only_not_used_for_decision",
    }

    edge = float(money.get("edge", 0.0) or 0.0)
    final_ratio = float(money.get("final_bet_ratio", 0.0) or 0.0)

    result = {
        "ok": True,
        "engine": "BIG_ROAD_TIME_DECAY_MARKOV",
        "model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "DISABLED_FOR_FORMAL_DECISION",
        "system_model_version": POLICY_VERSION,
        "model_variant": "ROAD_ONLY_VARIABLE_ORDER_DECAY_MARKOV_WITH_PENALTY_OBSERVE",
        "model_core": "dynamic_prediction_policy",
        "decision_pipeline": (
            "big_road_bp_only_to_time_decay_variable_order_markov_"
            "to_two_miss_penalty_observe_to_virtual_ev_to_quarter_kelly"
        ),
        "prediction_fingerprint": fingerprint,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "raw_direction_probabilities": {"B": p_b, "P": p_p},
        "banker_rate": round(p_b * 100.0, 2),
        "player_rate": round(p_p * 100.0, 2),
        "tie_rate": round(p_t * 100.0, 2),
        "recommend": action,
        "recommend_text": text,
        "action": action,
        "action_text": text,
        "internal_recommend": action,
        "internal_action": action,
        "next_round_direction": action,
        "next_round_direction_text": text,
        "direction": direction,
        "direction_text": direction_text,
        "adaptive_only_direction": direction,
        "signal_allowed": bet_allowed,
        "risk_gate_open": bet_allowed,
        "mandatory_bet": False,
        "signal_status_code": signal_status_code,
        "signal_status_text": signal_status_text,
        "signal_reason": (
            f"BigRoadBP={len(big_road)}；"
            f"order={int(policy.get('selected_order', 0) or 0)}；"
            f"state={str(policy.get('state_key') or 'START')}；"
            f"decay={float(policy.get('decay', 0.0) or 0.0):.3f}；"
            f"P(B)={p_b:.3f}；P(P)={p_p:.3f}；"
            f"virtualEV={float(money.get('virtual_ev', 0.0) or 0.0):.4f}；"
            f"penalty={penalty_active}；"
            f"sizing={str(money.get('reason') or '')}。"
        ),
        "internal_signal_reason": "",
        "direction_source": "time_decay_markov_big_road_bp_only",
        "confidence": confidence,
        "raw_markov_confidence": confidence,
        "pattern_calibrated_confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": (
            "較高" if confidence >= 0.60
            else "中等" if confidence >= 0.54
            else "偏低"
        ),
        "entropy_bits": 0.0,
        "entropy_base_weight": confidence,
        "shoe_progress": float(len(big_road) / 70.0),
        "shoe_depth_estimate": {
            "rounds": len(big_road),
            "source": "big_road_count_only",
        },
        "remaining_card_state": dict(neutral_shoe["remaining_card_state"]),
        "estimated_remaining_cards": float(remaining_cards_value),
        "estimated_remaining_interval": {
            "low": float(remaining_cards_value),
            "high": float(remaining_cards_value),
        },
        "shoe_stage": "ROAD_ONLY",
        "pattern_survival": {
            "score": 1.0,
            "mode": "not_used",
        },
        "pattern_survival_score": 1.0,
        "run_length_hazard": {},
        "run_length_hazard_weight": 0.0,
        "probabilistic_shoe_estimate": dict(neutral_shoe),
        "tie_risk_active": False,
        "direction_edge": edge,
        "direction_edge_percent": round(edge * 100.0, 4),
        "selected_expected_return": float(
            money.get("virtual_ev", 0.0) or 0.0
        ) if bet_allowed else 0.0,
        "selected_expected_return_percent": float(
            money.get("virtual_ev_percent", 0.0) or 0.0
        ) if bet_allowed else 0.0,
        "bet_allowed": bet_allowed,
        "markov": dict(markov_predict),
        "markov_probs": {"B": p_b, "P": p_p, "T": p_t},
        "final_probs": {"B": p_b, "P": p_p, "T": p_t},
        "direction_probs": {"B": p_b, "P": p_p, "T": p_t},
        "economic_probs": {"B": p_b, "P": p_p, "T": p_t},
        "final_probability": float(probabilities[direction]),
        "economic_probability_for_direction": float(probabilities[direction]),
        "markov_predict": markov_predict,
        "probabilistic_shoe_predict": dict(neutral_shoe),
        "fusion_decision": {
            "direction": direction,
            "probabilities": {"B": p_b, "P": p_p, "T": p_t},
            "markov_prior_weight": 1.0,
            "probabilistic_shoe_weight": 0.0,
            "derived_road_likelihood_power": 0.0,
            "run_length_hazard_likelihood_power": 0.0,
            "road_applied": False,
            "hazard_applied": False,
            "method": "road_only_time_decay_markov",
            "semantics": "formal_decision_uses_only_big_road_bp_sequence",
        },
        "fusion": {
            "method": "road_only_time_decay_markov",
            "shoe_reliability": 0.0,
            "road_reliability": 1.0,
            "hazard_reliability": 0.0,
        },
        "markov_state": {
            "state_key": str(policy.get("state_key") or ""),
            "direction_context": str(policy.get("state_key") or ""),
            "density": "RoadOnly",
            "tie_trigger": "Ignored",
            "sample_count": len(big_road),
            "effective_support": float(
                policy.get("effective_support", 0.0) or 0.0
            ),
            "state_count": len(
                list(policy.get("order_diagnostics") or [])
            ),
            "selected_order": int(policy.get("selected_order", 0) or 0),
            "change_point": penalty_active,
            "shoe_stage": "ROAD_ONLY",
            "pattern_survival_score": 1.0,
        },
        "road_predict": road_predict,
        "road_support": road,
        "derived_road_analysis": {},
        "road_fusion": {
            "applied": False,
            "mode": "diagnostic_only",
            "reliability": 0.0,
            "raw_reliability": 0.0,
            "pattern_survival_score": 1.0,
            "likelihood": None,
            "reason": "正式方向只使用大路 B/P 時間衰減馬可夫。",
        },
        "run_length_hazard_fusion": {
            "applied": False,
            "reliability": 0.0,
            "raw_reliability": 0.0,
            "likelihood": None,
            "continue_probability": 0.5,
            "turn_probability": 0.5,
            "selected_context": "",
            "support": 0.0,
            "reason": "正式方向只使用大路 B/P 時間衰減馬可夫。",
        },
        "component_probabilities": component_probabilities,
        "money_management": dict(money),
        "kelly_fraction": float(money.get("kelly_fraction", 0.0) or 0.0),
        "pre_tie_adjusted_ratio": float(
            money.get("pre_tie_adjusted_ratio", 0.0) or 0.0
        ),
        "adjusted_ratio": float(money.get("adjusted_ratio", 0.0) or 0.0),
        "final_bet_ratio": final_ratio,
        "bet_percentage": float(money.get("bet_percentage", 0.0) or 0.0),
        "suggested_bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_multiplier": (
            min(1.0, final_ratio / MAX_BET_RATIO)
            if MAX_BET_RATIO > 0.0 else 0.0
        ),
        "context_vector": list(road.get("road_features") or []),
        "bandit_context": [],
        "context_feature_names": list(ROAD_FEATURE_NAMES),
        "context_dim": len(list(road.get("road_features") or [])),
        "contextual_bandit_enabled": False,
        "contextual_bandit_update_enabled": False,
        "cusum_linucb_enabled": False,
        "force_observe": bool(action == "O"),
        "hard_brake_active": False,
        "post_reset_vacuum_active": penalty_active,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": (
            "time_decayed_big_road_bp_transition_probability_"
            "not_exact_card_probability_and_not_guaranteed_outcome"
        ),
        "dynamic_prediction_policy": {
            "version": POLICY_VERSION,
            "road_only": True,
            "big_road_rounds": len(big_road),
            "forecast": dict(policy),
            "penalty_observe": penalty,
            "exact_card_dependency": False,
            "shoe_probability_decision_weight": 0.0,
            "ocr_or_screen_flow_modified": False,
        },
        "dynamic_policy_version": POLICY_VERSION,
        "online_performance_feedback": feedback,
        "decision": action,
        "decision_text": text,
        "skip": bool(action == "O"),
        "skip_reason": observe_reason,
        "decision_gate": {
            "decision": action,
            "allowed": bet_allowed,
            "reason": observe_reason or "road_model_positive_virtual_ev",
            "direction": direction,
            "resolved_confidence": confidence,
            "expected_net_ev": float(
                money.get("virtual_ev", 0.0) or 0.0
            ),
            "penalty_observe": penalty_active,
        },
        "timeline_alignment": {
            "raw_round_count": len(raw_history),
            "bp_round_count": len(big_road),
            "ties_ignored_for_model": len(raw_history) - len(big_road),
        },
    }
    result["internal_signal_reason"] = result["signal_reason"]
    return result


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compatibility API for the existing virtual-shoe endpoint."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")

    outcome_history = _normalize_outcome_history(
        list(session.get("round_history") or [])
    )
    seed = int(
        run_seed if run_seed is not None else secrets.randbits(32)
    ) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        run_seed=seed,
        shoe_context={
            "bankroll": float(session.get("bankroll", 0.0) or 0.0),
            "remaining_cards": len(hidden_shoe),
            "remaining_cards_reliability": 1.0,
        },
    )

    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(prediction.get("action") or "").upper()
    actual = str(hand.outcome or "").upper()
    if actual == "T":
        verdict = "TIE_SKIPPED"
    elif predicted_side in {"B", "P"}:
        verdict = "HIT" if predicted_side == actual else "MISS"
    else:
        verdict = "OBSERVE"

    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_road_only_markov_compatibility",
        "virtual_hand": hand_data,
        "virtual_outcome": actual,
        "virtual_outcome_text": hand_data["outcome_text"],
        "verdict": verdict,
        "verdict_text": {
            "HIT": "命中",
            "MISS": "未命中",
            "TIE_SKIPPED": "和局不計",
            "OBSERVE": "觀望／虛擬下注",
        }[verdict],
        "cards_consumed": int(hand.cards_used),
        "remaining_cards_after": len(remaining_shoe),
        "remaining_counts_after": counts_from_shoe(remaining_shoe),
        "round_number": int(session.get("hand_number", 0) or 0) + 1,
        "bandit_learning_applied": False,
        "disclaimer": (
            "正式方向只使用大路 B/P 時間衰減馬可夫；"
            "精確牌面與剩餘牌數不參與方向預測。"
        ),
    })
    return {
        "prediction": prediction,
        "hand": hand_data,
        "remaining_shoe": remaining_shoe,
    }


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = [
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
