"""BGS 正式預測器：已揭曉 B/P 歷史 -> 因果式 road_forecaster -> Kelly。

公開 predict() 參數與主要回傳欄位維持相容。OCR / screenshot / road adapter
仍可傳入既有 context，但正式下一局方向只由 forecaster 機率 argmax 決定。
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import (
    POLICY_VERSION,
    linucb_policy,
    normalize_big_road,
    recent_user_direction_feedback,
    record_online_feedback,
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
        return [part for part in history.replace("|", ",").split(",") if part.strip()]
    return list(history)


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    """既有牌路模型只保留 UI/稽核診斷，不參與正式 forecaster 方向。"""
    try:
        banker = max(0.0, min(1.0, float(road.get("banker_probability", 0.5) or 0.5)))
    except (TypeError, ValueError):
        banker = 0.5
    try:
        player = max(0.0, min(1.0, float(road.get("player_probability", 1.0 - banker) or 0.0)))
    except (TypeError, ValueError):
        player = 1.0 - banker
    total = banker + player
    if total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / total, player / total
    try:
        confidence = max(0.0, min(1.0, float(road.get("confidence_score", 0.0) or 0.0)))
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

    # OCR/掃描資料只讀取既有輸出，不改動任何掃描程式；road_context 只做診斷。
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
            initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0),
            manual_count=int(supplied_road.get("manual_count", 0) or 0),
        )
        if supplied_road:
            road["scan_metadata"] = supplied_road

    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))

    # 保留 linucb_policy 呼叫介面；其正式來源已切換為逐手訓練的 forecaster。
    # 只使用已揭曉大路 B/P。牌組欄位、OCR 診斷及事後斜率不能覆蓋方向。
    policy = linucb_policy(
        raw_history,
        shoe_context=context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    probabilities = dict(policy["probabilities"])
    direction = str(policy["direction"])
    confidence = float(policy["selected_win_probability"])
    action = direction
    text = "莊" if action == "B" else "閒"
    direction_text = text

    # 強制 5%～30%：每局都要有明確可執行注碼；Kelly 仍負責相對風險尺度，
    # 最後由硬下限/上限控制實際資金暴露。
    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )
    bet_allowed = True
    signal_status_code = "LINUCB_TWO_ARM_KELLY_5_30"
    signal_status_text = (
        f"下一手模型：{direction_text} {confidence:.1%}；"
        f"Kelly {float(money.get('bet_percentage', 5.0) or 5.0):.2f}%"
    )

    fingerprint = sha256(
        "|".join(
            (
                "".join(raw_history),
                str(venue or "").upper().strip(),
                str(room or "").strip(),
                str(shoe_id or "").strip(),
                POLICY_VERSION,
            )
        ).encode("utf-8")
    ).hexdigest()[:24]

    road_predict = _road_diagnostic(road)
    p_b = float(probabilities["B"])
    p_p = float(probabilities["P"])
    p_t = float(probabilities.get("T", 0.0) or 0.0)
    feedback = recent_user_direction_feedback(user_id)
    context_vector = list(policy.get("context_vector") or [])
    context_feature_names = list(policy.get("context_feature_names") or [])
    context_meta = dict(policy.get("context_metadata") or {})
    bandit_scores = dict(policy.get("scores") or {})
    bandit_feedback_update = dict(policy.get("feedback_update") or {})

    markov_predict = {
        "direction": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "pattern_calibrated_probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "state": {
            "direction_context": "LINUCB",
            "density": "CardCompositionFirst",
            "tie_trigger": "RewardZero",
        },
        "state_key": "LINUCB",
        "transition_counts": {},
        "effective_support": float(sum(int((bandit_scores.get(side) or {}).get("uncertainty", 0.0) >= 0.0) for side in ("B", "P"))),
        "entropy_bits": 0.0,
        "base_weight": confidence,
        "raw_final_weight": confidence,
        "final_weight": confidence,
        "pattern_survival_score": 1.0,
        "decay": 0.0,
        "retention_lambda": 0.0,
        "decay_intensity": 0.0,
        "selected_order": 0,
        "support_threshold": 0,
        "backoff_steps": 0,
        "backoff_penalty": 1.0,
        "focus_applied": True,
        "regime_profile": {
            "change_point": False,
            "bandit_feedback_update": bandit_feedback_update,
        },
        "prior": {"B": 0.5, "P": 0.5},
        "prior_strength": float(policy.get("ridge", 1.0) or 1.0),
        "order_diagnostics": [],
        "bandit_scores": bandit_scores,
    }

    component_probabilities = {
        "road_forecaster": {"B": p_b, "P": p_p, "T": p_t},
        # Legacy alias retained; these are forecaster probabilities, not UCB output.
        "contextual_linucb": {"B": p_b, "P": p_p, "T": p_t},
        "road_diagnostic": {
            "B": float(road_predict["banker_probability"]),
            "P": float(road_predict["player_probability"]),
            "T": 0.0,
        },
    }

    remaining_cards_value = float(context_meta.get("remaining_cards", 0.0) or 0.0)
    estimated_counts = list(context_meta.get("estimated_remaining_counts_0_to_9") or [])
    supplied_remaining = bool(context.get("remaining_cards") or context.get("remaining_counts"))
    depth_constraint = {
        "applied": supplied_remaining,
        "reason": (
            "supplied_remaining_card_depth_used_in_linucb_context"
            if supplied_remaining
            else "remaining_depth_estimated_from_round_count"
        ),
        "target_remaining_cards": remaining_cards_value,
    }
    shoe_estimate = {
        "direction": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": 0.0},
        "bp_conditional_probabilities": {"B": p_b, "P": p_p},
        "expected_remaining_cards": remaining_cards_value,
        "expected_remaining_decks": remaining_cards_value / 52.0,
        "expected_remaining_counts": estimated_counts,
        "remaining_count_std": [],
        "remaining_card_state": {
            "available": True,
            "conditioned_rounds": len(big_road),
            "source": str(context_meta.get("remaining_cards_source") or "estimated"),
        },
        "conditioned_rounds": len(big_road),
        "particle_count": 0,
        "reliability": 1.0 if supplied_remaining else 0.5,
        "fusion_weight": 0.0,
        "target_remaining_cards": remaining_cards_value,
        "depth_constraint_applied": supplied_remaining,
        "depth_constraint": depth_constraint,
        "shoe_tendency": {
            "high_vs_four_ratio_delta": float(context_vector[11]) if len(context_vector) > 11 else 0.0,
        },
        "inference_semantics": "linucb_context_estimate_not_exact_next_card_probability",
    }

    edge = float(money.get("edge", 0.0) or 0.0)
    final_ratio = float(money.get("final_bet_ratio", 0.05) or 0.05)
    bet_percentage = float(money.get("bet_percentage", final_ratio * 100.0) or final_ratio * 100.0)

    result = {
        "ok": True,
        "engine": "CAUSAL_ROAD_FORECASTER_BP",
        "model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "CONTEXT_GENERATOR_V1",
        "system_model_version": POLICY_VERSION,
        "model_variant": "ONLINE_L2_LOGISTIC_ROAD_FORECASTER_KELLY_5_30",
        "model_core": "road_forecaster",
        "decision_pipeline": "observed_BP_prefix_to_causal_logistic_argmax_to_existing_kelly",
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
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": True,
        "signal_status_code": signal_status_code,
        "signal_status_text": signal_status_text,
        "signal_reason": (
            f"Forecaster={str(policy.get('road_forecaster', {}).get('model_id', ''))}；"
            f"support={float(policy.get('effective_support', 0.0)):.0f}；"
            f"P(B)={p_b:.3f}；P(P)={p_p:.3f}；"
            f"Kelly={bet_percentage:.2f}%；feedback={bool(bandit_feedback_update.get('updated', False))}。"
        ),
        "internal_signal_reason": "",
        "direction_source": "road_forecaster_probability_argmax",
        "confidence": confidence,
        "raw_markov_confidence": confidence,
        "pattern_calibrated_confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": "較高" if confidence >= 0.56 else "中等" if confidence >= 0.52 else "保守",
        "entropy_bits": 0.0,
        "entropy_base_weight": confidence,
        "shoe_progress": float(min(1.0, len(big_road) / 70.0)),
        "shoe_depth_estimate": {
            "rounds": len(big_road),
            "remaining_cards": remaining_cards_value,
            "source": str(context_meta.get("remaining_cards_source") or "estimated"),
        },
        "remaining_card_state": dict(shoe_estimate["remaining_card_state"]),
        "estimated_remaining_cards": remaining_cards_value,
        "estimated_remaining_interval": {"low": remaining_cards_value, "high": remaining_cards_value},
        "shoe_stage": "EARLY" if len(big_road) <= 20 else "MID" if len(big_road) < 41 else "LATE",
        "pattern_survival": {"score": 1.0, "mode": "diagnostic_only"},
        "pattern_survival_score": 1.0,
        "run_length_hazard": {},
        "run_length_hazard_weight": 0.0,
        "probabilistic_shoe_estimate": dict(shoe_estimate),
        "tie_risk_active": False,
        "direction_edge": edge,
        "direction_edge_percent": round(edge * 100.0, 4),
        "selected_expected_return": float(money.get("virtual_ev", 0.0) or 0.0),
        "selected_expected_return_percent": float(money.get("virtual_ev_percent", 0.0) or 0.0),
        "bet_allowed": True,
        "markov": dict(markov_predict),
        "markov_probs": {"B": p_b, "P": p_p, "T": p_t},
        "final_probs": {"B": p_b, "P": p_p, "T": p_t},
        "direction_probs": {"B": p_b, "P": p_p, "T": p_t},
        "economic_probs": {"B": p_b, "P": p_p, "T": p_t},
        "final_probability": float(probabilities[direction]),
        "economic_probability_for_direction": float(money.get("resolved_win_probability", confidence) or confidence),
        "markov_predict": markov_predict,
        "probabilistic_shoe_predict": dict(shoe_estimate),
        "fusion_decision": {
            "direction": direction,
            "probabilities": {"B": p_b, "P": p_p, "T": p_t},
            "markov_prior_weight": 0.0,
            "probabilistic_shoe_weight": 0.0,
            "road_forecaster_weight": 1.0,
            "derived_road_likelihood_power": 0.0,
            "run_length_hazard_likelihood_power": 0.0,
            "road_applied": False,
            "hazard_applied": False,
            "method": "road_forecaster_probability_argmax",
            "semantics": "formal_direction_uses_only_causal_BP_prefix_features",
        },
        "fusion": {
            "method": "road_forecaster_probability_argmax",
            "shoe_reliability": 1.0 if supplied_remaining else 0.5,
            "road_reliability": 0.25,
            "hazard_reliability": 0.0,
        },
        "markov_state": {
            "state_key": "LINUCB",
            "direction_context": "LINUCB",
            "density": "CardCompositionFirst",
            "tie_trigger": "RewardZero",
            "sample_count": len(big_road),
            "effective_support": int((bandit_scores.get("P") or {}).get("uncertainty", 0.0) >= 0.0) + int((bandit_scores.get("B") or {}).get("uncertainty", 0.0) >= 0.0),
            "state_count": 2,
            "selected_order": 0,
            "change_point": False,
            "shoe_stage": "SHORT_SHOE_50_70_TARGET",
            "pattern_survival_score": 1.0,
        },
        "road_predict": road_predict,
        "road_support": road,
        "derived_road_analysis": dict(policy.get("regression_analysis") or {}),
        "road_fusion": {
            "applied": False,
            "mode": "diagnostic_only",
            "reliability": 0.25,
            "raw_reliability": 0.25,
            "pattern_survival_score": 1.0,
            "likelihood": None,
            "reason": "本區事後牌路診斷不參與方向；正式方向由逐手訓練 forecaster 決定。",
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
            "reason": "非正式方向來源。",
        },
        "component_probabilities": component_probabilities,
        "money_management": dict(money),
        "kelly_fraction": float(money.get("kelly_fraction", final_ratio) or final_ratio),
        "pre_tie_adjusted_ratio": float(money.get("pre_tie_adjusted_ratio", final_ratio) or final_ratio),
        "adjusted_ratio": float(money.get("adjusted_ratio", final_ratio) or final_ratio),
        "final_bet_ratio": final_ratio,
        "bet_percentage": bet_percentage,
        "suggested_bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_multiplier": min(1.0, final_ratio / MAX_BET_RATIO) if MAX_BET_RATIO > 0.0 else 1.0,
        "context_vector": context_vector,
        "bandit_context": context_vector,
        "context_feature_names": context_feature_names,
        "context_dim": len(context_vector),
        "bandit_scores": bandit_scores,
        "bandit_selected_arm": direction,
        "bandit_scope_key": str(policy.get("scope_key") or ""),
        "bandit_feedback_update": bandit_feedback_update,
        "contextual_bandit_enabled": True,
        "contextual_bandit_update_enabled": True,
        "linucb_enabled": True,
        "linucb_diagnostic_only": True,
        "linucb_direction_weight": 0.0,
        "road_forecaster": dict(policy["road_forecaster"]),
        "cusum_linucb_enabled": False,
        "force_observe": False,
        "hard_brake_active": False,
        "post_reset_vacuum_active": False,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": "causal_online_logistic_next_resolved_BP_probability",
        "dynamic_prediction_policy": {
            "version": POLICY_VERSION,
            "road_only": True,
            "big_road_rounds": len(big_road),
            "forecast": dict(policy),
            "penalty_observe": {"active": False, "force_observe": False},
            "exact_card_dependency": False,
            "shoe_probability_decision_weight": 0.0,
            "ocr_or_screen_flow_modified": False,
            "formal_direction_source": "road_forecaster",
        },
        "dynamic_policy_version": POLICY_VERSION,
        "online_performance_feedback": feedback,
        "decision": action,
        "decision_text": text,
        "skip": False,
        "skip_reason": "",
        "decision_gate": {
            "decision": action,
            "allowed": True,
            "reason": "road_forecaster_argmax_always_returns_BP",
            "direction": direction,
            "resolved_confidence": confidence,
            "expected_net_ev": float(money.get("virtual_ev", 0.0) or 0.0),
            "penalty_observe": False,
        },
        "timeline_alignment": {
            "raw_round_count": len(raw_history),
            "bp_round_count": len(big_road),
            "ties_ignored_for_direction_context": len(raw_history) - len(big_road),
        },
    }
    result["internal_signal_reason"] = result["signal_reason"]
    return result


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """先由 forecaster 預測，再開牌；LinUCB 診斷回饋與既有回傳介面保留。"""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")

    outcome_history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        user_id=str(session.get("user_id") or ""),
        run_seed=seed,
        shoe_context={
            "bankroll": float(session.get("bankroll", 0.0) or 0.0),
            "remaining_cards": len(hidden_shoe),
            "remaining_cards_reliability": 1.0,
            "remaining_cards_source": "virtual_shoe_exact_total",
        },
    )

    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(prediction.get("action") or "P").upper()
    actual = str(hand.outcome or "").upper()
    if actual == "T":
        verdict = "TIE_SKIPPED"
    else:
        verdict = "HIT" if predicted_side == actual else "MISS"

    bandit_update = record_online_feedback(
        scope_key=str(prediction.get("bandit_scope_key") or ""),
        action=predicted_side,
        context_vector=list(prediction.get("context_vector") or []),
        actual_outcome=actual,
    )

    prediction.update(
        {
            "ok": True,
            "mode": "virtual_shoe_road_forecaster",
            "virtual_hand": hand_data,
            "virtual_outcome": actual,
            "virtual_outcome_text": hand_data["outcome_text"],
            "verdict": verdict,
            "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict],
            "cards_consumed": int(hand.cards_used),
            "remaining_cards_after": len(remaining_shoe),
            "remaining_counts_after": counts_from_shoe(remaining_shoe),
            "round_number": int(session.get("hand_number", 0) or 0) + 1,
            "bandit_learning_applied": bool(bandit_update.get("updated", False)),
            "bandit_update": bandit_update,
            "disclaimer": (
                "正式方向由已揭曉大路歷史的因果式線上邏輯迴歸產生；"
                "單局結果具有高變異，模型機率不代表保證獲利。"
            ),
        }
    )
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
