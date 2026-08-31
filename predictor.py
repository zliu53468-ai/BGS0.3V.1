"""BGS 正式預測器：精確牌靴 EV 優先，缺資料時回退因果式 road forecaster。

公開 predict() 參數與主要回傳欄位維持相容。OCR / screenshot / road adapter
不做修改。正式下一局方向永遠只會是 B/P：
1) remaining_counts；2) observed_cards；3) road forecaster fallback。
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
from shoe_composition import analyze_shoe_composition

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


def _shoe_source(context: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    if not bool(analysis.get("available")):
        return "none"
    raw_counts = context.get("remaining_counts")
    if isinstance(raw_counts, (list, tuple)) and len(raw_counts) == 10:
        return "remaining_counts"
    observed = context.get("observed_cards")
    if isinstance(observed, (list, tuple)) and len(observed) > 0:
        return "observed_cards"
    source = str(analysis.get("source") or "").strip().lower()
    if source in {"remaining_counts", "exact_remaining_counts"}:
        return "remaining_counts"
    if source in {"observed_cards", "observed_card_values"}:
        return "observed_cards"
    return "none"


def _resolved_confidence(probabilities: Mapping[str, Any], direction: str) -> float:
    p_b = max(0.0, float(probabilities.get("B", 0.0) or 0.0))
    p_p = max(0.0, float(probabilities.get("P", 0.0) or 0.0))
    resolved = p_b + p_p
    if resolved <= 1e-12:
        return 0.5
    return float((p_b if direction == "B" else p_p) / resolved)


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
            initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0),
            manual_count=int(supplied_road.get("manual_count", 0) or 0),
        )
        if supplied_road:
            road["scan_metadata"] = supplied_road

    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))

    # Road path is always evaluated so diagnostics/online feedback/public interfaces
    # remain compatible. Exact shoe composition, when available, owns formal direction.
    policy = linucb_policy(
        raw_history,
        shoe_context=context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    road_probabilities = dict(policy["probabilities"])
    road_direction = str(policy["direction"])
    road_confidence = float(policy["selected_win_probability"])

    shoe_analysis = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe_analysis.get("available"))
    composition_source = _shoe_source(context, shoe_analysis)

    if shoe_available:
        probabilities = dict(shoe_analysis.get("probabilities") or {})
        returns = dict(shoe_analysis.get("expected_returns") or {})
        b_ev = float(returns.get("B", 0.0) or 0.0)
        p_ev = float(returns.get("P", 0.0) or 0.0)
        # No EV gate and no third arm: even if both sides are negative, choose the
        # mathematically better B/P side. Low/negative edge is handled by Kelly floor.
        direction = "B" if b_ev >= p_ev else "P"
        confidence = _resolved_confidence(probabilities, direction)
        formal_source = "shoe_composition_ev_argmax"
        card_weight = 1.0
        road_weight = 0.0
        selected_physical_ev = b_ev if direction == "B" else p_ev
        shoe_analysis["action"] = direction
        shoe_analysis["action_text"] = "莊" if direction == "B" else "閒"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True
    else:
        probabilities = dict(road_probabilities)
        direction = road_direction if road_direction in {"B", "P"} else "B"
        confidence = road_confidence
        formal_source = "road_forecaster_probability_argmax"
        card_weight = 0.0
        road_weight = 1.0
        selected_physical_ev = None
        shoe_analysis["action"] = None
        shoe_analysis["action_text"] = "牌靴資料不可用，正式方向回退牌路"
        shoe_analysis["formal_direction"] = direction
        shoe_analysis["formal_no_observe_arm"] = True

    action = direction
    text = "莊" if action == "B" else "閒"
    direction_text = text

    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )
    bet_allowed = True
    signal_status_code = (
        "SHOE_EV_TWO_ARM_KELLY_5_30"
        if shoe_available
        else "LINUCB_TWO_ARM_KELLY_5_30"
    )
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
                composition_source,
            )
        ).encode("utf-8")
    ).hexdigest()[:24]

    road_predict = _road_diagnostic(road)
    p_b = float(probabilities.get("B", 0.5) or 0.0)
    p_p = float(probabilities.get("P", 0.5) or 0.0)
    p_t = float(probabilities.get("T", 0.0) or 0.0)
    feedback = recent_user_direction_feedback(user_id)
    context_vector = list(policy.get("context_vector") or [])
    context_feature_names = list(policy.get("context_feature_names") or [])
    context_meta = dict(policy.get("context_metadata") or {})
    bandit_scores = dict(policy.get("scores") or {})
    bandit_feedback_update = dict(policy.get("feedback_update") or {})

    exact_counts = list(shoe_analysis.get("remaining_counts") or []) if shoe_available else []
    remaining_cards_value = (
        float(sum(exact_counts))
        if exact_counts
        else float(context_meta.get("remaining_cards", 0.0) or 0.0)
    )
    estimated_counts = exact_counts or list(context_meta.get("estimated_remaining_counts_0_to_9") or [])
    if not remaining_cards_value:
        try:
            remaining_cards_value = max(0.0, float(context.get("remaining_cards", 0.0) or 0.0))
        except (TypeError, ValueError):
            remaining_cards_value = 0.0

    context_meta.update(
        {
            "formal_direction_source": formal_source,
            "shoe_context_used_for_formal_direction": shoe_available,
            "card_composition_direction_weight": card_weight,
            "road_context_direction_weight": road_weight,
            "card_composition_source": composition_source,
            "remaining_counts_source": composition_source,
            "remaining_cards": remaining_cards_value,
            "remaining_cards_source": composition_source if shoe_available else str(
                context_meta.get("remaining_cards_source") or "estimated"
            ),
            "estimated_remaining_counts_0_to_9": estimated_counts,
        }
    )
    policy["context_metadata"] = context_meta
    policy["road_direction_before_shoe_override"] = road_direction
    policy["road_probabilities_before_shoe_override"] = dict(road_probabilities)
    policy["direction"] = direction
    policy["selected_arm"] = direction
    policy["action"] = direction
    policy["action_text"] = text
    policy["latent_direction"] = direction
    policy["probabilities"] = dict(probabilities)
    policy["selected_win_probability"] = confidence
    policy["confidence"] = confidence
    policy["confidence_prob"] = confidence
    policy["selection_reason"] = formal_source
    policy["formal_context_source"] = (
        composition_source if shoe_available else "screenshot_big_road_plus_manual_history"
    )
    policy["road_context_direction_weight"] = road_weight
    policy["card_composition_direction_weight"] = card_weight
    policy["shoe_context_used_for_formal_direction"] = shoe_available

    markov_predict = {
        "direction": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "pattern_calibrated_probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "state": {
            "direction_context": "SHOE_EV" if shoe_available else "LINUCB",
            "density": "CardCompositionFirst",
            "tie_trigger": "RewardZero",
        },
        "state_key": "SHOE_EV" if shoe_available else "LINUCB",
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
        "road_forecaster": {
            "B": float(road_probabilities.get("B", 0.5) or 0.5),
            "P": float(road_probabilities.get("P", 0.5) or 0.5),
            "T": float(road_probabilities.get("T", 0.0) or 0.0),
        },
        "contextual_linucb": {
            "B": float(road_probabilities.get("B", 0.5) or 0.5),
            "P": float(road_probabilities.get("P", 0.5) or 0.5),
            "T": float(road_probabilities.get("T", 0.0) or 0.0),
        },
        "road_diagnostic": {
            "B": float(road_predict["banker_probability"]),
            "P": float(road_predict["player_probability"]),
            "T": 0.0,
        },
    }
    if shoe_available:
        component_probabilities["card_composition"] = {"B": p_b, "P": p_p, "T": p_t}

    supplied_remaining = bool(
        context.get("remaining_counts") is not None
        or context.get("observed_cards") is not None
        or context.get("remaining_cards")
    )
    depth_constraint = {
        "applied": shoe_available or supplied_remaining,
        "reason": (
            f"{composition_source}_used_for_exact_shoe_ev"
            if shoe_available
            else "remaining_depth_estimated_or_diagnostic_only"
        ),
        "target_remaining_cards": remaining_cards_value,
    }
    shoe_estimate = {
        "direction": direction,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "bp_conditional_probabilities": {
            "B": p_b / max(1e-12, p_b + p_p),
            "P": p_p / max(1e-12, p_b + p_p),
        },
        "expected_remaining_cards": remaining_cards_value,
        "expected_remaining_decks": remaining_cards_value / 52.0,
        "expected_remaining_counts": estimated_counts,
        "remaining_count_std": [],
        "remaining_card_state": {
            "available": shoe_available,
            "conditioned_rounds": len(big_road),
            "source": composition_source if shoe_available else "none",
        },
        "conditioned_rounds": len(big_road),
        "particle_count": 0,
        "reliability": 1.0 if shoe_available else 0.0,
        "fusion_weight": card_weight,
        "target_remaining_cards": remaining_cards_value,
        "depth_constraint_applied": shoe_available,
        "depth_constraint": depth_constraint,
        "shoe_tendency": dict(shoe_analysis.get("composition") or {}),
        "inference_semantics": (
            "exact_nonreplacement_next_hand_probability"
            if shoe_available
            else "unavailable_road_fallback"
        ),
        "source": composition_source,
        "expected_returns": dict(shoe_analysis.get("expected_returns") or {}),
    }

    edge = float(money.get("edge", 0.0) or 0.0)
    final_ratio = float(money.get("final_bet_ratio", 0.05) or 0.05)
    bet_percentage = float(money.get("bet_percentage", final_ratio * 100.0) or final_ratio * 100.0)
    shoe_returns = dict(shoe_analysis.get("expected_returns") or {})
    banker_ev = shoe_returns.get("B") if shoe_available else None
    player_ev = shoe_returns.get("P") if shoe_available else None
    selected_ev = (
        float(selected_physical_ev)
        if selected_physical_ev is not None
        else float(money.get("virtual_ev", 0.0) or 0.0)
    )

    if shoe_available:
        signal_reason = (
            f"ShoeSource={composition_source}；EV(B)={float(banker_ev):.6f}；"
            f"EV(P)={float(player_ev):.6f}；formal={direction}；"
            f"Kelly={bet_percentage:.2f}%。"
        )
    else:
        signal_reason = (
            f"Forecaster={str(policy.get('road_forecaster', {}).get('model_id', ''))}；"
            f"support={float(policy.get('effective_support', 0.0)):.0f}；"
            f"P(B)={p_b:.3f}；P(P)={p_p:.3f}；"
            f"Kelly={bet_percentage:.2f}%；feedback={bool(bandit_feedback_update.get('updated', False))}。"
        )

    result = {
        "ok": True,
        "engine": "EXACT_SHOE_EV_BP" if shoe_available else "CAUSAL_ROAD_FORECASTER_BP",
        "model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "EXACT_NON_REPLACEMENT_V1" if shoe_available else "CONTEXT_GENERATOR_V1",
        "system_model_version": POLICY_VERSION,
        "model_variant": "EXACT_SHOE_EV_ARGMAX_KELLY_5_30" if shoe_available else "ONLINE_L2_LOGISTIC_ROAD_FORECASTER_KELLY_5_30",
        "model_core": "shoe_composition" if shoe_available else "road_forecaster",
        "decision_pipeline": (
            "remaining_counts_or_observed_cards_to_exact_nonreplacement_EV_argmax_to_existing_kelly"
            if shoe_available
            else "observed_BP_prefix_to_causal_logistic_argmax_to_existing_kelly"
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
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": True,
        "signal_status_code": signal_status_code,
        "signal_status_text": signal_status_text,
        "signal_reason": signal_reason,
        "internal_signal_reason": "",
        "direction_source": formal_source,
        "formal_direction_source": formal_source,
        "shoe_context_used_for_formal_direction": shoe_available,
        "card_composition_direction_weight": card_weight,
        "road_context_direction_weight": road_weight,
        "card_composition_source": composition_source,
        "remaining_counts_source": composition_source,
        "banker_expected_return": banker_ev,
        "player_expected_return": player_ev,
        "banker_ev": banker_ev,
        "player_ev": player_ev,
        "shoe_composition": dict(shoe_analysis),
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
            "source": composition_source if shoe_available else str(context_meta.get("remaining_cards_source") or "estimated"),
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
        "selected_expected_return": selected_ev,
        "selected_expected_return_percent": selected_ev * 100.0,
        "bet_allowed": bet_allowed,
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
            "probabilistic_shoe_weight": card_weight,
            "road_forecaster_weight": road_weight,
            "derived_road_likelihood_power": 0.0,
            "run_length_hazard_likelihood_power": 0.0,
            "road_applied": not shoe_available,
            "hazard_applied": False,
            "method": formal_source,
            "semantics": "exact_shoe_EV_has_formal_priority" if shoe_available else "road_fallback_when_exact_shoe_unavailable",
        },
        "fusion": {
            "method": formal_source,
            "shoe_reliability": 1.0 if shoe_available else 0.0,
            "road_reliability": road_weight,
            "hazard_reliability": 0.0,
        },
        "markov_state": {
            "state_key": "SHOE_EV" if shoe_available else "LINUCB",
            "direction_context": "SHOE_EV" if shoe_available else "LINUCB",
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
            "applied": not shoe_available,
            "mode": "formal_fallback" if not shoe_available else "diagnostic_only",
            "reliability": road_weight,
            "raw_reliability": road_weight,
            "pattern_survival_score": 1.0,
            "likelihood": None,
            "reason": "精確牌組可用，牌路不覆蓋牌靴 EV。" if shoe_available else "缺少精確牌組，正式方向回退既有 road forecaster。",
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
        "probability_semantics": "exact_nonreplacement_BPT_probability" if shoe_available else "causal_online_logistic_next_resolved_BP_probability",
        "dynamic_prediction_policy": {
            "version": POLICY_VERSION,
            "road_only": not shoe_available,
            "big_road_rounds": len(big_road),
            "forecast": dict(policy),
            "penalty_observe": {"active": False, "force_observe": False},
            "exact_card_dependency": shoe_available,
            "shoe_probability_decision_weight": card_weight,
            "ocr_or_screen_flow_modified": False,
            "formal_direction_source": formal_source,
            "card_composition_source": composition_source,
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
            "reason": "exact_shoe_EV_argmax_always_returns_BP" if shoe_available else "road_forecaster_argmax_always_returns_BP",
            "direction": direction,
            "resolved_confidence": confidence,
            "expected_net_ev": selected_ev,
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
    """先以虛擬牌靴精確剩餘組成預測，再開牌；公開回傳介面維持相容。"""
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
            "remaining_counts": counts_from_shoe(hidden_shoe),
            "remaining_cards_reliability": 1.0,
            "remaining_cards_source": "virtual_shoe_exact_counts",
            "source": "remaining_counts",
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
            "mode": "virtual_shoe_exact_composition",
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
                "有精確牌組時正式方向由不放回牌靴 EV 決定；缺少牌組時回退 road forecaster。"
                "單局結果具有高變異，模型機率不代表保證獲利。"
            ),
        }
    )
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
