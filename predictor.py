"""BGS production predictor: Road-Primary B/P core for 50-70 hand shoes.

Formal direction is produced only by the Big-Road sequence model in
``road_pattern_core`` through ``road_only_policy``. The core combines:

* 6/10/16/24-hand multi-window SAME/SWITCH behaviour;
* orientation-invariant historical pattern replay;
* relation n-grams, orders 2-5;
* pattern-survival / run-lifecycle evidence.

Shoe depth, remaining ratio and cut-card position only shrink confidence and
therefore Kelly sizing. Exact remaining composition is diagnostic only and can
never flip B/P. OCR, screenshot recognition and road detection are untouched.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import (
    POLICY_VERSION,
    normalize_big_road,
    recent_user_direction_feedback,
    record_online_feedback,
    road_only_policy,
)
from money_management import MAX_BET_RATIO, MoneyManagementModel
from road_model import ROAD_FEATURE_NAMES, build_road_context
from shoe_composition import analyze_shoe_composition
from shoe_constants import (
    AVERAGE_CARDS_PER_HAND,
    BURN_CARDS,
    CARDS_PER_DECK,
    REFERENCE_HANDS,
    SHOE_DECKS,
)
from shoe_depth_estimator import (
    DEFAULT_CUT_CARD_REMAINING,
    TARGET_HANDS_MAX,
    TARGET_HANDS_MIN,
    ShoeDepthEstimator,
)

OUTCOMES = ("B", "P", "T")
_MONEY = MoneyManagementModel()
MODEL_VERSION = POLICY_VERSION


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, number))


def _history_values(history: Union[str, Iterable[Any], None]) -> List[Any]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in OUTCOMES for char in compact):
            return list(compact)
        return [part for part in history.replace("|", ",").split(",") if part.strip()]
    return list(history)


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


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    banker = _clip(road.get("banker_probability", 0.5))
    player = _clip(road.get("player_probability", 1.0 - banker))
    total = banker + player
    banker, player = (0.5, 0.5) if total <= 1e-12 else (banker / total, player / total)
    confidence = _clip(road.get("confidence_score", 0.0))
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
    counts = context.get("remaining_counts")
    observed = context.get("observed_cards")
    if isinstance(counts, (list, tuple)) and len(counts) == 10:
        return "remaining_counts"
    if isinstance(observed, (list, tuple)) and observed:
        return "observed_cards"
    source = str(analysis.get("source") or "").lower().strip()
    return source if source in {"remaining_counts", "observed_cards"} else "none"


def _calibrate_road_confidence(
    *,
    direction: str,
    raw_probabilities: Mapping[str, Any],
    shoe_confidence_factor: float,
) -> tuple[dict[str, float], float, float]:
    p_b = _clip(raw_probabilities.get("B", 0.5))
    p_p = _clip(raw_probabilities.get("P", 0.5))
    total = p_b + p_p
    p_b, p_p = (0.5, 0.5) if total <= 1e-12 else (p_b / total, p_p / total)
    side = "B" if str(direction).upper() == "B" else "P"
    raw_confidence = p_b if side == "B" else p_p
    factor = _clip(shoe_confidence_factor, 0.85, 1.0)
    calibrated_confidence = _clip(0.5 + (raw_confidence - 0.5) * factor, 0.5, 0.75)
    probabilities = {
        "B": calibrated_confidence if side == "B" else 1.0 - calibrated_confidence,
        "P": calibrated_confidence if side == "P" else 1.0 - calibrated_confidence,
        "T": 0.0,
    }
    return probabilities, float(raw_confidence), float(calibrated_confidence)


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
    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))
    cut_override = context.get("cut_card_remaining_cards")

    depth = ShoeDepthEstimator(
        shoe_decks=int(context.get("decks", SHOE_DECKS) or SHOE_DECKS),
        average_cards_per_hand=AVERAGE_CARDS_PER_HAND,
        reference_hands=REFERENCE_HANDS,
        burn_cards=BURN_CARDS,
        cut_card_remaining_cards=cut_override,
    ).estimate(raw_history).as_dict()

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

    shoe = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe.get("available"))
    composition_source = _shoe_source(context, shoe)

    policy = road_only_policy(
        raw_history,
        shoe_context=context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    road_pattern = dict(policy.get("road_pattern") or {})
    direction = str(policy.get("direction") or "B").upper().strip()
    if direction not in {"B", "P"}:
        direction = "B"

    raw_probabilities = dict(policy.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    shoe_factor = float(depth.get("shoe_confidence_factor", 1.0) or 1.0)
    probabilities, raw_confidence, confidence = _calibrate_road_confidence(
        direction=direction,
        raw_probabilities=raw_probabilities,
        shoe_confidence_factor=shoe_factor,
    )
    p_b = float(probabilities["B"])
    p_p = float(probabilities["P"])
    text = "莊" if direction == "B" else "閒"
    formal_source = "road_pattern_core"

    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )
    final_ratio = float(money.get("final_bet_ratio", 0.05) or 0.05)
    bet_percentage = float(money.get("bet_percentage", final_ratio * 100.0) or final_ratio * 100.0)

    remaining_cards = float(depth.get("remaining_cards", 0.0) or 0.0)
    remaining_source = str(depth.get("remaining_cards_source") or "round_count_estimate")
    remaining_ratio = float(depth.get("remaining_ratio", 1.0) or 0.0)
    penetration = float(depth.get("penetration", 0.0) or 0.0)
    shoe_stage = str(depth.get("shoe_stage", "UNKNOWN") or "UNKNOWN")
    cut_progress = float(depth.get("cut_progress", 0.0) or 0.0)
    cut_remaining = float(depth.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING) or DEFAULT_CUT_CARD_REMAINING)
    exact_counts = list(shoe.get("remaining_counts") or []) if shoe_available else []

    markov_predict = dict(policy.get("fallback_markov") or {})
    road_diag = _road_diagnostic(road)
    road_forecaster_diag = dict(policy.get("road_forecaster_diagnostic") or {})
    road_forecaster = dict(policy.get("road_forecaster") or {})
    feedback = recent_user_direction_feedback(user_id)
    context_vector = list(policy.get("context_vector") or [])
    feature_names = list(policy.get("context_feature_names") or [])
    bandit_scores = dict(policy.get("scores") or {})
    bandit_update = dict(policy.get("feedback_update") or {})

    shoe_returns = dict(shoe.get("expected_returns") or {})
    banker_ev = shoe_returns.get("B") if shoe_available else None
    player_ev = shoe_returns.get("P") if shoe_available else None
    selected_ev = float(money.get("virtual_ev", 0.0) or 0.0)
    edge = float(money.get("edge", 0.0) or 0.0)

    fingerprint = sha256(
        "|".join(
            (
                "".join(raw_history),
                str(venue).upper().strip(),
                str(room).strip(),
                str(shoe_id).strip(),
                POLICY_VERSION,
                f"cut={cut_remaining:.2f}",
            )
        ).encode("utf-8")
    ).hexdigest()[:24]

    survival_component = dict((road_pattern.get("components") or {}).get("pattern_survival") or {})
    component_probabilities = {
        "road_pattern_primary": dict(raw_probabilities),
        "road_pattern_after_shoe_confidence": dict(probabilities),
        "multi_window": dict((road_pattern.get("components") or {}).get("multi_window") or {}),
        "pattern_replay": dict((road_pattern.get("components") or {}).get("pattern_replay") or {}),
        "ngram": dict((road_pattern.get("components") or {}).get("ngram") or {}),
        "pattern_survival": survival_component,
        "time_decay_markov_diagnostic": dict(markov_predict.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}),
        "road_diagnostic": {"B": road_diag["banker_probability"], "P": road_diag["player_probability"], "T": 0.0},
    }
    if shoe_available:
        component_probabilities["exact_shoe_diagnostic_only"] = dict(shoe.get("probabilities") or {})

    remaining_state = {
        "available": True,
        "conditioned_rounds": len(raw_history),
        "remaining_cards": remaining_cards,
        "mean_remaining_cards": remaining_cards,
        "remaining_ratio": remaining_ratio,
        "penetration": penetration,
        "shoe_stage": shoe_stage,
        "shoe_stage_factor": shoe_factor,
        "cut_card_remaining_cards": cut_remaining,
        "cut_progress": cut_progress,
        "cut_proximity": cut_progress,
        "cards_until_cut": float(depth.get("cards_until_cut", 0.0) or 0.0),
        "estimated_hands_until_cut": float(depth.get("estimated_hands_until_cut", 0.0) or 0.0),
        "reliability": float(depth.get("remaining_cards_reliability", 0.0) or 0.0),
        "exact_composition": shoe_available,
        "source": remaining_source,
        "direction_authority": False,
        "semantics": "shoe_depth_and_cut_confidence_only_never_BP_direction",
    }

    shoe_estimate = {
        "direction": None,
        "probabilities": dict(shoe.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}),
        "expected_remaining_cards": remaining_cards,
        "expected_remaining_decks": remaining_cards / float(CARDS_PER_DECK),
        "expected_remaining_counts": exact_counts,
        "remaining_card_state": remaining_state,
        "conditioned_rounds": len(raw_history),
        "reliability": float(depth.get("remaining_cards_reliability", 0.0) or 0.0),
        "fusion_weight": 0.0,
        "depth_constraint_applied": True,
        "source": composition_source,
        "expected_returns": shoe_returns,
        "direction_authority": False,
        "inference_semantics": "shoe_is_diagnostic_and_confidence_only_road_owns_BP",
    }

    pattern_name = str(road_pattern.get("pattern") or survival_component.get("pattern") or "GENERIC")
    survival_score = float(road_pattern.get("pattern_survival_score", 0.5) or 0.5)
    signal_code = "ROAD_PATTERN_PRIMARY_KELLY_5_30"
    signal_reason = (
        f"Primary=RoadPattern；pattern={pattern_name}；raw={raw_confidence:.3f}；"
        f"shoeFactor={shoe_factor:.3f}；final={confidence:.3f}；"
        f"P(B)={p_b:.3f}；P(P)={p_p:.3f}；Kelly={bet_percentage:.2f}%。"
    )

    context_meta = dict(policy.get("context_metadata") or {})
    context_meta.update(
        {
            "formal_direction_source": formal_source,
            "primary_model": "ROAD_PATTERN_CORE",
            "road_direction_weight": 1.0,
            "road_context_direction_weight": 1.0,
            "shoe_context_used_for_formal_direction": False,
            "exact_shoe_composition_used_for_direction": False,
            "card_composition_direction_weight": 0.0,
            "lstm_direction_weight": 0.0,
            "fallback_markov_direction_weight": 0.0,
            "remaining_cards": remaining_cards,
            "remaining_ratio": remaining_ratio,
            "penetration": penetration,
            "shoe_stage": shoe_stage,
            "cut_progress": cut_progress,
            "target_hands_min": int(TARGET_HANDS_MIN),
            "target_hands_max": int(TARGET_HANDS_MAX),
        }
    )

    result: Dict[str, Any] = {
        "ok": True,
        "engine": "ROAD_PATTERN_PRIMARY_BP",
        "model_version": POLICY_VERSION,
        "system_model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "SHOE_DEPTH_CONFIDENCE_ONLY_V1",
        "model_variant": "ROAD_MULTIWINDOW_PATTERN_REPLAY_NGRAM_SURVIVAL_50_70_KELLY_5_30",
        "model_core": "road_pattern_core",
        "primary_model": "ROAD_PATTERN_CORE",
        "decision_pipeline": "big_road_BP_to_multiwindow_pattern_replay_ngram_survival_to_shoe_depth_confidence_to_kelly",
        "prediction_fingerprint": fingerprint,
        "probabilities": probabilities,
        "raw_direction_probabilities": dict(raw_probabilities),
        "banker_rate": round(p_b * 100.0, 2),
        "player_rate": round(p_p * 100.0, 2),
        "tie_rate": 0.0,
        "recommend": direction,
        "recommend_text": text,
        "action": direction,
        "action_text": text,
        "internal_recommend": direction,
        "internal_action": direction,
        "next_round_direction": direction,
        "next_round_direction_text": text,
        "direction": direction,
        "direction_text": text,
        "adaptive_only_direction": direction,
        "signal_allowed": True,
        "risk_gate_open": True,
        "mandatory_bet": True,
        "signal_status_code": signal_code,
        "signal_status_text": f"下一手模型：{text} {confidence:.1%}；Kelly {bet_percentage:.2f}%",
        "signal_reason": signal_reason,
        "internal_signal_reason": signal_reason,
        "direction_source": formal_source,
        "formal_direction_source": formal_source,
        "road_direction_weight": 1.0,
        "road_context_direction_weight": 1.0,
        "shoe_context_used_for_formal_direction": False,
        "exact_shoe_composition_used_for_direction": False,
        "card_composition_direction_weight": 0.0,
        "lstm_direction_weight": 0.0,
        "structure_direction_weight": 0.0,
        "fallback_markov_direction_weight": 0.0,
        "card_composition_source": composition_source,
        "remaining_counts_source": composition_source,
        "remaining_cards_source": remaining_source,
        "remaining_ratio": remaining_ratio,
        "penetration": penetration,
        "cut_card_remaining_cards": cut_remaining,
        "cut_progress": cut_progress,
        "target_hands_min": int(TARGET_HANDS_MIN),
        "target_hands_max": int(TARGET_HANDS_MAX),
        "average_cards_per_hand": float(AVERAGE_CARDS_PER_HAND),
        "shoe_decks": int(context.get("decks", SHOE_DECKS) or SHOE_DECKS),
        "burn_cards": int(BURN_CARDS),
        "reference_hands": int(REFERENCE_HANDS),
        "banker_expected_return": banker_ev,
        "player_expected_return": player_ev,
        "banker_ev": banker_ev,
        "player_ev": player_ev,
        "shoe_composition": {**shoe, "formal_direction_authority": False, "direction_weight_inside_fusion": 0.0},
        "confidence": confidence,
        "raw_lstm_confidence": 0.5,
        "raw_model_confidence": raw_confidence,
        "raw_markov_confidence": float(markov_predict.get("selected_win_probability", 0.5) or 0.5),
        "pattern_calibrated_confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": "較高" if confidence >= 0.56 else "中等" if confidence >= 0.52 else "保守",
        "confidence_calibration": {
            "applied": bool(shoe_factor < 0.999999),
            "raw_confidence": raw_confidence,
            "shoe_stage_factor": shoe_factor,
            "final_confidence": confidence,
            "direction_override": False,
            "semantics": "shrink_road_margin_only_never_flip_direction",
        },
        "transition_calibration": {"applied": False, "disabled": True, "formal_direction_weight": 0.0},
        "entropy_bits": 0.0,
        "entropy_base_weight": confidence,
        "shoe_progress": float(depth.get("shoe_progress", 0.0) or 0.0),
        "shoe_depth_estimate": {**depth, "rounds": len(raw_history), "direction_authority": False},
        "remaining_card_state": remaining_state,
        "estimated_remaining_cards": remaining_cards,
        "estimated_remaining_interval": {"low": remaining_cards, "high": remaining_cards},
        "shoe_stage": shoe_stage,
        "pattern_survival": {
            **survival_component,
            "score": survival_score,
            "mode": "formal_road_component",
            "direction_override": False,
        },
        "pattern_survival_score": survival_score,
        "run_length_hazard": {},
        "run_length_hazard_weight": 0.0,
        "probabilistic_shoe_estimate": shoe_estimate,
        "tie_risk_active": False,
        "direction_edge": edge,
        "direction_edge_percent": round(edge * 100.0, 4),
        "selected_expected_return": selected_ev,
        "selected_expected_return_percent": selected_ev * 100.0,
        "bet_allowed": True,
        "markov": {**markov_predict, "diagnostic_only": True, "formal_direction_weight": 0.0},
        "markov_probs": dict(markov_predict.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}),
        "final_probs": probabilities,
        "direction_probs": probabilities,
        "economic_probs": probabilities,
        "final_probability": probabilities[direction],
        "economic_probability_for_direction": float(money.get("resolved_win_probability", confidence) or confidence),
        "markov_predict": markov_predict,
        "probabilistic_shoe_predict": shoe_estimate,
        "fusion_decision": {
            "direction": direction,
            "probabilities": probabilities,
            "road_pattern_weight": 1.0,
            "shoe_composition_weight": 0.0,
            "lstm_weight": 0.0,
            "fallback_markov_weight": 0.0,
            "road_applied": True,
            "hazard_applied": False,
            "method": "road_pattern_core",
            "semantics": "road_only_direction_shoe_cut_confidence_only",
            "details": dict(road_pattern.get("component_weights") or {}),
        },
        "fusion": {
            "method": "road_pattern_primary",
            "road_reliability": float(road_pattern.get("effective_weight_sum", 0.0) or 0.0),
            "hazard_reliability": 0.0,
            "lstm_reliability": 0.0,
            "cut_progress": cut_progress,
        },
        "markov_state": {
            "state_key": "MARKOV_DIAGNOSTIC_ONLY",
            "direction_context": "diagnostic_only",
            "density": "unused_for_formal_direction",
            "tie_trigger": "TiesSkipped",
            "sample_count": len(big_road),
            "effective_support": float(markov_predict.get("context_support", 0.0) or 0.0),
            "state_count": 2,
            "selected_order": int(markov_predict.get("selected_order", 1) or 1),
            "change_point": False,
            "shoe_stage": shoe_stage,
            "pattern_survival_score": survival_score,
        },
        "road_predict": {
            "direction": direction,
            "banker_probability": float(raw_probabilities.get("B", 0.5) or 0.5),
            "player_probability": float(raw_probabilities.get("P", 0.5) or 0.5),
            "confidence": raw_confidence,
            "decision_weight": 1.0,
            "diagnostic_only": False,
            "model_id": road_pattern.get("model_id"),
        },
        "road_support": road,
        "road_pattern_model": road_pattern,
        "derived_road_analysis": dict(policy.get("regression_analysis") or {}),
        "road_fusion": {
            "applied": True,
            "mode": "formal_road_pattern_primary",
            "reliability": float(road_pattern.get("effective_weight_sum", 0.0) or 0.0),
            "raw_reliability": float(road_pattern.get("effective_weight_sum", 0.0) or 0.0),
            "pattern_survival_score": survival_score,
            "likelihood": dict(raw_probabilities),
            "reason": "formal direction is owned by road_pattern_core",
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
            "confidence_factor": 1.0,
            "direction_override": False,
            "reason": "disabled_formal_core_is_road_pattern",
        },
        "component_probabilities": component_probabilities,
        "money_management": money,
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
        "context_feature_names": feature_names,
        "context_dim": len(context_vector),
        "bandit_scores": bandit_scores,
        "bandit_selected_arm": str(road_forecaster_diag.get("direction") or ""),
        "bandit_scope_key": str(policy.get("scope_key") or ""),
        "bandit_feedback_update": bandit_update,
        "contextual_bandit_enabled": True,
        "contextual_bandit_update_enabled": True,
        "linucb_enabled": True,
        "linucb_diagnostic_only": True,
        "linucb_direction_weight": 0.0,
        "road_forecaster": road_forecaster,
        "lstm_enabled": False,
        "lstm_primary": False,
        "lstm_shoe_cut_fusion": False,
        "lstm_model": {"available": False, "formal_direction_weight": 0.0, "reason": "disabled_formal_core_is_road_pattern"},
        "lstm_fallback_active": False,
        "lstm_fallback_reason": "disabled",
        "cusum_linucb_enabled": False,
        "force_observe": False,
        "hard_brake_active": False,
        "post_reset_vacuum_active": False,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": "resolved_BP_probability_from_road_pattern_core_then_shoe_depth_margin_shrink",
        "dynamic_prediction_policy": {
            "version": POLICY_VERSION,
            "road_primary": True,
            "lstm_primary": False,
            "fallback_active": False,
            "big_road_rounds": len(big_road),
            "forecast": policy,
            "penalty_observe": {"active": False, "force_observe": False},
            "exact_card_dependency": False,
            "exact_card_used_when_available": False,
            "shoe_probability_decision_weight": 0.0,
            "ocr_or_screen_flow_modified": False,
            "formal_direction_source": formal_source,
            "card_composition_source": composition_source,
            "shoe_direction_authority": False,
            "cut_progress": cut_progress,
        },
        "dynamic_policy_version": POLICY_VERSION,
        "online_performance_feedback": feedback,
        "decision": direction,
        "decision_text": text,
        "skip": False,
        "skip_reason": "",
        "decision_gate": {
            "decision": direction,
            "allowed": True,
            "reason": "road_pattern_core_always_returns_BP",
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
        "context_metadata": context_meta,
    }
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    """Predict from road history, then reveal one virtual hand."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(
        history=history,
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
            "remaining_cards_source": "virtual_shoe_exact_counts_diagnostic_only",
            "source": "remaining_counts",
            "cut_card_remaining_cards": float(session.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING) or DEFAULT_CUT_CARD_REMAINING),
        },
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted = str(prediction.get("action") or "B").upper()
    actual = str(hand.outcome or "").upper()
    verdict = "TIE_SKIPPED" if actual == "T" else ("HIT" if predicted == actual else "MISS")
    update = record_online_feedback(
        scope_key=str(prediction.get("bandit_scope_key") or ""),
        action=predicted,
        context_vector=list(prediction.get("context_vector") or []),
        actual_outcome=actual,
    )
    prediction.update(
        {
            "ok": True,
            "mode": "virtual_shoe_road_pattern_primary",
            "virtual_hand": hand_data,
            "virtual_outcome": actual,
            "virtual_outcome_text": hand_data["outcome_text"],
            "verdict": verdict,
            "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict],
            "cards_consumed": int(hand.cards_used),
            "remaining_cards_after": len(remaining_shoe),
            "remaining_counts_after": counts_from_shoe(remaining_shoe),
            "round_number": int(session.get("hand_number", 0) or 0) + 1,
            "bandit_learning_applied": bool(update.get("updated", False)),
            "bandit_update": update,
            "disclaimer": (
                "正式方向僅由大路 Road-Pattern 模型產生；牌靴與切牌只調整信心與注碼。"
                "LSTM/LinUCB/Markov/hazard/HSMM 不參與正式方向。單局結果具有高變異，"
                "模型機率不代表保證獲利。"
            ),
        }
    )
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
