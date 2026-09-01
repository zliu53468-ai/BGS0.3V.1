"""BGS production predictor: Single-Brain Contextual LinUCB B/P core.

Formal direction, probability and confidence are produced only by the two-arm
Contextual LinUCB policy. Shoe composition, HSMM regime, run-length hazard and
derived-road structure enter only through the fixed 16-D context vector.
Legacy road/Markov/geometry diagnostics remain outward-compatible at zero
formal direction weight. OCR and screenshot recognition are untouched.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import (
    POLICY_VERSION,
    normalize_big_road,
    recent_user_direction_feedback,
    road_only_policy,
)
from money_management import MAX_BET_RATIO, MIN_BET_RATIO, MoneyManagementModel
from road_model import ROAD_FEATURE_NAMES, build_road_context
from shoe_composition import analyze_shoe_composition
from shoe_constants import AVERAGE_CARDS_PER_HAND, BURN_CARDS, CARDS_PER_DECK, REFERENCE_HANDS, SHOE_DECKS
from shoe_depth_estimator import DEFAULT_CUT_CARD_REMAINING, TARGET_HANDS_MAX, TARGET_HANDS_MIN, ShoeDepthEstimator

OUTCOMES = ("B", "P", "T")
_MONEY = MoneyManagementModel()
MODEL_VERSION = POLICY_VERSION


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try: number = float(value)
    except (TypeError, ValueError): return lo
    return max(lo, min(hi, number))


def _history_values(history: Union[str, Iterable[Any], None]) -> List[Any]:
    if history is None: return []
    if isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in OUTCOMES for char in compact): return list(compact)
        return [part for part in history.replace("|", ",").split(",") if part.strip()]
    return deepcopy(list(history))


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        raw = (item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")) if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in OUTCOMES: result.append(value)
    return result[-2000:]


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    banker = _clip(road.get("banker_probability", 0.5)); player = _clip(road.get("player_probability", 1.0 - banker)); total = banker + player
    banker, player = (0.5, 0.5) if total <= 1e-12 else (banker / total, player / total)
    confidence = _clip(road.get("confidence_score", 0.0))
    return {"direction": "B" if banker >= player else "P", "banker_probability": float(banker), "player_probability": float(player), "confidence": float(confidence), "decision_weight": 0.0, "diagnostic_only": True}


def _shoe_source(context: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    if not bool(analysis.get("available")): return "none"
    counts = context.get("remaining_counts"); observed = context.get("observed_cards")
    if isinstance(counts, (list, tuple)) and len(counts) == 10: return "remaining_counts"
    if isinstance(observed, (list, tuple)) and observed: return "observed_cards"
    source = str(analysis.get("source") or "").lower().strip()
    return source if source in {"remaining_counts", "observed_cards"} else "none"


def _normalize_policy_probabilities(policy: Mapping[str, Any]) -> tuple[dict[str, float], str, float]:
    raw = dict(policy.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0})
    p_b = _clip(raw.get("B", 0.5)); p_p = _clip(raw.get("P", 0.5)); total = p_b + p_p
    p_b, p_p = (0.5, 0.5) if total <= 1e-12 else (p_b / total, p_p / total)
    probabilities = {"B": float(p_b), "P": float(p_p), "T": 0.0}
    direction = str(policy.get("direction") or "").upper().strip()
    if direction not in {"B", "P"}: direction = "B" if p_b >= p_p else "P"
    confidence = _clip(float(policy.get("selected_win_probability", probabilities[direction]) or probabilities[direction]), 0.0, 1.0)
    return probabilities, direction, confidence


def predict(history: Union[str, Iterable[Any], None] = None, venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "", run_seed: Optional[int] = None, shoe_context: Optional[Mapping[str, Any]] = None, road_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    del run_seed
    raw_history = _normalize_outcome_history(_history_values(deepcopy(history))); big_road = normalize_big_road(raw_history); context = deepcopy(dict(shoe_context or {}))
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0)); cut_override = context.get("cut_card_remaining_cards")
    depth = ShoeDepthEstimator(shoe_decks=int(context.get("decks", SHOE_DECKS) or SHOE_DECKS), average_cards_per_hand=AVERAGE_CARDS_PER_HAND, reference_hands=REFERENCE_HANDS, burn_cards=BURN_CARDS, cut_card_remaining_cards=cut_override).estimate(raw_history).as_dict()

    supplied_road = deepcopy(dict(road_context or {}))
    if isinstance(supplied_road.get("road_features"), list) and len(supplied_road.get("road_features") or []) == len(ROAD_FEATURE_NAMES): road = supplied_road
    else:
        road = build_road_context(raw_history, grid_cells=list(supplied_road.get("grid_cells") or []), initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0), manual_count=int(supplied_road.get("manual_count", 0) or 0))
        if supplied_road: road["scan_metadata"] = supplied_road

    shoe = dict(analyze_shoe_composition(context)); shoe_available = bool(shoe.get("available")); composition_source = _shoe_source(context, shoe)
    policy = road_only_policy(raw_history, shoe_context=deepcopy(context), user_id=str(user_id or ""), venue=str(venue or ""), room=str(room or ""), shoe_id=str(shoe_id or ""))
    probabilities, direction, confidence = _normalize_policy_probabilities(policy)
    raw_probabilities = dict(probabilities); p_b = float(probabilities["B"]); p_p = float(probabilities["P"]); text = "莊" if direction == "B" else "閒"; formal_source = "contextual_linucb"

    money = _MONEY.allocate(direction=direction, probabilities=probabilities, final_weight=confidence, bankroll=bankroll)
    final_ratio = min(float(MAX_BET_RATIO), max(float(MIN_BET_RATIO), float(money.get("final_bet_ratio", MIN_BET_RATIO) or MIN_BET_RATIO)))
    bet_percentage = final_ratio * 100.0
    money.update({"final_bet_ratio": final_ratio, "bet_percentage": bet_percentage, "bet_amount": bankroll * final_ratio, "bet_allowed": True, "mandatory_bet": True})

    remaining_cards = float(depth.get("remaining_cards", 0.0) or 0.0); remaining_source = str(depth.get("remaining_cards_source") or "round_count_estimate"); remaining_ratio = float(depth.get("remaining_ratio", 1.0) or 0.0); penetration = float(depth.get("penetration", 0.0) or 0.0); shoe_stage = str(depth.get("shoe_stage", "UNKNOWN") or "UNKNOWN"); cut_progress = float(depth.get("cut_progress", 0.0) or 0.0); cut_remaining = float(depth.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING) or DEFAULT_CUT_CARD_REMAINING)
    exact_counts = list(shoe.get("remaining_counts") or []) if shoe_available else []
    markov_predict = dict(policy.get("fallback_markov") or {}); road_pattern = dict(policy.get("road_pattern") or {}); road_diag = _road_diagnostic(road); feedback = recent_user_direction_feedback(user_id)
    context_vector = list(policy.get("context_vector") or []); feature_names = list(policy.get("context_feature_names") or []); bandit_scores = dict(policy.get("scores") or {}); bandit_update = dict(policy.get("feedback_update") or {}); context_meta = deepcopy(dict(policy.get("context_metadata") or {}))
    shoe_returns = dict(shoe.get("expected_returns") or {}); banker_ev = shoe_returns.get("B") if shoe_available else None; player_ev = shoe_returns.get("P") if shoe_available else None; selected_ev = float(money.get("virtual_ev", 0.0) or 0.0); edge = float(money.get("edge", 0.0) or 0.0)

    fingerprint = sha256("|".join(("".join(raw_history), str(venue).upper().strip(), str(room).strip(), str(shoe_id).strip(), POLICY_VERSION, f"cut={cut_remaining:.2f}")).encode("utf-8")).hexdigest()[:24]
    remaining_state = {"available": True, "conditioned_rounds": len(raw_history), "remaining_cards": remaining_cards, "mean_remaining_cards": remaining_cards, "remaining_ratio": remaining_ratio, "penetration": penetration, "shoe_stage": shoe_stage, "shoe_stage_factor": 1.0, "cut_card_remaining_cards": cut_remaining, "cut_progress": cut_progress, "cut_proximity": cut_progress, "cards_until_cut": float(depth.get("cards_until_cut", 0.0) or 0.0), "estimated_hands_until_cut": float(depth.get("estimated_hands_until_cut", 0.0) or 0.0), "reliability": float(depth.get("remaining_cards_reliability", 0.0) or 0.0), "exact_composition": shoe_available, "source": remaining_source, "direction_authority": False, "semantics": "shoe_depth_is_context_or_diagnostic_not_separate_vote"}
    shoe_estimate = {"direction": None, "probabilities": dict(shoe.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "expected_remaining_cards": remaining_cards, "expected_remaining_decks": remaining_cards / float(CARDS_PER_DECK), "expected_remaining_counts": exact_counts, "remaining_card_state": remaining_state, "conditioned_rounds": len(raw_history), "reliability": float(context_meta.get("probabilistic_shoe_reliability", 0.0) or 0.0), "fusion_weight": 0.0, "depth_constraint_applied": False, "depth_constraint": {}, "source": composition_source, "expected_returns": shoe_returns, "direction_authority": False, "inference_semantics": "shoe_is_context_feature_not_independent_BP_vote"}

    context_meta.update({"formal_direction_source": formal_source, "primary_model": "CONTEXTUAL_LINUCB", "linucb_direction_weight": 1.0, "road_direction_weight": 0.0, "road_context_direction_weight": 0.0, "shoe_context_used_for_formal_direction": False, "shoe_context_used_as_linucb_features": False, "history_estimated_shoe_features_used": True, "shoe_independent_direction_vote": False, "exact_shoe_composition_used_for_direction": False, "card_composition_direction_weight": 0.0, "derived_road_independent_vote": False, "geometry_independent_vote": False, "anti_echo_external_penalty": False, "lstm_direction_weight": 0.0, "fallback_markov_direction_weight": 0.0, "remaining_cards": remaining_cards, "remaining_ratio": remaining_ratio, "penetration": penetration, "shoe_stage": shoe_stage, "cut_progress": cut_progress, "target_hands_min": int(TARGET_HANDS_MIN), "target_hands_max": int(TARGET_HANDS_MAX)})
    signal_reason = f"Primary=ContextualLinUCB；scoreGap={float(policy.get('score_gap', 0.0) or 0.0):+.6f}；P(B)={p_b:.3f}；P(P)={p_p:.3f}；Kelly={bet_percentage:.2f}%。"
    component_probabilities = {"contextual_linucb_primary": dict(probabilities), "time_decay_markov_diagnostic": dict(markov_predict.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "road_diagnostic": {"B": road_diag["banker_probability"], "P": road_diag["player_probability"], "T": 0.0}}
    if shoe_available: component_probabilities["exact_shoe_feature_diagnostic"] = dict(shoe.get("probabilities") or {})

    result: Dict[str, Any] = {
        "ok": True, "engine": "CONTEXTUAL_LINUCB_SINGLE_BRAIN_BP", "model_version": POLICY_VERSION, "system_model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "SHOE_FEATURE_PROVIDER_V5", "model_variant": "LINUCB_SINGLE_BRAIN_16D_50_70_KELLY_5_30", "model_core": "contextual_linucb", "primary_model": "CONTEXTUAL_LINUCB",
        "decision_pipeline": "16d_context_to_two_arm_linucb_ucb_argmax_to_kelly", "prediction_fingerprint": fingerprint, "probabilities": probabilities, "raw_direction_probabilities": dict(raw_probabilities),
        "banker_rate": round(p_b * 100.0, 2), "player_rate": round(p_p * 100.0, 2), "tie_rate": 0.0,
        "recommend": direction, "recommend_text": text, "action": direction, "action_text": text, "internal_recommend": direction, "internal_action": direction,
        "next_round_direction": direction, "next_round_direction_text": text, "direction": direction, "direction_text": text, "adaptive_only_direction": direction,
        "signal_allowed": True, "risk_gate_open": True, "mandatory_bet": True, "signal_status_code": "LINUCB_SINGLE_BRAIN_KELLY_5_30", "signal_status_text": f"下一手模型：{text} {confidence:.1%}；Kelly {bet_percentage:.2f}%", "signal_reason": signal_reason, "internal_signal_reason": signal_reason,
        "direction_source": formal_source, "formal_direction_source": formal_source, "linucb_direction_weight": 1.0, "road_direction_weight": 0.0, "road_context_direction_weight": 0.0,
        "shoe_context_used_for_formal_direction": False, "shoe_context_used_as_linucb_features": False, "history_estimated_shoe_features_used": True, "exact_shoe_composition_used_for_direction": False,
        "card_composition_direction_weight": 0.0, "lstm_direction_weight": 0.0, "structure_direction_weight": 0.0, "fallback_markov_direction_weight": 0.0,
        "card_composition_source": composition_source, "remaining_counts_source": composition_source, "remaining_cards_source": remaining_source, "remaining_ratio": remaining_ratio, "penetration": penetration,
        "cut_card_remaining_cards": cut_remaining, "cut_progress": cut_progress, "target_hands_min": int(TARGET_HANDS_MIN), "target_hands_max": int(TARGET_HANDS_MAX), "average_cards_per_hand": float(AVERAGE_CARDS_PER_HAND), "shoe_decks": int(context.get("decks", SHOE_DECKS) or SHOE_DECKS), "burn_cards": int(BURN_CARDS), "reference_hands": int(REFERENCE_HANDS),
        "banker_expected_return": banker_ev, "player_expected_return": player_ev, "banker_ev": banker_ev, "player_ev": player_ev,
        "shoe_composition": {**shoe, "formal_direction_authority": False, "direction_weight_inside_fusion": 0.0, "used_as_context_feature": True},
        "confidence": confidence, "raw_lstm_confidence": 0.5, "raw_model_confidence": confidence, "raw_markov_confidence": float(markov_predict.get("selected_win_probability", 0.5) or 0.5), "pattern_calibrated_confidence": confidence, "ensemble_confidence": confidence, "quality_score": confidence,
        "confidence_label": "較高" if confidence >= 0.56 else "中等" if confidence >= 0.52 else "保守",
        "confidence_calibration": {"applied": False, "raw_confidence": confidence, "shoe_stage_factor": 1.0, "final_confidence": confidence, "direction_override": False, "semantics": "no_external_probability_or_confidence_calibration"},
        "transition_calibration": {"applied": False, "disabled": True, "formal_direction_weight": 0.0}, "entropy_bits": 0.0, "entropy_base_weight": confidence,
        "shoe_progress": float(depth.get("shoe_progress", 0.0) or 0.0), "shoe_depth_estimate": {**depth, "rounds": len(raw_history), "direction_authority": False}, "remaining_card_state": remaining_state, "estimated_remaining_cards": remaining_cards, "estimated_remaining_interval": {"low": remaining_cards, "high": remaining_cards}, "shoe_stage": shoe_stage,
        "pattern_survival": {"mode": "diagnostic_only", "direction_override": False, "formal_direction_weight": 0.0}, "pattern_survival_score": 0.5,
        "run_length_hazard": {"turn_probability": context_meta.get("hazard_rate", 0.5), "diagnostic_only": False, "used_as_context_feature": True, "formal_direction_weight": 0.0}, "run_length_hazard_weight": 0.0,
        "probabilistic_shoe_estimate": shoe_estimate, "tie_risk_active": False, "direction_edge": edge, "direction_edge_percent": round(edge * 100.0, 4), "selected_expected_return": selected_ev, "selected_expected_return_percent": selected_ev * 100.0, "bet_allowed": True,
        "markov": {**markov_predict, "diagnostic_only": True, "formal_direction_weight": 0.0}, "markov_probs": dict(markov_predict.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}),
        "final_probs": probabilities, "direction_probs": probabilities, "economic_probs": probabilities, "final_probability": probabilities[direction], "economic_probability_for_direction": float(money.get("resolved_win_probability", confidence) or confidence), "markov_predict": markov_predict, "probabilistic_shoe_predict": shoe_estimate,
        "fusion_decision": {"direction": direction, "probabilities": probabilities, "linucb_weight": 1.0, "road_pattern_weight": 0.0, "shoe_composition_weight": 0.0, "lstm_weight": 0.0, "fallback_markov_weight": 0.0, "road_applied": False, "hazard_applied": False, "method": "contextual_linucb", "semantics": "single_brain_no_external_vote", "details": {}},
        "fusion": {"method": "contextual_linucb_single_brain", "road_reliability": 0.0, "hazard_reliability": 0.0, "lstm_reliability": 0.0, "cut_progress": cut_progress},
        "markov_state": {"state_key": "MARKOV_DIAGNOSTIC_ONLY", "direction_context": "diagnostic_only", "density": "unused_for_formal_direction", "tie_trigger": "TiesSkipped", "sample_count": len(big_road), "effective_support": float(markov_predict.get("context_support", 0.0) or 0.0), "state_count": 2, "selected_order": int(markov_predict.get("selected_order", 1) or 1), "change_point": False, "shoe_stage": shoe_stage, "pattern_survival_score": 0.5},
        "road_predict": {**road_diag, "decision_weight": 0.0, "diagnostic_only": True}, "road_support": road, "road_pattern_model": {**road_pattern, "diagnostic_only": True, "formal_direction_weight": 0.0},
        "derived_road_analysis": {**dict(policy.get("regression_analysis") or {}), "diagnostic_only": True, "formal_direction_weight": 0.0},
        "road_fusion": {"applied": False, "mode": "diagnostic_only", "reliability": 0.0, "raw_reliability": 0.0, "likelihood": None, "reason": "formal direction is owned by contextual_linucb"},
        "run_length_hazard_fusion": {"applied": False, "reliability": 0.0, "raw_reliability": 0.0, "likelihood": None, "continue_probability": 1.0 - float(context_meta.get("hazard_rate", 0.5) or 0.5), "turn_probability": float(context_meta.get("hazard_rate", 0.5) or 0.5), "selected_context": "", "support": 0.0, "confidence_factor": 1.0, "direction_override": False, "reason": "hazard_is_context_feature_only"},
        "component_probabilities": component_probabilities, "money_management": money, "kelly_fraction": float(money.get("kelly_fraction", final_ratio) or final_ratio), "pre_tie_adjusted_ratio": final_ratio, "adjusted_ratio": final_ratio, "final_bet_ratio": final_ratio, "bet_percentage": bet_percentage, "suggested_bet_amount": bankroll * final_ratio, "bet_amount": bankroll * final_ratio,
        "bet_multiplier": min(1.0, final_ratio / MAX_BET_RATIO) if MAX_BET_RATIO > 0.0 else 1.0,
        "context_vector": context_vector, "bandit_context": context_vector, "context_feature_names": feature_names, "context_dim": len(context_vector), "bandit_scores": bandit_scores, "bandit_selected_arm": direction, "bandit_scope_key": str(policy.get("scope_key") or ""), "bandit_feedback_update": bandit_update,
        "contextual_bandit_enabled": True, "contextual_bandit_update_enabled": False, "linucb_enabled": True, "linucb_diagnostic_only": False,
        "road_forecaster": {**dict(policy.get("road_forecaster") or {}), "diagnostic_only": True, "formal_direction_weight": 0.0},
        "lstm_enabled": False, "lstm_primary": False, "lstm_shoe_cut_fusion": False, "lstm_model": {"available": False, "formal_direction_weight": 0.0, "reason": "disabled_single_brain_linucb"}, "lstm_fallback_active": False, "lstm_fallback_reason": "disabled", "cusum_linucb_enabled": False,
        "force_observe": False, "hard_brake_active": False, "post_reset_vacuum_active": False, "input_required": False, "venue": str(venue or ""), "room": str(room or ""), "shoe_id": str(shoe_id or ""),
        "probability_semantics": "bounded_logistic_mapping_of_linucb_ucb_score_gap",
        "dynamic_prediction_policy": {"version": POLICY_VERSION, "road_primary": False, "linucb_primary": True, "lstm_primary": False, "fallback_active": False, "big_road_rounds": len(big_road), "forecast": policy, "penalty_observe": {"active": False, "force_observe": False}, "exact_card_dependency": False, "exact_card_used_when_available": False, "shoe_probability_decision_weight": 0.0, "ocr_or_screen_flow_modified": False, "formal_direction_source": formal_source, "card_composition_source": composition_source, "shoe_direction_authority": False, "shoe_context_as_features": False, "history_estimated_shoe_features_used": True, "cut_progress": cut_progress},
        "dynamic_policy_version": POLICY_VERSION, "online_performance_feedback": feedback, "decision": direction, "decision_text": text, "skip": False, "skip_reason": "",
        "decision_gate": {"decision": direction, "allowed": True, "reason": "contextual_linucb_always_returns_BP", "direction": direction, "resolved_confidence": confidence, "expected_net_ev": selected_ev, "penalty_observe": False},
        "timeline_alignment": {"raw_round_count": len(raw_history), "bp_round_count": len(big_road), "ties_ignored_for_direction_context": len(raw_history) - len(big_road)}, "context_metadata": context_meta,
    }
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    from particle_filter_points import counts_from_shoe, deal_ordered_hand
    isolated_session = deepcopy(dict(session or {})); hidden_shoe = [int(card) for card in list(isolated_session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6: raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    history = _normalize_outcome_history(list(isolated_session.get("round_history") or [])); seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(history=deepcopy(history), venue=str(isolated_session.get("venue") or ""), room=str(isolated_session.get("room") or ""), shoe_id=str(isolated_session.get("shoe_id") or ""), user_id=str(isolated_session.get("user_id") or ""), run_seed=seed, shoe_context={"bankroll": float(isolated_session.get("bankroll", 0.0) or 0.0), "remaining_cards": len(hidden_shoe), "remaining_counts": counts_from_shoe(hidden_shoe), "remaining_cards_reliability": 1.0, "remaining_cards_source": "virtual_shoe_exact_counts_feature", "source": "remaining_counts", "cut_card_remaining_cards": float(isolated_session.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING) or DEFAULT_CUT_CARD_REMAINING)})
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe); hand_data = hand.as_dict(); predicted = str(prediction.get("action") or "B").upper(); actual = str(hand.outcome or "").upper(); verdict = "TIE_SKIPPED" if actual == "T" else ("HIT" if predicted == actual else "MISS")
    update = {"updated": False, "reason": "web_panel_direct_no_feedback_update", "formal_model": "contextual_linucb"}
    prediction.update({"ok": True, "mode": "virtual_shoe_contextual_linucb_single_brain", "virtual_hand": hand_data, "virtual_outcome": actual, "virtual_outcome_text": hand_data["outcome_text"], "verdict": verdict, "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict], "cards_consumed": int(hand.cards_used), "remaining_cards_after": len(remaining_shoe), "remaining_counts_after": counts_from_shoe(remaining_shoe), "round_number": int(isolated_session.get("hand_number", 0) or 0) + 1, "bandit_learning_applied": False, "bandit_update": update, "disclaimer": "正式方向由 BBB Frozen Direct 32D Contextual LinUCB 的兩臂 UCB Score 單獨產生；正式流程不 bootstrap、不自動更新 A/b。"})
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
