"""BGS formal predictor: Big-Road LSTM primary with confidence-only shoe calibration.

Public predict() arguments and major response fields stay compatible. OCR,
screenshot and road-detector modules are untouched. Formal B/P direction comes
from LSTM when available, otherwise a time-decay Markov cold-start fallback.
Shoe/depth, run-length hazard and HSMM may only shrink confidence; they never
reverse B/P direction.
"""
from __future__ import annotations
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import POLICY_VERSION, lstm_primary_policy, normalize_big_road, recent_user_direction_feedback, record_online_feedback
from hsmm_regime import analyze_lstm_transition
from money_management import MAX_BET_RATIO, MoneyManagementModel
from pattern_survival import calibrate_lstm_confidence
from road_model import ROAD_FEATURE_NAMES, build_road_context
from run_length_hazard import analyze_run_length_hazard, lstm_hazard_confidence_calibration
from shoe_composition import analyze_shoe_composition
from shoe_constants import AVERAGE_CARDS_PER_HAND, BURN_CARDS, CARDS_PER_DECK, REFERENCE_HANDS, SHOE_DECKS
from shoe_depth_estimator import ShoeDepthEstimator, build_shoe_depth_features

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
        if compact and all(ch in OUTCOMES for ch in compact): return list(compact)
        return [part for part in history.replace("|", ",").split(",") if part.strip()]
    return list(history)


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        raw = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in OUTCOMES: result.append(value)
    return result[-2000:]


def _road_diagnostic(road: Mapping[str, Any]) -> Dict[str, Any]:
    banker = _clip(road.get("banker_probability", 0.5)); player = _clip(road.get("player_probability", 1.0 - banker)); total = banker + player
    banker, player = ((0.5, 0.5) if total <= 1e-12 else (banker / total, player / total))
    confidence = _clip(road.get("confidence_score", 0.0))
    return {"direction": "B" if banker >= player else "P", "banker_probability": float(banker), "player_probability": float(player), "confidence": float(confidence), "decision_weight": 0.0, "diagnostic_only": True}


def _shoe_source(context: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    if not bool(analysis.get("available")): return "none"
    counts = context.get("remaining_counts"); observed = context.get("observed_cards")
    if isinstance(counts, (list, tuple)) and len(counts) == 10: return "remaining_counts"
    if isinstance(observed, (list, tuple)) and observed: return "observed_cards"
    source = str(analysis.get("source") or "").lower().strip()
    return source if source in {"remaining_counts", "observed_cards"} else "none"


def _resolved_confidence(probabilities: Mapping[str, Any], direction: str) -> float:
    p_b = max(0.0, float(probabilities.get("B", 0.0) or 0.0)); p_p = max(0.0, float(probabilities.get("P", 0.0) or 0.0)); total = p_b + p_p
    return 0.5 if total <= 1e-12 else float((p_b if direction == "B" else p_p) / total)


def _remaining_state(raw_history: list[str], depth: Mapping[str, Any], shoe: Mapping[str, Any], source: str, context: Mapping[str, Any]) -> dict[str, Any]:
    if bool(shoe.get("available")):
        counts = list(shoe.get("remaining_counts") or []); remaining = float(sum(counts)); decks = max(1, min(16, int(shoe.get("shoe_decks", context.get("decks", SHOE_DECKS)) or SHOE_DECKS)))
        f = build_shoe_depth_features(remaining, shoe_decks=decks, reliability=1.0, source="exact_counts" if source == "remaining_counts" else "observed_cards")
        reliability, exact = 1.0, True
    else:
        supplied = context.get("remaining_cards")
        try: supplied_remaining = float(supplied) if supplied is not None else None
        except (TypeError, ValueError): supplied_remaining = None
        if supplied_remaining is not None and supplied_remaining >= 0.0:
            reliability = _clip(context.get("remaining_cards_reliability", 0.70)); f = build_shoe_depth_features(supplied_remaining, shoe_decks=int(context.get("decks", SHOE_DECKS) or SHOE_DECKS), reliability=reliability, source=str(context.get("remaining_cards_source") or "supplied_remaining_cards")); remaining = supplied_remaining
        else:
            remaining = float(depth.get("remaining_cards", 0.0) or 0.0); reliability = _clip(depth.get("remaining_cards_reliability", 0.60)); f = build_shoe_depth_features(remaining, shoe_decks=SHOE_DECKS, reliability=reliability, source="round_count_estimate")
        exact = False
    return {"available": True, "conditioned_rounds": len(raw_history), "remaining_cards": float(remaining), "mean_remaining_cards": float(remaining), "remaining_ratio": float(f["remaining_ratio"]), "penetration": float(f["penetration"]), "shoe_stage": str(f["shoe_stage"]), "shoe_confidence_factor": float(f["shoe_confidence_factor"]), "lstm_shoe_stage_factor": float(f["shoe_stage_anchor"]), "reliability": float(reliability), "exact_composition": exact, "source": str(f["depth_feature_source"]), "direction_authority": False, "semantics": "shoe_depth_for_confidence_and_sizing_only_never_BP_direction"}


def predict(history: Union[str, Iterable[Any], None] = None, venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "", run_seed: Optional[int] = None, shoe_context: Optional[Mapping[str, Any]] = None, road_context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    del run_seed
    raw_history = _normalize_outcome_history(_history_values(history)); big_road = normalize_big_road(raw_history)
    depth = ShoeDepthEstimator(shoe_decks=SHOE_DECKS, average_cards_per_hand=AVERAGE_CARDS_PER_HAND, reference_hands=REFERENCE_HANDS, burn_cards=BURN_CARDS).estimate(raw_history).as_dict()
    supplied_road = dict(road_context or {})
    if isinstance(supplied_road.get("road_features"), list) and len(supplied_road.get("road_features") or []) == len(ROAD_FEATURE_NAMES): road = supplied_road
    else:
        road = build_road_context(raw_history, grid_cells=list(supplied_road.get("grid_cells") or []), initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0), manual_count=int(supplied_road.get("manual_count", 0) or 0))
        if supplied_road: road["scan_metadata"] = supplied_road
    context = dict(shoe_context or {}); bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))
    shoe = dict(analyze_shoe_composition(context)); shoe_available = bool(shoe.get("available")); composition_source = _shoe_source(context, shoe)
    remaining = _remaining_state(raw_history, depth, shoe, composition_source, context)

    policy = lstm_primary_policy(raw_history, shoe_context=context, user_id=user_id, venue=venue, room=room, shoe_id=shoe_id)
    direction = str(policy.get("direction") or "B").upper(); direction = direction if direction in {"B", "P"} else "B"
    raw_probs = dict(policy.get("probabilities") or {}); raw_confidence = _resolved_confidence(raw_probs, direction)
    formal_source = str(policy.get("formal_direction_source") or policy.get("policy_source") or "time_decay_markov_fallback")
    lstm_active = formal_source == "lstm_road_model"; fallback_active = not lstm_active

    hazard = analyze_run_length_hazard(big_road); hazard_cal = lstm_hazard_confidence_calibration(analysis=hazard); transition = analyze_lstm_transition(big_road, hazard=hazard)
    calibration = calibrate_lstm_confidence(direction=direction, raw_confidence=raw_confidence, remaining_card_state=remaining, hazard_calibration=hazard_cal, transition_calibration=transition)
    confidence = float(calibration["confidence"]); probabilities = dict(calibration["probabilities"]); p_b, p_p, p_t = float(probabilities["B"]), float(probabilities["P"]), float(probabilities.get("T", 0.0))
    text = "莊" if direction == "B" else "閒"; money = _MONEY.allocate(direction=direction, probabilities=probabilities, final_weight=confidence, bankroll=bankroll)
    final_ratio = float(money.get("final_bet_ratio", 0.05) or 0.05); bet_percentage = float(money.get("bet_percentage", final_ratio * 100.0) or final_ratio * 100.0)
    fallback_markov = dict(policy.get("fallback_markov") or {}); road_diag = _road_diagnostic(road); feedback = recent_user_direction_feedback(user_id)
    context_vector = list(policy.get("context_vector") or []); feature_names = list(policy.get("context_feature_names") or []); bandit_scores = dict(policy.get("scores") or {}); bandit_update = dict(policy.get("feedback_update") or {})
    remaining_cards = float(remaining["remaining_cards"]); remaining_source = str(remaining["source"]); exact_counts = list(shoe.get("remaining_counts") or []) if shoe_available else []
    road_forecaster_diag = dict(policy.get("road_forecaster_diagnostic") or {}); road_forecaster = dict(policy.get("road_forecaster") or {})
    shoe_returns = dict(shoe.get("expected_returns") or {}); banker_ev = shoe_returns.get("B") if shoe_available else None; player_ev = shoe_returns.get("P") if shoe_available else None
    selected_ev = float(money.get("virtual_ev", 0.0) or 0.0); edge = float(money.get("edge", 0.0) or 0.0)
    fingerprint = sha256("|".join(("".join(raw_history), str(venue).upper().strip(), str(room).strip(), str(shoe_id).strip(), POLICY_VERSION, formal_source, composition_source)).encode()).hexdigest()[:24]

    context_meta = dict(policy.get("context_metadata") or {}); context_meta.update({"formal_direction_source": formal_source, "primary_model": "LSTM" if lstm_active else "TIME_DECAY_MARKOV_FALLBACK", "shoe_context_used_for_formal_direction": False, "card_composition_direction_weight": 0.0, "lstm_direction_weight": 1.0 if lstm_active else 0.0, "fallback_markov_direction_weight": 1.0 if fallback_active else 0.0, "road_context_direction_weight": 1.0 if fallback_active else 0.0, "road_direction_weight": 1.0 if fallback_active else 0.0, "card_composition_source": composition_source, "remaining_counts_source": composition_source, "remaining_cards": remaining_cards, "remaining_cards_source": remaining_source, "remaining_ratio": float(remaining["remaining_ratio"]), "penetration": float(remaining["penetration"]), "shoe_stage": str(remaining["shoe_stage"]), "estimated_remaining_counts_0_to_9": exact_counts})
    policy.update({"context_metadata": context_meta, "direction": direction, "selected_arm": direction, "action": direction, "action_text": text, "latent_direction": direction, "raw_direction_probabilities": raw_probs, "probabilities": probabilities, "selected_win_probability": confidence, "confidence": confidence, "confidence_prob": confidence, "selection_reason": formal_source, "formal_context_source": "big_road_lstm" if lstm_active else "big_road_time_decay_markov_fallback", "road_context_direction_weight": 1.0 if fallback_active else 0.0, "road_direction_weight": 1.0 if fallback_active else 0.0, "card_composition_direction_weight": 0.0, "shoe_context_used_for_formal_direction": False, "confidence_calibration": calibration})

    markov_predict = {**fallback_markov, "pattern_calibrated_probabilities": probabilities, "state": {"direction_context": formal_source, "density": "ShortShoeSequence", "tie_trigger": "TiesSkipped", "recent5": big_road[-5:]}, "state_key": "LSTM" if lstm_active else "MARKOV_FALLBACK", "final_weight": confidence, "pattern_survival_score": float(calibration["combined_factor"]), "regime_profile": {"change_point": bool(transition.get("transition_probability", 0.0) > 0.65), "current_run_length": int(hazard.get("current_run_length", 0) or 0), "recent_run_lengths": list(hazard.get("run_lengths") or []), "alternation_ratio": float(transition.get("alternation_ratio", 0.5) or 0.5), "bandit_feedback_update": bandit_update}}
    component_probabilities = {"lstm_road_model": dict((policy.get("lstm") or {}).get("probabilities") or {"B": 0.5, "P": 0.5}), "time_decay_markov_fallback": dict(fallback_markov.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "road_forecaster": dict(road_forecaster_diag.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "road_diagnostic": {"B": road_diag["banker_probability"], "P": road_diag["player_probability"], "T": 0.0}}
    if shoe_available: component_probabilities["card_composition_diagnostic"] = dict(shoe.get("probabilities") or {})
    shoe_estimate = {"direction": None, "probabilities": dict(shoe.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "expected_remaining_cards": remaining_cards, "expected_remaining_decks": remaining_cards / float(CARDS_PER_DECK), "expected_remaining_counts": exact_counts, "remaining_card_state": remaining, "conditioned_rounds": len(raw_history), "reliability": float(remaining["reliability"]), "fusion_weight": 0.0, "depth_constraint_applied": True, "depth_constraint": {"applied": True, "target_remaining_cards": remaining_cards, "source": remaining_source, "remaining_ratio": remaining["remaining_ratio"], "penetration": remaining["penetration"], "shoe_stage": remaining["shoe_stage"], "direction_authority": False}, "source": composition_source, "expected_returns": shoe_returns, "inference_semantics": "shoe_state_for_confidence_only_no_BP_vote"}

    signal_code = "LSTM_TWO_ARM_KELLY_5_30" if lstm_active else "MARKOV_FALLBACK_TWO_ARM_KELLY_5_30"
    signal_reason = f"Primary={formal_source}；raw={raw_confidence:.3f}；stage={remaining['shoe_stage']}/{float(calibration['shoe_stage_factor']):.3f}；hazard={float(calibration['hazard_factor']):.3f}；transition={float(calibration['transition_factor']):.3f}；final={confidence:.3f}；Kelly={bet_percentage:.2f}%。"
    result = {
        "ok": True, "engine": "LSTM_ROAD_PRIMARY_BP" if lstm_active else "TIME_DECAY_MARKOV_FALLBACK_BP", "model_version": POLICY_VERSION, "system_model_version": POLICY_VERSION,
        "shoe_posterior_model_version": "EXACT_NON_REPLACEMENT_DIAGNOSTIC_V1" if shoe_available else "DEPTH_CONFIDENCE_ONLY_V1", "model_variant": "LSTM_SOFTMAX_SHOE_DEPTH_CALIBRATED_KELLY_5_30" if lstm_active else "TIME_DECAY_MARKOV_COLD_START_KELLY_5_30", "model_core": "lstm_road_model" if lstm_active else "time_decay_markov_fallback", "primary_model": "LSTM" if lstm_active else "TIME_DECAY_MARKOV_FALLBACK", "decision_pipeline": "big_road_BP_to_LSTM_or_markov_fallback_to_shoe_depth_hazard_HSMM_confidence_to_existing_kelly", "prediction_fingerprint": fingerprint,
        "probabilities": probabilities, "raw_direction_probabilities": {"B": float(raw_probs.get("B", 0.5) or 0.5), "P": float(raw_probs.get("P", 0.5) or 0.5)}, "banker_rate": round(p_b * 100, 2), "player_rate": round(p_p * 100, 2), "tie_rate": round(p_t * 100, 2),
        "recommend": direction, "recommend_text": text, "action": direction, "action_text": text, "internal_recommend": direction, "internal_action": direction, "next_round_direction": direction, "next_round_direction_text": text, "direction": direction, "direction_text": text, "adaptive_only_direction": direction,
        "signal_allowed": True, "risk_gate_open": True, "mandatory_bet": True, "signal_status_code": signal_code, "signal_status_text": f"下一手模型：{text} {confidence:.1%}；Kelly {bet_percentage:.2f}%", "signal_reason": signal_reason, "internal_signal_reason": signal_reason,
        "direction_source": formal_source, "formal_direction_source": formal_source, "shoe_context_used_for_formal_direction": False, "card_composition_direction_weight": 0.0, "lstm_direction_weight": 1.0 if lstm_active else 0.0, "fallback_markov_direction_weight": 1.0 if fallback_active else 0.0, "road_context_direction_weight": 1.0 if fallback_active else 0.0, "road_direction_weight": 1.0 if fallback_active else 0.0,
        "card_composition_source": composition_source, "remaining_counts_source": composition_source, "remaining_cards_source": remaining_source, "remaining_ratio": float(remaining["remaining_ratio"]), "penetration": float(remaining["penetration"]), "average_cards_per_hand": float(AVERAGE_CARDS_PER_HAND), "shoe_decks": int(SHOE_DECKS), "burn_cards": int(BURN_CARDS), "reference_hands": int(REFERENCE_HANDS),
        "banker_expected_return": banker_ev, "player_expected_return": player_ev, "banker_ev": banker_ev, "player_ev": player_ev, "shoe_composition": {**shoe, "formal_direction_authority": False, "diagnostic_only_for_direction": True},
        "confidence": confidence, "raw_lstm_confidence": float((policy.get("lstm") or {}).get("raw_confidence", raw_confidence) or raw_confidence) if lstm_active else None, "raw_model_confidence": raw_confidence, "raw_markov_confidence": float(fallback_markov.get("selected_win_probability", 0.5) or 0.5), "pattern_calibrated_confidence": confidence, "ensemble_confidence": confidence, "quality_score": confidence, "confidence_label": "較高" if confidence >= 0.56 else "中等" if confidence >= 0.52 else "保守", "confidence_calibration": calibration, "transition_calibration": transition,
        "entropy_bits": 0.0, "entropy_base_weight": raw_confidence, "shoe_progress": float(depth["shoe_progress"]), "shoe_depth_estimate": {**depth, "rounds": len(raw_history), "remaining_cards": remaining_cards, "remaining_cards_source": remaining_source, "source": remaining_source, "exact_composition": shoe_available, "remaining_ratio": remaining["remaining_ratio"], "penetration": remaining["penetration"], "shoe_stage": remaining["shoe_stage"], "direction_authority": False}, "remaining_card_state": remaining, "estimated_remaining_cards": remaining_cards, "estimated_remaining_interval": {"low": remaining_cards, "high": remaining_cards}, "shoe_stage": remaining["shoe_stage"],
        "pattern_survival": {"score": float(calibration["combined_factor"]), "mode": "lstm_confidence_calibration_only", "direction_override": False}, "pattern_survival_score": float(calibration["combined_factor"]), "run_length_hazard": hazard, "run_length_hazard_weight": float(1.0 - hazard_cal["confidence_factor"]), "probabilistic_shoe_estimate": shoe_estimate, "tie_risk_active": False,
        "direction_edge": edge, "direction_edge_percent": round(edge * 100, 4), "selected_expected_return": selected_ev, "selected_expected_return_percent": selected_ev * 100, "bet_allowed": True, "markov": markov_predict, "markov_probs": dict(fallback_markov.get("probabilities") or {"B": 0.5, "P": 0.5, "T": 0.0}), "final_probs": probabilities, "direction_probs": probabilities, "economic_probs": probabilities, "final_probability": probabilities[direction], "economic_probability_for_direction": float(money.get("resolved_win_probability", confidence) or confidence), "markov_predict": markov_predict, "probabilistic_shoe_predict": shoe_estimate,
        "fusion_decision": {"direction": direction, "probabilities": probabilities, "lstm_weight": 1.0 if lstm_active else 0.0, "fallback_markov_weight": 1.0 if fallback_active else 0.0, "markov_prior_weight": 1.0 if fallback_active else 0.0, "probabilistic_shoe_weight": 0.0, "card_composition_direction_weight": 0.0, "road_forecaster_weight": 0.0, "run_length_hazard_likelihood_power": 0.0, "road_applied": fallback_active, "hazard_applied": bool(hazard_cal.get("applied", False)), "method": formal_source, "semantics": "direction_from_lstm_or_markov_fallback_shoe_hazard_HSMM_confidence_only"},
        "fusion": {"method": formal_source, "shoe_reliability": float(remaining["reliability"]), "road_reliability": 1.0 if fallback_active else 0.0, "hazard_reliability": float(hazard.get("reliability", 0.0) or 0.0), "lstm_reliability": 1.0 if lstm_active else 0.0},
        "markov_state": {"state_key": "LSTM" if lstm_active else "MARKOV_FALLBACK", "direction_context": formal_source, "density": "ShortShoeSequence", "tie_trigger": "TiesSkipped", "sample_count": len(big_road), "effective_support": float(fallback_markov.get("context_support", 0.0) or 0.0), "state_count": 2, "selected_order": int(fallback_markov.get("selected_order", 1) or 1), "change_point": bool(transition.get("transition_probability", 0.0) > 0.65), "shoe_stage": remaining["shoe_stage"], "pattern_survival_score": float(calibration["combined_factor"])},
        "road_predict": road_diag, "road_support": road, "derived_road_analysis": dict(policy.get("regression_analysis") or {}), "road_fusion": {"applied": fallback_active, "mode": "time_decay_markov_fallback" if fallback_active else "diagnostic_only", "reliability": 1.0 if fallback_active else 0.0, "raw_reliability": 1.0 if fallback_active else 0.0, "pattern_survival_score": float(calibration["combined_factor"]), "likelihood": None, "reason": "LSTM available; old road forecaster diagnostic only." if lstm_active else "LSTM cold/unavailable; time-decay Markov fallback."},
        "run_length_hazard_fusion": {"applied": bool(hazard_cal.get("applied", False)), "reliability": float(hazard.get("reliability", 0.0) or 0.0), "raw_reliability": float(hazard.get("reliability", 0.0) or 0.0), "likelihood": dict(hazard.get("likelihood") or {}), "continue_probability": float(hazard.get("continue_probability", 0.5) or 0.5), "turn_probability": float(hazard.get("turn_probability", 0.5) or 0.5), "selected_context": str(hazard.get("selected_context", "") or ""), "support": float(hazard.get("support", 0.0) or 0.0), "confidence_factor": float(hazard_cal.get("confidence_factor", 1.0) or 1.0), "direction_override": False, "reason": "hazard only shrinks confidence margin"},
        "component_probabilities": component_probabilities, "money_management": money, "kelly_fraction": float(money.get("kelly_fraction", final_ratio) or final_ratio), "pre_tie_adjusted_ratio": float(money.get("pre_tie_adjusted_ratio", final_ratio) or final_ratio), "adjusted_ratio": float(money.get("adjusted_ratio", final_ratio) or final_ratio), "final_bet_ratio": final_ratio, "bet_percentage": bet_percentage, "suggested_bet_amount": float(money.get("bet_amount", 0.0) or 0.0), "bet_amount": float(money.get("bet_amount", 0.0) or 0.0), "bet_multiplier": min(1.0, final_ratio / MAX_BET_RATIO) if MAX_BET_RATIO > 0 else 1.0,
        "context_vector": context_vector, "bandit_context": context_vector, "context_feature_names": feature_names, "context_dim": len(context_vector), "bandit_scores": bandit_scores, "bandit_selected_arm": str(road_forecaster_diag.get("direction") or ""), "bandit_scope_key": str(policy.get("scope_key") or ""), "bandit_feedback_update": bandit_update, "contextual_bandit_enabled": True, "contextual_bandit_update_enabled": True, "linucb_enabled": True, "linucb_diagnostic_only": True, "linucb_direction_weight": 0.0, "road_forecaster": road_forecaster,
        "lstm_enabled": True, "lstm_primary": True, "lstm_model": dict(policy.get("lstm") or {}), "lstm_fallback_active": fallback_active, "lstm_fallback_reason": str(policy.get("fallback_reason") or ""), "cusum_linucb_enabled": False, "force_observe": False, "hard_brake_active": False, "post_reset_vacuum_active": False, "input_required": False, "venue": str(venue or ""), "room": str(room or ""), "shoe_id": str(shoe_id or ""), "probability_semantics": "lstm_BP_softmax_margin_calibrated_toward_50" if lstm_active else "time_decay_markov_BP_margin_calibrated_toward_50",
        "dynamic_prediction_policy": {"version": POLICY_VERSION, "lstm_primary": True, "fallback_active": fallback_active, "big_road_rounds": len(big_road), "forecast": policy, "penalty_observe": {"active": False, "force_observe": False}, "exact_card_dependency": False, "shoe_probability_decision_weight": 0.0, "ocr_or_screen_flow_modified": False, "formal_direction_source": formal_source, "card_composition_source": composition_source, "shoe_direction_authority": False}, "dynamic_policy_version": POLICY_VERSION, "online_performance_feedback": feedback,
        "decision": direction, "decision_text": text, "skip": False, "skip_reason": "", "decision_gate": {"decision": direction, "allowed": True, "reason": "lstm_or_markov_fallback_always_returns_BP", "direction": direction, "resolved_confidence": confidence, "expected_net_ev": selected_ev, "penalty_observe": False}, "timeline_alignment": {"raw_round_count": len(raw_history), "bp_round_count": len(big_road), "ties_ignored_for_direction_context": len(raw_history) - len(big_road)},
    }
    return result


def run_virtual_round(session: Mapping[str, Any], run_seed: Optional[int] = None) -> Dict[str, Any]:
    """Predict first, then reveal a virtual-shoe hand; exact cards remain confidence-only."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand
    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6: raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    history = _normalize_outcome_history(list(session.get("round_history") or [])); seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(history=history, venue=str(session.get("venue") or ""), room=str(session.get("room") or ""), shoe_id=str(session.get("shoe_id") or ""), user_id=str(session.get("user_id") or ""), run_seed=seed,
                         shoe_context={"bankroll": float(session.get("bankroll", 0.0) or 0.0), "remaining_cards": len(hidden_shoe), "remaining_counts": counts_from_shoe(hidden_shoe), "remaining_cards_reliability": 1.0, "remaining_cards_source": "virtual_shoe_exact_counts", "source": "remaining_counts"})
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe); hand_data = hand.as_dict(); predicted = str(prediction.get("action") or "P").upper(); actual = str(hand.outcome or "").upper(); verdict = "TIE_SKIPPED" if actual == "T" else ("HIT" if predicted == actual else "MISS")
    update = record_online_feedback(scope_key=str(prediction.get("bandit_scope_key") or ""), action=predicted, context_vector=list(prediction.get("context_vector") or []), actual_outcome=actual)
    prediction.update({"ok": True, "mode": "virtual_shoe_lstm_primary", "virtual_hand": hand_data, "virtual_outcome": actual, "virtual_outcome_text": hand_data["outcome_text"], "verdict": verdict, "verdict_text": {"HIT": "命中", "MISS": "未命中", "TIE_SKIPPED": "和局不計"}[verdict], "cards_consumed": int(hand.cards_used), "remaining_cards_after": len(remaining_shoe), "remaining_counts_after": counts_from_shoe(remaining_shoe), "round_number": int(session.get("hand_number", 0) or 0) + 1, "bandit_learning_applied": bool(update.get("updated", False)), "bandit_update": update, "disclaimer": "正式方向由大路 LSTM 決定；冷啟動使用 time-decay Markov。牌靴/切牌深度、hazard、HSMM 僅調節信心與注碼，不反轉方向。單局結果具有高變異，模型機率不代表保證獲利。"})
    return {"prediction": prediction, "hand": hand_data, "remaining_shoe": remaining_shoe}


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
