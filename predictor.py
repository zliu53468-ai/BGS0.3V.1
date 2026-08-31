"""BGS production predictor: one LSTM + shoe + cut-card fusion core.

Formal B/P direction is produced once by ``lstm_primary_policy``.  The fused
model combines:

1. masked, class-balanced Big-Road LSTM sequence evidence;
2. exact non-replacement remaining-shoe B/P evidence when exact composition is
   available;
3. continuous cut-card depth weighting tuned for a 50-70 hand shoe.

Legacy road/Markov/LinUCB data is retained only for API diagnostics.  Hazard,
HSMM and pattern-survival are no longer part of the formal prediction path.
OCR, screenshot recognition and road detection are untouched.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from dynamic_prediction_policy import (
    POLICY_VERSION,
    lstm_primary_policy,
    normalize_big_road,
    recent_user_direction_feedback,
    record_online_feedback,
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
        compact = (
            history.replace("|", "")
            .replace(",", "")
            .replace(" ", "")
            .upper()
        )
        if compact and all(char in OUTCOMES for char in compact):
            return list(compact)
        return [
            part
            for part in history.replace("|", ",").split(",")
            if part.strip()
        ]
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
    if total <= 1e-12:
        banker, player = 0.5, 0.5
    else:
        banker, player = banker / total, player / total
    confidence = _clip(road.get("confidence_score", 0.0))
    return {
        "direction": "B" if banker >= player else "P",
        "banker_probability": float(banker),
        "player_probability": float(player),
        "confidence": float(confidence),
        "decision_weight": 0.0,
        "diagnostic_only": True,
    }


def _shoe_source(
    context: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> str:
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


def _resolved_confidence(
    probabilities: Mapping[str, Any],
    direction: str,
) -> float:
    p_b = max(0.0, float(probabilities.get("B", 0.0) or 0.0))
    p_p = max(0.0, float(probabilities.get("P", 0.0) or 0.0))
    total = p_b + p_p
    if total <= 1e-12:
        return 0.5
    return float((p_b if direction == "B" else p_p) / total)


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

    # Exact shoe analysis is retained as a transparent physical diagnostic.  The
    # same exact probabilities are already consumed inside the fused LSTM model.
    shoe = dict(analyze_shoe_composition(context))
    shoe_available = bool(shoe.get("available"))
    composition_source = _shoe_source(context, shoe)

    policy = lstm_primary_policy(
        raw_history,
        shoe_context=context,
        user_id=user_id,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
    )
    lstm_model = dict(policy.get("lstm") or {})
    lstm_fusion = dict(lstm_model.get("fusion") or {})
    shoe_fusion = dict(lstm_model.get("shoe_fusion") or {})
    neural = dict(lstm_model.get("neural") or {})

    direction = str(policy.get("direction") or "B").upper().strip()
    if direction not in {"B", "P"}:
        direction = "B"
    probabilities = dict(policy.get("probabilities") or {})
    p_b = _clip(probabilities.get("B", 0.5))
    p_p = _clip(probabilities.get("P", 0.5))
    total = p_b + p_p
    if total <= 1e-12:
        p_b = p_p = 0.5
    else:
        p_b, p_p = p_b / total, p_p / total
    probabilities = {"B": float(p_b), "P": float(p_p), "T": 0.0}
    confidence = _resolved_confidence(probabilities, direction)
    raw_confidence = confidence
    formal_source = "lstm_road_model"
    text = "莊" if direction == "B" else "閒"

    money = _MONEY.allocate(
        direction=direction,
        probabilities=probabilities,
        final_weight=confidence,
        bankroll=bankroll,
    )
    final_ratio = float(money.get("final_bet_ratio", 0.05) or 0.05)
    bet_percentage = float(
        money.get("bet_percentage", final_ratio * 100.0)
        or final_ratio * 100.0
    )

    remaining_cards = float(
        shoe_fusion.get("remaining_cards", depth.get("remaining_cards", 0.0)) or 0.0
    )
    remaining_source = str(
        shoe_fusion.get("depth_feature_source")
        or depth.get("remaining_cards_source")
        or "round_count_estimate"
    )
    remaining_ratio = float(
        shoe_fusion.get("remaining_ratio", depth.get("remaining_ratio", 1.0)) or 0.0
    )
    penetration = float(
        shoe_fusion.get("penetration", depth.get("penetration", 0.0)) or 0.0
    )
    shoe_stage = str(
        shoe_fusion.get("shoe_stage", depth.get("shoe_stage", "UNKNOWN"))
        or "UNKNOWN"
    )
    cut_progress = float(
        shoe_fusion.get("cut_progress", depth.get("cut_progress", 0.0)) or 0.0
    )
    cut_remaining = float(
        shoe_fusion.get(
            "cut_card_remaining_cards",
            depth.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING),
        )
        or DEFAULT_CUT_CARD_REMAINING
    )
    exact_counts = list(shoe.get("remaining_counts") or []) if shoe_available else []
    shoe_reliability = float(
        shoe_fusion.get("remaining_cards_reliability", 0.0) or 0.0
    )

    lstm_weight = float(lstm_fusion.get("lstm_weight", 0.0) or 0.0)
    shoe_direction_weight = float(lstm_fusion.get("shoe_weight", 0.0) or 0.0)
    structure_weight = float(lstm_fusion.get("structure_weight", 0.0) or 0.0)
    exact_shoe_used_for_direction = bool(
        shoe_fusion.get("exact_composition_available", False)
        and shoe_direction_weight > 0.0
    )

    fallback_markov = dict(policy.get("fallback_markov") or {})
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
                composition_source,
                f"cut={cut_remaining:.2f}",
            )
        ).encode("utf-8")
    ).hexdigest()[:24]

    context_meta = dict(policy.get("context_metadata") or {})
    context_meta.update(
        {
            "formal_direction_source": formal_source,
            "primary_model": "LSTM_SHOE_CUT_FUSION",
            "shoe_context_used_for_formal_direction": True,
            "exact_shoe_composition_used_for_direction": exact_shoe_used_for_direction,
            "card_composition_direction_weight": shoe_direction_weight,
            "lstm_direction_weight": lstm_weight,
            "structure_direction_weight": structure_weight,
            "fallback_markov_direction_weight": 0.0,
            "road_context_direction_weight": 0.0,
            "road_direction_weight": 0.0,
            "card_composition_source": composition_source,
            "remaining_counts_source": composition_source,
            "remaining_cards": remaining_cards,
            "remaining_cards_source": remaining_source,
            "remaining_ratio": remaining_ratio,
            "penetration": penetration,
            "shoe_stage": shoe_stage,
            "cut_card_remaining_cards": cut_remaining,
            "cut_progress": cut_progress,
            "target_hands_min": int(TARGET_HANDS_MIN),
            "target_hands_max": int(TARGET_HANDS_MAX),
            "estimated_remaining_counts_0_to_9": exact_counts,
        }
    )
    policy.update(
        {
            "context_metadata": context_meta,
            "direction": direction,
            "selected_arm": direction,
            "action": direction,
            "action_text": text,
            "latent_direction": direction,
            "probabilities": probabilities,
            "selected_win_probability": confidence,
            "confidence": confidence,
            "confidence_prob": confidence,
            "selection_reason": "single_lstm_shoe_cut_fusion",
            "formal_context_source": "big_road_plus_shoe_plus_cut_depth",
            "road_context_direction_weight": 0.0,
            "road_direction_weight": 0.0,
            "card_composition_direction_weight": shoe_direction_weight,
            "shoe_context_used_for_formal_direction": True,
        }
    )

    markov_predict = {
        **fallback_markov,
        "direction": str(fallback_markov.get("direction") or ""),
        "probabilities": dict(
            fallback_markov.get("probabilities")
            or {"B": 0.5, "P": 0.5, "T": 0.0}
        ),
        "diagnostic_only": True,
        "formal_direction_weight": 0.0,
        "state_key": "MARKOV_DIAGNOSTIC_ONLY",
    }

    component_probabilities = {
        "lstm_shoe_cut_fusion": probabilities,
        "lstm_neural_branch": {
            "B": float(neural.get("probability_b", 0.5) or 0.5),
            "P": float(neural.get("probability_p", 0.5) or 0.5),
            "T": 0.0,
        },
        "time_decay_markov_diagnostic": dict(
            fallback_markov.get("probabilities")
            or {"B": 0.5, "P": 0.5, "T": 0.0}
        ),
        "road_forecaster_diagnostic": dict(
            road_forecaster_diag.get("probabilities")
            or {"B": 0.5, "P": 0.5, "T": 0.0}
        ),
        "road_diagnostic": {
            "B": road_diag["banker_probability"],
            "P": road_diag["player_probability"],
            "T": 0.0,
        },
    }
    if shoe_available:
        component_probabilities["exact_shoe_nonreplacement"] = dict(
            shoe.get("probabilities") or {}
        )

    remaining_state = {
        "available": True,
        "conditioned_rounds": len(raw_history),
        "remaining_cards": remaining_cards,
        "mean_remaining_cards": remaining_cards,
        "remaining_ratio": remaining_ratio,
        "penetration": penetration,
        "shoe_stage": shoe_stage,
        "cut_card_remaining_cards": cut_remaining,
        "cut_progress": cut_progress,
        "cut_proximity": cut_progress,
        "cards_until_cut": float(shoe_fusion.get("cards_until_cut", 0.0) or 0.0),
        "estimated_hands_until_cut": float(
            shoe_fusion.get("estimated_hands_until_cut", 0.0) or 0.0
        ),
        "reliability": shoe_reliability,
        "exact_composition": shoe_available,
        "source": remaining_source,
        "direction_authority": "through_lstm_shoe_cut_fusion",
        "semantics": "physical_shoe_and_cut_features_inside_single_formal_fusion",
    }

    shoe_estimate = {
        "direction": str(shoe_fusion.get("shoe_direction") or "") or None,
        "probabilities": dict(
            shoe.get("probabilities")
            or {"B": 0.5, "P": 0.5, "T": 0.0}
        ),
        "expected_remaining_cards": remaining_cards,
        "expected_remaining_decks": remaining_cards / float(CARDS_PER_DECK),
        "expected_remaining_counts": exact_counts,
        "remaining_card_state": remaining_state,
        "conditioned_rounds": len(raw_history),
        "reliability": shoe_reliability,
        "fusion_weight": shoe_direction_weight,
        "depth_constraint_applied": True,
        "depth_constraint": {
            "applied": True,
            "target_remaining_cards": remaining_cards,
            "source": remaining_source,
            "remaining_ratio": remaining_ratio,
            "penetration": penetration,
            "shoe_stage": shoe_stage,
            "cut_card_remaining_cards": cut_remaining,
            "cut_progress": cut_progress,
            "direction_authority": "weighted_inside_lstm_fusion",
        },
        "source": composition_source,
        "expected_returns": shoe_returns,
        "inference_semantics": "exact_shoe_direction_if_available_plus_cut_depth_weighted_in_lstm_fusion",
    }

    signal_code = "LSTM_SHOE_CUT_TWO_ARM_KELLY_5_30"
    signal_reason = (
        f"Primary=LSTM+Shoe+Cut；LSTMw={lstm_weight:.3f}；"
        f"Shoew={shoe_direction_weight:.3f}；Structw={structure_weight:.3f}；"
        f"cut={cut_progress:.3f}；P(B)={p_b:.3f}；P(P)={p_p:.3f}；"
        f"Kelly={bet_percentage:.2f}%。"
    )

    result: Dict[str, Any] = {
        "ok": True,
        "engine": "LSTM_SHOE_CUT_FUSION_BP",
        "model_version": POLICY_VERSION,
        "system_model_version": POLICY_VERSION,
        "shoe_posterior_model_version": (
            "EXACT_NON_REPLACEMENT_IN_FUSION_V2"
            if shoe_available
            else "CUT_DEPTH_WITHOUT_EXACT_COMPOSITION_V2"
        ),
        "model_variant": "BALANCED_MASKED_LSTM_EXACT_SHOE_CUT_50_70_KELLY_5_30",
        "model_core": "lstm_shoe_cut_fusion",
        "primary_model": "LSTM_SHOE_CUT_FUSION",
        "decision_pipeline": "big_road_BP_to_balanced_masked_LSTM_plus_exact_shoe_logit_plus_50_70_cut_weight_to_BP_to_kelly",
        "prediction_fingerprint": fingerprint,
        "probabilities": probabilities,
        "raw_direction_probabilities": {"B": p_b, "P": p_p},
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
        "shoe_context_used_for_formal_direction": True,
        "exact_shoe_composition_used_for_direction": exact_shoe_used_for_direction,
        "card_composition_direction_weight": shoe_direction_weight,
        "lstm_direction_weight": lstm_weight,
        "structure_direction_weight": structure_weight,
        "fallback_markov_direction_weight": 0.0,
        "road_context_direction_weight": 0.0,
        "road_direction_weight": 0.0,
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
        "shoe_composition": {
            **shoe,
            "formal_direction_authority": exact_shoe_used_for_direction,
            "direction_weight_inside_fusion": shoe_direction_weight,
        },
        "confidence": confidence,
        "raw_lstm_confidence": float(
            max(
                float(neural.get("probability_b", 0.5) or 0.5),
                float(neural.get("probability_p", 0.5) or 0.5),
            )
        ),
        "raw_model_confidence": raw_confidence,
        "raw_markov_confidence": float(
            fallback_markov.get("selected_win_probability", 0.5) or 0.5
        ),
        "pattern_calibrated_confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": (
            "較高" if confidence >= 0.56 else "中等" if confidence >= 0.52 else "保守"
        ),
        "confidence_calibration": {
            "applied": False,
            "reason": "formal_probability_already_calibrated_inside_single_lstm_shoe_cut_fusion",
            "direction_override": False,
        },
        "transition_calibration": {
            "applied": False,
            "disabled": True,
            "formal_direction_weight": 0.0,
        },
        "entropy_bits": 0.0,
        "entropy_base_weight": confidence,
        "shoe_progress": float(depth["shoe_progress"]),
        "shoe_depth_estimate": {
            **depth,
            "rounds": len(raw_history),
            "remaining_cards": remaining_cards,
            "remaining_cards_source": remaining_source,
            "source": remaining_source,
            "exact_composition": shoe_available,
            "remaining_ratio": remaining_ratio,
            "penetration": penetration,
            "shoe_stage": shoe_stage,
            "cut_card_remaining_cards": cut_remaining,
            "cut_progress": cut_progress,
            "direction_authority": "weighting_inside_lstm_fusion",
        },
        "remaining_card_state": remaining_state,
        "estimated_remaining_cards": remaining_cards,
        "estimated_remaining_interval": {
            "low": remaining_cards,
            "high": remaining_cards,
        },
        "shoe_stage": shoe_stage,
        "pattern_survival": {
            "score": 1.0,
            "mode": "disabled_formal_core",
            "direction_override": False,
        },
        "pattern_survival_score": 1.0,
        "run_length_hazard": {},
        "run_length_hazard_weight": 0.0,
        "probabilistic_shoe_estimate": shoe_estimate,
        "tie_risk_active": False,
        "direction_edge": edge,
        "direction_edge_percent": round(edge * 100.0, 4),
        "selected_expected_return": selected_ev,
        "selected_expected_return_percent": selected_ev * 100.0,
        "bet_allowed": True,
        "markov": markov_predict,
        "markov_probs": dict(
            fallback_markov.get("probabilities")
            or {"B": 0.5, "P": 0.5, "T": 0.0}
        ),
        "final_probs": probabilities,
        "direction_probs": probabilities,
        "economic_probs": probabilities,
        "final_probability": probabilities[direction],
        "economic_probability_for_direction": float(
            money.get("resolved_win_probability", confidence) or confidence
        ),
        "markov_predict": markov_predict,
        "probabilistic_shoe_predict": shoe_estimate,
        "fusion_decision": {
            "direction": direction,
            "probabilities": probabilities,
            "lstm_weight": lstm_weight,
            "shoe_composition_weight": shoe_direction_weight,
            "structure_weight": structure_weight,
            "cut_progress": cut_progress,
            "fallback_markov_weight": 0.0,
            "markov_prior_weight": 0.0,
            "road_forecaster_weight": 0.0,
            "run_length_hazard_likelihood_power": 0.0,
            "road_applied": False,
            "hazard_applied": False,
            "method": "single_lstm_shoe_cut_fusion",
            "semantics": "one_fused_logit_owns_direction",
            "details": lstm_fusion,
        },
        "fusion": {
            "method": "lstm_shoe_cut_fusion",
            "shoe_reliability": shoe_reliability,
            "road_reliability": 0.0,
            "hazard_reliability": 0.0,
            "lstm_reliability": float(neural.get("maturity", 0.0) or 0.0),
            "cut_progress": cut_progress,
        },
        "markov_state": {
            "state_key": "MARKOV_DIAGNOSTIC_ONLY",
            "direction_context": "diagnostic_only",
            "density": "unused_for_formal_direction",
            "tie_trigger": "TiesSkipped",
            "sample_count": len(big_road),
            "effective_support": float(
                fallback_markov.get("context_support", 0.0) or 0.0
            ),
            "state_count": 2,
            "selected_order": int(fallback_markov.get("selected_order", 1) or 1),
            "change_point": False,
            "shoe_stage": shoe_stage,
            "pattern_survival_score": 1.0,
        },
        "road_predict": road_diag,
        "road_support": road,
        "derived_road_analysis": dict(policy.get("regression_analysis") or {}),
        "road_fusion": {
            "applied": False,
            "mode": "diagnostic_only",
            "reliability": 0.0,
            "raw_reliability": 0.0,
            "pattern_survival_score": 1.0,
            "likelihood": None,
            "reason": "formal direction is owned by LSTM+shoe+cut fusion",
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
            "reason": "disabled in production LSTM-shoe-cut core",
        },
        "component_probabilities": component_probabilities,
        "money_management": money,
        "kelly_fraction": float(money.get("kelly_fraction", final_ratio) or final_ratio),
        "pre_tie_adjusted_ratio": float(
            money.get("pre_tie_adjusted_ratio", final_ratio) or final_ratio
        ),
        "adjusted_ratio": float(money.get("adjusted_ratio", final_ratio) or final_ratio),
        "final_bet_ratio": final_ratio,
        "bet_percentage": bet_percentage,
        "suggested_bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_amount": float(money.get("bet_amount", 0.0) or 0.0),
        "bet_multiplier": (
            min(1.0, final_ratio / MAX_BET_RATIO)
            if MAX_BET_RATIO > 0.0
            else 1.0
        ),
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
        "lstm_enabled": True,
        "lstm_primary": True,
        "lstm_shoe_cut_fusion": True,
        "lstm_model": lstm_model,
        "lstm_fallback_active": False,
        "lstm_fallback_reason": "",
        "cusum_linucb_enabled": False,
        "force_observe": False,
        "hard_brake_active": False,
        "post_reset_vacuum_active": False,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": "resolved_BP_probability_from_single_lstm_shoe_cut_fusion",
        "dynamic_prediction_policy": {
            "version": POLICY_VERSION,
            "lstm_primary": True,
            "lstm_shoe_cut_fusion": True,
            "fallback_active": False,
            "big_road_rounds": len(big_road),
            "forecast": policy,
            "penalty_observe": {"active": False, "force_observe": False},
            "exact_card_dependency": False,
            "exact_card_used_when_available": exact_shoe_used_for_direction,
            "shoe_probability_decision_weight": shoe_direction_weight,
            "ocr_or_screen_flow_modified": False,
            "formal_direction_source": formal_source,
            "card_composition_source": composition_source,
            "shoe_direction_authority": "inside_single_fusion",
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
            "reason": "single_lstm_shoe_cut_fusion_always_returns_BP",
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
    return result


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Predict with exact hidden-shoe composition, then reveal one virtual hand."""
    from particle_filter_points import counts_from_shoe, deal_ordered_hand

    hidden_shoe = [int(card) for card in list(session.get("virtual_shoe") or [])]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")
    history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(
        run_seed if run_seed is not None else secrets.randbits(32)
    ) & 0xFFFFFFFF
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
            "remaining_cards_source": "virtual_shoe_exact_counts",
            "source": "remaining_counts",
            "cut_card_remaining_cards": float(
                session.get("cut_card_remaining_cards", DEFAULT_CUT_CARD_REMAINING)
                or DEFAULT_CUT_CARD_REMAINING
            ),
        },
    )
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted = str(prediction.get("action") or "B").upper()
    actual = str(hand.outcome or "").upper()
    verdict = (
        "TIE_SKIPPED"
        if actual == "T"
        else ("HIT" if predicted == actual else "MISS")
    )
    update = record_online_feedback(
        scope_key=str(prediction.get("bandit_scope_key") or ""),
        action=predicted,
        context_vector=list(prediction.get("context_vector") or []),
        actual_outcome=actual,
    )
    prediction.update(
        {
            "ok": True,
            "mode": "virtual_shoe_lstm_shoe_cut_fusion",
            "virtual_hand": hand_data,
            "virtual_outcome": actual,
            "virtual_outcome_text": hand_data["outcome_text"],
            "verdict": verdict,
            "verdict_text": {
                "HIT": "命中",
                "MISS": "未命中",
                "TIE_SKIPPED": "和局不計",
            }[verdict],
            "cards_consumed": int(hand.cards_used),
            "remaining_cards_after": len(remaining_shoe),
            "remaining_counts_after": counts_from_shoe(remaining_shoe),
            "round_number": int(session.get("hand_number", 0) or 0) + 1,
            "bandit_learning_applied": bool(update.get("updated", False)),
            "bandit_update": update,
            "disclaimer": (
                "正式方向由單一 LSTM+精確牌靴+50-70 局切牌深度融合產生；"
                "Markov/LinUCB/hazard/HSMM 不參與正式方向。"
                "單局結果具有高變異，模型機率不代表保證獲利。"
            ),
        }
    )
    return {
        "prediction": prediction,
        "hand": hand_data,
        "remaining_shoe": remaining_shoe,
    }


def parse_point_observation(value: Any) -> None:
    del value
    return None


__all__ = ["parse_point_observation", "predict", "run_virtual_round"]
