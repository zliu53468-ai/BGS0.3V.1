"""BGS predictor: image road parsing -> Quant Markov + road + shoe posterior.

Formal direction comes from BaccaratQuantEngine:
- support-aware variable-order Markov (1..4),
- adaptive entropy/regime forgetting,
- remaining-card-aware Pattern Survival calibration,
- standard derived-road Markov ask-road likelihood,
- run-length structural CONTINUE/TURN hazard likelihood,
- probabilistic shoe posterior optionally conditioned on a plausible screen depth,
- positive-Edge-only bankroll sizing.

The screen remaining-card total and road structure are probabilistic evidence only.
They do not identify hidden cards or guarantee the next baccarat outcome.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from baccarat_quant_engine import BaccaratQuantEngine
from markov_model import MODEL_VERSION
from money_management import MAX_BET_RATIO
from pattern_survival import build_remaining_card_state
from probabilistic_shoe_estimator import (
    MODEL_VERSION as SHOE_POSTERIOR_MODEL_VERSION,
    MAX_FUSION_WEIGHT as MAX_SHOE_FUSION_WEIGHT,
    estimate_probabilistic_shoe,
)
from road_model import ROAD_FEATURE_NAMES, build_road_context

OUTCOMES = ("B", "P", "T")
_QUANT_ENGINE = BaccaratQuantEngine()


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
    del user_id, run_seed

    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        compact = history.replace("|", "").replace(",", "").replace(" ", "").upper()
        if compact and all(char in OUTCOMES for char in compact):
            history_values = list(compact)
        else:
            history_values = [
                part for part in history.replace("|", ",").split(",") if part.strip()
            ]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)

    supplied_road = dict(road_context or {})
    if (
        isinstance(supplied_road.get("road_features"), list)
        and len(supplied_road.get("road_features") or []) == len(ROAD_FEATURE_NAMES)
    ):
        road = supplied_road
    else:
        road = build_road_context(
            cleaned,
            grid_cells=list(supplied_road.get("grid_cells") or []),
            initial_image_count=int(supplied_road.get("initial_image_count", 0) or 0),
            manual_count=int(supplied_road.get("manual_count", 0) or 0),
        )
        if supplied_road:
            road["scan_metadata"] = supplied_road

    context = dict(shoe_context or {})
    bankroll = max(0.0, float(context.get("bankroll", 0.0) or 0.0))

    try:
        decks = max(1, min(16, int(context.get("decks", 8) or 8)))
    except (TypeError, ValueError):
        decks = 8

    target_remaining_cards: Optional[int]
    try:
        raw_target = context.get("remaining_cards")
        target_remaining_cards = (
            int(raw_target) if raw_target is not None and int(raw_target) > 0 else None
        )
    except (TypeError, ValueError):
        target_remaining_cards = None

    try:
        depth_reliability = max(
            0.0,
            min(
                1.0,
                float(context.get("remaining_cards_reliability", 0.65) or 0.0),
            ),
        )
    except (TypeError, ValueError):
        depth_reliability = 0.65

    shoe_posterior = estimate_probabilistic_shoe(
        cleaned,
        decks=decks,
        target_remaining_cards=target_remaining_cards,
        depth_reliability=depth_reliability,
    )
    shoe_probabilities = dict(shoe_posterior["probabilities"])
    depth_constraint = dict(shoe_posterior.get("depth_constraint") or {})
    remaining_card_state = build_remaining_card_state(
        shoe_posterior,
        decks=decks,
    )

    external_shoe_probs = context.get("shoe_probs")
    if external_shoe_probs is not None:
        fusion_shoe_probs = external_shoe_probs
        shoe_reliability = max(
            0.0,
            min(1.0, float(context.get("shoe_reliability", 1.0) or 0.0)),
        )
        shoe_source = "external_shoe_probs"
    else:
        fusion_shoe_probs = shoe_probabilities
        shoe_reliability = float(shoe_posterior.get("fusion_weight", 0.0) or 0.0)
        shoe_source = (
            "probabilistic_shoe_particle_v3_depth_conditioned"
            if depth_constraint.get("applied")
            else "probabilistic_shoe_particle_v3_outcome_conditioned"
        )

    quant = _QUANT_ENGINE.predict(
        cleaned,
        shoe_probs=fusion_shoe_probs,
        shoe_reliability=shoe_reliability,
        remaining_card_state=remaining_card_state,
        bankroll=bankroll,
    )

    markov = dict(quant["markov"])
    markov_probabilities = dict(quant["markov_probs"])
    pattern_markov_probabilities = dict(
        quant.get("pattern_calibrated_markov_probs") or markov_probabilities
    )
    pattern_survival = dict(quant.get("pattern_survival") or {})
    survival_score = float(pattern_survival.get("score", 0.0) or 0.0)
    hazard_analysis = dict(quant.get("run_length_hazard") or {})
    shoe_stage = str(remaining_card_state.get("shoe_stage") or "UNKNOWN")
    markov_direction = str(markov["direction"])
    probabilities = dict(quant["final_probs"])
    direction = str(quant["direction"])
    confidence = float(
        quant.get("pattern_calibrated_final_weight", markov["final_weight"])
    )
    raw_markov_confidence = float(markov["final_weight"])
    text = "莊" if direction == "B" else "閒"
    money = dict(quant["money_management"])
    fusion = dict(quant["fusion"])
    quant_road = dict(quant.get("road_analysis") or {})

    road_predict = _road_diagnostic(road)
    system_model_version = (
        f"{MODEL_VERSION}+{SHOE_POSTERIOR_MODEL_VERSION}+PATTERN-SURVIVAL+RUN-HAZARD+QUANT"
    )
    fingerprint = sha256(
        "|".join((
            "".join(cleaned),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
            system_model_version,
        )).encode("utf-8")
    ).hexdigest()[:24]

    p_b = float(probabilities["B"])
    p_p = float(probabilities["P"])
    p_t = float(probabilities["T"])
    edge = float(money.get("edge", 0.0) or 0.0)
    bet_allowed = bool(money.get("bet_allowed", False))
    tie_risk_active = bool(money.get("tie_risk_active", p_t > 0.15))
    shoe_weight = float(fusion.get("shoe_reliability", shoe_reliability) or 0.0)
    road_weight = float(fusion.get("road_reliability", 0.0) or 0.0)
    hazard_weight = float(fusion.get("hazard_reliability", 0.0) or 0.0)
    road_fusion_detail = dict(fusion.get("road") or {})
    hazard_fusion_detail = dict(fusion.get("hazard") or {})
    road_fusion_applied = bool(road_fusion_detail.get("applied", False))
    hazard_fusion_applied = bool(hazard_fusion_detail.get("applied", False))

    if bet_allowed:
        signal_status_code = "POSITIVE_EDGE_DYNAMIC_BET"
        signal_status_text = (
            f"正 Edge；依 Edge×波動係數動態配置 "
            f"{float(money['bet_percentage']):.2f}%"
        )
    else:
        signal_status_code = "NO_BET_NONPOSITIVE_EDGE"
        signal_status_text = "目前 Edge ≤ 0，保留方向預測但不配置資金"

    return {
        "ok": True,
        "engine": "BACCARAT_QUANT_MARKOV_PATTERN_SURVIVAL_ROAD_RUN_HAZARD_DEPTH_SHOE",
        "model_version": MODEL_VERSION,
        "shoe_posterior_model_version": SHOE_POSTERIOR_MODEL_VERSION,
        "system_model_version": system_model_version,
        "model_variant": (
            "SUPPORT_BACKOFF_MARKOV_PLUS_REMAINING_CARD_PATTERN_SURVIVAL_"
            "PLUS_DERIVED_ROAD_PLUS_RUN_LENGTH_HAZARD_PLUS_DEPTH_SHOE_EDGE_GATED"
        ),
        "model_core": "baccarat_quant_engine",
        "decision_pipeline": (
            "image_scan_to_support_backoff_markov_to_remaining_card_stage_"
            "to_pattern_survival_plus_standard_derived_road_plus_run_length_hazard_"
            "to_depth_conditioned_probabilistic_shoe_to_bayesian_fusion_"
            "to_positive_edge_capital_sizing"
        ),
        "prediction_fingerprint": fingerprint,
        "probabilities": {"B": p_b, "P": p_p, "T": p_t},
        "raw_direction_probabilities": {"B": p_b, "P": p_p},
        "banker_rate": round(p_b * 100.0, 2),
        "player_rate": round(p_p * 100.0, 2),
        "tie_rate": round(p_t * 100.0, 2),
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
        "signal_allowed": bet_allowed,
        "risk_gate_open": bet_allowed,
        "mandatory_bet": False,
        "signal_status_code": signal_status_code,
        "signal_status_text": signal_status_text,
        "signal_reason": (
            f"state={markov['state_key'] or 'START'}；"
            f"order={int(markov.get('selected_order', 0))}；"
            f"supportK={int(markov.get('support_threshold', 4))}；"
            f"backoff={int(markov.get('backoff_steps', 0))}；"
            f"lambda={float(markov.get('retention_lambda', markov['decay'])):.3f}；"
            f"change={bool(markov.get('regime_profile', {}).get('change_point', False))}；"
            f"shoe_stage={shoe_stage}；"
            f"pattern_survival={survival_score:.3f}；"
            f"remaining≈{float(remaining_card_state.get('mean_remaining_cards', 0.0) or 0.0):.1f}；"
            f"hazard_turn={float(hazard_analysis.get('turn_probability', 0.5) or 0.5):.3f}；"
            f"hazard_w={hazard_weight:.3f}；"
            f"shoe_source={shoe_source}；"
            f"shoe_w={shoe_weight:.3f}；"
            f"depth={bool(depth_constraint.get('applied', False))}；"
            f"road_w={road_weight:.3f}；"
            f"edge={edge:.4f}；"
            f"sizing={money['reason']}。"
        ),
        "direction_source": (
            "pattern_calibrated_markov_plus_shoe_plus_derived_road_plus_run_length_hazard"
        ),
        "confidence": confidence,
        "raw_markov_confidence": raw_markov_confidence,
        "pattern_calibrated_confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": (
            "較高" if confidence >= 0.40 else
            "中等" if confidence >= 0.22 else "偏低"
        ),
        "entropy_bits": float(markov["entropy_bits"]),
        "entropy_base_weight": float(markov["base_weight"]),
        "shoe_progress": float(markov["shoe_progress"]),
        "shoe_depth_estimate": dict(markov["shoe_depth"]),
        "remaining_card_state": remaining_card_state,
        "estimated_remaining_cards": float(
            remaining_card_state.get("mean_remaining_cards", 0.0) or 0.0
        ),
        "estimated_remaining_interval": {
            "low": float(
                remaining_card_state.get("plausible_interval_low", 0.0) or 0.0
            ),
            "high": float(
                remaining_card_state.get("plausible_interval_high", 0.0) or 0.0
            ),
        },
        "shoe_stage": shoe_stage,
        "pattern_survival": pattern_survival,
        "pattern_survival_score": survival_score,
        "run_length_hazard": hazard_analysis,
        "run_length_hazard_weight": hazard_weight,
        "probabilistic_shoe_estimate": dict(shoe_posterior),
        "tie_risk_active": tie_risk_active,
        "direction_edge": edge,
        "direction_edge_percent": round(edge * 100.0, 4),
        "bet_allowed": bet_allowed,
        "markov_predict": {
            "direction": markov_direction,
            "probabilities": {
                "B": float(markov_probabilities["B"]),
                "P": float(markov_probabilities["P"]),
                "T": float(markov_probabilities["T"]),
            },
            "pattern_calibrated_probabilities": {
                "B": float(pattern_markov_probabilities["B"]),
                "P": float(pattern_markov_probabilities["P"]),
                "T": float(pattern_markov_probabilities["T"]),
            },
            "state": dict(markov["state"]),
            "state_key": str(markov["state_key"]),
            "transition_counts": dict(markov["transition_counts"]),
            "effective_support": float(markov["effective_support"]),
            "entropy_bits": float(markov["entropy_bits"]),
            "base_weight": float(markov["base_weight"]),
            "raw_final_weight": raw_markov_confidence,
            "final_weight": confidence,
            "pattern_survival_score": survival_score,
            "decay": float(markov["decay"]),
            "retention_lambda": float(markov.get("retention_lambda", markov["decay"])),
            "decay_intensity": float(markov.get("decay_intensity", 0.0)),
            "selected_order": int(markov.get("selected_order", 0)),
            "support_threshold": int(markov.get("support_threshold", 4)),
            "backoff_steps": int(markov.get("backoff_steps", 0)),
            "backoff_penalty": float(markov.get("backoff_penalty", 1.0)),
            "focus_applied": bool(markov.get("focus_applied", False)),
            "regime_profile": dict(markov.get("regime_profile") or {}),
            "prior": dict(markov["prior"]),
            "prior_strength": float(markov["prior_strength"]),
        },
        "probabilistic_shoe_predict": {
            "direction": str(shoe_posterior["direction"]),
            "probabilities": {
                "B": float(shoe_probabilities["B"]),
                "P": float(shoe_probabilities["P"]),
                "T": float(shoe_probabilities["T"]),
            },
            "bp_conditional_probabilities": dict(
                shoe_posterior.get("bp_conditional_probabilities") or {}
            ),
            "expected_remaining_cards": float(shoe_posterior["expected_remaining_cards"]),
            "expected_remaining_decks": float(shoe_posterior["expected_remaining_decks"]),
            "expected_remaining_counts": list(shoe_posterior["expected_remaining_counts"]),
            "remaining_count_std": list(shoe_posterior["remaining_count_std"]),
            "remaining_card_state": remaining_card_state,
            "conditioned_rounds": int(shoe_posterior["conditioned_rounds"]),
            "particle_count": int(shoe_posterior["particle_count"]),
            "reliability": float(shoe_posterior["reliability"]),
            "fusion_weight": float(shoe_weight),
            "target_remaining_cards": shoe_posterior.get("target_remaining_cards"),
            "depth_constraint_applied": bool(
                shoe_posterior.get("depth_constraint_applied", False)
            ),
            "depth_constraint": depth_constraint,
            "shoe_tendency": dict(shoe_posterior.get("shoe_tendency") or {}),
            "inference_semantics": str(shoe_posterior["inference_semantics"]),
        },
        "fusion_decision": {
            "direction": direction,
            "probabilities": {"B": p_b, "P": p_p, "T": p_t},
            "raw_markov_prior": {
                "B": float(markov_probabilities["B"]),
                "P": float(markov_probabilities["P"]),
                "T": float(markov_probabilities["T"]),
            },
            "pattern_calibrated_markov_prior": {
                "B": float(pattern_markov_probabilities["B"]),
                "P": float(pattern_markov_probabilities["P"]),
                "T": float(pattern_markov_probabilities["T"]),
            },
            "pattern_survival_score": survival_score,
            "markov_prior_weight": 1.0,
            "probabilistic_shoe_likelihood_power": float(shoe_weight),
            "derived_road_likelihood_power": float(road_weight),
            "run_length_hazard_likelihood_power": float(hazard_weight),
            "markov_weight": 1.0,
            "probabilistic_shoe_weight": float(shoe_weight),
            "max_probabilistic_shoe_weight": float(MAX_SHOE_FUSION_WEIGHT),
            "shoe_source": shoe_source,
            "depth_constraint_applied": bool(depth_constraint.get("applied", False)),
            "road_applied": road_fusion_applied,
            "hazard_applied": hazard_fusion_applied,
            "method": str(
                fusion.get("method")
                or "pattern_calibrated_sequential_tempered_bayes_with_run_hazard"
            ),
            "semantics": (
                "posterior_proportional_to_pattern_calibrated_markov_prior_"
                "times_shoe_times_derived_road_times_run_length_hazard_likelihoods"
            ),
        },
        "markov_state": {
            "state_key": str(markov["state_key"]),
            "direction_context": str(markov["state"].get("direction_context") or ""),
            "density": str(markov["state"].get("density") or "Medium"),
            "tie_trigger": str(markov["state"].get("tie_trigger") or "NoTie"),
            "sample_count": int(markov["sample_count"]),
            "effective_support": float(markov["effective_support"]),
            "state_count": int(markov["state_count"]),
            "selected_order": int(markov.get("selected_order", 0)),
            "change_point": bool(
                dict(markov.get("regime_profile") or {}).get("change_point", False)
            ),
            "shoe_stage": shoe_stage,
            "pattern_survival_score": survival_score,
        },
        "road_predict": road_predict,
        "road_support": road,
        "derived_road_analysis": quant_road,
        "road_fusion": {
            "applied": road_fusion_applied,
            "mode": (
                "pattern_survival_gated_standard_derived_road_markov_likelihood"
                if road_fusion_applied
                else "derived_road_not_mature_or_pattern_survival_too_low"
            ),
            "reliability": float(road_weight),
            "raw_reliability": float(fusion.get("raw_road_reliability", road_weight) or 0.0),
            "pattern_survival_score": survival_score,
            "likelihood": road_fusion_detail.get("likelihood"),
            "reason": (
                "標準下三路由完整大路重建；R/U Markov 問路 likelihood 先受 Pattern Survival "
                "降權後才參與 Quant Engine，避免失效規律與主 Markov 重複放大。"
                if road_fusion_applied
                else "下三路樣本尚未成熟、規律存活度過低或沒有可用問路 likelihood。"
            ),
        },
        "run_length_hazard_fusion": {
            "applied": hazard_fusion_applied,
            "reliability": float(hazard_weight),
            "raw_reliability": float(fusion.get("raw_hazard_reliability", hazard_weight) or 0.0),
            "likelihood": hazard_fusion_detail.get("likelihood"),
            "continue_probability": float(
                hazard_analysis.get("continue_probability", 0.5) or 0.5
            ),
            "turn_probability": float(
                hazard_analysis.get("turn_probability", 0.5) or 0.5
            ),
            "selected_context": str(hazard_analysis.get("selected_context") or ""),
            "support": float(hazard_analysis.get("support", 0.0) or 0.0),
            "reason": (
                "柱高模型依已完成大路柱學習 CONTINUE/TURN hazard；只以封頂 likelihood power "
                "參與融合，不把樓梯路、長龍等名稱寫死成必然規則。"
            ),
        },
        "component_probabilities": {
            "markov_raw": {
                "B": float(markov_probabilities["B"]),
                "P": float(markov_probabilities["P"]),
                "T": float(markov_probabilities["T"]),
            },
            "markov_pattern_calibrated": {
                "B": float(pattern_markov_probabilities["B"]),
                "P": float(pattern_markov_probabilities["P"]),
                "T": float(pattern_markov_probabilities["T"]),
            },
            "probabilistic_shoe": {
                "B": float(shoe_probabilities["B"]),
                "P": float(shoe_probabilities["P"]),
                "T": float(shoe_probabilities["T"]),
            },
            "derived_road_likelihood": dict(
                road_fusion_detail.get("likelihood") or {}
            ),
            "run_length_hazard_likelihood": dict(
                hazard_fusion_detail.get("likelihood") or {}
            ),
            "fused": {"B": p_b, "P": p_p, "T": p_t},
            "road_diagnostic": {
                "B": float(road_predict["banker_probability"]),
                "P": float(road_predict["player_probability"]),
                "T": 0.0,
            },
        },
        "money_management": dict(money),
        "kelly_fraction": float(money["kelly_fraction"]),
        "pre_tie_adjusted_ratio": float(money["pre_tie_adjusted_ratio"]),
        "adjusted_ratio": float(money["adjusted_ratio"]),
        "final_bet_ratio": float(money["final_bet_ratio"]),
        "bet_percentage": float(money["bet_percentage"]),
        "suggested_bet_amount": float(money["bet_amount"]),
        "bet_amount": float(money["bet_amount"]),
        "bet_multiplier": (
            min(1.0, float(money["final_bet_ratio"]) / MAX_BET_RATIO)
            if MAX_BET_RATIO > 0.0 else 0.0
        ),
        "context_vector": list(road.get("road_features") or []),
        "bandit_context": [],
        "context_feature_names": list(ROAD_FEATURE_NAMES),
        "context_dim": len(list(road.get("road_features") or [])),
        "contextual_bandit_enabled": False,
        "contextual_bandit_update_enabled": False,
        "cusum_linucb_enabled": False,
        "force_observe": False,
        "hard_brake_active": False,
        "post_reset_vacuum_active": False,
        "input_required": False,
        "venue": str(venue or ""),
        "room": str(room or ""),
        "shoe_id": str(shoe_id or ""),
        "probability_semantics": (
            "remaining_card_pattern_calibrated_support_backoff_markov_times_"
            "depth_conditioned_shoe_times_bounded_derived_road_times_bounded_"
            "run_length_hazard_likelihood_not_guaranteed_outcome"
        ),
    }


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
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
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
    verdict = (
        "TIE_SKIPPED" if actual == "T" else
        "HIT" if predicted_side == actual else "MISS"
    )

    prediction.update({
        "ok": True,
        "mode": "virtual_shoe_baccarat_quant_compatibility",
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
        "bandit_learning_applied": False,
        "disclaimer": (
            "虛擬相容模式使用 Support-aware Markov + Remaining-card Pattern Survival + "
            "Derived-road Markov + Run-Length Hazard + depth-conditioned probabilistic shoe；"
            "資金配置需 Edge > 0。"
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
