"""BGS direct predictor: image road analysis -> Markov -> B/P.

Only two predictive layers remain:
1. road_model.py converts recognized road history into the original 21 road features;
2. markov_model.py is the primary next-hand B/P predictor using the original 8 Markov
   probabilities, with only a small road calibration.

There is no Adaptive Ensemble, Stacking, LinUCB, CUSUM, or cross-resonance layer.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import secrets

from markov_model import MODEL_VERSION, predict_markov
from road_model import build_road_context


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
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-2000:]


def _tie_probability(history: List[str]) -> float:
    prior = 0.095156
    strength = 40.0
    p = (history.count("T") + prior * strength) / (len(history) + strength)
    return max(0.04, min(0.18, float(p)))


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
    del user_id, run_seed, shoe_context
    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [
            part for part in history.replace("|", ",").split(",") if part.strip()
        ]
    else:
        history_values = list(history)
    cleaned = _normalize_outcome_history(history_values)

    supplied_road = dict(road_context or {})
    if (
        isinstance(supplied_road.get("road_features"), list)
        and len(supplied_road.get("road_features") or []) == 21
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

    markov = predict_markov(cleaned, road_context=road)
    conditional_b = float(markov["banker_probability"])
    conditional_p = float(markov["player_probability"])
    direction = str(markov["direction"])
    confidence = float(markov["confidence"])

    tie = _tie_probability(cleaned)
    bp_mass = 1.0 - tie
    banker_probability = bp_mass * conditional_b
    player_probability = bp_mass * conditional_p
    text = "莊" if direction == "B" else "閒"

    fingerprint = sha256(
        "|".join((
            "".join(cleaned),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
            str(shoe_id or "").strip(),
            MODEL_VERSION,
        )).encode("utf-8")
    ).hexdigest()[:24]

    return {
        "ok": True,
        "engine": "ROAD_MARKOV_DIRECT",
        "model_version": MODEL_VERSION,
        "model_variant": "ROAD_21D_PLUS_MARKOV_8D_DIRECT",
        "model_core": "markov_primary_road_calibration",
        "decision_pipeline": "image_scan_to_road21_to_markov8_to_direct_bp",
        "prediction_fingerprint": fingerprint,
        "probabilities": {
            "B": float(banker_probability),
            "P": float(player_probability),
            "T": float(tie),
        },
        "raw_direction_probabilities": {
            "B": conditional_b,
            "P": conditional_p,
        },
        "banker_rate": round(banker_probability * 100.0, 2),
        "player_rate": round(player_probability * 100.0, 2),
        "tie_rate": round(tie * 100.0, 2),
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
        "signal_allowed": True,
        "signal_status_code": "DIRECT_MARKOV_DIRECTION",
        "signal_status_text": "Markov 主模型直接預測",
        "signal_reason": (
            f"Markov state={markov['markov_predict']['state'] or '-'}；"
            f"Markov 權重={markov['fusion']['markov_weight']:.3f}；"
            f"Road 校準權重={markov['fusion']['road_weight']:.3f}。"
        ),
        "direction_source": "markov_primary_road_calibrated",
        "confidence": confidence,
        "ensemble_confidence": confidence,
        "quality_score": confidence,
        "confidence_label": (
            "較高" if confidence >= 0.68 else
            "中等" if confidence >= 0.50 else "偏低"
        ),
        "bet_multiplier": min(1.0, max(0.35, 0.35 + 0.65 * confidence)),
        "direction_edge": float(markov["edge"]),
        "direction_edge_percent": round(float(markov["edge"]) * 100.0, 4),
        "road_predict": dict(markov["road_predict"]),
        "markov_predict": dict(markov["markov_predict"]),
        "markov_features": dict(markov["markov_features"]),
        "markov_state": dict(markov["markov_state"]),
        "road_markov_weights": dict(markov["fusion"]),
        "road_support": road,
        "component_probabilities": {
            "road": {
                "B": float(markov["road_predict"]["banker_probability"]),
                "P": float(markov["road_predict"]["player_probability"]),
                "T": 0.0,
            },
            "markov": {
                "B": float(markov["markov_predict"]["banker_probability"]),
                "P": float(markov["markov_predict"]["player_probability"]),
                "T": 0.0,
            },
        },
        "context_vector": list(markov["context_vector"]),
        "bandit_context": list(markov["context_vector"]),
        "context_feature_names": list(markov["context_feature_names"]),
        "context_dim": int(markov["context_dim"]),
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
        "probability_semantics": "model_direction_score_not_guaranteed_outcome_probability",
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
    outcome_history = _normalize_outcome_history(list(session.get("round_history") or []))
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    prediction = predict(
        history=outcome_history,
        venue=str(session.get("venue") or ""),
        room=str(session.get("room") or ""),
        shoe_id=str(session.get("shoe_id") or ""),
        run_seed=seed,
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
        "mode": "virtual_shoe_road_markov_compatibility",
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
        "disclaimer": "虛擬相容模式方向使用 Road + Markov Direct。",
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
