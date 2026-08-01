"""Click-only virtual-shoe baccarat predictor.

The public entry point is ``run_virtual_round(session)``.  V7 uses an exact
finite-population hypergeometric core from remaining card counts, with Monte
Carlo and hidden-order particle validation.  The hidden ordered shoe is not
revealed until after the prediction is complete.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import os
import secrets

from particle_filter_points import (
    DB_HOLDOUT,
    EngineSettings,
    VirtualShoeParticleEngine,
    counts_from_shoe,
    deal_ordered_hand,
    fresh_counts,
)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


_ENGINE = VirtualShoeParticleEngine(
    EngineSettings(
        decks=_env_int("PF_DECKS", 8, 1, 16),
        particles=_env_int("PF_PARTICLES", 500, 64, 4000),
        replicas=_env_int("PF_REPLICAS", 5, 3, 11),
        simulations_per_replica=_env_int(
            "PF_PREDICT_SIMULATIONS_PER_REPLICA",
            1200,
            200,
            20_000,
        ),
        particle_draws_per_particle=_env_int(
            "PF_DRAWS_PER_PARTICLE",
            2,
            1,
            12,
        ),
    )
)


def _normalize_outcome_history(values: Iterable[Any]) -> List[str]:
    history: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = (
                item.get("outcome")
                or item.get("actual")
                or item.get("virtual_outcome")
            )
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            history.append(value)
    return history


def _normalize_path_history(values: Iterable[Any]) -> List[str]:
    history: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("draw_path") or item.get("path")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"N", "P", "B", "D"}:
            history.append(value)
    return history


def _prediction_label(prediction: Mapping[str, Any]) -> str:
    quality = float(prediction.get("quality_score", 0.0) or 0.0)
    if quality >= 0.78:
        return "較高"
    if quality >= 0.58:
        return "中等"
    return "偏低"


def run_virtual_round(
    session: Mapping[str, Any],
    run_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Predict first, then deal one hidden virtual round."""
    hidden_shoe = [
        int(card)
        for card in list(session.get("virtual_shoe") or [])
    ]
    if len(hidden_shoe) < 6:
        raise ValueError("虛擬牌靴不足，請重新建立牌靴。")

    remaining_counts = session.get("remaining_counts")
    if not isinstance(remaining_counts, list) or len(remaining_counts) != 10:
        remaining_counts = counts_from_shoe(hidden_shoe)

    round_history = list(session.get("round_history") or [])
    outcome_history = _normalize_outcome_history(round_history)
    path_history = _normalize_path_history(round_history)

    seed = int(
        run_seed if run_seed is not None else secrets.randbits(32)
    ) & 0xFFFFFFFF
    prediction = _ENGINE.analyze(
        remaining_counts=remaining_counts,
        history=outcome_history,
        draw_path_history=path_history,
        seed=seed,
    )

    # The hidden order is consumed only after the probability calculation.
    hand, remaining_shoe = deal_ordered_hand(hidden_shoe)
    hand_data = hand.as_dict()
    predicted_side = str(prediction.get("recommend") or "").upper()
    action = str(prediction.get("action") or "O").upper()
    actual = hand.outcome

    if action == "O":
        verdict = "OBSERVE"
    elif actual == "T" and predicted_side in {"B", "P"}:
        verdict = "TIE_SKIPPED"
    elif predicted_side == actual:
        verdict = "HIT"
    else:
        verdict = "MISS"

    prediction.update(
        {
            "model_version": "V7.0-HYPERGEOMETRIC-PARTICLE-MC",
            "mode": "virtual_shoe_click_only",
            "input_required": False,
            "confidence_label": _prediction_label(prediction),
            "virtual_hand": hand_data,
            "virtual_outcome": actual,
            "virtual_outcome_text": hand_data["outcome_text"],
            "verdict": verdict,
            "verdict_text": {
                "HIT": "命中",
                "MISS": "未命中",
                "TIE_SKIPPED": "和局不計",
                "OBSERVE": "觀望不計",
            }[verdict],
            "cards_consumed": hand.cards_used,
            "remaining_cards_after": len(remaining_shoe),
            "remaining_counts_after": counts_from_shoe(remaining_shoe),
            "shoe_id": str(session.get("shoe_id") or ""),
            "venue": str(session.get("venue") or ""),
            "room": str(session.get("room") or ""),
            "round_number": int(session.get("hand_number", 0) or 0) + 1,
            "warmup_rounds": int(session.get("warmup_rounds", 0) or 0),
            "disclaimer": (
                "此結果只對程式內建虛擬牌靴有效，"
                "與外部真人桌無資料連線。"
            ),
        }
    )

    return {
        "prediction": prediction,
        "hand": hand_data,
        "remaining_shoe": remaining_shoe,
    }


def predict(
    history: Union[str, Iterable[Any], None] = None,
    venue: str = "",
    room: str = "",
    shoe_id: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compatibility probability-only API."""
    context = dict(shoe_context or {})
    counts = context.get("remaining_counts")
    if not isinstance(counts, Sequence) or len(counts) != 10:
        counts = fresh_counts(_ENGINE.settings.decks)

    if history is None:
        history_values: List[Any] = []
    elif isinstance(history, str):
        history_values = [
            part
            for part in history.replace("|", ",").split(",")
            if part.strip()
        ]
    else:
        history_values = list(history)

    prediction = _ENGINE.analyze(
        remaining_counts=[int(value) for value in counts],
        history=_normalize_outcome_history(history_values),
        draw_path_history=_normalize_path_history(history_values),
        seed=run_seed,
    )
    prediction.update(
        {
            "venue": venue,
            "room": room,
            "shoe_id": shoe_id,
            "user_id": user_id,
            "input_required": False,
            "mode": "probability_only_compatibility",
            "model_version": "V7.0-HYPERGEOMETRIC-PARTICLE-MC",
        }
    )
    return prediction


def parse_point_observation(value: Any) -> Optional[Dict[str, Any]]:
    """Point input remains disabled in click-only V7."""
    return None


__all__ = [
    "DB_HOLDOUT",
    "parse_point_observation",
    "predict",
    "run_virtual_round",
]
