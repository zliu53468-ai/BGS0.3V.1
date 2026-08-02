"""遊戲截圖的 B/P/T 牌路先行預測轉接層。

- raw_outcomes 保存每一局 B/P/T。
- road_sequence 只保留 B/P，和局不新增大路格位。
- 牌路 context、有限牌組核心、自適應集成與三方校準完成後，建立待結算預測。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import os
import secrets

from particle_filter_points import fresh_counts
from performance_tracker import record_prediction
from predictor import predict
from road_model import build_road_context


PERFORMANCE_TRACKING_ENABLED = os.getenv("PERFORMANCE_TRACKING_ENABLED", "1").strip() == "1"


def _clean_raw_outcomes(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in values:
        if isinstance(item, Mapping):
            raw = item.get("outcome") or item.get("actual")
        else:
            raw = item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result[-1000:]


def _clean_sequence(values: Iterable[Any]) -> List[str]:
    return [value for value in _clean_raw_outcomes(values) if value in {"B", "P"}][-500:]


def _largest_remainder_allocation(weights: Sequence[float], total: int) -> List[int]:
    if total <= 0:
        return [0] * len(weights)
    positive = [max(0.0, float(value)) for value in weights]
    weight_sum = sum(positive)
    if weight_sum <= 0:
        positive = [1.0] * len(weights)
        weight_sum = float(len(weights))
    exact = [value / weight_sum * total for value in positive]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - floors[index], positive[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        floors[index] += 1
    return floors


def estimate_point_counts(
    remaining_cards: int,
    *,
    prior_counts: Optional[Sequence[int]] = None,
    decks: int = 8,
) -> Tuple[List[int], str]:
    maximum = 52 * max(1, min(16, int(decks)))
    total = max(6, min(maximum, int(remaining_cards)))
    if (
        isinstance(prior_counts, Sequence)
        and len(prior_counts) == 10
        and sum(max(0, int(value)) for value in prior_counts) >= 6
    ):
        weights = [max(0, int(value)) for value in prior_counts]
        source = "session_scaled"
    else:
        weights = fresh_counts(decks)
        source = "fresh_shoe_scaled"
    return _largest_remainder_allocation(weights, total), source


def predict_from_screenshot(
    sequence: Iterable[Any],
    *,
    remaining_cards: Optional[int],
    raw_outcomes: Optional[Iterable[Any]] = None,
    tie_markers: Optional[Mapping[str, Any]] = None,
    prior_counts: Optional[Sequence[int]] = None,
    venue: str = "",
    room: str = "",
    user_id: str = "",
    run_seed: Optional[int] = None,
    road_context: Optional[Mapping[str, Any]] = None,
    screen_metadata: Optional[Mapping[str, Any]] = None,
    record_for_learning: bool = True,
) -> Dict[str, Any]:
    cleaned = _clean_sequence(sequence)
    raw_history = _clean_raw_outcomes(raw_outcomes if raw_outcomes is not None else cleaned)
    if not raw_history:
        raw_history = list(cleaned)
    fallback_total = sum(int(value) for value in prior_counts or [] if int(value) >= 0)
    total = int(remaining_cards or fallback_total or 416)
    counts, source = estimate_point_counts(total, prior_counts=prior_counts, decks=8)

    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    road_seed = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    context = dict(road_context or build_road_context(raw_history, seed=road_seed))
    metadata = dict(screen_metadata or {})
    markers = {str(key): max(0, int(value or 0)) for key, value in dict(tie_markers or {}).items()}

    result = predict(
        history=raw_history,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=seed,
        shoe_context={"remaining_counts": counts},
        road_context=context,
    )
    result.update({
        "model_version": "V9.7-BPT-TIE-AWARE-CALIBRATED",
        "mode": "screen_estimated_composition_bpt_road_first",
        "model_core": "B/P/T完整歷史＋牌路先行＋有限牌組超幾何＋自適應校準",
        "screen_remaining_cards": total,
        "estimated_remaining_counts": counts,
        "composition_source": source,
        "composition_quality": "estimated",
        "road_sequence_length": len(cleaned),
        "raw_outcome_length": len(raw_history),
        "tie_count": sum(1 for value in raw_history if value == "T"),
        "tie_markers": markers,
        "road_support": context,
        "road_pipeline_completed": True,
        "screen_metadata": metadata,
        "screen_input_type": str(metadata.get("input_type") or "unknown"),
        "room_source": str(metadata.get("room_source") or "unknown"),
        "venue_source": str(metadata.get("venue_source") or "unknown"),
        "virtual_only": False,
        "external_screen_input": True,
        "disclaimer": (
            "截圖未包含每個點值的真實剩餘張數；系統保存 B/P/T 實際結果並做保守校準，"
            "但無法取得真人桌尚未公開的隱藏牌序。"
        ),
    })
    result["road_fusion"] = dict(result.get("road_integration") or {})

    if PERFORMANCE_TRACKING_ENABLED and record_for_learning and user_id:
        prediction_id = record_prediction(
            user_id,
            result,
            venue=venue,
            room=room,
            metadata={
                **metadata,
                "road_sequence_length": len(cleaned),
                "raw_outcome_length": len(raw_history),
                "tie_count": result["tie_count"],
            },
        )
        result["prediction_id"] = prediction_id
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["estimate_point_counts", "predict_from_screenshot"]
