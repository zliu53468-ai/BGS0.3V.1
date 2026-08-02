"""遊戲截圖 B/P/T 全歷史預測轉接層 V10.3。

重點：圖片初始化歷史與後續人工輸入分開保存，再合併成模型完整歷史；
全盤牌路模型可取得初始格位與辨識品質，不再只看最新一局。
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


def _clean_raw(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in values:
        raw = item.get("outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _clean_bp(values: Iterable[Any]) -> List[str]:
    return [v for v in _clean_raw(values) if v in {"B", "P"}][-1000:]


def _largest_remainder_allocation(weights: Sequence[float], total: int) -> List[int]:
    positive = [max(0.0, float(v)) for v in weights]
    if total <= 0:
        return [0] * len(positive)
    weight_sum = sum(positive) or float(len(positive))
    if sum(positive) <= 0:
        positive = [1.0] * len(positive)
    exact = [v / weight_sum * total for v in positive]
    floors = [int(v) for v in exact]
    for index in sorted(range(len(exact)), key=lambda i: exact[i] - floors[i], reverse=True)[: total - sum(floors)]:
        floors[index] += 1
    return floors


def estimate_point_counts(remaining_cards: int, *, prior_counts: Optional[Sequence[int]] = None, decks: int = 8) -> Tuple[List[int], str]:
    total = max(6, min(52 * max(1, min(16, int(decks))), int(remaining_cards)))
    if isinstance(prior_counts, Sequence) and len(prior_counts) == 10 and sum(max(0, int(v)) for v in prior_counts) >= 6:
        return _largest_remainder_allocation([max(0, int(v)) for v in prior_counts], total), "session_scaled"
    return _largest_remainder_allocation(fresh_counts(decks), total), "fresh_shoe_scaled"


def predict_from_screenshot(
    sequence: Iterable[Any], *, remaining_cards: Optional[int],
    raw_outcomes: Optional[Iterable[Any]] = None,
    tie_markers: Optional[Mapping[str, Any]] = None,
    prior_counts: Optional[Sequence[int]] = None,
    venue: str = "", room: str = "", user_id: str = "",
    run_seed: Optional[int] = None,
    road_context: Optional[Mapping[str, Any]] = None,
    screen_metadata: Optional[Mapping[str, Any]] = None,
    initial_grid_cells: Optional[Sequence[Mapping[str, Any]]] = None,
    initial_image_history: Optional[Iterable[Any]] = None,
    manual_outcome_history: Optional[Iterable[Any]] = None,
    record_for_learning: bool = True,
) -> Dict[str, Any]:
    initial_raw = _clean_raw(initial_image_history or [])
    manual_raw = _clean_raw(manual_outcome_history or [])
    supplied_raw = _clean_raw(raw_outcomes or [])
    combined_raw = initial_raw + manual_raw if (initial_raw or manual_raw) else supplied_raw
    if not combined_raw:
        combined_raw = _clean_raw(sequence)
    cleaned = _clean_bp(combined_raw)

    fallback_total = sum(int(v) for v in prior_counts or [] if int(v) >= 0)
    total = int(remaining_cards or fallback_total or 416)
    counts, source = estimate_point_counts(total, prior_counts=prior_counts, decks=8)
    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    context = dict(road_context or build_road_context(
        combined_raw,
        seed=(seed ^ 0x9E3779B9) & 0xFFFFFFFF,
        grid_cells=list(initial_grid_cells or []),
        initial_image_count=len(initial_raw),
        manual_count=len(manual_raw),
    ))
    metadata = dict(screen_metadata or {})
    result = predict(
        history=combined_raw, venue=venue, room=room, user_id=user_id, run_seed=seed,
        shoe_context={"remaining_counts": counts}, road_context=context,
    )
    result.update({
        "model_version": "V10.3-FULL-HISTORY-GRID-AWARE",
        "mode": "screen_full_history_grid_aware",
        "screen_remaining_cards": total,
        "estimated_remaining_counts": counts,
        "composition_source": source,
        "composition_quality": "estimated",
        "road_sequence_length": len(cleaned),
        "raw_outcome_length": len(combined_raw),
        "initial_image_count": len(initial_raw),
        "manual_round_count": len(manual_raw),
        "combined_round_count": len(combined_raw),
        "full_history_used_count": len(combined_raw),
        "initial_grid_cells": list(initial_grid_cells or []),
        "tie_count": sum(1 for v in combined_raw if v == "T"),
        "tie_markers": dict(tie_markers or {}),
        "road_support": context,
        "road_pipeline_completed": True,
        "screen_metadata": metadata,
        "screen_input_type": str(metadata.get("input_type") or "unknown"),
        "virtual_only": False,
        "external_screen_input": True,
    })
    result["road_fusion"] = dict(result.get("road_integration") or {})
    if PERFORMANCE_TRACKING_ENABLED and record_for_learning and user_id:
        result["prediction_id"] = record_prediction(user_id, result, venue=venue, room=room, metadata={
            **metadata, "initial_image_count": len(initial_raw), "manual_round_count": len(manual_raw),
            "combined_round_count": len(combined_raw),
        })
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["estimate_point_counts", "predict_from_screenshot"]
