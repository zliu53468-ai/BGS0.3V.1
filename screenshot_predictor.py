"""遊戲截圖 cMAB 預測轉接層。

完整 B/P/T 歷史與牌路上下文送入 LinUCB cMAB。
不再建立估計牌組，也不使用 fresh_counts、粒子、超幾何、蒙地卡羅或 Stacking。

為了不修改 app.py 的非模型流程，本檔在人工回報後會先結算上一筆 reward，
再產生下一局預測；績效紀錄器會略過 app.py 稍後的重複結算呼叫。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import os
import secrets

from performance_tracker import record_prediction, resolve_latest_prediction
from predictor import predict
from road_model import build_road_context

PERFORMANCE_TRACKING_ENABLED = os.getenv("PERFORMANCE_TRACKING_ENABLED", "1").strip() == "1"


def _clean_raw(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in values:
        raw = item.get("outcome") if isinstance(item, Mapping) else item
        value = str(raw or "").upper().strip()
        if value in {"B", "P", "T"}:
            out.append(value)
    return out[-2000:]


def _clean_bp(values: Iterable[Any]) -> list[str]:
    return [value for value in _clean_raw(values) if value in {"B", "P"}][-1000:]


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

    seed = int(run_seed if run_seed is not None else secrets.randbits(32)) & 0xFFFFFFFF
    context = dict(road_context or build_road_context(
        combined_raw,
        seed=(seed ^ 0x9E3779B9) & 0xFFFFFFFF,
        grid_cells=list(initial_grid_cells or []),
        initial_image_count=len(initial_raw),
        manual_count=len(manual_raw),
    ))
    metadata = dict(screen_metadata or {})

    previous_resolution = None
    if PERFORMANCE_TRACKING_ENABLED and record_for_learning and user_id and manual_raw:
        previous_resolution = resolve_latest_prediction(
            user_id,
            manual_raw[-1],
            venue=venue,
            room=room,
            mark_duplicate_guard=True,
        )

    result = predict(
        history=combined_raw,
        venue=venue,
        room=room,
        user_id=user_id,
        run_seed=seed,
        shoe_context=None,
        road_context=context,
    )
    result.update({
        "model_version": "CMAB-LINUCB-V1-FULL-HISTORY-GRID-AWARE",
        "mode": "screen_full_history_contextual_bandit",
        "screen_remaining_cards": int(remaining_cards or 0),
        "estimated_remaining_counts": [],
        "composition_source": "not_used_cmab",
        "composition_quality": "not_applicable_cmab",
        "prior_counts_ignored": bool(prior_counts),
        "road_sequence_length": len(cleaned),
        "raw_outcome_length": len(combined_raw),
        "initial_image_count": len(initial_raw),
        "manual_round_count": len(manual_raw),
        "combined_round_count": len(combined_raw),
        "full_history_used_count": len(combined_raw),
        "initial_grid_cells": list(initial_grid_cells or []),
        "tie_count": sum(value == "T" for value in combined_raw),
        "tie_markers": dict(tie_markers or {}),
        "road_support": context,
        "road_pipeline_completed": True,
        "screen_metadata": metadata,
        "screen_input_type": str(metadata.get("input_type") or "unknown"),
        "virtual_only": False,
        "external_screen_input": True,
        "previous_prediction_resolved_before_next": bool(previous_resolution),
    })
    result["road_fusion"] = {
        "applied": True,
        "mode": "context_features_only",
        "reason": "牌路資訊作為 cMAB 上下文，不再以 Stacking 機率融合",
    }

    if PERFORMANCE_TRACKING_ENABLED and record_for_learning and user_id:
        result["prediction_id"] = record_prediction(
            user_id,
            result,
            venue=venue,
            room=room,
            metadata={
                **metadata,
                "initial_image_count": len(initial_raw),
                "manual_round_count": len(manual_raw),
                "combined_round_count": len(combined_raw),
            },
        )
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["predict_from_screenshot"]
