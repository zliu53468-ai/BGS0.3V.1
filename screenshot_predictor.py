"""Screenshot adapter for the three-way Markov + shoe-depth predictor.

The image pipeline preserves chronological B/P/T outcomes. No tie is filtered before
prediction. Performance tracking remains audit-only and does not update model weights.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from performance_tracker import record_prediction, resolve_latest_prediction
from predictor import predict
from road_model import build_road_context

PERFORMANCE_TRACKING_ENABLED = True


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
    observed_cards: Optional[Iterable[Any]] = None,
    shoe_context: Optional[Mapping[str, Any]] = None,
    venue: str = "", room: str = "", shoe_id: str = "", user_id: str = "",
    run_seed: Optional[int] = None,
    road_context: Optional[Mapping[str, Any]] = None,
    screen_metadata: Optional[Mapping[str, Any]] = None,
    initial_grid_cells: Optional[Sequence[Mapping[str, Any]]] = None,
    initial_image_history: Optional[Iterable[Any]] = None,
    manual_outcome_history: Optional[Iterable[Any]] = None,
    previous_prediction_id: str = "",
    latest_actual_outcome: str = "",
    record_for_learning: bool = True,
) -> Dict[str, Any]:
    initial_raw = _clean_raw(initial_image_history or [])
    manual_raw = _clean_raw(manual_outcome_history or [])
    supplied_raw = _clean_raw(raw_outcomes or [])
    combined_raw = initial_raw + manual_raw if (initial_raw or manual_raw) else supplied_raw
    if not combined_raw:
        combined_raw = _clean_raw(sequence)
    cleaned_bp = _clean_bp(combined_raw)

    if run_seed is None:
        seed_payload = "|".join((
            "".join(combined_raw),
            str(venue or "").upper().strip(),
            str(room or "").strip(),
        ))
        seed = int.from_bytes(
            sha256(seed_payload.encode("utf-8")).digest()[:4],
            byteorder="big",
            signed=False,
        )
    else:
        seed = int(run_seed) & 0xFFFFFFFF

    scan_context = dict(road_context or {})
    context = build_road_context(
        combined_raw,
        seed=(seed ^ 0x9E3779B9) & 0xFFFFFFFF,
        grid_cells=list(initial_grid_cells or []),
        initial_image_count=len(initial_raw),
        manual_count=len(manual_raw),
    )
    if scan_context:
        context["scan_metadata"] = scan_context

    metadata = dict(screen_metadata or {})
    latest_actual = str(latest_actual_outcome or "").upper().strip()
    latest_actual_is_new = latest_actual in {"B", "P", "T"}
    previous_resolution = None
    if (
        PERFORMANCE_TRACKING_ENABLED
        and record_for_learning
        and user_id
        and latest_actual_is_new
    ):
        previous_resolution = resolve_latest_prediction(
            user_id,
            latest_actual,
            venue=venue,
            room=room,
            prediction_id=str(previous_prediction_id or ""),
        )

    prior_counts_ignored = bool(prior_counts)
    observed_cards_ignored = bool(observed_cards)

    # Preserve bankroll from the app session for MoneyManagementModel. Exact card
    # counts may remain present for legacy APIs, but round-count ShoeDepthEstimator
    # does not treat them as physical composition evidence.
    model_shoe_context = dict(shoe_context or {})
    result = predict(
        history=combined_raw,
        venue=venue,
        room=room,
        shoe_id=shoe_id,
        user_id=user_id,
        run_seed=seed,
        shoe_context=model_shoe_context,
        road_context=context,
    )
    result.update({
        "screen_pipeline_version": "THREEWAY-MARKOV-SHOE-DEPTH-SCREEN-V2",
        "mode": "screen_threeway_markov_shoe_depth",
        "shoe_id": str(shoe_id or ""),
        "screen_remaining_cards": int(remaining_cards or 0),
        "estimated_remaining_counts": [],
        "composition_source": "round_count_depth_estimator",
        "composition_quality": "maturity_estimate_not_exact_card_composition",
        "exact_remaining_counts_supplied": bool(
            isinstance(model_shoe_context.get("remaining_counts"), (list, tuple))
            and len(model_shoe_context.get("remaining_counts") or []) == 10
        ),
        "prior_counts_ignored": prior_counts_ignored,
        "observed_cards_ignored": observed_cards_ignored,
        "shoe_context_ignored": False,
        "road_sequence_length": len(cleaned_bp),
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
        "previous_prediction_id": str(previous_prediction_id or ""),
        "learning_update_triggered_by_new_actual": latest_actual_is_new,
        "deterministic_feature_seed": True,
    })
    result["road_fusion"] = {
        "applied": False,
        "mode": "diagnostic_only",
        "reason": "圖片保留完整 B/P/T；正式方向由三元 Markov 後驗 B/P 機率直接決定。",
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
                "shoe_id": str(shoe_id or ""),
                "prediction_pipeline": "threeway_markov_shoe_depth_v2",
            },
        )
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["predict_from_screenshot"]
