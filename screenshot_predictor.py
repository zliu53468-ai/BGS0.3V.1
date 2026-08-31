"""Screenshot adapter for the BGS quant predictor.

Image/road recognition remains unchanged. This adapter only forwards verified
shoe evidence into predictor with strict priority:
remaining_counts > observed_cards > screenshot/session remaining-card total.
A total card count is only a soft depth hint and is never promoted into an
exact composition claim.
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


def _as_exact_counts(values: Any) -> Optional[list[Any]]:
    if isinstance(values, Mapping):
        ordered: list[Any] = []
        for point in range(10):
            if point in values:
                ordered.append(values[point])
            elif str(point) in values:
                ordered.append(values[str(point)])
            else:
                return None
        return ordered
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw = list(values)
        return raw if len(raw) == 10 else None
    return None


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

    model_shoe_context = dict(shoe_context or {})

    # Priority 1: exact remaining point counts. Keep the existing shoe_context
    # field first; legacy prior_counts is accepted only as a compatibility alias.
    exact_counts = _as_exact_counts(model_shoe_context.get("remaining_counts"))
    prior_counts_used = False
    if exact_counts is None:
        exact_counts = _as_exact_counts(prior_counts)
        if exact_counts is not None:
            model_shoe_context["remaining_counts"] = exact_counts
            model_shoe_context.setdefault("source", "remaining_counts")
            prior_counts_used = True
    elif exact_counts is not None:
        model_shoe_context["remaining_counts"] = exact_counts

    if exact_counts is not None:
        try:
            exact_total = sum(max(0, int(value)) for value in exact_counts)
        except (TypeError, ValueError):
            exact_total = 0
        if exact_total > 0:
            model_shoe_context["remaining_cards"] = int(exact_total)
            model_shoe_context["remaining_cards_reliability"] = 1.0
            model_shoe_context["remaining_cards_source"] = (
                "user_exact_remaining_counts_total"
            )
            model_shoe_context.setdefault("source", "remaining_counts")

    # Priority 2: verified observed card faces/point values. Do not overwrite
    # remaining_counts because exact remaining counts are higher priority.
    observed_cards_used = False
    if exact_counts is None and "observed_cards" not in model_shoe_context:
        if observed_cards is not None:
            observed_list = list(observed_cards)
            if observed_list:
                model_shoe_context["observed_cards"] = observed_list
                model_shoe_context.setdefault("source", "observed_cards")
                observed_cards_used = True
    elif exact_counts is None:
        existing_observed = model_shoe_context.get("observed_cards")
        if isinstance(existing_observed, Iterable) and not isinstance(
            existing_observed, (str, bytes, Mapping)
        ):
            observed_cards_used = bool(list(existing_observed))

    # Priority 3: screenshot/session total is only a soft physical depth hint.
    if "remaining_cards" not in model_shoe_context:
        try:
            screen_remaining = int(remaining_cards or 0)
        except (TypeError, ValueError):
            screen_remaining = 0
        if screen_remaining > 0:
            model_shoe_context["remaining_cards"] = screen_remaining
            model_shoe_context.setdefault("remaining_cards_reliability", 0.65)
            model_shoe_context.setdefault(
                "remaining_cards_source",
                "screenshot_or_session_total_soft_depth",
            )

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

    shoe_estimate = dict(result.get("probabilistic_shoe_estimate") or {})
    depth_constraint = dict(shoe_estimate.get("depth_constraint") or {})
    result.update({
        "screen_pipeline_version": "THREEWAY-MARKOV-SHOE-DEPTH-SCREEN-V3",
        "mode": "screen_quant_markov_depth_conditioned_shoe",
        "shoe_id": str(shoe_id or ""),
        "screen_remaining_cards": int(remaining_cards or 0),
        "estimated_remaining_counts": list(
            shoe_estimate.get("expected_remaining_counts") or []
        ),
        "composition_source": str(
            result.get("card_composition_source")
            or (
                "outcome_conditioned_particle_posterior_plus_soft_screen_depth"
                if depth_constraint.get("applied")
                else "outcome_conditioned_particle_posterior"
            )
        ),
        "composition_quality": (
            "exact_card_composition"
            if bool(result.get("shoe_context_used_for_formal_direction"))
            else "probabilistic_not_exact_card_composition"
        ),
        "screen_depth_constraint_applied": bool(
            depth_constraint.get("applied", False)
        ),
        "screen_depth_constraint": depth_constraint,
        "remaining_cards_source": str(
            model_shoe_context.get("remaining_cards_source") or ""
        ),
        "exact_remaining_counts_supplied": bool(exact_counts is not None),
        "prior_counts_ignored": bool(prior_counts) and not prior_counts_used,
        "prior_counts_used_as_remaining_counts": prior_counts_used,
        "observed_cards_ignored": bool(observed_cards) and not observed_cards_used,
        "observed_cards_forwarded": observed_cards_used,
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
                "prediction_pipeline": (
                    "support_markov_derived_road_depth_conditioned_shoe_v3"
                ),
            },
        )
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["predict_from_screenshot"]
