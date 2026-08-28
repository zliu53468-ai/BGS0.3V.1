"""Screenshot adapter for the BGS quant predictor.

The image pipeline preserves chronological B/P/T outcomes. For the LINE/screenshot
workflow, remaining-card depth is re-estimated from the full observed history on every
analysis instead of trusting a rolling session total such as "previous remaining - 5".
A user-supplied exact remaining point-count vector remains authoritative.

This keeps the PR32 Shoe diagnostics (current remaining-card posterior and next-hand
4/5/6-card consumption distribution) while preventing guessed per-round card usage from
feeding back into the next Shoe posterior.
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

    model_shoe_context = dict(shoe_context or {})

    # The normal LINE/screenshot workflow always carries initial_image_history
    # and/or manual_outcome_history. In that mode the Shoe depth must be rebuilt
    # from the complete B/P/T history on every button press. A legacy rolling
    # screen/session total (for example previous remaining - 5) is diagnostic
    # only and must not become a soft depth constraint.
    history_reestimate_mode = bool(
        initial_raw
        or manual_raw
        or metadata.get("manual_update")
        or metadata.get("history_reestimate_mode")
    )

    try:
        legacy_screen_remaining = int(remaining_cards or 0)
    except (TypeError, ValueError):
        legacy_screen_remaining = 0

    # Priority 1: explicitly verified remaining point-counts. We still pass only
    # their total to this depth channel; exact composition remains a separate
    # evidence type and is never fabricated from a screenshot.
    exact_counts = model_shoe_context.get("remaining_counts")
    if isinstance(exact_counts, (list, tuple)) and len(exact_counts) == 10:
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

    # Priority 2 (compatibility only): callers outside the normal screenshot
    # history flow may still explicitly supply a total as soft depth evidence.
    # The LINE/screenshot flow intentionally skips this branch so every press
    # re-estimates depth from the full current history.
    if (
        "remaining_cards" not in model_shoe_context
        and not history_reestimate_mode
        and legacy_screen_remaining > 0
    ):
        model_shoe_context["remaining_cards"] = legacy_screen_remaining
        model_shoe_context.setdefault("remaining_cards_reliability", 0.65)
        model_shoe_context.setdefault(
            "remaining_cards_source",
            "explicit_screen_total_soft_depth",
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
    next_hand_draw_profile = dict(shoe_estimate.get("next_hand_draw_profile") or {})

    try:
        posterior_remaining = int(round(float(
            result.get("estimated_remaining_cards")
            or shoe_estimate.get("expected_remaining_cards")
            or 0.0
        )))
    except (TypeError, ValueError):
        posterior_remaining = 0

    exact_remaining_supplied = bool(
        isinstance(model_shoe_context.get("remaining_counts"), (list, tuple))
        and len(model_shoe_context.get("remaining_counts") or []) == 10
    )
    if exact_remaining_supplied:
        remaining_source = "user_exact_remaining_counts_total"
    elif history_reestimate_mode:
        remaining_source = "full_history_particle_posterior"
    else:
        remaining_source = str(
            model_shoe_context.get("remaining_cards_source") or ""
        )

    reported_remaining = (
        posterior_remaining
        if history_reestimate_mode and posterior_remaining > 0
        else legacy_screen_remaining
    )

    result.update({
        "screen_pipeline_version": "THREEWAY-MARKOV-SHOE-HISTORY-DEPTH-SCREEN-V4",
        "mode": (
            "screen_quant_markov_history_reestimated_shoe"
            if history_reestimate_mode
            else "screen_quant_markov_depth_conditioned_shoe"
        ),
        "shoe_id": str(shoe_id or ""),
        "screen_remaining_cards": int(reported_remaining),
        "legacy_screen_remaining_cards_input": int(legacy_screen_remaining),
        "history_reestimated_remaining_cards": bool(history_reestimate_mode),
        "history_depth_semantics": (
            "full_current_bpt_history_recomputed_each_analysis_not_rolling_fixed_decrement"
        ),
        "estimated_remaining_counts": list(
            shoe_estimate.get("expected_remaining_counts") or []
        ),
        "next_hand_draw_profile": next_hand_draw_profile,
        "composition_source": (
            "outcome_conditioned_particle_posterior_plus_verified_depth"
            if depth_constraint.get("applied")
            else "outcome_conditioned_particle_posterior"
        ),
        "composition_quality": (
            "probabilistic_not_exact_card_composition"
        ),
        "screen_depth_constraint_applied": bool(
            depth_constraint.get("applied", False)
        ),
        "screen_depth_constraint": depth_constraint,
        "remaining_cards_source": remaining_source,
        "exact_remaining_counts_supplied": exact_remaining_supplied,
        "legacy_screen_remaining_ignored_for_shoe": bool(
            history_reestimate_mode and not exact_remaining_supplied
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
                "history_reestimated_remaining_cards": bool(history_reestimate_mode),
                "prediction_pipeline": (
                    "support_markov_derived_road_history_reestimated_shoe_v4"
                    if history_reestimate_mode
                    else "support_markov_derived_road_depth_conditioned_shoe_v3"
                ),
            },
        )
        result["performance_tracking"] = True
    else:
        result["performance_tracking"] = False
    return result


__all__ = ["predict_from_screenshot"]
