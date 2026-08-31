"""Display-geometry analysis for baccarat derived roads.

Big Eye Boy, Small Road and Cockroach Pig are red/blue structural sequences.
This module lays each R/U sequence onto the standard six-row road grid, then
extracts table-player-style shape information (vertical drops, dragon tails,
collision turns, column-height rhythm and stair/shape breaks).

Geometry is deliberately auxiliary: it is derived from the same R/U history as
the human sequence model and therefore receives a smaller reliability cap.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import math

DERIVED_SYMBOLS = ("R", "U")
DERIVED_GRID_ROWS = 6
MAX_GEOMETRY_RELIABILITY = 0.10
MIN_GEOMETRY_ACTIVE_ROADS = 2
GEOMETRY_ROAD_WEIGHTS = {
    "big_eye": 1.00,
    "small_road": 0.85,
    "cockroach_road": 0.70,
}


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(number):
        return lo
    return max(lo, min(hi, number))


def _clean(values: Sequence[Any]) -> list[str]:
    return [
        str(value).upper().strip()
        for value in values
        if str(value).upper().strip() in DERIVED_SYMBOLS
    ][-300:]


def _runs(values: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _support_reliability(support: float, half_saturation: float) -> float:
    support = max(0.0, float(support))
    return _clip(support / (support + max(1e-9, float(half_saturation))))


def _probability_to_red(last_mark: str, p_same: float) -> float:
    p_same = _clip(p_same)
    return p_same if last_mark == "R" else 1.0 - p_same


def _run_survival(values: Sequence[str], current_length: int) -> tuple[float, int]:
    runs = _runs(values)
    completed = [length for _, length in runs[:-1]]
    eligible = [length for length in completed if length >= current_length]
    survived = sum(length > current_length for length in eligible)
    return (survived + 2.0) / (len(eligible) + 4.0), len(eligible)


def build_derived_road_geometry(values: Sequence[Any]) -> Dict[str, Any]:
    """Lay one derived-road R/U sequence on the standard six-row road grid.

    Same colour first moves downward. At row six, or when the cell below is
    already occupied by an earlier road tail, the run turns right. A colour
    change starts a new logical column at the top. Coordinates returned to
    callers are one-based for easy comparison with casino road displays.
    """
    sequence = _clean(values)
    occupied: dict[tuple[int, int], str] = {}
    positions: list[dict[str, Any]] = []
    run_base_col = -1
    run_index = -1
    current_mark = ""
    last_row = 0
    last_col = -1

    for index, mark in enumerate(sequence):
        new_run = mark != current_mark
        collision_turn = False
        bottom_turn = False
        if new_run:
            run_index += 1
            run_base_col += 1
            row = 0
            col = run_base_col
            while (row, col) in occupied:
                col += 1
            run_base_col = col
            placement = "NEW_COLUMN"
            current_mark = mark
        else:
            down = (last_row + 1, last_col)
            if last_row + 1 < DERIVED_GRID_ROWS and down not in occupied:
                row, col = down
                placement = "DOWN"
            else:
                row = last_row
                col = last_col + 1
                while (row, col) in occupied:
                    col += 1
                collision_turn = last_row + 1 < DERIVED_GRID_ROWS and down in occupied
                bottom_turn = last_row + 1 >= DERIVED_GRID_ROWS
                placement = "RIGHT_COLLISION" if collision_turn else "RIGHT_BOTTOM"

        occupied[(row, col)] = mark
        positions.append(
            {
                "index": int(index),
                "mark": mark,
                "color": "RED" if mark == "R" else "BLUE",
                "run_index": int(run_index),
                "row": int(row + 1),
                "column": int(col + 1),
                "placement": placement,
                "collision_turn": bool(collision_turn),
                "bottom_turn": bool(bottom_turn),
            }
        )
        last_row, last_col = row, col

    runs = _runs(sequence)
    run_summaries: list[dict[str, Any]] = []
    for idx, (mark, length) in enumerate(runs):
        cells = [item for item in positions if item["run_index"] == idx]
        right_cells = [item for item in cells if str(item["placement"]).startswith("RIGHT")]
        collision_cells = [item for item in cells if item["collision_turn"]]
        run_summaries.append(
            {
                "run_index": int(idx),
                "mark": mark,
                "length": int(length),
                "start_row": int(cells[0]["row"]) if cells else 0,
                "start_column": int(cells[0]["column"]) if cells else 0,
                "end_row": int(cells[-1]["row"]) if cells else 0,
                "end_column": int(cells[-1]["column"]) if cells else 0,
                "vertical_cells": int(len(cells) - len(right_cells)),
                "horizontal_tail_length": int(len(right_cells)),
                "has_horizontal_tail": bool(right_cells),
                "collision_turn": bool(collision_cells),
            }
        )

    heights = [length for _, length in runs]
    recent_heights = heights[-6:]
    completed = heights[:-1]
    current_run = run_summaries[-1] if run_summaries else {}
    shape_family = "COLD_START"
    staircase_score = 0.0

    if heights:
        shape_family = "GENERIC"
    if current_run.get("collision_turn"):
        shape_family = "COLLISION_TAIL"
    elif int(current_run.get("horizontal_tail_length", 0) or 0) > 0:
        shape_family = "HORIZONTAL_TAIL"
    elif len(recent_heights) >= 4 and all(v == 1 for v in recent_heights[-4:]):
        shape_family = "SINGLE_COLUMN_JUMP"
    elif len(completed) >= 3 and all(v == 2 for v in completed[-3:]) and heights[-1] <= 2:
        shape_family = "DOUBLE_COLUMN_RHYTHM"
    elif len(completed) >= 4:
        a, b, c, d = completed[-4:]
        if a == c and b == d and a != b:
            shape_family = f"COLUMN_RHYTHM_{a}_{b}"
    if shape_family == "GENERIC" and heights and heights[-1] >= 3:
        shape_family = "VERTICAL_DRAGON"

    if len(recent_heights) >= 4:
        diffs = [b - a for a, b in zip(recent_heights[-4:], recent_heights[-3:])]
        if diffs and all(delta > 0 for delta in diffs):
            shape_family = "ASCENDING_STAIR"
            staircase_score = min(1.0, sum(abs(v) for v in diffs) / 4.0)
        elif diffs and all(delta < 0 for delta in diffs):
            shape_family = "DESCENDING_STAIR"
            staircase_score = min(1.0, sum(abs(v) for v in diffs) / 4.0)

    if recent_heights:
        mean_height = sum(recent_heights) / len(recent_heights)
        variance = sum((value - mean_height) ** 2 for value in recent_heights) / len(recent_heights)
        height_consistency = _clip(1.0 - math.sqrt(variance) / 2.5)
    else:
        height_consistency = 0.0

    display_width = max((item["column"] for item in positions), default=0)
    return {
        "sequence": sequence,
        "positions": positions,
        "runs": run_summaries,
        "column_heights": heights,
        "recent_column_heights": recent_heights,
        "current_run": current_run,
        "shape_family": shape_family,
        "staircase_score": float(staircase_score),
        "height_consistency": float(height_consistency),
        "display_rows": DERIVED_GRID_ROWS,
        "display_width": int(display_width),
        "horizontal_tail_active": bool(current_run.get("has_horizontal_tail", False)),
        "collision_turn_active": bool(current_run.get("collision_turn", False)),
        "semantics": "standard_six_row_derived_road_display_geometry",
    }


def predict_next_geometry_mark(values: Sequence[Any]) -> Dict[str, Any]:
    """Estimate next R/U using grid-shape rhythm without hard follow/reversal."""
    sequence = _clean(values)
    geometry = build_derived_road_geometry(sequence)
    if not sequence:
        return {
            "model_id": "DERIVED-ROAD-GEOMETRY-V1",
            "probabilities": {"R": 0.5, "U": 0.5},
            "direction": "R",
            "confidence": 0.0,
            "geometry": geometry,
            "desired_relation": "NEUTRAL",
            "expected_run_length": None,
            "shape_break_probability": 0.5,
        }

    runs = _runs(sequence)
    last_mark, current_length = runs[-1]
    completed = [length for _, length in runs[:-1]]
    expected_run_length: int | None = None
    desired_same: bool | None = None
    pattern_reliability = 0.0

    if len(completed) >= 3 and all(value == 1 for value in completed[-3:]):
        expected_run_length = 1
        desired_same = False
        pattern_reliability = 0.68
    elif len(completed) >= 3 and len(set(completed[-3:])) == 1 and 2 <= completed[-1] <= 4:
        expected_run_length = completed[-1]
        desired_same = current_length < expected_run_length
        pattern_reliability = 0.60
    elif len(completed) >= 4:
        a, b, c, d = completed[-4:]
        if a == c and b == d and a != b and 1 <= a <= 4 and 1 <= b <= 4:
            expected_run_length = a
            desired_same = current_length < expected_run_length
            pattern_reliability = 0.55

    empirical_same, support = _run_survival(sequence, current_length)
    support_rel = _support_reliability(support, 4.0)
    current_geometry = dict(geometry.get("current_run") or {})
    tail_active = bool(current_geometry.get("has_horizontal_tail", False))
    collision_active = bool(current_geometry.get("collision_turn", False))

    if desired_same is None:
        p_same = empirical_same
        reliability = 0.24 * support_rel
        if tail_active:
            reliability *= 0.90
    else:
        rule_same = 0.66 if desired_same else 0.34
        empirical_weight = 0.40 * support_rel
        p_same = (1.0 - empirical_weight) * rule_same + empirical_weight * empirical_same
        reliability = pattern_reliability * (0.45 + 0.55 * support_rel)

    if collision_active:
        reliability *= 0.88

    maturity = _clip(len(sequence) / 12.0)
    reliability = _clip(reliability * (0.35 + 0.65 * maturity))
    p_r = _clip(_probability_to_red(last_mark, p_same), 0.36, 0.64)

    if expected_run_length is None:
        shape_break_probability = 0.5
    elif current_length > expected_run_length:
        shape_break_probability = _clip(0.65 + 0.08 * (current_length - expected_run_length))
    elif current_length == expected_run_length:
        shape_break_probability = 0.30 if desired_same is False else 0.55
    else:
        shape_break_probability = 0.30

    return {
        "model_id": "DERIVED-ROAD-GEOMETRY-V1",
        "probabilities": {"R": float(p_r), "U": float(1.0 - p_r)},
        "direction": "R" if p_r >= 0.5 else "U",
        "confidence": float(reliability),
        "sample_count": len(sequence),
        "maturity": float(maturity),
        "desired_relation": (
            "SAME" if desired_same is True else "SWITCH" if desired_same is False else "EMPIRICAL"
        ),
        "expected_run_length": expected_run_length,
        "run_survival_probability": float(empirical_same),
        "run_survival_support": int(support),
        "shape_break_probability": float(shape_break_probability),
        "geometry": geometry,
        "semantics": "derived_grid_geometry_shape_probability_auxiliary_only",
    }


def score_geometry_ask_road_scenarios(
    derived_roads: Mapping[str, Sequence[Any]],
    scenario_marks: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Score hypothetical B/P by how well their ask marks fit current geometry."""
    models = {
        name: predict_next_geometry_mark(list(derived_roads.get(name) or []))
        for name in GEOMETRY_ROAD_WEIGHTS
    }
    weighted_logs = {"B": 0.0, "P": 0.0}
    weight_sums = {"B": 0.0, "P": 0.0}
    scenario_details: Dict[str, Any] = {"B": {}, "P": {}}
    preference_num = 0.0
    preference_den = 0.0
    active: set[str] = set()

    for name, road_weight in GEOMETRY_ROAD_WEIGHTS.items():
        model = models[name]
        confidence = float(model.get("confidence", 0.0) or 0.0)
        mark_b = str((scenario_marks.get("B") or {}).get(name) or "").upper()
        mark_p = str((scenario_marks.get("P") or {}).get(name) or "").upper()
        if mark_b not in DERIVED_SYMBOLS or mark_p not in DERIVED_SYMBOLS or confidence <= 0.02:
            for side, mark in (("B", mark_b), ("P", mark_p)):
                scenario_details[side][name] = {
                    "mark": mark,
                    "active": False,
                    "reason": "geometry_not_mature_or_no_standard_ask_mark",
                }
            continue

        active.add(name)
        prob_b = max(1e-6, float(model["probabilities"].get(mark_b, 0.5) or 0.5))
        prob_p = max(1e-6, float(model["probabilities"].get(mark_p, 0.5) or 0.5))
        effective = road_weight * confidence
        weighted_logs["B"] += effective * math.log(prob_b)
        weighted_logs["P"] += effective * math.log(prob_p)
        weight_sums["B"] += effective
        weight_sums["P"] += effective
        edge = prob_b - prob_p
        preference_num += effective * edge
        preference_den += effective * abs(edge)
        scenario_details["B"][name] = {
            "mark": mark_b,
            "active": True,
            "mark_probability": float(prob_b),
            "geometry_confidence": confidence,
            "shape_family": model["geometry"].get("shape_family"),
        }
        scenario_details["P"][name] = {
            "mark": mark_p,
            "active": True,
            "mark_probability": float(prob_p),
            "geometry_confidence": confidence,
            "shape_family": model["geometry"].get("shape_family"),
        }

    log_scores = {"B": math.log(0.5), "P": math.log(0.5)}
    for side in ("B", "P"):
        if weight_sums[side] > 1e-12:
            log_scores[side] = weighted_logs[side] / weight_sums[side]
    max_log = max(log_scores.values())
    exp_b = math.exp(log_scores["B"] - max_log)
    exp_p = math.exp(log_scores["P"] - max_log)
    total = exp_b + exp_p
    likelihood = {
        "B": exp_b / total if total > 1e-12 else 0.5,
        "P": exp_p / total if total > 1e-12 else 0.5,
    }

    active_list = sorted(active)
    active_fraction = len(active_list) / 3.0
    if active_list:
        denom = sum(GEOMETRY_ROAD_WEIGHTS[name] for name in active_list)
        mean_confidence = sum(
            GEOMETRY_ROAD_WEIGHTS[name] * float(models[name]["confidence"])
            for name in active_list
        ) / max(1e-12, denom)
    else:
        mean_confidence = 0.0
    agreement = abs(preference_num) / preference_den if preference_den > 1e-12 else 0.0
    separation = abs(likelihood["B"] - likelihood["P"])
    raw = (
        MAX_GEOMETRY_RELIABILITY
        * mean_confidence
        * active_fraction
        * (0.50 + 0.50 * _clip(separation / 0.18))
        * (0.70 + 0.30 * _clip(agreement))
    )
    reliability = min(MAX_GEOMETRY_RELIABILITY, raw) if len(active_list) >= MIN_GEOMETRY_ACTIVE_ROADS else 0.0

    return {
        "model_id": "DERIVED-GEOMETRY-ASK-ROAD-V1",
        "likelihood": likelihood,
        "reliability": float(reliability),
        "raw_reliability": float(min(MAX_GEOMETRY_RELIABILITY, raw)),
        "max_reliability": float(MAX_GEOMETRY_RELIABILITY),
        "active_roads": active_list,
        "active_road_count": len(active_list),
        "minimum_formal_active_roads": int(MIN_GEOMETRY_ACTIVE_ROADS),
        "cross_road_agreement": float(_clip(agreement)),
        "models": models,
        "scenario_details": scenario_details,
        "scenario_marks": {side: dict(scenario_marks.get(side) or {}) for side in ("B", "P")},
        "semantics": "derived_six_row_geometry_ask_road_auxiliary_with_shared_history_cap",
    }


__all__ = [
    "DERIVED_GRID_ROWS",
    "MAX_GEOMETRY_RELIABILITY",
    "MIN_GEOMETRY_ACTIVE_ROADS",
    "build_derived_road_geometry",
    "predict_next_geometry_mark",
    "score_geometry_ask_road_scenarios",
]
