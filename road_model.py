"""Baccarat Big Road + standard derived-road analyzer.

The screenshot detector reconstructs chronological B/P/T history. This module
rebuilds the logical Big Road and derives Big Eye Boy, Small Road and Cockroach
Pig from the standard Big-Road cell-comparison rules.

Important:
- Derived-road red/blue marks are NOT Banker/Player.
- Ties do not create a new Big-Road cell or derived-road mark.
- Internally R = red/regular and U = blue/irregular.
- The derived-road Markov channel evaluates standard ask-road outcomes under
  hypothetical next Banker vs Player extensions.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math

from derived_road_markov import score_ask_road_scenarios

ROAD_HISTORY_LIMIT = 500
DERIVED_ROAD_RULE_VERSION = "STANDARD-BIG-ROAD-CELL-COMPARISON-V1"
DERIVED_ROAD_OFFSETS = {
    "big_eye": 1,
    "small_road": 2,
    "cockroach_road": 3,
}

ROAD_FEATURE_NAMES = (
    "bias", "history_maturity", "global_banker_balance", "recent3_banker_balance",
    "recent8_banker_balance", "current_streak_direction", "current_streak_length",
    "alternation6", "alternation12", "transition_acceleration", "streak_break_signal",
    "long_dragon_tail_pressure", "observed_tie_rate", "road_planning_balance",
    "road_recent_balance", "road_confidence", "road_agreement", "big_eye_saturation",
    "small_road_saturation", "cockroach_road_saturation", "derived_road_consensus",
)


def _clip(value: Any, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(lo, min(hi, number))


def normalize_raw_outcomes(values: Iterable[Any]) -> List[str]:
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
    return result[-max(ROAD_HISTORY_LIMIT * 2, 200):]


def normalize_road_sequence(values: Iterable[Any]) -> List[str]:
    return [
        value for value in normalize_raw_outcomes(values)
        if value in {"B", "P"}
    ][-ROAD_HISTORY_LIMIT:]


def _runs(sequence: Sequence[str]) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for value in sequence:
        if result and result[-1][0] == value:
            side, length = result[-1]
            result[-1] = (side, length + 1)
        else:
            result.append((value, 1))
    return result


def _build_big_road_timeline(sequence: Sequence[str]) -> Dict[str, Any]:
    """Build logical streak columns and per-hand (column,row) positions.

    For derived-road mathematics a dragon tail is treated as the logical streak
    continuing downward beyond display row six. Therefore logical run height,
    not rendered screen coordinates, is the comparison state.
    """
    seq = [
        str(v).upper().strip()
        for v in sequence
        if str(v).upper().strip() in {"B", "P"}
    ]
    column_sides: List[str] = []
    column_heights: List[int] = []
    positions: List[Dict[str, Any]] = []

    for hand_index, side in enumerate(seq):
        new_column = not column_sides or side != column_sides[-1]
        if new_column:
            column_sides.append(side)
            column_heights.append(1)
        else:
            column_heights[-1] += 1

        column_index = len(column_heights) - 1
        row = int(column_heights[-1])
        positions.append({
            "hand_index": int(hand_index),
            "side": side,
            "column_index": int(column_index),
            "column_number": int(column_index + 1),
            "row": row,
            "new_column": bool(new_column),
        })

    columns = [
        {
            "side": side,
            "height": int(column_heights[index]),
            "index": int(index),
            "column_number": int(index + 1),
        }
        for index, side in enumerate(column_sides)
    ]
    return {
        "sequence": seq,
        "columns": columns,
        "column_heights": [int(v) for v in column_heights],
        "column_sides": list(column_sides),
        "column_count": len(columns),
        "positions": positions,
    }


def build_big_road(sequence: Sequence[str]) -> Dict[str, Any]:
    return _build_big_road_timeline(sequence)


def _derived_mark_for_current_position(
    column_heights: Sequence[int],
    *,
    column_index: int,
    row: int,
    new_column: bool,
    offset: int,
) -> tuple[str, Dict[str, Any]]:
    """Return one standard derived-road mark for the new Big-Road cell.

    d=1/2/3 maps to Big Eye Boy / Small Road / Cockroach Pig.

    New streak column (row=1):
      compare heights of c-1 and c-1-d. Equal -> red, unequal -> blue.

    Continuing streak (row>1):
      inspect column c-d at current row and row directly above. Same occupancy
      state (both occupied or both blank) -> red; different state -> blue.

    Natural start points:
      Big Eye: 2nd entry col 2 or 1st entry col 3.
      Small: 2nd entry col 3 or 1st entry col 4.
      Cockroach: 2nd entry col 4 or 1st entry col 5.
    """
    c = int(column_index)
    r = int(row)
    d = max(1, int(offset))

    if new_column:
        if c < d + 1:
            return "", {"emitted": False, "reason": "insufficient_completed_columns"}
        near_index = c - 1
        far_index = c - 1 - d
        near_height = int(column_heights[near_index])
        far_height = int(column_heights[far_index])
        mark = "R" if near_height == far_height else "U"
        return mark, {
            "emitted": True,
            "rule": "new_column_compare_completed_heights",
            "near_column_index": int(near_index),
            "far_column_index": int(far_index),
            "near_height": near_height,
            "far_height": far_height,
            "equal": bool(near_height == far_height),
        }

    if c < d:
        return "", {"emitted": False, "reason": "insufficient_left_reference_column"}

    reference_index = c - d
    reference_height = int(column_heights[reference_index])
    same_row_occupied = reference_height >= r
    above_row_occupied = reference_height >= (r - 1)
    same_state = same_row_occupied == above_row_occupied
    mark = "R" if same_state else "U"
    return mark, {
        "emitted": True,
        "rule": "continuation_compare_same_row_vs_above_occupancy",
        "reference_column_index": int(reference_index),
        "reference_height": reference_height,
        "current_row": r,
        "same_row_occupied": bool(same_row_occupied),
        "above_row_occupied": bool(above_row_occupied),
        "same_state": bool(same_state),
    }


def build_standard_derived_roads(sequence: Sequence[str]) -> Dict[str, Any]:
    """Generate all three derived roads hand-by-hand by standard rules."""
    seq = [
        str(v).upper().strip()
        for v in sequence
        if str(v).upper().strip() in {"B", "P"}
    ]
    column_sides: List[str] = []
    column_heights: List[int] = []
    derived: Dict[str, List[str]] = {
        name: [] for name in DERIVED_ROAD_OFFSETS
    }
    events: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in DERIVED_ROAD_OFFSETS
    }

    for hand_index, side in enumerate(seq):
        new_column = not column_sides or side != column_sides[-1]
        if new_column:
            column_sides.append(side)
            column_heights.append(1)
        else:
            column_heights[-1] += 1

        column_index = len(column_heights) - 1
        row = int(column_heights[-1])

        for name, offset in DERIVED_ROAD_OFFSETS.items():
            mark, diagnostic = _derived_mark_for_current_position(
                column_heights,
                column_index=column_index,
                row=row,
                new_column=new_column,
                offset=offset,
            )
            if not mark:
                continue
            derived[name].append(mark)
            events[name].append({
                "derived_index": len(derived[name]) - 1,
                "hand_index": int(hand_index),
                "big_road_column_index": int(column_index),
                "big_road_column_number": int(column_index + 1),
                "big_road_row": row,
                "big_road_side": side,
                "mark": mark,
                "color": "RED" if mark == "R" else "BLUE",
                "offset": int(offset),
                **diagnostic,
            })

    return {
        "rule_version": DERIVED_ROAD_RULE_VERSION,
        "encoding": {"R": "RED_REGULAR", "U": "BLUE_IRREGULAR"},
        "big_eye": derived["big_eye"],
        "small_road": derived["small_road"],
        "cockroach_road": derived["cockroach_road"],
        "events": events,
        "start_rules": {
            "big_eye": "2nd entry column 2 or 1st entry column 3",
            "small_road": "2nd entry column 3 or 1st entry column 4",
            "cockroach_road": "2nd entry column 4 or 1st entry column 5",
        },
    }


def _scenario_marks(
    sequence: Sequence[str],
    current_derived: Mapping[str, Sequence[str]],
) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {"B": {}, "P": {}}
    base_lengths = {
        name: len(list(current_derived.get(name) or []))
        for name in DERIVED_ROAD_OFFSETS
    }
    for side in ("B", "P"):
        scenario = build_standard_derived_roads(list(sequence) + [side])
        for name in DERIVED_ROAD_OFFSETS:
            values = list(scenario.get(name) or [])
            result[side][name] = (
                str(values[-1]) if len(values) > base_lengths[name] else ""
            )
    return result


def _derived_stats(values: Sequence[str]) -> Dict[str, float]:
    sequence = [
        str(v).upper() for v in values
        if str(v).upper() in {"R", "U"}
    ]
    if not sequence:
        return {
            "continuation": 0.5,
            "recent_continuation": 0.5,
            "balance": 0.0,
            "saturation": 0.0,
        }
    if len(sequence) >= 2:
        continuation = sum(
            a == b for a, b in zip(sequence, sequence[1:])
        ) / (len(sequence) - 1)
    else:
        continuation = 0.5
    recent = sequence[-5:]
    if len(recent) >= 2:
        recent_continuation = sum(
            a == b for a, b in zip(recent, recent[1:])
        ) / (len(recent) - 1)
    else:
        recent_continuation = 0.5
    balance = abs(sequence.count("R") / len(sequence) - 0.5) * 2.0
    saturation = max(balance, abs(2.0 * recent_continuation - 1.0))
    return {
        "continuation": float(continuation),
        "recent_continuation": float(recent_continuation),
        "balance": float(balance),
        "saturation": float(_clip(saturation, 0.0, 1.0)),
    }


def _transition_rate(sequence: Sequence[str], size: int) -> float:
    values = list(sequence[-size:])
    if len(values) < 2:
        return 0.5
    return sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)


def _balance(sequence: Sequence[str], size: int | None = None) -> float:
    values = list(sequence[-size:] if size else sequence)
    if not values:
        return 0.0
    return _clip((values.count("B") / len(values) - 0.5) * 2.0)


def _streak(sequence: Sequence[str]) -> Tuple[str, int]:
    if not sequence:
        return "", 0
    side = sequence[-1]
    length = 1
    for value in reversed(sequence[:-1]):
        if value != side:
            break
        length += 1
    return side, length


def _streak_break(sequence: Sequence[str]) -> float:
    values = list(sequence)
    if len(values) < 4 or values[-1] == values[-2]:
        return 0.0
    previous_side = values[-2]
    run = 1
    for value in reversed(values[:-2]):
        if value != previous_side:
            break
        run += 1
    if run < 3:
        return 0.0
    return (
        (1.0 if values[-1] == "B" else -1.0)
        * min(1.0, run / 6.0)
    )


def _road_structural_probability(
    sequence: Sequence[str],
    *,
    derived_consensus: float,
) -> Tuple[float, str, float]:
    """Legacy structural diagnostic; formal road fusion uses derived Markov."""
    seq = list(sequence)
    if not seq:
        return 0.5, "", 0.0

    last_side, run = _streak(seq)
    alternation6 = _transition_rate(seq, 6)
    alternation12 = _transition_rate(seq, 12)
    lengths = [length for _, length in _runs(seq)]
    direction = last_side
    edge = 0.0

    if len(lengths) >= 4 and all(length == 1 for length in lengths[-4:]):
        direction = "P" if last_side == "B" else "B"
        edge = 0.10
    elif len(lengths) >= 3 and all(length == 2 for length in lengths[-3:]):
        direction = "P" if last_side == "B" else "B"
        edge = 0.085
    elif run == 1 and len(lengths) >= 3 and all(length == 2 for length in lengths[-3:-1]):
        direction = last_side
        edge = 0.08
    elif run >= 3:
        direction = last_side
        edge = min(0.09, 0.045 + 0.01 * (run - 3))
    elif alternation6 >= 0.80 and alternation12 >= 0.70:
        direction = "P" if last_side == "B" else "B"
        edge = 0.075
    else:
        recent_balance = _balance(seq, 8)
        direction = "B" if recent_balance >= 0.0 else "P"
        edge = min(0.045, abs(recent_balance) * 0.045)

    confidence = _clip(
        min(0.75, len(seq) / 36.0) * (0.55 + 0.45 * derived_consensus),
        0.0,
        0.75,
    )
    signed = edge if direction == "B" else -edge
    p_b = _clip(0.5 + signed * confidence, 0.35, 0.65)
    return float(p_b), direction, float(confidence)


def calculate_road_probabilities(
    values: Iterable[Any],
    seed: int | None = None,
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    del seed
    raw = normalize_raw_outcomes(values)
    sequence = [
        value for value in raw if value in {"B", "P"}
    ][-ROAD_HISTORY_LIMIT:]

    road = build_big_road(sequence)
    heights = list(road["column_heights"])
    standard = build_standard_derived_roads(sequence)
    big_eye = list(standard["big_eye"])
    small_road = list(standard["small_road"])
    cockroach_road = list(standard["cockroach_road"])
    derived_map = {
        "big_eye": big_eye,
        "small_road": small_road,
        "cockroach_road": cockroach_road,
    }

    derived_stats = {
        name: _derived_stats(items)
        for name, items in derived_map.items()
    }
    saturations = [
        derived_stats[name]["saturation"]
        for name in ("big_eye", "small_road", "cockroach_road")
    ]
    mean_sat = sum(saturations) / 3.0
    dispersion = (
        abs(saturations[0] - saturations[1])
        + abs(saturations[1] - saturations[2])
        + abs(saturations[2] - saturations[0])
    ) / 3.0
    derived_consensus = _clip(mean_sat * (1.0 - dispersion), 0.0, 1.0)

    scenarios = _scenario_marks(sequence, derived_map)
    derived_markov = score_ask_road_scenarios(derived_map, scenarios)
    road_likelihood = dict(derived_markov["likelihood"])
    road_markov_reliability = float(derived_markov["reliability"])

    side, run = _streak(sequence)
    streak_sign = 1.0 if side == "B" else -1.0 if side == "P" else 0.0
    tie_rate = raw.count("T") / max(1, len(raw))
    alt6 = (_transition_rate(sequence, 6) - 0.5) * 2.0
    alt12 = (_transition_rate(sequence, 12) - 0.5) * 2.0
    acceleration = _transition_rate(sequence, 6) - _transition_rate(sequence, 14)

    road_b, road_direction, road_confidence = _road_structural_probability(
        sequence,
        derived_consensus=derived_consensus,
    )
    recent_balance = _balance(sequence, 8)
    road_agreement = _clip(
        1.0 - abs(road_b - (0.5 + 0.08 * recent_balance)) / 0.20,
        0.0,
        1.0,
    )

    road_features = [
        1.0,
        min(1.0, len(sequence) / 60.0),
        _balance(sequence),
        _balance(sequence, 3),
        recent_balance,
        streak_sign,
        min(1.0, run / 8.0),
        _clip(alt6),
        _clip(alt12),
        _clip(acceleration),
        _streak_break(sequence),
        streak_sign * min(1.0, max(0, run - 3) / 5.0),
        _clip(tie_rate / 0.20, 0.0, 1.0),
        _clip((road_b - 0.5) * 2.0),
        _clip(recent_balance),
        road_confidence,
        road_agreement,
        saturations[0],
        saturations[1],
        saturations[2],
        derived_consensus,
    ]
    if len(road_features) != len(ROAD_FEATURE_NAMES):
        raise RuntimeError("Road 21D feature dimension mismatch")

    derived_colors = {
        name: ["RED" if mark == "R" else "BLUE" for mark in items]
        for name, items in derived_map.items()
    }

    return {
        "ok": bool(sequence),
        "engine": "ROAD_FEATURE_EXTRACTOR_21D_V2_STANDARD_DERIVED",
        "pipeline_stage": "image_history_to_standard_derived_roads_to_markov_ask_road",
        "sequence": sequence,
        "raw_outcomes": raw,
        "sample_count": len(sequence),
        "raw_sample_count": len(raw),
        "tie_count": raw.count("T"),
        "observed_tie_rate": tie_rate,
        "banker_probability": float(road_b),
        "player_probability": float(1.0 - road_b),
        "direction": road_direction or ("B" if road_b >= 0.5 else "P"),
        "direction_text": "莊" if (road_direction or "B") == "B" else "閒",
        "confidence_score": float(road_confidence),
        "confidence_label": (
            "較高" if road_confidence >= 0.65 else
            "中等" if road_confidence >= 0.45 else "偏低"
        ),
        "road_feature_names": list(ROAD_FEATURE_NAMES),
        "road_features": [round(float(v), 10) for v in road_features],
        "geometry": road,
        "column_heights": heights,
        "run_lengths": [length for _, length in _runs(sequence)],
        "derived_road_rule_version": DERIVED_ROAD_RULE_VERSION,
        "derived_road_encoding": dict(standard["encoding"]),
        "derived_road_start_rules": dict(standard["start_rules"]),
        "derived_roads": derived_map,
        "derived_road_colors": derived_colors,
        "derived_road_events": dict(standard["events"]),
        "derived_stats": derived_stats,
        "derived_road_consensus": float(derived_consensus),
        "ask_road_scenarios": scenarios,
        "derived_road_markov": derived_markov,
        "derived_markov_likelihood": road_likelihood,
        "derived_markov_reliability": road_markov_reliability,
        "component_probabilities": {
            "road_structure": {
                "B": float(road_b),
                "P": float(1.0 - road_b),
                "T": 0.0,
            },
            "derived_road_markov_likelihood": {
                "B": float(road_likelihood["B"]),
                "P": float(road_likelihood["P"]),
                "T": 0.0,
            },
        },
        "models": {
            "road_structure": {
                "active": bool(sequence),
                "banker_probability": float(road_b),
                "player_probability": float(1.0 - road_b),
                "direction": road_direction,
                "reliability": float(road_confidence),
                "formal_fusion": False,
            },
            "derived_road_markov": {
                "active": bool(road_markov_reliability > 0.0),
                "banker_likelihood": float(road_likelihood["B"]),
                "player_likelihood": float(road_likelihood["P"]),
                "direction": "B" if road_likelihood["B"] >= road_likelihood["P"] else "P",
                "reliability": road_markov_reliability,
                "formal_fusion": True,
                "max_reliability": float(derived_markov["max_reliability"]),
            },
        },
        "grid_cell_count": len(list(grid_cells or [])),
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "data_scope": "recognized_big_road_history_rebuilds_standard_derived_roads",
        "derived_road_semantics": "red_blue_are_structure_marks_not_banker_player",
    }


def build_road_context(
    values: Iterable[Any],
    seed: int | None = None,
    *,
    grid_cells: Sequence[Mapping[str, Any]] | None = None,
    initial_image_count: int = 0,
    manual_count: int = 0,
) -> Dict[str, Any]:
    return calculate_road_probabilities(
        values,
        seed=seed,
        grid_cells=grid_cells,
        initial_image_count=initial_image_count,
        manual_count=manual_count,
    )


def fuse_road_with_main_prediction(
    main_prediction: Mapping[str, Any],
    road_analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    result = dict(main_prediction or {})
    result["road_support"] = dict(road_analysis or {})
    return result


__all__ = [
    "ROAD_FEATURE_NAMES",
    "DERIVED_ROAD_RULE_VERSION",
    "DERIVED_ROAD_OFFSETS",
    "build_big_road",
    "build_standard_derived_roads",
    "build_road_context",
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_raw_outcomes",
    "normalize_road_sequence",
]
