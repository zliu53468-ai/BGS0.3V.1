"""Lightweight baccarat road analyzer for the direct Markov predictor.

This module does not contain an ensemble, bandit, stacking model, or Monte Carlo
predictor.  It converts the recognized B/P/T history into the original 21 road
features used by the first 29D design and exposes a small structural road signal
for Markov calibration.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math

ROAD_HISTORY_LIMIT = 500

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


def build_big_road(sequence: Sequence[str]) -> Dict[str, Any]:
    """Return the logical big-road columns.

    For prediction we only need the ordered run/column structure. The detector
    remains responsible for pixel/grid reconstruction.
    """
    seq = [str(v).upper().strip() for v in sequence if str(v).upper().strip() in {"B", "P"}]
    runs = _runs(seq)
    columns = [
        {"side": side, "height": int(length), "index": index}
        for index, (side, length) in enumerate(runs)
    ]
    return {
        "columns": columns,
        "column_heights": [item["height"] for item in columns],
        "column_sides": [item["side"] for item in columns],
        "column_count": len(columns),
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
    return (1.0 if values[-1] == "B" else -1.0) * min(1.0, run / 6.0)


def _derived_road(heights: Sequence[int], offset: int) -> List[str]:
    result: List[str] = []
    for index in range(offset + 1, len(heights)):
        current = int(heights[index])
        reference = int(heights[index - offset])
        previous_reference = int(heights[index - offset - 1])
        delta_current = current - reference
        delta_previous = reference - previous_reference
        regular = (
            current == reference
            or delta_current == delta_previous
            or abs(current - reference) <= 1
        )
        result.append("R" if regular else "U")
    return result


def _derived_stats(values: Sequence[str]) -> Dict[str, float]:
    sequence = [str(v).upper() for v in values if str(v).upper() in {"R", "U"}]
    if not sequence:
        return {
            "continuation": 0.5,
            "recent_continuation": 0.5,
            "balance": 0.0,
            "saturation": 0.0,
        }
    if len(sequence) >= 2:
        continuation = sum(a == b for a, b in zip(sequence, sequence[1:])) / (len(sequence) - 1)
    else:
        continuation = 0.5
    recent = sequence[-5:]
    if len(recent) >= 2:
        recent_continuation = sum(a == b for a, b in zip(recent, recent[1:])) / (len(recent) - 1)
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


def _road_structural_probability(
    sequence: Sequence[str],
    *,
    derived_consensus: float,
) -> Tuple[float, str, float]:
    """Small B/P road calibration signal; Markov remains the primary predictor."""
    seq = list(sequence)
    if not seq:
        return 0.5, "", 0.0

    last_side, run = _streak(seq)
    alternation6 = _transition_rate(seq, 6)
    alternation12 = _transition_rate(seq, 12)
    runs = _runs(seq)
    lengths = [length for _, length in runs]

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
    sequence = [v for v in raw if v in {"B", "P"}][-ROAD_HISTORY_LIMIT:]
    road = build_big_road(sequence)
    heights = list(road["column_heights"])

    big_eye = _derived_road(heights, 1)
    small_road = _derived_road(heights, 2)
    cockroach_road = _derived_road(heights, 3)
    derived_stats = {
        "big_eye": _derived_stats(big_eye),
        "small_road": _derived_stats(small_road),
        "cockroach_road": _derived_stats(cockroach_road),
    }
    saturations = [derived_stats[name]["saturation"] for name in ("big_eye", "small_road", "cockroach_road")]
    mean_sat = sum(saturations) / 3.0
    dispersion = (
        abs(saturations[0] - saturations[1])
        + abs(saturations[1] - saturations[2])
        + abs(saturations[2] - saturations[0])
    ) / 3.0
    derived_consensus = _clip(mean_sat * (1.0 - dispersion), 0.0, 1.0)

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
    road_agreement = _clip(1.0 - abs(road_b - (0.5 + 0.08 * recent_balance)) / 0.20, 0.0, 1.0)

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

    return {
        "ok": bool(sequence),
        "engine": "ROAD_FEATURE_EXTRACTOR_21D_V1",
        "pipeline_stage": "image_history_to_road_features",
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
        "derived_roads": {
            "big_eye": big_eye,
            "small_road": small_road,
            "cockroach_road": cockroach_road,
        },
        "derived_stats": derived_stats,
        "derived_road_consensus": float(derived_consensus),
        "component_probabilities": {
            "road_structure": {"B": float(road_b), "P": float(1.0 - road_b), "T": 0.0}
        },
        "models": {
            "road_structure": {
                "active": bool(sequence),
                "banker_probability": float(road_b),
                "player_probability": float(1.0 - road_b),
                "direction": road_direction,
                "reliability": float(road_confidence),
            }
        },
        "grid_cell_count": len(list(grid_cells or [])),
        "initial_image_count": max(0, int(initial_image_count or 0)),
        "manual_count": max(0, int(manual_count or 0)),
        "data_scope": "recognized_history_only",
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
    "build_big_road",
    "build_road_context",
    "calculate_road_probabilities",
    "fuse_road_with_main_prediction",
    "normalize_raw_outcomes",
    "normalize_road_sequence",
]
