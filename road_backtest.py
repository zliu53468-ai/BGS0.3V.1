"""Sequence-aware walk-forward evaluation for the formal road predictor.

This evaluator intentionally refuses to treat aggregate shoe-composition counts as
road history.  A valid road backtest needs ordered outcomes grouped by shoe.

The helper can rebuild ordered shoes from ``prediction_performance_v3.json`` when
resolved records contain ``shoe_id`` and actual outcomes, then replay every shoe
prefix without future leakage.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import argparse
import json
import math

ForecastFn = Callable[[str | Iterable[Any] | None], Mapping[str, Any]]


def _clean_outcomes(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
        value = str(item or "").upper().strip()
        if value in {"B", "P", "T"}:
            result.append(value)
    return result


def _last_bp(values: Sequence[str]) -> str:
    for value in reversed(values):
        if value in {"B", "P"}:
            return value
    return ""


def _bp_runs(values: Sequence[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for value in values:
        if value not in {"B", "P"}:
            continue
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _is_fresh_switch(values: Sequence[str]) -> bool:
    runs = _bp_runs(values)
    return bool(len(runs) >= 2 and runs[-1][1] == 1 and runs[-2][1] >= 2)


def load_shoes_from_performance_file(
    path: str | Path,
    *,
    min_resolved_hands: int = 8,
) -> list[list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = list(payload.get("records") or [])
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        actual = str(record.get("actual_outcome") or "").upper().strip()
        shoe_id = str(record.get("shoe_id") or "").strip()
        if actual not in {"B", "P", "T"} or not shoe_id:
            continue
        timestamp = int(
            record.get("resolved_at")
            or record.get("created_at")
            or index
            or 0
        )
        grouped[shoe_id].append((timestamp, index, actual))

    shoes: list[list[str]] = []
    for items in grouped.values():
        ordered = [actual for _, _, actual in sorted(items)]
        if sum(value in {"B", "P"} for value in ordered) >= min_resolved_hands:
            shoes.append(ordered)
    return shoes


def evaluate_shoes(
    shoes: Sequence[Sequence[Any]],
    forecast_fn: ForecastFn,
    *,
    min_history_bp: int = 4,
) -> dict[str, Any]:
    sample_count = 0
    correct_count = 0
    brier_total = 0.0
    log_loss_total = 0.0
    follow_last_count = 0
    actual_same_count = 0
    echo_eligible = 0
    fresh_count = 0
    fresh_correct = 0
    regular_count = 0
    regular_correct = 0
    per_shoe: list[dict[str, Any]] = []

    for shoe_index, raw_shoe in enumerate(shoes):
        outcomes = _clean_outcomes(raw_shoe)
        history: list[str] = []
        shoe_samples = 0
        shoe_correct = 0
        for actual in outcomes:
            bp_history_count = sum(value in {"B", "P"} for value in history)
            if actual in {"B", "P"} and bp_history_count >= min_history_bp:
                result = dict(forecast_fn("".join(history)))
                probabilities = dict(result.get("probabilities") or {})
                p_b = max(1e-6, min(1.0 - 1e-6, float(probabilities.get("B", 0.5) or 0.5)))
                p_p = max(1e-6, min(1.0 - 1e-6, float(probabilities.get("P", 0.5) or 0.5)))
                norm = p_b + p_p
                p_b, p_p = p_b / norm, p_p / norm
                direction = str(result.get("direction") or result.get("action") or "").upper().strip()
                if direction not in {"B", "P"}:
                    direction = "B" if p_b >= p_p else "P"

                is_correct = direction == actual
                sample_count += 1
                correct_count += int(is_correct)
                shoe_samples += 1
                shoe_correct += int(is_correct)
                target_b = 1.0 if actual == "B" else 0.0
                brier_total += 0.5 * ((p_b - target_b) ** 2 + (p_p - (1.0 - target_b)) ** 2)
                log_loss_total += -math.log(p_b if actual == "B" else p_p)

                last = _last_bp(history)
                if last:
                    echo_eligible += 1
                    follow_last_count += int(direction == last)
                    actual_same_count += int(actual == last)

                if _is_fresh_switch(history):
                    fresh_count += 1
                    fresh_correct += int(is_correct)
                else:
                    regular_count += 1
                    regular_correct += int(is_correct)

            history.append(actual)

        per_shoe.append(
            {
                "shoe_index": shoe_index,
                "samples": shoe_samples,
                "correct": shoe_correct,
                "accuracy": shoe_correct / shoe_samples if shoe_samples else 0.0,
            }
        )

    accuracy = correct_count / sample_count if sample_count else 0.0
    follow_rate = follow_last_count / echo_eligible if echo_eligible else 0.0
    actual_same_rate = actual_same_count / echo_eligible if echo_eligible else 0.0
    return {
        "shoe_count": len(shoes),
        "sample_count": sample_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "brier_score_bp": brier_total / sample_count if sample_count else 0.0,
        "log_loss_bp": log_loss_total / sample_count if sample_count else 0.0,
        "follow_last_prediction_rate": follow_rate,
        "actual_same_rate": actual_same_rate,
        "last_hand_echo_gap": follow_rate - actual_same_rate,
        "fresh_switch_sample_count": fresh_count,
        "fresh_switch_accuracy": fresh_correct / fresh_count if fresh_count else 0.0,
        "regular_sample_count": regular_count,
        "regular_accuracy": regular_correct / regular_count if regular_count else 0.0,
        "per_shoe": per_shoe,
        "semantics": "prefix_only_walk_forward_no_future_outcomes_used",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("performance_file", type=Path)
    parser.add_argument("--min-hands", type=int, default=8)
    args = parser.parse_args()

    from road_pattern_core import forecast_road_pattern

    shoes = load_shoes_from_performance_file(
        args.performance_file,
        min_resolved_hands=args.min_hands,
    )
    if not shoes:
        print(json.dumps({"error": "no_sequence_shoes_available"}, ensure_ascii=False, indent=2))
        return 2
    report = evaluate_shoes(shoes, forecast_road_pattern)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_shoes_from_performance_file", "evaluate_shoes"]
