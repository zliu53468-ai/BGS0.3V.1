"""Causal next-hand B/P forecasting with L2-regularized online logistic regression.

No full-shoe fitting, future labels, pretrained chase prior, or LSTM. Each
observe() trains on features from BEFORE that observed outcome, then appends it.
forecast_next() replays that same protocol on the supplied observed prefix only.
Repeated requests and different users cannot contaminate one another's model.

Walk-forward CLI (one shoe per text line, or JSON {"shoes": ["BPT...", ...]}):
    python road_forecaster.py --backtest shoes.json
    python road_forecaster.py --demo
    python road_forecaster.py --backtest shoes.json --require-pass
Metrics exclude ties and the first resolved hand of EACH shoe (no follow-last
baseline exists yet). Each shoe starts with zero weights. Demo data is synthetic,
not evidence of live profitability or outperformance. The default acceptance
tolerance is an explicit 2 percentage points, not a statistical significance test.
With --demo, --require-pass also checks each subgroup, so a favorable mixed
average cannot mask a subgroup failure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MODEL_ID = "road_online_l2_logistic"
VERSION = "ROAD-FORECASTER-WALK-FORWARD-V1"
MIN_TRANSITION_SUPPORT = 5
LEARNING_RATE = 0.35
L2_REGULARIZATION = 0.05
FEATURE_NAMES = (
    "intercept", "run_direction_signed", "signed_run_length_norm",
    "signed_run_eq2", "signed_run_eq3", "signed_run_ge4",
    "switch6_centered", "switch12_centered", "run_direction_x_switch6",
    "order1_transition_banker_edge", "order2_transition_banker_edge",
)


def normalize_history(history: Iterable[Any] | str | None) -> list[str]:
    if history is None:
        return []
    if isinstance(history, str):
        compact = "".join(c for c in history.upper() if not c.isspace() and c not in ",|")
        if any(c not in "BPT" for c in compact):
            raise ValueError("History must contain only B/P/T outcomes")
        return [c for c in compact if c in "BP"]
    values = []
    for item in history:
        if isinstance(item, Mapping):
            item = item.get("outcome") or item.get("actual") or item.get("actual_outcome") or item.get("virtual_outcome")
        value = str(item or "").upper().strip()
        if value not in {"B", "P", "T"}:
            raise ValueError("History must contain only B/P/T outcomes")
        if value != "T":
            values.append(value)
    return values


def _switch_rate(history: Sequence[str], window: int) -> float:
    values = history[-window:]
    if len(values) < 2:
        return 0.5
    return sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)


class RoadForecaster:
    """One shoe, one causal online model; weights always start at zero."""

    def __init__(self) -> None:
        self.history: list[str] = []
        self.weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.gradient_squares = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        self.transitions: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0])
        self.run_side = ""
        self.run_length = 0
        self.updates = 0

    def _features(self) -> tuple[np.ndarray, dict[str, Any]]:
        sign = 1.0 if self.run_side == "B" else -1.0 if self.run_side == "P" else 0.0
        switch6 = 2.0 * _switch_rate(self.history, 6) - 1.0
        switch12 = 2.0 * _switch_rate(self.history, 12) - 1.0
        edges: list[float] = []
        supports: list[int] = []
        for order in (1, 2):
            counts = self.transitions.get(tuple(self.history[-order:]), [0, 0]) if len(self.history) >= order else [0, 0]
            support = sum(counts)
            supports.append(support)
            # Gating occurs BEFORE prediction and training; insufficient support
            # gives exactly zero input/gradient, regardless of learned weights.
            edges.append((counts[0] - counts[1]) / (support + 2.0) if support >= MIN_TRANSITION_SUPPORT else 0.0)
        x = np.asarray([
            1.0, sign, sign * min(self.run_length / 8.0, 1.0),
            sign * float(self.run_length == 2), sign * float(self.run_length == 3),
            sign * float(self.run_length >= 4), switch6, switch12,
            sign * switch6, edges[0], edges[1],
        ], dtype=np.float64)
        return x, {"run_side": self.run_side, "run_length": self.run_length,
                   "order1_support": supports[0], "order2_support": supports[1],
                   "order1_gate": 1.0 if supports[0] >= MIN_TRANSITION_SUPPORT else 0.0,
                   "order2_gate": 1.0 if supports[1] >= MIN_TRANSITION_SUPPORT else 0.0,
                   "transition_min_support": MIN_TRANSITION_SUPPORT}

    def _estimate(self) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
        x, meta = self._features()
        factors = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        contributions = self.weights * x
        sign = 1.0 if self.run_side == "B" else -1.0 if self.run_side == "P" else 0.0
        if self.run_length >= 3:
            factor = 0.55 if self.run_length == 3 else 0.35 if self.run_length == 4 else 0.20
            # Include an aligned intercept too, so learned one-sided bias cannot
            # bypass long-run shrinkage. Opposing learned contributions stay intact.
            factors[contributions * sign > 0.0] = factor
        effective_x = x * factors
        logit = float(np.dot(self.weights, effective_x))
        p_b = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        meta.update({"anti_chase_applied": bool(np.any(factors < 1.0)),
                     "feature_decay_factors": dict(zip(FEATURE_NAMES, map(float, factors))),
                     "raw_logit": float(np.sum(contributions)), "logit": logit})
        return p_b, x, effective_x, meta

    def predict_next(self) -> dict[str, Any]:
        p_b, x, effective_x, meta = self._estimate()
        return {
            "p_b": p_b, "p_p": 1.0 - p_b,
            "direction": "B" if p_b >= 0.5 else "P",
            "uncertainty": 1.0 / math.sqrt(1.0 + self.updates),
            "uncertainty_semantics": "inverse_sqrt_sample_support_not_a_calibrated_probability_interval",
            "effective_support": float(self.updates),
            "features_used": dict(zip(FEATURE_NAMES, map(float, effective_x))),
            "raw_features": dict(zip(FEATURE_NAMES, map(float, x))),
            "model_id": MODEL_ID, "version": VERSION,
            "metadata": dict(meta, training_updates=self.updates, history_round_count=len(self.history),
                             training_protocol="predict_prefix_then_observe_target_once",
                             l2_regularization=L2_REGULARIZATION, linucb_direction_weight=0.0,
                             probability_semantics="next_resolved_B_or_P_model_estimate_not_guaranteed_win_rate"),
        }

    def observe(self, outcome: str) -> dict[str, Any]:
        """Call only after the actual result is known; never with a prediction."""
        actual = str(outcome or "").upper().strip()
        if actual == "T":
            return {"updated": False, "reason": "tie_excluded_from_conditional_BP_model"}
        if actual not in {"B", "P"}:
            raise ValueError("Observed outcome must be B, P, or T")
        # Features and prediction are computed BEFORE appending actual.
        p_b, _, effective_x, _ = self._estimate()
        target = 1.0 if actual == "B" else 0.0
        gradient = (p_b - target) * effective_x + L2_REGULARIZATION * self.weights
        self.gradient_squares += gradient * gradient
        self.weights -= LEARNING_RATE * gradient / np.sqrt(self.gradient_squares)
        self.weights = np.clip(self.weights, -4.0, 4.0)
        for order in (1, 2):
            if len(self.history) >= order:
                self.transitions[tuple(self.history[-order:])][0 if actual == "B" else 1] += 1
        self.run_length = self.run_length + 1 if self.run_side == actual else 1
        self.run_side = actual
        self.history.append(actual)
        self.updates += 1
        return {"updated": True, "actual_outcome": actual, "training_updates": self.updates,
                "pre_update_p_b": p_b, "label_used_only_after_prediction": True}


def forecast_next(history: Iterable[Any] | str | None) -> dict[str, Any]:
    """Replay only supplied observations, then predict the genuinely unseen next hand.

    Replay avoids double updates on UI refresh, stale model state, shoe resets,
    and cross-user contamination. No persisted LinUCB parameters are imported.
    """
    model = RoadForecaster()
    for outcome in normalize_history(history):
        model.observe(outcome)
    return model.predict_next()


def _metrics(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"samples": 0, "accuracy": None, "brier_score": None, "continuation_rate": None}
    return {"samples": n,
            "accuracy": sum(r[model + "_direction"] == r["actual"] for r in rows) / n,
            "brier_score": sum((r[model + "_p_b"] - float(r["actual"] == "B")) ** 2 for r in rows) / n,
            "continuation_rate": sum(r[model + "_direction"] == r["last_side"] for r in rows) / n}


def walk_forward(shoes: Iterable[Iterable[Any] | str], *, max_accuracy_drop: float = 0.02, include_trace: bool = False) -> dict[str, Any]:
    """Prequential evaluation with independent shoes and a paired follow-last baseline."""
    if not 0.0 <= max_accuracy_drop <= 1.0:
        raise ValueError("max_accuracy_drop must be between zero and one")
    rows: list[dict[str, Any]] = []
    canonical: list[list[str]] = []
    for shoe_index, shoe in enumerate(shoes):
        sequence = normalize_history(shoe)
        canonical.append(sequence)
        model = RoadForecaster()
        for t, actual in enumerate(sequence):
            predicted = model.predict_next()  # weights/context contain ONLY sequence[:t]
            if t > 0:
                last = model.history[-1]
                rows.append({"shoe_index": shoe_index, "target_index": t, "actual": actual,
                             "last_side": last, "run_length_before_target": model.run_length,
                             "training_updates_before_target": model.updates,
                             "model_p_b": predicted["p_b"], "model_direction": predicted["direction"],
                             "baseline_p_b": 1.0 if last == "B" else 0.0, "baseline_direction": last})
            model.observe(actual)  # Only after the prediction is frozen/scored.
    if not rows:
        raise ValueError("At least one shoe with two resolved B/P outcomes is required")
    long_rows = [r for r in rows if r["run_length_before_target"] >= 3]
    model_metrics, baseline_metrics = _metrics(rows, "model"), _metrics(rows, "baseline")
    long_model, long_baseline = _metrics(long_rows, "model"), _metrics(long_rows, "baseline")
    accuracy_delta = model_metrics["accuracy"] - baseline_metrics["accuracy"]
    continuation_pass = long_model["continuation_rate"] <= long_baseline["continuation_rate"] if long_rows else None
    accuracy_pass = accuracy_delta >= -max_accuracy_drop
    result = {
        "model_id": MODEL_ID, "version": VERSION, "protocol": "walk_forward_predict_then_update_reset_each_shoe",
        "shoe_count": len(canonical), "resolved_rounds": sum(map(len, canonical)),
        "dataset_sha256": sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest(),
        "overall": {"forecaster": model_metrics, "follow_last": baseline_metrics,
                    "accuracy_delta": accuracy_delta, "brier_delta": model_metrics["brier_score"] - baseline_metrics["brier_score"],
                    "neutral_50_50_brier": 0.25},
        "run_length_ge3": {"forecaster": long_model, "follow_last": long_baseline},
        "acceptance": {"max_accuracy_drop": max_accuracy_drop, "accuracy_pass": accuracy_pass,
                       "long_run_continuation_pass": continuation_pass,
                       "passes": accuracy_pass and continuation_pass is True,
                       "note": "Fixed engineering tolerance, not a significance test or guarantee on unseen shoes"},
        "baseline_probability_semantics": "follow_last_as_deterministic_0_or_1; Brier is binary mean squared error",
    }
    if include_trace:
        result["trace"] = rows
    return result


def demo_shoes(seed: int = 20260831) -> list[str]:
    """Fixed reproducible synthetic suite; never presented as real betting data."""
    rng = random.Random(seed)
    shoes = []
    for _ in range(100):
        shoes.append("".join(rng.choice("BP") for _ in range(70)))
    for stay_probability in (0.25, 0.75):
        for _ in range(50):
            sequence = [rng.choice("BP")]
            for _ in range(69):
                sequence.append(sequence[-1] if rng.random() < stay_probability else ("P" if sequence[-1] == "B" else "B"))
            shoes.append("".join(sequence))
    return shoes


def load_shoes(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data["shoes"] if "shoes" in data else [data["history"]]
        if isinstance(data, str):
            return [data]
        if not isinstance(data, list):
            raise ValueError("JSON must be a history or list of shoes")
        return [data] if data and all(isinstance(v, str) and v in {"B", "P", "T"} for v in data) else data
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--backtest", type=Path, help="JSON shoes or one B/P/T shoe per text line")
    source.add_argument("--demo", action="store_true", help="Fixed synthetic suite, NOT a real-data validation")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02, help="Absolute accuracy tolerance; default 0.02 = 2pp")
    parser.add_argument("--require-pass", action="store_true", help="Return exit code 1 if acceptance fails or long-run data is absent")
    args = parser.parse_args(argv)
    try:
        shoes = demo_shoes() if args.demo else load_shoes(args.backtest)
        result = walk_forward(shoes, max_accuracy_drop=args.max_accuracy_drop)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    result["data_source"] = "synthetic_demo_not_real_shoes" if args.demo else "supplied_file_provenance_unverified"
    result["real_world_performance_validated"] = False
    if args.demo:
        # Do not let a mixed-suite average hide losses in a particular regime.
        result["demo_groups"] = {}
        for name, group in (("independent_random", shoes[:100]), ("switching", shoes[100:150]), ("persistent", shoes[150:])):
            evaluation = walk_forward(group, max_accuracy_drop=args.max_accuracy_drop)
            result["demo_groups"][name] = {key: evaluation[key] for key in ("overall", "run_length_ge3", "acceptance")}
        result["demo_all_groups_pass"] = all(group["acceptance"]["passes"] for group in result["demo_groups"].values())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    passes = result["acceptance"]["passes"] and result.get("demo_all_groups_pass", True)
    return 1 if args.require_pass and not passes else 0


if __name__ == "__main__":
    raise SystemExit(main())
