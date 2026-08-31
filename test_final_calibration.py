from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from road_anti_echo import MAX_CONTEXT_RESIDUAL_WEIGHT, MAX_ECHO_SHRINK, calibrate_fresh_switch
from road_backtest import evaluate_shoes, load_shoes_from_performance_file
from road_pattern_core import forecast_road_pattern
from road_pattern_v1_core import forecast_road_pattern as forecast_v1_only


def _swap_bp(history: str) -> str:
    return "".join("P" if ch == "B" else "B" if ch == "P" else ch for ch in history)


class FreshSwitchAntiEchoTests(unittest.TestCase):
    def test_v1_core_stays_exact_and_anti_echo_only_changes_wrapper_stage(self) -> None:
        history = "BBBBBBP"
        raw = forecast_v1_only(history)
        wrapped = forecast_road_pattern(history)
        self.assertEqual(wrapped["v1_probabilities"], raw["probabilities"])
        self.assertTrue(wrapped["anti_echo_calibration"]["fresh_switch"])
        raw_b = float(raw["probabilities"]["B"])
        post_b = float(wrapped["post_anti_echo_v1_probabilities"]["B"])
        self.assertLessEqual(abs(post_b - 0.5), abs(raw_b - 0.5) + 1e-12)

    def test_long_dragon_is_not_penalized(self) -> None:
        result = forecast_road_pattern("BBBBBBBB")
        self.assertFalse(result["anti_echo_applied"])
        self.assertFalse(result["anti_echo_calibration"]["fresh_switch"])
        self.assertEqual(
            result["post_anti_echo_v1_probabilities"],
            result["v1_probabilities"],
        )

    def test_single_jump_is_not_mistaken_for_long_run_break(self) -> None:
        result = forecast_road_pattern("BPBPBP")
        self.assertFalse(result["anti_echo_applied"])
        self.assertEqual(result["components"]["pattern_survival"]["pattern"], "SINGLE_JUMP")

    def test_context_uses_completed_comparable_fresh_switches(self) -> None:
        # Historical P runs after B3 are both length 1; the current final P is a
        # fresh switch after B3, so same-shoe context should prefer OLD_RETURN.
        result = calibrate_fresh_switch(list("BBBPBBBPBBBP"), 0.46)
        context = result["context"]
        self.assertGreaterEqual(context["exact_support"], 1.5)
        self.assertLess(context["p_new_continue"], 0.5)
        self.assertLessEqual(result["echo_shrink"], MAX_ECHO_SHRINK)
        self.assertLessEqual(result["context_residual_weight"], MAX_CONTEXT_RESIDUAL_WEIGHT)

    def test_orientation_mirror_symmetry(self) -> None:
        history = "BBBPPBBBPPBBBP"
        mirrored = _swap_bp(history)
        left_raw = forecast_v1_only(history)["probabilities"]["B"]
        right_raw = forecast_v1_only(mirrored)["probabilities"]["B"]
        left = calibrate_fresh_switch(list(history), float(left_raw))
        right = calibrate_fresh_switch(list(mirrored), float(right_raw))
        self.assertAlmostEqual(left["final_p_b"], 1.0 - right["final_p_b"], places=12)

        full_left = forecast_road_pattern(history)
        full_right = forecast_road_pattern(mirrored)
        self.assertAlmostEqual(
            float(full_left["probabilities"]["B"]),
            float(full_right["probabilities"]["P"]),
            places=12,
        )

    def test_anti_echo_does_not_force_a_material_reversal(self) -> None:
        result = calibrate_fresh_switch(list("BBBBBBP"), 0.44)
        self.assertGreater(result["final_p_b"], 0.44)
        self.assertLessEqual(result["final_p_b"], 0.500001)


class WalkForwardBacktestTests(unittest.TestCase):
    def test_performance_file_is_grouped_by_shoe_and_ordered(self) -> None:
        payload = {
            "records": [
                {"shoe_id": "A", "resolved_at": 2, "actual_outcome": "P"},
                {"shoe_id": "A", "resolved_at": 1, "actual_outcome": "B"},
                {"shoe_id": "B", "resolved_at": 1, "actual_outcome": "T"},
                {"shoe_id": "A", "resolved_at": 3, "actual_outcome": "B"},
                {"shoe_id": "B", "resolved_at": 2, "actual_outcome": "P"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "perf.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            shoes = load_shoes_from_performance_file(path, min_resolved_hands=1)
        self.assertIn(["B", "P", "B"], shoes)
        self.assertIn(["T", "P"], shoes)

    def test_backtester_exposes_last_hand_echo_gap(self) -> None:
        def follow_last(history: str | None):
            values = [ch for ch in str(history or "") if ch in {"B", "P"}]
            last = values[-1] if values else "B"
            return {
                "direction": last,
                "probabilities": {last: 0.60, ("P" if last == "B" else "B"): 0.40},
            }

        report = evaluate_shoes([list("BPBPBPBPBPBP")], follow_last, min_history_bp=2)
        self.assertAlmostEqual(report["follow_last_prediction_rate"], 1.0)
        self.assertAlmostEqual(report["actual_same_rate"], 0.0)
        self.assertGreater(report["last_hand_echo_gap"], 0.9)
        self.assertEqual(report["semantics"], "prefix_only_walk_forward_no_future_outcomes_used")


if __name__ == "__main__":
    unittest.main()
